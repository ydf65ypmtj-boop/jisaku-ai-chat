"use strict";
/*
 * Cube（GitHub Pages版） script.js
 *
 * 全部ブラウザの中で動く版：
 * - 自作LLMの推論エンジン（Cube Nano / Flash は int8量子化、Cube Twin /
 *   Cube Pro / Cube Max はグループ単位 int4量子化。KVキャッシュ付き。
 *   Python版とNodeで10問全一致を検証済みのエンジンと同一計算）
 * - Cube Ultra は自作エンジンではなく、Hugging Face（FMA03/cube-ultra）
 *   のモデルを @huggingface/transformers（WebGPU）で動かす。ライブラリと
 *   モデルは、Ultra を初めて選んだときだけ遅延読み込みする。Ultra では
 *   エンジンを自動で開始せず、ステータスバーの選択ブロックから通常
 *   エンジン（ONNX）と軽量エンジン（GGUF版）のどちらかをユーザーが
 *   選んで読み込む。軽量エンジンは、同じモデルの GGUF 版を
 *   @wllama/wllama（llama.cpp の WASM 版）で動かす。GGUF 版は読み込み・
 *   生成の各フェーズを localStorage に記録し、タブが落ちた場合は次回に
 *   「どこで落ちたか」を具体的に通知できるようにする
 * - 回答の照合は、ブラウザから日本語Wikipedia APIを直接呼んで行う
 * - 知識メモ帳は、この端末のブラウザ内（localStorage）に保存する
 * 外部AIのAPIは使っていない。
 */

/* ================================================================== */
/* 推論エンジン（llm-engine.js と同一）                                */
/* ================================================================== */

// ---------- erf (Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7) ----------
function erf(x) {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return s * y;
}
function gelu(x) { return x * 0.5 * (1 + erf(x * Math.SQRT1_2)); }

// ---------- BPE (bpe.py と同一仕様) ----------
const SPLIT_RE = /[ぁ-んゝゞ]+|[ァ-ヶー]+|[一-鿿々]+|[a-zA-Z]+|[0-9]+|[^\s]/gu;
const PAD = 0, EOS = 1, UNK = 2;

class BPE {
  constructor(bpeJson) {
    this.vocab = bpeJson.vocab;                 // token string -> id
    this.inv = new Array(Object.keys(this.vocab).length);
    for (const tok of Object.keys(this.vocab)) this.inv[this.vocab[tok]] = tok;
    this.rank = new Map();
    bpeJson.merges.forEach((p, i) => this.rank.set(p[0] + "\u0000" + p[1], i));
  }
  pretokenize(text) { return text.match(SPLIT_RE) || []; }
  encodeWord(word) {
    let toks = Array.from(word).map(ch =>
      Object.prototype.hasOwnProperty.call(this.vocab, ch) ? ch : "<unk>");
    while (toks.length > 1) {
      let bestI = -1, bestRank = Infinity;
      for (let i = 0; i < toks.length - 1; i++) {
        const r = this.rank.get(toks[i] + "\u0000" + toks[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
      }
      if (bestI < 0) break;
      toks = toks.slice(0, bestI).concat([toks[bestI] + toks[bestI + 1]], toks.slice(bestI + 2));
    }
    return toks.map(t => this.vocab[t]);
  }
  encode(text) {
    const ids = [];
    const cache = new Map();
    for (const w of this.pretokenize(text)) {
      let got = cache.get(w);
      if (got === undefined) { got = this.encodeWord(w); cache.set(w, got); }
      for (const id of got) ids.push(id);
    }
    return ids;
  }
  decode(ids) {
    let out = "";
    for (const id of ids) if (id >= 3) { const s = this.inv[id]; if (s) out += s; }
    return out;
  }
}

// ---------- 量子化テンソル ----------
class QTensor {
  constructor(int8, scale, shape) { this.q = int8; this.scale = scale; this.shape = shape; }
  // y[o] = bias[o] + scale * sum_i q[o*in+i] * x[i]
  matvec(x, bias, out) {
    const q = this.q, s = this.scale;
    const nIn = this.shape[1], nOut = this.shape[0];
    for (let o = 0; o < nOut; o++) {
      const base = o * nIn;
      let acc = 0;
      for (let i = 0; i < nIn; i++) acc += q[base + i] * x[i];
      out[o] = (bias ? bias[o] : 0) + s * acc;
    }
    return out;
  }
  rowFloat(rowIdx, out) { // 埋め込み行を float で取り出す
    const n = this.shape[1], base = rowIdx * n, s = this.scale, q = this.q;
    for (let i = 0; i < n; i++) out[i] = q[base + i] * s;
    return out;
  }
}

function dequantSmall(t) { // LayerNorm等の小さなテンソルは float 化しておく
  // int4形式では1次元テンソルは fp32 のまま渡ってくる（raw.bin）。
  if (t instanceof Float32Array) return t;
  const out = new Float32Array(t.q.length);
  for (let i = 0; i < t.q.length; i++) out[i] = t.q[i] * t.scale;
  return out;
}

// ---------- fp16 デコード（int4形式のグループスケール用） ----------
/** IEEE 754 半精度1値を float に戻す。スケールは fp16 で保存されている。 */
function fp16ToFloat32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * (frac / 1024) * 2 ** -14;
  if (exp === 31) return frac === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

// ---------- int4 量子化テンソル（Cube Twin / Pro / Max） ----------
/**
 * グループ単位 int4 の2次元重み。packed はダウンロードしたシャードへの
 * ビューのまま保持し、fp32 へは絶対に展開しない（233Mパラメータを
 * 展開すると約932MBになり破綻する）。要素 (row, col) は全体 index
 * row*cols+col にあり、バイト (index>>1) の、偶数 index では下位
 * ニブル・奇数 index では上位ニブル（low-first）。実値は
 * (code - 8) * scales[row * groupsPerRow + col / groupSize]。
 * matvec はグループごとに内積へスケールを掛ける。
 */
class Int4Tensor {
  constructor(packed, scales, shape, groupSize) {
    this.packed = packed; // Uint8Array（ニブル詰め）
    this.scales = scales; // Float32Array（行×グループ数、fp16から復号済み）
    this.shape = shape;
    this.groupSize = groupSize;
    this.rows = shape[0];
    this.cols = shape[1];
    this.groupsPerRow = this.cols / groupSize;
    this.count = this.rows * this.cols;
  }
  // y[o] = bias[o] + sum_g scale[o,g] * sum_j (code-8) * x[g*group+j]
  matvec(x, bias, out) {
    const packed = this.packed, scales = this.scales;
    const nIn = this.cols, nOut = this.rows;
    const groupSize = this.groupSize, groupsPerRow = this.groupsPerRow;
    for (let o = 0; o < nOut; o++) {
      let sum = bias ? bias[o] : 0;
      const rowBase = o * nIn, scaleBase = o * groupsPerRow;
      for (let g = 0; g < groupsPerRow; g++) {
        const s = scales[scaleBase + g];
        if (s === 0) continue;
        const base = g * groupSize;
        let acc = 0;
        for (let j = 0; j < groupSize; j += 2) {
          const idx = rowBase + base + j;
          const byte = packed[idx >> 1];
          acc += x[base + j] * ((byte & 0xf) - 8) +
            x[base + j + 1] * (((byte >> 4) & 0xf) - 8);
        }
        sum += acc * s;
      }
      out[o] = sum;
    }
    return out;
  }
  rowFloat(rowIdx, out) { // 埋め込み行を float で取り出す（1行だけ展開する）
    const packed = this.packed, scales = this.scales;
    const cols = this.cols, groupSize = this.groupSize, groupsPerRow = this.groupsPerRow;
    const rowBase = rowIdx * cols;
    for (let g = 0; g < groupsPerRow; g++) {
      const s = scales[rowIdx * groupsPerRow + g];
      const base = g * groupSize;
      for (let j = 0; j < groupSize; j++) {
        const idx = rowBase + base + j;
        const byte = packed[idx >> 1];
        const code = idx % 2 === 0 ? byte & 0xf : (byte >> 4) & 0xf;
        out[base + j] = (code - 8) * s;
      }
    }
    return out;
  }
}

/** テンソルの要素数（形式によらず検証・集計に使う）。 */
function tensorElementCount(t) {
  if (t instanceof Float32Array) return t.length;
  if (t instanceof Int4Tensor) return t.count;
  if (t && t.q) return t.q.length;
  return 0;
}

function layernorm(x, w, b, out) {
  const n = x.length;
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varr = 0; for (let i = 0; i < n; i++) { const d = x[i] - mean; varr += d * d; }
  varr /= n;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// ---------- GPT 本体 (KVキャッシュ付き) ----------
class GPT {
  constructor(manifest, tensors) {
    this.cfg = manifest.config;
    this.T = tensors; // name -> QTensor
    const L = this.cfg.n_layer;
    this.layers = [];
    for (let i = 0; i < L; i++) {
      const p = `blocks.${i}.`;
      this.layers.push({
        ln1w: dequantSmall(tensors[p + "ln1.weight"]), ln1b: dequantSmall(tensors[p + "ln1.bias"]),
        qkv: tensors[p + "attn.qkv.weight"], qkvb: dequantSmall(tensors[p + "attn.qkv.bias"]),
        proj: tensors[p + "attn.proj.weight"], projb: dequantSmall(tensors[p + "attn.proj.bias"]),
        ln2w: dequantSmall(tensors[p + "ln2.weight"]), ln2b: dequantSmall(tensors[p + "ln2.bias"]),
        fc1: tensors[p + "mlp.fc.weight"], fc1b: dequantSmall(tensors[p + "mlp.fc.bias"]),
        fc2: tensors[p + "mlp.proj.weight"], fc2b: dequantSmall(tensors[p + "mlp.proj.bias"]),
        K: new Float32Array(this.cfg.block_size * this.cfg.n_embd),
        V: new Float32Array(this.cfg.block_size * this.cfg.n_embd),
      });
    }
    this.emb = tensors["tok_emb.weight"];
    this.pos = tensors["pos_emb.weight"];
    this.lnfw = dequantSmall(tensors["ln_f.weight"]);
    this.lnfb = dequantSmall(tensors["ln_f.bias"]);
    this.reset();
  }
  reset() { this.len = 0; }
  // 1トークン進めて最終 logits を返す
  step(id) {
    const C = this.cfg.n_embd, H = this.cfg.n_head, D = C / H, pos = this.len;
    const embRow = this.emb.rowFloat(id, new Float32Array(C));
    const posRow = this.pos.rowFloat(pos, new Float32Array(C));
    const x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = embRow[i] + posRow[i];
    const h = new Float32Array(C), qkv = new Float32Array(3 * C);
    const attnOut = new Float32Array(C), tmp = new Float32Array(C);
    const scale = 1 / Math.sqrt(D);
    for (const ly of this.layers) {
      layernorm(x, ly.ln1w, ly.ln1b, h);
      ly.qkv.matvec(h, ly.qkvb, qkv);
      ly.K.set(qkv.subarray(C, 2 * C), pos * C);
      ly.V.set(qkv.subarray(2 * C, 3 * C), pos * C);
      for (let hd = 0; hd < H; hd++) {
        const base = hd * D;
        const scores = new Float32Array(pos + 1);
        let mx = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let acc = 0; const kb = t * C + base;
          for (let j = 0; j < D; j++) acc += qkv[base + j] * ly.K[kb + j];
          scores[t] = acc * scale; if (scores[t] > mx) mx = scores[t];
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - mx); sum += scores[t]; }
        for (let j = 0; j < D; j++) {
          let acc = 0;
          for (let t = 0; t <= pos; t++) acc += (scores[t] / sum) * ly.V[t * C + base + j];
          attnOut[base + j] = acc;
        }
      }
      ly.proj.matvec(attnOut, ly.projb, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      layernorm(x, ly.ln2w, ly.ln2b, h);
      const u4 = new Float32Array(4 * C);
      ly.fc1.matvec(h, ly.fc1b, u4);
      for (let i = 0; i < 4 * C; i++) u4[i] = gelu(u4[i]);
      ly.fc2.matvec(u4, ly.fc2b, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    this.len++;
    layernorm(x, this.lnfw, this.lnfb, h);
    const V = this.cfg.vocab_size;
    const logits = new Float32Array(V);
    // lm_head は tok_emb と tied。matvec 経由にすることで、int8
    // （QTensor）でも int4（Int4Tensor）でも同じ経路で計算できる。
    // int8 の計算結果は従来の直接ループと同一になる。
    this.emb.matvec(h, null, logits);
    return logits;
  }
  // promptIds を処理し、続きを生成して生成分の id 配列を返す
  generate(promptIds, opts = {}) {
    const maxNew = opts.maxNew ?? 48;
    const greedy = opts.greedy ?? false;
    const temperature = opts.temperature ?? 0.8;
    const topK = opts.topK ?? 40;
    this.reset();
    let ids = promptIds.slice();
    if (ids.length > this.cfg.block_size) ids = ids.slice(ids.length - this.cfg.block_size);
    if (ids.length === 0) ids = [UNK];
    let logits = null;
    for (const id of ids) logits = this.step(id);
    const out = [];
    for (let n = 0; n < maxNew; n++) {
      if (this.len >= this.cfg.block_size) break;
      let nxt;
      if (greedy) {
        nxt = 0; let best = -Infinity;
        for (let v = 0; v < logits.length; v++) if (logits[v] > best) { best = logits[v]; nxt = v; }
      } else {
        nxt = sampleTopK(logits, temperature, topK);
      }
      if (nxt === EOS) break;
      out.push(nxt);
      logits = this.step(nxt);
    }
    return out;
  }
}

function sampleTopK(logits, temperature, topK) {
  const V = logits.length;
  const scaled = new Float32Array(V);
  for (let i = 0; i < V; i++) scaled[i] = logits[i] / Math.max(temperature, 1e-5);
  const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
  const keep = idx.slice(0, Math.min(topK, V));
  let mx = -Infinity; for (const i of keep) if (scaled[i] > mx) mx = scaled[i];
  let sum = 0; const probs = keep.map(i => { const p = Math.exp(scaled[i] - mx); sum += p; return p; });
  let r = Math.random() * sum;
  for (let j = 0; j < keep.length; j++) { r -= probs[j]; if (r <= 0) return keep[j]; }
  return keep[keep.length - 1];
}

// ---------- ローダ ----------
/**
 * 書き出した重みのMLPは学習側の連番名（blocks.N.mlp.0 / mlp.2）で
 * 保存されているが、推論側は mlp.fc / mlp.proj で引く。読み込み時に
 * どちらの綴りも fc/proj へ対応づける。これが無いと生成が必ず例外になる。
 */
function canonicalTensorName(name) {
  return name
    .replace(/^(blocks\.\d+\.mlp)\.0\./, "$1.fc.")
    .replace(/^(blocks\.\d+\.mlp)\.2\./, "$1.proj.");
}

/**
 * 推論が触る全テンソルと、モデル設定から決まる要素数。
 * 読み込み時にこれと照合し、名前が引けない・大きさが合わない重みは
 * 「読み込み完了」と言わず、その場で名前を挙げて失敗させる。
 */
function expectedTensors(cfg) {
  const C = cfg.n_embd;
  const expected = new Map([
    ["tok_emb.weight", cfg.vocab_size * C],
    ["pos_emb.weight", cfg.block_size * C],
    ["ln_f.weight", C],
    ["ln_f.bias", C],
  ]);
  for (let l = 0; l < cfg.n_layer; l++) {
    const p = `blocks.${l}.`;
    expected.set(`${p}ln1.weight`, C);
    expected.set(`${p}ln1.bias`, C);
    expected.set(`${p}attn.qkv.weight`, 3 * C * C);
    expected.set(`${p}attn.qkv.bias`, 3 * C);
    expected.set(`${p}attn.proj.weight`, C * C);
    expected.set(`${p}attn.proj.bias`, C);
    expected.set(`${p}ln2.weight`, C);
    expected.set(`${p}ln2.bias`, C);
    expected.set(`${p}mlp.fc.weight`, 4 * C * C);
    expected.set(`${p}mlp.fc.bias`, 4 * C);
    expected.set(`${p}mlp.proj.weight`, C * 4 * C);
    expected.set(`${p}mlp.proj.bias`, C);
  }
  return expected;
}

/**
 * fileBuffers から、テンソルが属するファイルのバイト列を引く。
 * fileBuffers は次のどれでもよい：ファイル名→Uint8Array の Map、
 * ファイル名キーのプレーンオブジェクト、manifest.files 順の配列。
 * int4形式では scales.bin / raw.bin もファイル名で引ける必要がある。
 */
function bufferForFile(fileBuffers, manifest, fileRef) {
  const name = typeof fileRef === "number" ? manifest.files[fileRef] : fileRef;
  if (!fileBuffers) return undefined;
  if (Array.isArray(fileBuffers)) {
    return typeof fileRef === "number"
      ? fileBuffers[fileRef]
      : fileBuffers[manifest.files.indexOf(fileRef)];
  }
  if (fileBuffers instanceof Map) return fileBuffers.get(name);
  return fileBuffers[name];
}

/**
 * テンソルのバイト列を読む（ファイルまたぎ＝span read 対応）。
 * 開始ファイルは bufferForFile で引き、offset + length がそのファイルに
 * 収まる場合は従来どおりのゼロコピーのビューを返す。収まらない場合のみ、
 * manifest.files の開始インデックスから後続ファイルへ順に連結して
 * length バイトを集める（Cube Max の weights-0.bin を GitHub の25MB
 * 上限に合わせて2分割し、tok_emb.weight が files[0]+files[1] に
 * またがる新レイアウトを読むため）。後続ファイルが尽きても足りなければ
 * エラーにする。またぎを含まない manifest では常にゼロコピー経路になる。
 */
function readTensorBytes(fileBuffers, manifest, fileRef, offset, length) {
  const buf = bufferForFile(fileBuffers, manifest, fileRef);
  if (!buf) throw new Error("テンソルのファイルがありません");
  if (offset + length <= buf.byteLength) {
    return new Uint8Array(buf.buffer, buf.byteOffset + offset, length);
  }
  const startIdx = typeof fileRef === "number"
    ? fileRef
    : manifest.files.indexOf(fileRef);
  if (startIdx < 0) {
    throw new Error(`テンソルの開始ファイルが manifest.files にありません（${fileRef}）`);
  }
  const out = new Uint8Array(length);
  let written = 0;
  for (let i = startIdx; written < length; i++) {
    const cur = i === startIdx
      ? buf
      : bufferForFile(fileBuffers, manifest, i);
    if (!cur) break;
    const curOff = i === startIdx ? offset : 0;
    const avail = cur.byteLength - curOff;
    if (avail <= 0) continue;
    const take = Math.min(avail, length - written);
    out.set(cur.subarray(curOff, curOff + take), written);
    written += take;
  }
  if (written < length) {
    throw new Error(
      `テンソルのデータがファイル範囲を超えています（${length} バイト必要ですが ${written} バイトしか読めませんでした）`,
    );
  }
  return out;
}

function buildEngine(manifest, bpeJson, fileBuffers) {
  const tensors = {};
  const isInt4 = manifest.format === "jisaku-llm-int4";
  if (!isInt4) {
    // int8形式（Cube Nano / Flash）：従来通りの読み込み。
    for (const t of manifest.tensors) {
      const buf = bufferForFile(fileBuffers, manifest, t.file);
      if (!buf) throw new Error(`テンソル ${t.name} のファイルがありません`);
      const u8 = readTensorBytes(fileBuffers, manifest, t.file, t.offset, t.length);
      const bytes = new Int8Array(u8.buffer, u8.byteOffset, u8.length);
      tensors[canonicalTensorName(t.name)] = new QTensor(bytes, t.scale, t.shape);
    }
  } else {
    // int4形式（Cube Twin / Pro / Max）：2次元重みはニブル詰めのまま保持し、
    // 1次元テンソルは raw.bin の fp32 をそのまま使う。
    if (!manifest.scales_file || !manifest.raw_file) {
      throw new Error("int4形式の manifest に scales_file / raw_file がありません");
    }
    const scalesBuf = bufferForFile(fileBuffers, manifest, manifest.scales_file);
    const rawBuf = bufferForFile(fileBuffers, manifest, manifest.raw_file);
    if (!scalesBuf || !rawBuf) {
      throw new Error("int4形式の scales / raw ファイルがありません");
    }
    const scalesView = new DataView(
      scalesBuf.buffer, scalesBuf.byteOffset, scalesBuf.byteLength,
    );
    for (const t of manifest.tensors) {
      const canon = canonicalTensorName(t.name);
      if (t.kind === "fp32") {
        const rawOff = t.raw_off ?? 0;
        const rawLen = t.raw_len ?? 0;
        let floats;
        if ((rawBuf.byteOffset + rawOff) % 4 === 0) {
          floats = new Float32Array(rawBuf.buffer, rawBuf.byteOffset + rawOff, rawLen / 4);
        } else {
          floats = new Float32Array(rawLen / 4);
          const view = new DataView(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength);
          for (let i = 0; i < floats.length; i++) {
            floats[i] = view.getFloat32(rawOff + i * 4, true);
          }
        }
        tensors[canon] = floats;
      } else if (t.kind === "int4") {
        const buf = bufferForFile(fileBuffers, manifest, t.file);
        if (!buf) throw new Error(`テンソル ${t.name} のファイルがありません`);
        const groupSize = t.group ?? 128;
        const scaleCount = (t.scale_shape ?? []).reduce((a, b) => a * b, 1);
        const scales = new Float32Array(scaleCount);
        const scaleOff = t.scale_off ?? 0;
        for (let i = 0; i < scaleCount; i++) {
          scales[i] = fp16ToFloat32(scalesView.getUint16(scaleOff + i * 2, true));
        }
        const packed = readTensorBytes(
          fileBuffers, manifest, t.file, t.offset ?? 0, t.packed_len ?? 0,
        );
        tensors[canon] = new Int4Tensor(packed, scales, t.shape, groupSize);
      } else {
        throw new Error(`テンソル ${t.name} の kind が不明です（${t.kind ?? "なし"}）`);
      }
    }
  }
  // 全テンソル検証：生成に必要な重みが全部そろい、大きさも合う時だけ通す。
  const problems = [];
  for (const [name, len] of expectedTensors(manifest.config)) {
    const t = tensors[name];
    if (!t) problems.push(`${name} がありません`);
    else if (tensorElementCount(t) !== len) {
      problems.push(`${name} の大きさが合いません（${tensorElementCount(t)} ≠ ${len}）`);
    } else if (t instanceof Int4Tensor && t.packed.length * 2 !== t.count) {
      problems.push(`${name} のパック済みデータ長が合いません（${t.packed.length * 2} ≠ ${t.count}）`);
    }
  }
  if (problems.length > 0) {
    throw new Error(
      `重みの検証に失敗しました: ${problems.slice(0, 4).join("、")}${
        problems.length > 4 ? ` ほか${problems.length - 4}件` : ""
      }`,
    );
  }
  return { bpe: new BPE(bpeJson), model: new GPT(manifest, tensors), manifest };
}

/* ------------------------------------------------------------------ */
/* 重みの端末キャッシュ（IndexedDB）                                   */
/*                                                                    */
/* 約54MBのシャードをブラウザのHTTPキャッシュだけに頼ると、開くたび   */
/* に再ダウンロードになる。シャードと bpe.json を IndexedDB に       */
/* 「ベースURL｜manifestの指紋｜ファイル名」で保存し、2回目以降は      */
/* ダウンロードなしで開く。manifest.json 自体は常に取り直し、その      */
/* 指紋が変われば（重みの上げ直し）古いキャッシュには届かなくなる。   */
/* IndexedDB が使えない環境では黙ってネットワークに落とす。           */
/* ------------------------------------------------------------------ */

const MODEL_CACHE_DB = "jisaku-ai-model-cache";
const MODEL_CACHE_STORE = "files";

/** FNV-1a ハッシュ：manifest本文の指紋（キャッシュの世代マーカー）。 */
function manifestFingerprint(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `${text.length}-${(h >>> 0).toString(36)}`;
}

function openModelCache() {
  return new Promise((resolve) => {
    try {
      if (typeof indexedDB === "undefined") {
        resolve(null);
        return;
      }
      const req = indexedDB.open(MODEL_CACHE_DB, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(MODEL_CACHE_STORE)) {
          req.result.createObjectStore(MODEL_CACHE_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function modelCacheGet(db, key) {
  return new Promise((resolve) => {
    try {
      const req = db
        .transaction(MODEL_CACHE_STORE, "readonly")
        .objectStore(MODEL_CACHE_STORE)
        .get(key);
      req.onsuccess = () => {
        const v = req.result;
        if (v instanceof ArrayBuffer || typeof v === "string") resolve(v);
        else if (typeof Blob !== "undefined" && v instanceof Blob) {
          v.arrayBuffer().then(resolve).catch(() => resolve(null));
        } else resolve(null);
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function modelCachePut(db, key, value) {
  return new Promise((resolve) => {
    try {
      const req = db
        .transaction(MODEL_CACHE_STORE, "readwrite")
        .objectStore(MODEL_CACHE_STORE)
        .put(value, key);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

/** 同じベースURLの、古い世代のキャッシュを掃除する。 */
function pruneModelCache(db, basePrefix, keepPrefix) {
  return new Promise((resolve) => {
    try {
      const store = db
        .transaction(MODEL_CACHE_STORE, "readwrite")
        .objectStore(MODEL_CACHE_STORE);
      const keysReq = store.getAllKeys();
      keysReq.onsuccess = () => {
        for (const k of keysReq.result) {
          if (
            typeof k === "string" &&
            k.startsWith(basePrefix) &&
            !k.startsWith(keepPrefix)
          ) {
            store.delete(k);
          }
        }
        resolve();
      };
      keysReq.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

/* ================================================================== */
/* モデル定義（Cube Nano / Flash / Twin / Pro / Max / Ultra 切り替え）*/
/*                                                                    */
/* Nano と Flash と Twin は語彙数こそ同じ 6000 だが BPE ファイルが    */
/* 別物でトークンIDに互換性がない。Pro と Max の BPE は語彙32,000      */
/* （bpe_v3）で、他の自作モデルとは互換性がない。そのため BPE は      */
/* 必ず選択中モデルのベースパスから読み込む（LocalModel.load は      */
/* モデルのベースURL配下からしか取らない）。manifest の期待値は        */
/* ファイル取り違え防止の照合  */
/* 用で、エンジンを構築する前に total_params・層数・幅が一致しなけ   */
/* れば読み込みエラーにする。重みの端末キャッシュ（IndexedDB）の     */
/* キーはベースURLを含むので、各モデルの重みが混ざることはない。      */
/*                                                                    */
/* Cube Ultra は例外：自作エンジンではなく、Hugging Face              */
/* （FMA03/cube-ultra）のモデルを @huggingface/transformers           */
/* （WebGPU）で動かす。ベースURL・manifest 照合・ベースURL上書きは    */
/* 自作モデル専用の仕組みなので Ultra には適用しない（engine が       */
/* "ultra" のモデルは LocalModel を一切通らない）。                   */
/* ================================================================== */

const MODEL_ID_KEY = "jisaku-ai-model-id";

const MODELS = {
  v1: {
    id: "v1",
    label: "Cube Nano",
    engine: "local",
    defaultBaseUrl: "https://ydf65ypmtj-boop.github.io/llm-chat/",
    urlKey: "jisaku-ai-model-base-url",
    maxNew: 48,
    expect: { totalParams: 31774848, nLayer: 12, nEmbd: 448 },
  },
  v2: {
    id: "v2",
    label: "Cube Flash",
    engine: "local",
    defaultBaseUrl: "https://ydf65ypmtj-boop.github.io/llm-chat/v2/",
    urlKey: "jisaku-ai-model-base-url-v2",
    maxNew: 96,
    expect: { totalParams: 59947008, nLayer: 18, nEmbd: 512 },
  },
  // Cube Twin：Flashの蒸留モデル（r6）と Flashの読解モデル（r9read2）を
  // 横並びに結合した1つのモデル（約2億3,314万パラメータ）を、
  // グループ単位 int4（グループ128）で書き出したもの。ホスト上の
  // 大きさは約121MB。生成設定は Flash と同じ（最大96トークン）。
  twin: {
    id: "twin",
    label: "Cube Twin",
    engine: "local",
    defaultBaseUrl: "https://ydf65ypmtj-boop.github.io/llm-chat/twin/",
    urlKey: "jisaku-ai-model-base-url-twin",
    maxNew: 96,
    expect: { totalParams: 233140224, nLayer: 18, nEmbd: 1024 },
  },
  // Cube Pro：v3モデル（約2億9,781万パラメータ、21層・幅1024、
  // 語彙32,000のBPE）の蒸留第3弾（r3。全ゲート合格で公開）。
  // Twin とまったく同じグループ単位 int4 形式（グループ128、
  // lm_head は tok_emb と tied）で書き出したものなので、読み込み
  // 経路は Twin と同一で、違うのはベースURL・manifest の照合値・
  // 生成上限だけ。生成設定は v3 の評価時と同一（greedy、最大48
  // トークン）。ホスト上の大きさは約156MB。
  pro: {
    id: "pro",
    label: "Cube Pro",
    engine: "local",
    defaultBaseUrl: "https://ydf65ypmtj-boop.github.io/llm-chat/pro/",
    urlKey: "jisaku-ai-model-base-url-pro",
    maxNew: 48,
    expect: { totalParams: 297815040, nLayer: 21, nEmbd: 1024 },
  },
  // Cube Max：v3モデル（語彙32,000の自作BPE）の蒸留 r1 と r2 を
  // 横並びに結合した1つのモデル（総パラメータ 1,189,648,384、
  // 21層・幅2048・32ヘッド）。全ゲート合格（回帰5/5・初見31/40・
  // 読解24/30・習得484/500）を受けて公開するもの。Pro・Twin と
  // まったく同じグループ単位 int4 形式（グループ128、lm_head は
  // tok_emb と tied）で書き出したものなので、読み込み経路は Pro と
  // 同一で、違うのはベースURL・manifest の照合値だけ。生成設定は
  // Pro と同じ（greedy、最大48トークン）。ホスト上の大きさは
  // 合計 582,679,390 バイト（約583MB）。
  max: {
    id: "max",
    label: "Cube Max",
    engine: "local",
    defaultBaseUrl: "https://ydf65ypmtj-boop.github.io/llm-chat/max/",
    urlKey: "jisaku-ai-model-base-url-max",
    maxNew: 48,
    expect: { totalParams: 1189648384, nLayer: 21, nEmbd: 2048 },
  },
  // Cube Ultra：Qwen2.5-1.5B に LoRA を統合したモデルを Hugging Face
  // （FMA03/cube-ultra）で公開したもの。自作エンジンではなく
  // @huggingface/transformers（WebGPU）で動かすため、ベースURL・
  // manifest 照合・ベースURL上書きは持たない（自作モデル専用の仕組み）。
  // 生成上限は Ultra 固定の 128 トークン（UltraModel 側で固定）。
  // 通常版の読み込みがタブ落下で中断されたときの予備経路として、
  // GGUF 版を @wllama/wllama で動かす軽量エンジンもある（UltraGgufModel）。
  ultra: {
    id: "ultra",
    label: "Cube Ultra",
    engine: "ultra",
    defaultBaseUrl: "",
    urlKey: "",
    maxNew: 128,
    expect: null,
  },
};

function modelDef(id) {
  return MODELS[id] || MODELS.v1;
}

/**
 * manifest が選択中モデルの期待値と一致するか検証する。
 * 一致すれば null、一致しなければ理由の文字列を返す。
 */
function validateManifest(manifest, def) {
  const cfg = manifest.config || {};
  const problems = [];
  if (manifest.total_params !== def.expect.totalParams) {
    problems.push(`パラメータ数 ${manifest.total_params} ≠ 期待値 ${def.expect.totalParams}`);
  }
  if (cfg.n_layer !== def.expect.nLayer) {
    problems.push(`層数 ${cfg.n_layer} ≠ 期待値 ${def.expect.nLayer}`);
  }
  if (cfg.n_embd !== def.expect.nEmbd) {
    problems.push(`幅 ${cfg.n_embd} ≠ 期待値 ${def.expect.nEmbd}`);
  }
  if (problems.length === 0) return null;
  return `${def.label} の重みではありません（${problems.join("、")}）。別のモデルのファイルが置かれている可能性があるため、読み込みを中止しました`;
}

/* モデル本体の薄いラッパ：状態表示つきで読み込み、質問文から生成する。
   保持するエンジンはアクティブな1つだけ。読み込みを始めた時点で前の
   エンジンは破棄するので、切り替え先の読み込みに失敗しても、前の
   モデルが新しいモデルの名前で回答することは絶対にない。「回答可能」
   の判定（readyFor）は、読み込み済みのモデルIDと選択中のIDが一致
   するときだけ真になる。 */
const LocalModel = {
  engine: null,
  loadedId: null, // 読み込み済みエンジンのモデルID（未読み込みは null）
  paramCount: 0,
  get ready() { return this.engine !== null; },
  readyFor(id) { return this.engine !== null && this.loadedId === id; },
  async load(baseUrl, modelId, onProgress) {
    const def = modelDef(modelId);
    // 切り替え開始時点で前のエンジンは破棄する。
    this.engine = null;
    this.loadedId = null;
    this.paramCount = 0;
    const base = baseUrl.replace(/\/?$/, "/");
    // manifest.json は常に取り直す（小さい）。その本文の指紋が重みの
    // 世代を決めるので、上げ直された重みを古いキャッシュで開くことはない。
    const manifestRes = await fetch(base + "manifest.json");
    if (!manifestRes.ok) {
      throw new Error(`manifest.json を取得できませんでした (HTTP ${manifestRes.status})`);
    }
    const manifestText = await manifestRes.text();
    const manifest = JSON.parse(manifestText);
    // ファイル取り違え防止：エンジンを構築する前に、manifest が
    // 選択中モデルの期待値と一致するか検証する。
    const mismatch = validateManifest(manifest, def);
    if (mismatch) throw new Error(mismatch);

    const cache = await openModelCache();
    const cachePrefix = `${base}|${manifestFingerprint(manifestText)}|`;
    const cacheKey = (name) => `${cachePrefix}${name}`;
    if (cache) void pruneModelCache(cache, `${base}|`, cachePrefix);

    // bpe.json：キャッシュにあればダウンロードしない。
    const bpeName = manifest.bpe_file || "bpe.json";
    let bpeText = null;
    if (cache) {
      const hit = await modelCacheGet(cache, cacheKey(bpeName));
      if (typeof hit === "string") bpeText = hit;
    }
    if (bpeText === null) {
      const r = await fetch(base + bpeName);
      if (!r.ok) throw new Error(`${bpeName} を取得できませんでした (HTTP ${r.status})`);
      bpeText = await r.text();
      if (cache) void modelCachePut(cache, cacheKey(bpeName), bpeText);
    }
    const bpeJson = JSON.parse(bpeText);

    // 取得するファイル：重みシャード一式に、int4形式ではグループ
    // スケールのファイルと fp32 の生ファイルも加える。
    const isInt4 = manifest.format === "jisaku-llm-int4";
    if (isInt4 && (!manifest.scales_file || !manifest.raw_file)) {
      throw new Error("int4形式の manifest に scales_file / raw_file がありません");
    }
    const fileNames = [...manifest.files];
    if (isInt4) fileNames.push(manifest.scales_file, manifest.raw_file);

    // シャード：キャッシュにあるものはそのまま使い、無いものだけ
    // バイト単位で進捗を数えながら読み、次回のために保存する。
    const totals = fileNames.map(() => 0);
    let loadedTotal = 0;
    const buffers = new Map();
    for (let i = 0; i < fileNames.length; i++) {
      const name = fileNames[i];
      if (cache) {
        const hit = await modelCacheGet(cache, cacheKey(name));
        if (hit instanceof ArrayBuffer) {
          const buf = new Uint8Array(hit);
          buffers.set(name, buf);
          totals[i] = buf.byteLength;
          loadedTotal += buf.byteLength;
          onProgress(loadedTotal, totals.reduce((a, b) => a + b, 0), `${name}（キャッシュから）`);
          continue;
        }
      }
      const r = await fetch(base + name);
      if (!r.ok) throw new Error(`${name} を取得できませんでした (HTTP ${r.status})`);
      const len = Number(r.headers.get("content-length") || 0);
      if (len > 0) totals[i] = len;
      if (!r.body || !r.body.getReader) {
        const buf = new Uint8Array(await r.arrayBuffer());
        buffers.set(name, buf);
        loadedTotal += totals[i] || buf.byteLength;
        onProgress(loadedTotal, totals.reduce((a, b) => a + b, 0), name);
        if (cache) {
          void modelCachePut(cache, cacheKey(name), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        }
        continue;
      }
      const reader = r.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.byteLength;
        onProgress(loadedTotal + got, totals.reduce((a, b) => a + b, 0), name);
      }
      const buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
      buffers.set(name, buf);
      loadedTotal += got;
      if (cache) {
        void modelCachePut(cache, cacheKey(name), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      }
    }
    this.engine = buildEngine(manifest, bpeJson, buffers);
    this.paramCount = manifest.total_params ||
      manifest.tensors.reduce(
        (a, t) => a + (t.length ?? (t.shape ?? []).reduce((x, y) => x * y, 1)),
        0,
      );
    // 試し生成：「読み込み完了」はファイルが届いたことではなく、
    // 実際に生成できること。ここで落ちる失敗は読み込みエラーとして出す。
    try {
      this.generate("テスト", 1);
    } catch (e) {
      this.engine = null;
      this.loadedId = null;
      throw new Error(
        `試し生成に失敗しました: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
    // ここまで来て初めて、このエンジンは選択中モデルのものになる。
    this.loadedId = def.id;
    return this.engine;
  },
  /* 学習時のQA形式「質問：…\n答え：」で貪欲生成し、答えの文字列を返す */
  generate(question, maxTokens = 48) {
    if (!this.engine) throw new Error("モデル未読み込み");
    const { bpe, model } = this.engine;
    const prompt = `質問：${question}\n答え：`;
    const pids = bpe.encode(prompt);
    const out = model.generate(pids, { greedy: true, maxNew: maxTokens });
    let text = bpe.decode(out);
    const cut = text.indexOf("質問：");
    if (cut >= 0) text = text.slice(0, cut);
    const nl = text.indexOf("\n");
    if (nl >= 0) text = text.slice(0, nl);
    return text.trim();
  },
};

/* ================================================================== */
/* Cube Ultra エンジン（@huggingface/transformers / WebGPU）           */
/*                                                                    */
/* Qwen2.5-1.5B に LoRA を統合したモデルを Hugging Face で公開した    */
/* FMA03/cube-ultra を、ブラウザの中で @huggingface/transformers v3   */
/* （3.8.1 に固定）＋ WebGPU で動かす。Ultra は自作エンジン           */
/* （LocalModel）を使わない：重みベースURL・manifest 照合・IndexedDB   */
/* 重みキャッシュはすべて transformers.js の pipeline 側が持つ。      */
/* ライブラリは ES モジュールとして、Ultra を初めて選んだときにだけ  */
/* 遅延読み込みする（他のモデルの読み込みでは読み込まない）。         */
/* 生成条件は検証済みのものと同一：固定の日本語システムプロンプト、   */
/* greedy（do_sample: false）、max_new_tokens 128。答えは             */
/* generated_text（メッセージ配列）の最後の assistant メッセージ。    */
/* pipeline は Ultra が選択中の間だけ保持し、他のモデルへ切り替えた   */
/* ら破棄する（アクティブなエンジンは常に1つだけ）。                  */
/*                                                                    */
/* ダウンロード完了直後のブラウザ再読み込み対策：約1.2GBの本体が      */
/* そろった直後は、ブラウザキャッシュへの保存コピーと WebGPU          */
/* セッション構築が重なってメモリ使用量が跳ね上がり、メモリの少ない   */
/* 端末（iPad 等）ではブラウザがタブごと再読み込みすることがある。     */
/* このページ自体はリロードを一切呼ばない。Ultra はエンジンを自動で    */
/* 開始しない（選択ブロックからユーザーが選ぶ）ので、落ちる→自動で    */
/* 読み直す→また落ちる、という繰り返しにはならないが、どこで落ちた    */
/* かを次回に伝えるため、読み込みの試行回数を sessionStorage に印と   */
/* して残す（タブを閉じれば消える）。1回落ちたら印が残り、次回は      */
/* 理由の表示と選択ブロックを出す。ユーザーが通常エンジンを選び直せ   */
/* ば、キャッシュ済みのファイルから再試行になる（保存コピーの重なり   */
/* が無く、ピークが低い）。2回続けて落ちたら（印が "build2"）自動の    */
/* 再試行を止め、理由をエラー表示にする。ユーザーがモデルを選び直し   */
/* たときだけ印を消し、最初から試せるようにする。                     */
/* ================================================================== */

const TRANSFORMERS_ESM_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm";

const ULTRA_REPO = "FMA03/cube-ultra";
const ULTRA_MAX_NEW_TOKENS = 128;
/* リポジトリの config.json 上の総パラメータ数（Qwen2：28層・幅1536・
   語彙151936・埋め込み共有）。表示専用で、manifest のような照合は
   Ultra には行わない。 */
const ULTRA_PARAM_COUNT = 1543714304;

const ULTRA_SYSTEM_PROMPT =
  "あなたは親切なAIアシスタントです。質問には日本語で分かりやすく答えてください。";

/* Ultra 読み込みの試行の印（sessionStorage）。値は通常版（ONNX）が
   "build"（1回目の試行中）→ "build2"（落ちた後の再試行中）、軽量
   エンジン（GGUF版）が "gguf"（1回目の試行中）→ "gguf2"（落ちた後の
   再試行中）。読み込みが成功するか、通常の例外で失敗したら消す。
   タブが落ちたときだけ印が残るので、次にページが開いたときに
   「前回は途中でブラウザが再読み込みされた」と分かる。sessionStorage
   はタブを閉じれば消えるので、端末に残り続けることはない。既存の
   "build" / "build2" の意味は変えない（旧バージョンが残した印でも
   壊れない）。 */
const ULTRA_STAGE_KEY = "cube-ultra-load-stage";
function readUltraStage() {
  try {
    return sessionStorage.getItem(ULTRA_STAGE_KEY);
  } catch {
    return null;
  }
}
function writeUltraStage(value) {
  try {
    if (value === null) sessionStorage.removeItem(ULTRA_STAGE_KEY);
    else sessionStorage.setItem(ULTRA_STAGE_KEY, value);
  } catch {
    /* 印が残せなくても読み込み自体は続ける */
  }
}

/* 読み込みに成功した Ultra のエンジン（"onnx" / "gguf"）の保存キー。
   次回エンジン選択ブロックを出すとき、「前回はどちらのエンジンで
   読み込めたか」を案内するために使う（store ラッパ経由）。 */
const ULTRA_ENGINE_KEY = "cube-ultra-engine";

/* GGUF 版のフェーズ記録（localStorage）。読み込み・生成のどの段階で
   タブが落ちたかを次回に具体的に伝えるためのもので、JSON
   {phase, pct, answered, reported} を残す。phase は
   "download"（ダウンロード中）→ "load"（ダウンロード完了・モデル
   読み込み中）→ "ready"（読み込み完了・未回答）→ "generating"
   （回答生成中）→ "answered"（1回以上回答できた）。pct はダウン
   ロード進捗（%）、answered は回答できた回数、reported は落下通知を
   既に出したかどうか。sessionStorage の試行の印と違い、タブを閉じ
   ても残る（読み込み完了後に落ちるケースはタブを跨いで検知する
   必要があるため）。記録に失敗しても読み込み自体は止めない。 */
const ULTRA_GGUF_PHASE_KEY = "cube-ultra-gguf-phase";
function readGgufPhase() {
  try {
    const raw = store.get(ULTRA_GGUF_PHASE_KEY);
    if (!raw) return null;
    const rec = JSON.parse(raw);
    if (!rec || typeof rec !== "object" || typeof rec.phase !== "string") return null;
    return rec;
  } catch {
    return null;
  }
}
function writeGgufPhase(rec) {
  try {
    if (rec === null) window.localStorage.removeItem(ULTRA_GGUF_PHASE_KEY);
    else store.set(ULTRA_GGUF_PHASE_KEY, JSON.stringify(rec));
  } catch {
    /* 記録できなくても読み込み自体は続ける */
  }
}

const UltraModel = {
  generator: null,
  generatorPromise: null,
  /* 前回の試行が途中でブラウザの再読み込みに遭い、今回がその
     再試行かどうか（表示の出し分けに使う）。 */
  recoveredFromCrash: false,
  get ready() { return this.generator !== null; },
  readyFor(id) { return id === "ultra" && this.generator !== null; },
  /* 前回の試行の印が残っているか（＝前回は読み込み完了前に
     ブラウザが再読み込みされた）。 */
  previousAttemptCrashed() { return readUltraStage() === "build"; },
  /**
   * FMA03/cube-ultra から pipeline を読み込む（dtype q4f16・device
   * webgpu）。progress_callback の進捗（ファイル名と、そのファイルの
   * ダウンロード済み／全体バイト数）を呼び出し側の読み込み表示へ流す。
   * 初回は onnx/model_q4f16.onnx（約1.23GB）を含む約1.2GBの
   * ダウンロードがある。失敗（WebGPU非対応・通信エラー等）は例外に
   * して、呼び出し側の読み込みエラー表示に渡す。
   * 読み込みの試行には sessionStorage の印（ULTRA_STAGE_KEY）を残し、
   * 途中でブラウザが再読み込みされたら次回に検知する。再試行でも
   * 落ちた場合は例外にして自動の再試行を止める（上のセクション
   * コメント参照）。
   */
  async load(onProgress) {
    if (this.generator) return;
    if (!this.generatorPromise) {
      const self = this;
      // 前回の試行の印を確認する。"build2" が残っているのは、
      // 再試行でもう一度タブが落ちたということなので、自動の
      // 再試行をここで止める（再読み込みの無限ループ防止）。
      // ユーザーがモデルを選び直すと印は消え、また試せる。
      const prevStage = readUltraStage();
      if (prevStage === "build2") {
        throw new Error(
          "Cube Ultra の起動中にブラウザが繰り返し再読み込みされたため、自動の再試行を止めました。" +
          "この端末ではメモリが足りず Ultra を起動できない可能性があります。" +
          "他のタブやアプリを閉じるか、他のモデルをお使いください。" +
          "他のモデルを選んでから Ultra を選び直すと、もう一度試せます",
        );
      }
      // "build" が残っている＝前回は読み込みの途中でブラウザが
      // 再読み込みされた。今回はキャッシュ済みのファイルから
      // 読み直す再試行になる（ダウンロードと保存コピーが無い分、
      // メモリのピークが低い）。
      const isRetry = prevStage === "build";
      this.recoveredFromCrash = isRetry;
      writeUltraStage(isRetry ? "build2" : "build");
      this.generatorPromise = (async () => {
        const mod = await import(TRANSFORMERS_ESM_URL);
        const gen = await mod.pipeline("text-generation", ULTRA_REPO, {
          dtype: "q4f16",
          device: "webgpu",
          progress_callback: (data) => {
            onProgress(
              data.loaded ?? 0,
              data.total ?? 0,
              data.file ?? data.name ?? ULTRA_REPO,
            );
          },
        });
        self.generator = gen;
        // 起動できたので印を消す。
        writeUltraStage(null);
        return gen;
      })();
      // 失敗した読み込みで次回の再試行まで塞がない。通常の例外で
      // 失敗したときも印を消す（印が残るのはタブが落ちたときだけ）。
      this.generatorPromise.catch(() => {
        this.generatorPromise = null;
        writeUltraStage(null);
      });
    }
    await this.generatorPromise;
  },
  /* アプリがモデルへ渡すプロンプト文字列から、Ultra の答えを生成する */
  async generate(prompt) {
    if (!this.generator) throw new Error("Cube Ultra が読み込まれていません");
    const messages = [
      { role: "system", content: ULTRA_SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ];
    const output = await this.generator(messages, {
      do_sample: false,
      max_new_tokens: ULTRA_MAX_NEW_TOKENS,
    });
    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1];
      if (
        last !== null &&
        typeof last === "object" &&
        "content" in last &&
        typeof last.content === "string"
      ) {
        return last.content.trim();
      }
      return "";
    }
    if (typeof generated === "string") return generated.trim();
    return "";
  },
  /* pipeline を破棄する（他のモデルが選択されたときに呼ぶ）。 */
  dispose() {
    const pending = this.generatorPromise;
    this.generator = null;
    this.generatorPromise = null;
    if (pending) {
      void pending
        .then((gen) => gen.dispose?.())
        .catch(() => {
          /* 破棄はベストエフォート */
        });
    }
  },
};

/* ================================================================== */
/* Cube Ultra 軽量エンジン（GGUF版 / @wllama/wllama）                  */
/*                                                                    */
/* 通常版（ONNX / transformers.js・約1.23GB）は、読み込み完了直後に   */
/* メモリ使用量が跳ね上がり、メモリの少ない端末（iPad Air 4 等）では  */
/* ブラウザがタブごと再読み込みしてしまうことがある。その予備経路と   */
/* して、同じモデル（FMA03/cube-ultra）の GGUF 版（Q4_K_M・           */
/* 986,048,320 バイト、ONNX 版との劣化ゼロを検証済み）を、llama.cpp   */
/* の WASM 版である @wllama/wllama 3.13.0 で動かす。ライブラリは      */
/* 通常版と同じ遅延読み込み方式で、軽量エンジンを始めるときだけ       */
/* CDN から動的 import する。GitHub Pages は COOP/COEP ヘッダを       */
/* 付けられないため、スレッドは単一（n_threads: 1）に固定する。       */
/* モデルのダウンロードと読み込みは wllama 標準のキャッシュを使う     */
/* ので、途中でタブが落ちても再ダウンロードにはならない（原則）。     */
/* 生成条件は通常版と完全同一：同じシステムプロンプトとユーザー発話を  */
/* ChatML 形式に組み立て（apply_chat_template 相当）、greedy          */
/* （temperature 0・top_k 1・seed 42）、最大128トークン、停止文字列は  */
/* <|im_end|> と <|endoftext|>。答えは choices[0].text を trim した    */
/* もの。通常版と軽量エンジンのどちらも自動では開始せず、Ultra を   */
/* 選んだ時点でステータスバーに出る選択ブロックからユーザーが選ぶ。 */
/* 試行の印は通常版と共用の ULTRA_STAGE_KEY に "gguf" / "gguf2" と   */
/* して残し、GGUF 版の落下が合計2回に達したら自動の再試行を止める。 */
/* さらに、読み込み・生成のどのフェーズで落ちたかを localStorage の  */
/* フェーズ記録（"cube-ultra-gguf-phase"）に残す：試行開始で         */
/* "download"、ダウンロード完了で "load"、読み込み成功で "ready"、   */
/* 生成開始で "generating"、生成成功で "answered"。読み込み完了後に  */
/* 落ちるケース（最初の生成時・常駐時）は stage の印では分からない   */
/* ため、この記録で次回に落下箇所を具体的に通知する。                 */
/* ================================================================== */

const WLLAMA_ESM_URL =
  "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.13.0/esm/index.js";
const WLLAMA_WASM_FROM_CDN_URL =
  "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.13.0/esm/wasm-from-cdn.js";
const ULTRA_GGUF_URL =
  "https://huggingface.co/FMA03/cube-ultra/resolve/main/cube-ultra-q4_k_m.gguf";
const ULTRA_GGUF_FILE_LABEL = "cube-ultra-q4_k_m.gguf";

const UltraGgufModel = {
  wllama: null,
  loadPromise: null,
  /* 前回の GGUF 試行が途中でブラウザの再読み込みに遭い、今回がその
     再試行かどうか（表示の出し分けに使う）。 */
  recoveredFromCrash: false,
  get ready() { return this.wllama !== null; },
  readyFor(id) { return id === "ultra" && this.wllama !== null; },
  /* 前回の GGUF 試行の印が残っているか（＝前回は GGUF の読み込み
     完了前にブラウザが再読み込みされた）。 */
  previousAttemptCrashed() { return readUltraStage() === "gguf"; },
  /**
   * FMA03/cube-ultra の GGUF 版を wllama で読み込む（n_ctx 2048・
   * n_threads 1）。progressCallback の loaded / total（全シャードの
   * 合計バイト数）を呼び出し側の読み込み表示へ流す。試行の印
   * （ULTRA_STAGE_KEY）の扱いは通常版と同じ：開始前に "gguf"（再試行
   * は "gguf2"）を残し、成功か通常の例外で失敗したら消す。再試行でも
   * 落ちていた（印が "gguf2"）場合は例外にして自動の再試行を止める。
   */
  async load(onProgress) {
    if (this.wllama) return;
    if (!this.loadPromise) {
      const self = this;
      const prevStage = readUltraStage();
      if (prevStage === "gguf2") {
        throw new Error(
          "Cube Ultra（軽量エンジン）の起動中にブラウザが繰り返し再読み込みされたため、自動の再試行を止めました。" +
          "この端末ではメモリが足りず Ultra を起動できない可能性があります。" +
          "他のタブやアプリを閉じるか、他のモデルをお使いください。" +
          "他のモデルを選んでから Ultra を選び直すと、もう一度試せます",
        );
      }
      // "gguf" が残っている＝前回は GGUF の読み込みの途中でブラウザが
      // 再読み込みされた。今回は wllama のキャッシュから読み直す
      // 再試行になる（再ダウンロードは原則不要）。
      // 通常版の印（"build" / "build2"）が残っている状態でボタンが
      // 押された場合は、GGUF の1回目の試行として "gguf" から始める。
      const isRetry = prevStage === "gguf";
      this.recoveredFromCrash = isRetry;
      writeUltraStage(isRetry ? "gguf2" : "gguf");
      // フェーズ記録：この試行はダウンロードから始まる。タブが落ちたら
      // この記録が残り、次回に「どこで落ちたか」を通知できる。
      writeGgufPhase({ phase: "download", pct: 0, answered: 0, reported: false });
      this.loadPromise = (async () => {
        const [mod, wasmMod] = await Promise.all([
          import(WLLAMA_ESM_URL),
          import(WLLAMA_WASM_FROM_CDN_URL),
        ]);
        const wllama = new mod.Wllama(wasmMod.default, {
          parallelDownloads: 1,
        });
        // 読み込みは loadModelFromUrl に任せる：wllama が OPFS
        // キャッシュ経由でストリーミング読み込みするので、986MB を
        // JS の ArrayBuffer / Blob に丸ごと保持しない最も軽い経路に
        // なる。手動 fetch + loadModel([blob]) のような二重保持に
        // なる方式には変えないこと。
        await wllama.loadModelFromUrl(ULTRA_GGUF_URL, {
          n_ctx: 2048,
          n_threads: 1,
          progressCallback: ({ loaded, total }) => {
            // フェーズ記録を進捗に合わせて更新する。loaded が total に
            // 達したらダウンロード完了＝モデルの読み込み中（"load"）。
            // キャッシュ済みの場合も wllama は loaded = total で1回
            // 呼ぶため、この判定でダウンロード完了を検知できる。
            const pct = total > 0 ? Math.floor((loaded / total) * 100) : 0;
            const done = total > 0 && loaded >= total;
            const cur = readGgufPhase();
            writeGgufPhase({
              phase: done ? "load" : "download",
              pct,
              answered: cur && typeof cur.answered === "number" ? cur.answered : 0,
              reported: cur ? !!cur.reported : false,
            });
            onProgress(loaded ?? 0, total ?? 0, ULTRA_GGUF_FILE_LABEL);
          },
        });
        self.wllama = wllama;
        // 起動できたので印を消し、フェーズは「読み込み完了・未回答」
        // に進める（この後にタブが落ちたら、生成前の落下と分かる）。
        writeUltraStage(null);
        writeGgufPhase({ phase: "ready", pct: 100, answered: 0, reported: false });
        return wllama;
      })();
      // 失敗した読み込みで次回の再試行まで塞がない。通常の例外で
      // 失敗したときも印とフェーズ記録を消す（両方が残るのはタブが
      // 落ちたときだけ）。
      this.loadPromise.catch(() => {
        this.loadPromise = null;
        writeUltraStage(null);
        writeGgufPhase(null);
      });
    }
    await this.loadPromise;
    // 成功が確定したら、読み込み約束への保持参照は手放す（モデル実体は
    // this.wllama が保持する）。余分な参照を残さないための最小化。
    this.loadPromise = null;
  },
  /* アプリがモデルへ渡すプロンプト文字列から、GGUF 版の答えを生成する。
     プロンプトの組み立てと生成条件は通常版と完全同一（上のセクション
     コメント参照）。 */
  async generate(prompt) {
    if (!this.wllama) throw new Error("Cube Ultra（軽量エンジン）が読み込まれていません");
    // フェーズ記録：生成の開始を残す。生成の途中でタブが落ちたら、
    // 次回に「回答生成の途中で落ちた」と通知できる（answered は維持）。
    const before = readGgufPhase();
    const answeredBefore = before && typeof before.answered === "number" ? before.answered : 0;
    writeGgufPhase({
      phase: "generating",
      pct: 100,
      answered: answeredBefore,
      reported: before ? !!before.reported : false,
    });
    const chatml =
      `<|im_start|>system\n${ULTRA_SYSTEM_PROMPT}<|im_end|>\n` +
      `<|im_start|>user\n${prompt}<|im_end|>\n` +
      `<|im_start|>assistant\n`;
    let res;
    try {
      res = await this.wllama.createCompletion({
        prompt: chatml,
        max_tokens: ULTRA_MAX_NEW_TOKENS,
        temperature: 0,
        temp: 0,
        top_k: 1,
        seed: 42,
        stop: ["<|im_end|>", "<|endoftext|>"],
      });
    } catch (e) {
      // 通常の例外で失敗したときは、落下と誤認しないようフェーズを
      // 生成前の状態へ戻してから投げ直す。
      writeGgufPhase({
        phase: answeredBefore > 0 ? "answered" : "ready",
        pct: 100,
        answered: answeredBefore,
        reported: false,
      });
      throw e;
    }
    // 生成できた：フェーズを "answered" に進め、回答できた回数を増やす。
    writeGgufPhase({
      phase: "answered",
      pct: 100,
      answered: answeredBefore + 1,
      reported: false,
    });
    const text = res?.choices?.[0]?.text;
    return typeof text === "string" ? text.trim() : "";
  },
  /* wllama を破棄する（他のモデルが選択されたときに呼ぶ）。 */
  dispose() {
    const pending = this.loadPromise;
    this.wllama = null;
    this.loadPromise = null;
    if (pending) {
      void pending
        .then((wllama) => wllama.exit())
        .catch(() => {
          /* 破棄はベストエフォート */
        });
    }
  },
};

/* ================================================================== */
/* Cube Ultra iMacサーバーエンジン                                      */
/*                                                                    */
/* 自宅の iMac（tailnet 内）で動く llama-server（GGUF Q4_K_M・同じ     */
/* モデル）を、Tailscale Serve の HTTPS 経由で呼ぶ第3のエンジン。      */
/* 端末側にモデルを読み込まないので、メモリの少ない端末でも Ultra を   */
/* 使える。生成条件は他のエンジンと同一（同じシステムプロンプト、       */
/* temperature 0・top_k 1・seed 42、最大128トークン）。公開範囲は      */
/* tailnet 内のみ（Tailscale Serve。Funnel の全体公開ではない）ため、  */
/* このエンジンが使えるのはユーザーの端末からだけ。iMac が起動して    */
/* いない・スリープしている等で届かない場合は、例外にして素直な文面で  */
/* 知らせ、エンジン選択に戻れるようにする。                           */
/* ================================================================== */

const ULTRA_IMAC_BASE_URL = "https://imac.tail9ed76a.ts.net";
const ULTRA_IMAC_CONNECT_TIMEOUT_MS = 30000;
const ULTRA_IMAC_GENERATE_TIMEOUT_MS = 120000;
const ULTRA_IMAC_UNREACHABLE =
  "iMacに届きませんでした。iMacが起動しているか確認してください";

/* fetch にタイムアウトを付ける（AbortController）。時間切れ・到達不能は
   呼び出し側でまとめて「iMacに届かない」として扱う。 */
async function fetchWithTimeout(url, options, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

const UltraImacModel = {
  connected: false,
  get ready() { return this.connected; },
  readyFor(id) { return id === "ultra" && this.connected; },
  /* 「読み込み」は iMac への到達確認だけ（端末へのダウンロードは無い）。
     llama-server の /health が 200 を返したら接続済みにする。 */
  async load(onProgress) {
    if (this.connected) return;
    onProgress(0, 0, "iMacサーバー（接続を確認しています）");
    let res;
    try {
      res = await fetchWithTimeout(
        `${ULTRA_IMAC_BASE_URL}/health`,
        { method: "GET" },
        ULTRA_IMAC_CONNECT_TIMEOUT_MS,
      );
    } catch {
      throw new Error(ULTRA_IMAC_UNREACHABLE);
    }
    if (!res || !res.ok) throw new Error(ULTRA_IMAC_UNREACHABLE);
    this.connected = true;
  },
  /* アプリがモデルへ渡すプロンプト文字列から、iMac 上の llama-server
     （OpenAI 互換 /v1/chat/completions）で答えを生成する。メッセージと
     生成条件は他のエンジンと同一。失敗したら接続を切り、素直な文面の
     例外にする（回答フロー側が注記として表示する）。 */
  async generate(prompt) {
    if (!this.connected) throw new Error("Cube Ultra（iMacサーバー）が接続されていません");
    let res;
    try {
      res = await fetchWithTimeout(
        `${ULTRA_IMAC_BASE_URL}/v1/chat/completions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [
              { role: "system", content: ULTRA_SYSTEM_PROMPT },
              { role: "user", content: prompt },
            ],
            temperature: 0,
            top_k: 1,
            seed: 42,
            max_tokens: ULTRA_MAX_NEW_TOKENS,
          }),
        },
        ULTRA_IMAC_GENERATE_TIMEOUT_MS,
      );
    } catch {
      this.connected = false;
      throw new Error(ULTRA_IMAC_UNREACHABLE);
    }
    if (!res || !res.ok) {
      this.connected = false;
      throw new Error(ULTRA_IMAC_UNREACHABLE);
    }
    const data = await res.json().catch(() => null);
    const text = data?.choices?.[0]?.message?.content;
    return typeof text === "string" ? text.trim() : "";
  },
  /* 接続を切る（他のモデルや他のエンジンが選択されたときに呼ぶ）。 */
  dispose() {
    this.connected = false;
  },
};

async function loadUltra(onProgress) {
  return UltraModel.load(onProgress);
}

async function loadUltraGguf(onProgress) {
  return UltraGgufModel.load(onProgress);
}

async function loadUltraImac(onProgress) {
  return UltraImacModel.load(onProgress);
}

/* Ultra の生成：このセッションで軽量エンジン（GGUF版）が読み込み
   済みならそちらを使い続け、それ以外は通常版（ONNX）を使う。回答
   フロー・バッジ・出典表示はどちらでも同一に扱う。 */
async function generateUltra(prompt) {
  if (UltraImacModel.ready) return UltraImacModel.generate(prompt);
  return UltraGgufModel.ready
    ? UltraGgufModel.generate(prompt)
    : UltraModel.generate(prompt);
}

/* Ultra のエンジンをすべて破棄する（他のモデルが選択されたとき）。 */
function disposeUltra() {
  UltraModel.dispose();
  UltraGgufModel.dispose();
  UltraImacModel.dispose();
}

/**
 * 「回答可能」の判定：読み込み完了したエンジンが選択中モデルと一致
 * するときだけ真。自作モデルは LocalModel、Ultra は UltraModel
 * （通常版）か UltraGgufModel（軽量エンジン）のどちらかが読み込み
 * 済みなら真。それぞれ判定し、切り替え先の読み込みに失敗した状態で
 * 前のモデルが新しいモデルの名前で回答することは絶対にない。
 */
function modelReadyFor(id) {
  return id === "ultra"
    ? UltraModel.readyFor(id) || UltraGgufModel.readyFor(id) || UltraImacModel.readyFor(id)
    : LocalModel.readyFor(id);
}

/* ================================================================== */
/* 回答コア：正規化・検算・照合・種類つき事実抽出                        */
/*                                                                    */
/* 事実抽出は「文の貼り付け」ではない。質問を事実の種類（作者／本社／    */
/* 日付／数値／場所／人物／定義）に分類し、その種類の置き場（対象記事の  */
/* インフォボックスの欄、または定義文のパターン）から取れた値だけを      */
/* 決まった文の形に組み立てて答えにする。取れなければ「確認できない」。  */
/* ================================================================== */

/** 知識メモ帳の照合用に質問を正規化する（空白・句読点を除く）。 */
function normalizeQuestion(q) {
  return q
    .replace(/[\s　]/g, "")
    .replace(/[、。・,.!?！？:：;；「」『』（）()［］""]/g, "")
    .toLowerCase();
}

/**
 * 繰り返し符号（ゝゞヽヾ）を展開し、こゝろ のような旧仮名表記を
 * こころ と同一視できるようにする。
 */
function normalizeJa(s) {
  const dakuten = {
    か: "が", き: "ぎ", く: "ぐ", け: "げ", こ: "ご",
    さ: "ざ", し: "じ", す: "ず", せ: "ぜ", そ: "ぞ",
    た: "だ", ち: "ぢ", つ: "づ", て: "で", と: "ど",
    は: "ば", ひ: "び", ふ: "ぶ", へ: "べ", ほ: "ぼ", う: "ゔ",
    カ: "ガ", キ: "ギ", ク: "グ", ケ: "ゲ", コ: "ゴ",
    サ: "ザ", シ: "ジ", ス: "ズ", セ: "ゼ", ソ: "ゾ",
    タ: "ダ", チ: "ヂ", ツ: "ヅ", テ: "デ", ト: "ド",
    ハ: "バ", ヒ: "ビ", フ: "ブ", ヘ: "ベ", ホ: "ボ", ウ: "ヴ",
  };
  let out = "";
  for (const ch of s) {
    if (ch === "ゝ" || ch === "ヽ") {
      out += out.length > 0 ? out.charAt(out.length - 1) : "";
    } else if (ch === "ゞ" || ch === "ヾ") {
      const prev = out.length > 0 ? out.charAt(out.length - 1) : "";
      out += dakuten[prev] ?? prev;
    } else {
      out += ch;
    }
  }
  return out;
}

/** 質問文中の『…』で名指しされた作品・対象。なければ null。 */
function subjectOf(question) {
  const m = question.match(/『(.+?)』/);
  const s = m?.[1]?.trim();
  return s ? s : null;
}

/** 記事タイトルがその対象を指すか（「 (小説)」などの接尾を外して比べる）。 */
function titleMatchesSubject(title, subject) {
  const normTitle = normalizeJa(title);
  const subj = normalizeJa(subject);
  if (!normTitle || !subj) return false;
  if (normTitle === subj) return true;
  const base = normTitle.replace(/[（(].*?[）)]/g, "").trim();
  if (base === subj) return true;
  return base.length >= 2 && subj.startsWith(base);
}

/**
 * 『…』の質問では、その作品自身の記事か『…』を引く記事の証拠だけを残す。
 * 同名の別人・別作品（『かがみの孤城』のこころ等）で「確認済み」にしないため。
 */
function scopeToSubject(question, results) {
  const subject = subjectOf(question);
  if (!subject) return results;
  const bracketed = `『${normalizeJa(subject)}』`;
  return results.filter(
    (r) =>
      titleMatchesSubject(r.title, subject) ||
      normalizeJa(r.text).includes(bracketed) ||
      normalizeJa(r.snippet).includes(bracketed),
  );
}

function toHalfWidth(s) {
  return s
    .replace(/[０-９]/g, (c) =>
      String.fromCharCode(c.charCodeAt(0) - 0xfee0),
    )
    .replace(/＋/g, "+")
    .replace(/－/g, "-")
    .replace(/×/g, "*")
    .replace(/÷/g, "/")
    .replace(/（/g, "(")
    .replace(/）/g, ")");
}

/**
 * 単純な計算の質問を検出して正確に計算する。
 * 計算でなければ null。
 */
function evalArithmetic(question) {
  const s = toHalfWidth(question);
  const m = /[\d.]+(?:\s*[+\-*/]\s*[\d.]+)+/.exec(s);
  if (!m) return null;
  let expr = m[0].replace(/\s+/g, "");
  const paren = /\([\d.+\-*/]+\)/.exec(s);
  if (paren && paren[0].length > expr.length) expr = paren[0];
  if (!/^[0-9.+\-*/()]+$/.test(expr)) return null;
  try {
    const matched = expr.match(/(\d+\.?\d*|[+\-*/()])/g);
    if (!matched) return null;
    const tokens = matched;
    let pos = 0;
    const peek = () => tokens[pos];
    function parseExpr() {
      let v = parseTerm();
      while (peek() === "+" || peek() === "-") {
        const op = tokens[pos++];
        const rhs = parseTerm();
        v = op === "+" ? v + rhs : v - rhs;
      }
      return v;
    }
    function parseTerm() {
      let v = parseFactor();
      while (peek() === "*" || peek() === "/") {
        const op = tokens[pos++];
        const rhs = parseFactor();
        v = op === "*" ? v * rhs : v / rhs;
      }
      return v;
    }
    function parseFactor() {
      const t = tokens[pos++];
      if (t === undefined) throw new Error("eof");
      if (t === "(") {
        const v = parseExpr();
        if (tokens[pos++] !== ")") throw new Error("paren");
        return v;
      }
      if (t === "-") return -parseFactor();
      const n = Number(t);
      if (Number.isNaN(n)) throw new Error("nan");
      return n;
    }
    const value = parseExpr();
    if (pos !== tokens.length || !Number.isFinite(value)) return null;
    return Math.round(value * 1e10) / 1e10;
  } catch {
    return null;
  }
}

"use strict";
/* calc_lane.js — Faithful JavaScript port of ~/workspace/hybrid/calc_lane.py
 * (hybrid system lane 3: pattern-based exact computation for Japanese
 * arithmetic questions). Returns {answer, steps, kind[, expr|remainder]}
 * or null when no pattern matches (the lane abstains).
 *
 * Port notes (behavior on the probe set is byte-identical to Python):
 * - Python's eval() for the restricted expression grammar is replaced by
 *   a two-pass evaluator (no eval): * and / bind tighter than + and -,
 *   left-associative, exactly like Python. Division by zero -> null
 *   (Python raises -> None). A numeric token with a leading zero such as
 *   "007" -> null (Python eval raises SyntaxError on such literals).
 * - Python's \d is Unicode-aware; JS \d is ASCII-only. All patterns run
 *   on norm()ed text (full-width digits are translated to ASCII first),
 *   except solve_sub's payment regex, which Python applies to the raw
 *   text: it is applied to norm(text) here so full-width digits behave
 *   the same. Keyword checks stay on the raw text, as in Python.
 * - Number formatting: integral floats are converted to integers before
 *   stringifying (same as Python's `val.is_integer()` -> int()).
 *
 * Exposes globalThis.CalcLane = { solve } (solveCalc). No module.exports
 * side effects, so the file can be embedded verbatim into Cube's
 * script.js and also loaded directly from Node for testing.
 */
const CalcLane = (() => {
  const ZEN_MAP = {
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "×": "×", "÷": "÷", "＊": "*", "／": "/", "＝": "=",
  };

  function norm(s) {
    let out = "";
    for (const ch of String(s)) out += ZEN_MAP[ch] ?? ch;
    return out.replace(/ /g, "").replace(/　/g, "").replace(/,/g, "");
  }

  function nums(text) {
    return (norm(text).match(/\d+/g) ?? []).map((x) => parseInt(x, 10));
  }

  const MINUS = "−"; // U+2212 (as in the Python f-strings)
  const FEQ = "＝"; // U+FF1D full-width equals
  const MUL = "×"; // U+00D7
  const DIV = "÷"; // U+00F7

  function subSteps(a, b) {
    if (b === 0) return `${a}${MINUS}0${FEQ}${a}`;
    const parts = [];
    let rem = b;
    for (const unit of [1000, 100, 10, 1]) {
      const d = Math.floor(rem / unit) * unit;
      if (d) parts.push(d);
      rem -= d;
    }
    const steps = [`${a}${MINUS}${b}を計算する。`];
    let cur = a;
    for (const p of parts) {
      steps.push(`${cur}${MINUS}${p}${FEQ}${cur - p}。`);
      cur -= p;
    }
    return steps.join("");
  }

  function divAnswer(a, b) {
    const q = Math.floor(a / b);
    const r = a % b;
    let steps = `${a}${DIV}${b}${FEQ}${q}`;
    if (r) steps += ` あまり${r}`;
    return [q, r, steps];
  }

  const COUNTERS = "個|枚|本|つ|冊|台|匹|人|円";

  function mulResult(a, b, unit, label = "全部で") {
    const ans = a * b;
    let steps = `${a}${MUL}${b}${FEQ}${ans}。`;
    if (label) steps += `${label}${ans}${unit}。`;
    return { answer: ans, steps, kind: "mul" };
  }

  function solveMul(text) {
    const t = norm(text);
    // direct: "6かける7は？" ("6×7" is handled by solveExpression)
    let m = new RegExp("(\\d+)かける(\\d+)").exec(t);
    if (m) return mulResult(parseInt(m[1], 10), parseInt(m[2], 10), "", "");
    // per-container: "1箱にボールが6個入っています。7箱では全部で何個？"
    if (t.includes("全部で") || t.includes("何個") || t.includes("いくつ")) {
      m = new RegExp(
        "1(.{1,6}?)に[^0-9]*?(\\d+)\\s*(" + COUNTERS + ")[^0-9]*?(\\d+)\\1",
      ).exec(t);
      if (m) {
        const per = parseInt(m[2], 10);
        const unit = m[3];
        const cnt = parseInt(m[4], 10);
        return mulResult(per, cnt, unit);
      }
    }
    // per-person handout: "1人に3個ずつ5人に配ると全部で何個？"
    m = new RegExp("(\\d+)\\s*(" + COUNTERS + ")ずつ\\D{0,8}?(\\d+)人").exec(t);
    if (m) return mulResult(parseInt(m[1], 10), parseInt(m[3], 10), m[2]);
    // same handout, people first: "2人に4枚ずつ配ると全部で何枚？"
    m = new RegExp("(\\d+)人に(\\d+)\\s*(" + COUNTERS + ")ずつ").exec(t);
    if (m) return mulResult(parseInt(m[2], 10), parseInt(m[1], 10), m[3]);
    return null;
  }

  // Two-pass evaluator for ^\d+(op\d+)+$ with Python eval semantics.
  function evalExpr(expr) {
    const tokens = expr.match(/\d+|[+\-×÷*/]/g);
    if (!tokens) return null;
    const values = [];
    const ops = [];
    for (let i = 0; i < tokens.length; i++) {
      if (i % 2 === 0) {
        if (/^0\d/.test(tokens[i])) return null; // Python: SyntaxError
        values.push(parseInt(tokens[i], 10));
      } else {
        ops.push(tokens[i]);
      }
    }
    // pass 1: × ÷ (and their ASCII forms * /)
    const v2 = [values[0]];
    const o2 = [];
    for (let i = 0; i < ops.length; i++) {
      const op = ops[i];
      const rhs = values[i + 1];
      if (op === "×" || op === "*") {
        v2[v2.length - 1] = v2[v2.length - 1] * rhs;
      } else if (op === "÷" || op === "/") {
        if (rhs === 0) return null; // Python: ZeroDivisionError
        v2[v2.length - 1] = v2[v2.length - 1] / rhs;
      } else {
        o2.push(op);
        v2.push(rhs);
      }
    }
    // pass 2: + -
    let val = v2[0];
    for (let i = 0; i < o2.length; i++) {
      val = o2[i] === "+" ? val + v2[i + 1] : val - v2[i + 1];
    }
    return val;
  }

  function fmtVal(val) {
    // Python: float with integral value -> int before str()
    return Number.isInteger(val) ? String(Math.trunc(val)) : String(val);
  }

  function solveExpression(text) {
    const t = norm(text);
    const m = /(\d+(?:[+\-×÷*/]\d+)+)/.exec(t);
    if (!m) return null;
    const expr = m[1];
    const val = evalExpr(expr);
    if (val === null || !Number.isFinite(val)) return null;
    const shown = expr.replace(/\*/g, "×").replace(/\//g, "÷");
    const v = Number.isInteger(val) ? Math.trunc(val) : val;
    return { answer: v, steps: `${shown}${FEQ}${fmtVal(val)}`, kind: "expr", expr: shown };
  }

  function solveSub(text) {
    const t = text;
    const ns = nums(t);
    if (ns.length < 2) return null;
    const kw = ["おつり", "残り", "差は", "差いくつ", "引く", "違い",
      "よりいくつ大きい", "より大きい"].some((k) => t.includes(k));
    if (!kw) return null;
    let a = ns[0];
    let b = ns[1];
    if (t.includes("より")) {
      a = Math.max(...ns);
      b = Math.min(...ns);
      const ans = a - b;
      return { answer: ans, steps: subSteps(a, b) + `差は${ans}。`, kind: "sub" };
    }
    // price first, payment second: "670円のおもちゃを買って2000円を出した"
    // (Python applies this regex to the raw text; JS \d is ASCII-only, so
    // it runs on norm(text) — see the port notes at the top.)
    if (/買って\d+円を出し|円札で|を出した/.test(norm(t))) {
      const price = ns[0];
      const pay = ns[ns.length - 1];
      if (pay >= price) {
        a = pay;
        b = price;
      }
    } else if (t.includes("差")) {
      a = Math.max(...ns);
      b = Math.min(...ns);
    }
    const ans = a - b;
    const unit = t.includes("円") ? "円" : t.includes("人") ? "人" : "";
    const label = t.includes("おつり") ? "おつり" : t.includes("残り") ? "残り" : "差";
    return {
      answer: ans,
      steps: subSteps(a, b) + `${label}は${ans}${unit}。`,
      kind: "sub",
    };
  }

  function solveDiv(text) {
    const t = text;
    const ns = nums(t);
    if (ns.length < 2) return null;
    let m = /(\d+)を(\d+)で割る/.exec(norm(t));
    if (m) {
      const a = parseInt(m[1], 10);
      const b = parseInt(m[2], 10);
      if (b) {
        const [q, r, steps] = divAnswer(a, b);
        return { answer: q, steps: steps + "。", kind: "div", remainder: r };
      }
    }
    // grouping by count: "24個を6個ずつ分けると何人分？" -> total / group size
    m = new RegExp(
      "(\\d+)\\s*(" + COUNTERS + ")[^0-9を]{0,6}を(\\d+)\\s*(" + COUNTERS + ")ずつ",
    ).exec(norm(t));
    if (m) {
      const a = parseInt(m[1], 10);
      const b = parseInt(m[3], 10);
      if (b) {
        const [q, r, steps] = divAnswer(a, b);
        let tail;
        if (t.includes("組")) tail = `${q}組。`;
        else if (t.includes("人分")) tail = `${q}人分。`;
        else tail = `${q}${m[4]}分。`;
        if (r) tail += `あまり${r}${m[2]}。`;
        return { answer: q, steps: steps + "。" + tail, kind: "div", remainder: r };
      }
    }
    const kw = ["分けると", "分けたら", "割り勘", "ずつ配", "ずつ分け",
      "何組", "何人分", "1人", "一人"].some((k) => t.includes(k));
    if (!kw) return null;
    const a = ns[0];
    const b = ns[1];
    if (b === 0) return null;
    const [q, r, steps] = divAnswer(a, b);
    const unit = t.includes("円") ? "円"
      : t.includes("個") ? "個"
      : t.includes("本") ? "本"
      : t.includes("枚") ? "枚" : "";
    let tail = (t.includes("1人") || t.includes("一人") || t.includes("分けると"))
      ? `1人${q}${unit}。`
      : `${q}${unit}。`;
    if (r) tail += `あまり${r}${unit}。`;
    return { answer: q, steps: steps + "。" + tail, kind: "div", remainder: r };
  }

  function solveCalc(text) {
    // multiplication must run before division: "1人に3個ずつ5人に配る"
    // contains the division keyword "1人" but is a multiplication.
    return (
      solveExpression(text) ||
      solveMul(text) ||
      solveDiv(text) ||
      solveSub(text)
    );
  }

  return { solve: solveCalc, solveCalc };
})();

globalThis.CalcLane = CalcLane;

/** 文中の数字（半角化してから拾う）。
 *  3桁区切りは先に外す：「3,776.12」は1つの数であり、「3」と「776.12」
 *  に割るとカンマ無しで書いたモデルの答えと照合できなくなる。 */
function numbersIn(text) {
  const s = toHalfWidth(text).replace(/(\d),(?=\d)/g, "$1");
  return s.match(/\d+(?:\.\d+)?/g) ?? [];
}

/** モデル回答のキーワード：数字＋2文字以上の漢字・かな・英字の連なり。 */
function keyTerms(text) {
  const terms = new Set();
  for (const n of numbersIn(text)) terms.add(n);
  const runs = text.match(/[一-鿿々]{2,}|[ぁ-んゝゞァ-ヶー]{2,}|[a-zA-Z]{2,}/g) ?? [];
  for (const r of runs) terms.add(r);
  return [...terms];
}

function articleEvidenceText(r) {
  return `${r.title}\n${r.snippet}\n${r.text}`;
}

/**
 * 2つの数字表記が、丸めの範囲で同じ値か。「3776」は「3776.12」の
 * 整数への丸めとして一致する。許容幅は粗い方（小数桁が少ない方）の
 * 0.5単位。
 */
function numbersMatchLoose(a, b) {
  const av = Number(a);
  const bv = Number(b);
  if (!Number.isFinite(av) || !Number.isFinite(bv)) return false;
  if (av === bv) return true;
  const dec = Math.min(
    (a.split(".")[1] ?? "").length,
    (b.split(".")[1] ?? "").length,
  );
  const tol = 0.5 * Math.pow(10, -dec) + 1e-9;
  return Math.abs(av - bv) <= tol;
}

/**
 * 事実の比較用に文字列を正規化する：旧仮名展開のうえ、空白・カンマ・
 * 引用符を落とし、末尾の句点を外す。「吉田 昭夫」と「吉田昭夫」、
 * 「「君が代」」と「君が代」、「3,776メートル」と「3776メートル」が
 * 同じと比べられるようにする。
 */
function normalizeForCompare(s) {
  return normalizeJa(s)
    .replace(/[\s　]/g, "")
    .replace(/,/g, "")
    .replace(/[「」『』""]/g, "")
    .replace(/[。．]+$/, "");
}

/**
 * モデルの答えが、ネット抽出の事実と同じ事実を言っているか。
 * モデル優先表示の関門：表記のゆれ（空白・カンマ・引用符）は同一視し、
 * 数値は丸めの範囲で一致すれば同じ事実とみなす。定義の質問では
 * 事実が文そのものなので、キーワードの過半がモデルの答えに
 * 現れるかで判定する。
 */
function modelAgreesWithFact(modelAnswer, fact) {
  if (!modelAnswer) return false;
  // 同語反復の文（「立方体」は「立方体」という意味です）は、事実と
  // 「一致」したことにしない。事実側が同語反復なら、それも事実として
  // 扱わない。
  if (isTautologicalAnswer(modelAnswer)) return false;
  if (isTautologicalAnswer(fact.answer) || isTautologicalAnswer(fact.value)) {
    return false;
  }
  const model = normalizeForCompare(modelAnswer);
  const value = normalizeForCompare(fact.value);
  if (!model || !value) return false;
  if (model.includes(value)) return true;
  const modelNums = numbersIn(modelAnswer);
  const factNums = numbersIn(fact.value);
  if (fact.kind === "measure" || fact.kind === "date") {
    if (modelNums.length === 0 || factNums.length === 0) return false;
    const allMatch = modelNums.every((mn) =>
      factNums.some((fn) =>
        fact.kind === "date" ? Number(mn) === Number(fn) : numbersMatchLoose(mn, fn),
      ),
    );
    if (!allMatch) return false;
    if (fact.kind === "date") return true;
    // 数値だけでなく単位もそろうこと：事実の単位と違う既知の単位を
    // モデルが言っているなら、数値の一致だけでは同じ事実とは言わない。
    const unit = value.replace(/[0-9.]/g, "");
    if (unit === "" || model.includes(unit)) return true;
    const KNOWN_UNITS = [
      "平方キロメートル",
      "キロメートル",
      "センチメートル",
      "ミリメートル",
      "メートル",
      "キログラム",
      "グラム",
      "トン",
      "人",
      "名",
      "円",
      "歳",
    ];
    return !KNOWN_UNITS.some((u) => u !== unit && model.includes(u));
  }
  if (fact.kind === "definition") {
    // 数字が食い違う定義は、言葉が似ていても同じ事実とは言わない。
    if (modelNums.length > 0 && factNums.length > 0) {
      const numsAgree = modelNums.every((mn) =>
        factNums.some((fn) => numbersMatchLoose(mn, fn)),
      );
      if (!numsAgree) return false;
    }
    const terms = keyTerms(fact.value);
    if (terms.length === 0) return false;
    let hits = 0;
    for (const t of terms) {
      if (model.includes(normalizeForCompare(t))) hits += 1;
    }
    return hits / terms.length >= 0.5;
  }
  if (modelNums.length > 0 && factNums.length > 0) {
    return modelNums.every((mn) =>
      factNums.some((fn) => numbersMatchLoose(mn, fn)),
    );
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Cube自身についての質問（固定情報）                                  */
/*                                                                    */
/* 作った人・名前・名前の意味・綴りは、Cube自身についての事実であり、  */
/* ウェブ検索では確かめようがない（WikipediaにCubeの記事は無い）。    */
/* モデルの生成任せにすると「私を作ったのは眞です」のように中心の字が  */
/* 欠けた答えが出ることがあるため、アプリに固定した正しい事実から      */
/* そのまま答える専用の種別にする。ネット照合もしないし、知識メモ帳    */
/* にも保存しない（固定情報は最初からここにある）。                    */
/* ------------------------------------------------------------------ */

const CUBE_SELF_FACTS = {
  creator: "佑眞",
  name: "Cube",
  meaning: "立方体",
  spelling: "C-u-b-e",
};

/**
 * 質問がCube自身についてのものなら、固定情報から組み立てた答えを返す。
 * そうでなければ null（通常の流れに任せる）。
 * 『…』で別の作品・対象が名指しされている質問や、第三者（彼・彼女・
 * 友達など）についての質問は、Cube自身の質問ではないので拾わない。
 * また「Cubeが作ったものは何？」のようにCubeが作り手側の質問も、
 * 作られた側の質問と取り違えない。
 */
function cubeSelfAnswer(question) {
  const q = question ?? "";
  const subject = subjectOf(q);
  if (subject && !/^(cube|キューブ)$/i.test(subject)) return null;
  const mentionsCube = /(cube|キューブ)/i.test(q);
  const secondPerson = /(あなた|あんた|君|きみ|お前|おまえ)/.test(q);
  const selfRef = mentionsCube || secondPerson;
  if (
    !mentionsCube &&
    /(彼|彼女|あの人|その人|友達|友だち|先生|お母さん|お父さん)/.test(q)
  ) {
    return null;
  }
  // 綴り：「cubeの綴りは？」「英語でどう書くの？」
  if (
    selfRef &&
    /(綴り|つづり|スペル|アルファベット|英語でどう書|どう書く)/.test(q)
  ) {
    return `${CUBE_SELF_FACTS.name}の綴りは「${CUBE_SELF_FACTS.spelling}」です。`;
  }
  // 名付けた人：「名付け親は誰？」「名前を付けたのは誰？」
  if (
    /(名付け|名づけ|名前を付け|名前をつけ|名前を決め)/.test(q) &&
    (selfRef || /名前/.test(q))
  ) {
    return `${CUBE_SELF_FACTS.name}と名付けたのは${CUBE_SELF_FACTS.creator}です。`;
  }
  // 名前の意味・由来：「cubeの意味は何ですか」「名前の意味は？」
  // 一般の語の意味（「立方体の意味は？」）は拾わない：Cubeか名前への
  // 言及があるときだけ、名前の意味として答える。
  if (
    /(意味|由来)/.test(q) &&
    (selfRef || /名前の(意味|由来)/.test(q))
  ) {
    return `${CUBE_SELF_FACTS.name}は英語で「${CUBE_SELF_FACTS.meaning}」という意味です。`;
  }
  // 作った人：「あなたを作ったのは誰ですか」「Cubeの作者は誰？」
  const makerSubjectAskingWhat =
    /(あなた|君|きみ|お前|おまえ|cube|キューブ)が\s*(作っ|開発し|生ん|つくっ)/i.test(q) &&
    /(何|なに|もの|作品)/.test(q);
  const creatorCue =
    /作ったのは|作った人|作り手|作った方|開発者|開発したのは|開発した人|生みの親|製作者|生んだのは|誰が[^。]{0,10}作っ|を(?:作っ|開発し|生ん)|作者|著者/.test(
      q,
    );
  if (selfRef && creatorCue && !makerSubjectAskingWhat) {
    return `私を作ったのは${CUBE_SELF_FACTS.creator}です。`;
  }
  // 自己紹介：「あなたは誰ですか」「自己紹介して」
  if (
    /(あなたは誰|君は誰|お前は誰|あなたは何者|何者なの|自己紹介)/.test(q)
  ) {
    return `私は${CUBE_SELF_FACTS.name}です。${CUBE_SELF_FACTS.creator}が作った自作LLMで、名前は英語で「${CUBE_SELF_FACTS.meaning}」という意味です。綴りは${CUBE_SELF_FACTS.spelling}です。`;
  }
  // 名前：「あなたの名前は何ですか」「名前を教えて」
  if (
    /名前|なまえ|名称/.test(q) &&
    (selfRef || q.length <= 14)
  ) {
    return `私の名前は${CUBE_SELF_FACTS.name}です。`;
  }
  return null;
}

/**
 * 同語反復の文か：「「立方体」は「立方体」という意味です」のように、
 * 説明される語と説明の中身が同じで、意味を何も伝えていない文。
 * 辞書ページ等からこういう文が事実として組み立てられたり、モデルの
 * 同じ文が照合を通ったりすると、中身の無い答えが「確認済み」に
 * なってしまうため、事実にも裏づけにも使わない。
 */
function isTautologicalAnswer(answer) {
  if (!answer) return false;
  const a = String(answer).trim();
  const same = (x, y) => {
    const nx = normalizeForCompare(x ?? "");
    const ny = normalizeForCompare(y ?? "");
    return nx.length > 0 && nx === ny;
  };
  const quoted = a.match(
    /「([^」]{1,40})」\s*(?:は|とは)[、,\s]*「([^」]{1,40})」/,
  );
  if (quoted && same(quoted[1], quoted[2])) return true;
  const plain = a.match(
    /^『?「?(.{1,40}?)』?」?\s*(?:は|とは)[、,\s]*『?「?(.{1,40}?)』?」?\s*(?:という意味|の意味|ということ|のこと|ことです|である|です)/,
  );
  if (plain && same(plain[1], plain[2])) return true;
  const meaningForm = a.match(
    /^(.{1,30}?)の意味は\s*『?「?(.{1,30}?)』?」?\s*(?:です|である|という意味)/,
  );
  if (meaningForm && same(meaningForm[1], meaningForm[2])) return true;
  return false;
}

/**
 * 助詞・活用など、事実の中身を持たない言い回し。質問にも証拠にも
 * ほぼ必ず現れるため、「裏づけ」の材料にしてはいけない。
 */
const EVIDENCE_STOP_TERMS = new Set([
  "です", "ます", "である", "でした", "だった", "こと", "もの", "ため",
  "それ", "これ", "どれ", "その", "この", "どの", "から", "まで", "など",
  "について", "という", "によって", "として", "には", "では",
]);

/**
 * モデルの答えのうち、その主張そのものを担う語：キーワードから、
 * 質問文に既に含まれている語（質問の言い換えは証拠にならない）と
 * 中身のない言い回しを除いたもの。「日本の総理大臣は誰ですか」に
 * 「日本の総理大臣は太子です。」と答えた場合、残るのは「太子」だけ
 * —— それが答えの肝心な部分である。
 */
function answerContentTerms(modelAnswer, question) {
  const questionTerms = new Set([
    ...keyTerms(normalizeJa(question)),
    ...numbersIn(question),
  ]);
  return keyTerms(normalizeJa(modelAnswer)).filter(
    (t) => !questionTerms.has(t) && !EVIDENCE_STOP_TERMS.has(t),
  );
}

/**
 * 照合用の語に正規化する：文尾の丁寧・断定の言い回し（〜している
 * ことです／です／である／した 等）と先頭の助詞を落とし、語の芯だけ
 * にする。モデルの答えは「…で活躍していることです。」と文として
 * 閉じるため、かなの連なりがそのままでは記事の文（「…で活躍して
 * いる。」）と一致せず、正しい答えまで「未確認」に落ちていた。
 * 漢字の語は語尾が漢字なので、ほぼ影響を受けない。
 */
function canonicalTerm(t) {
  let s = t;
  for (let i = 0; i < 3; i += 1) {
    const next = s.replace(
      /(ということ|ことです|である|でした|だった|です|ます|している|していた|されている|されます|された|される|になり|になって|になる|であり|であって|して|した|する)$/,
      "",
    );
    if (next === s) break;
    s = next;
  }
  return s.replace(/^(で|に|を|は|が|と|の|へ|から|まで|より)/, "");
}

/**
 * 実際に照合へ使う大事な言葉：answerContentTerms を canonicalTerm で
 * 芯だけにし、言い回しだけになって消えた語は数えない。
 */
function effectiveContentTerms(modelAnswer, question) {
  const out = [];
  for (const t of answerContentTerms(modelAnswer, question)) {
    if (/^\d/.test(t)) {
      if (!out.includes(t)) out.push(t);
      continue;
    }
    const c = canonicalTerm(t);
    if (c.length >= 2 && !EVIDENCE_STOP_TERMS.has(c) && !out.includes(c)) {
      out.push(c);
    }
  }
  return out;
}

/**
 * 証拠として使えない検索結果か：曖昧さ回避ページと、外国語主体の
 * ページ（【KR】のような他言語の辞書項目など）は、たとえ答えの語が
 * 含まれていても裏づけには使わない。
 */
function isUsableEvidenceResult(result) {
  const title = result?.title ?? "";
  const text = result?.text ?? "";
  if (/曖昧さ回避/.test(title)) return false;
  if (/^【/.test(title)) return false;
  if (/曖昧さ回避のためのページ/.test(text.slice(0, 600))) return false;
  // 外国語主体の判定：冒頭800字でハングルが かな の数を超えるページは
  // 日本語の記事ではない（他言語の項目が混ざった場合の保険）。
  const head = text.slice(0, 800);
  const hangul = (head.match(/[가-힣]/g) ?? []).length;
  if (hangul > 0) {
    const kana = (head.match(/[ぁ-んァ-ヶー]/g) ?? []).length;
    if (hangul > kana) return false;
  }
  return true;
}

/**
 * 単位の略記を日本語の単位名に寄せる：記事は「3,776.12m」のように
 * 略記し、モデルの答えは「3776メートル」と書くため、そのままでは
 * 同じ事実でも窓の中で一致しない。数字の直後の単位だけを対象にする。
 */
function normalizeUnits(text) {
  return text.replace(
    /(\d)\s*(km²|km2|km|cm|mm|kg|m|t|g)(?![a-zA-Z])/g,
    (m0, d, u) => {
      const map = {
        "km²": "平方キロメートル",
        km2: "平方キロメートル",
        km: "キロメートル",
        cm: "センチメートル",
        mm: "ミリメートル",
        kg: "キログラム",
        m: "メートル",
        t: "トン",
        g: "グラム",
      };
      return d + (map[u] ?? u);
    },
  );
}

/**
 * 照合の「窓」：タイトル＋冒頭のリード約1600字と、本文の各文。
 * 裏づけは、このどれか1つの窓の中で完結しなければならない。
 * 記事全体に語が散らばっているだけ（別々の節に偶然出てくるだけ）
 * では、その主張を裏づけたことにはならないため。
 */
function evidenceWindows(result) {
  const head = `${result.title}\n${result.snippet}\n${result.text.slice(0, 1600)}`;
  const windows = [normalizeUnits(normalizeJa(head))];
  for (const s of splitSentences(result.text)) {
    windows.push(normalizeUnits(normalizeJa(s)));
  }
  return windows;
}

function windowSupportsAnswer(windowText, modelAnswer, content) {
  const evNumbers = numbersIn(windowText);
  const ansNumbers = numbersIn(modelAnswer);
  if (ansNumbers.length > 0) {
    const allSupported = ansNumbers.every((n) =>
      evNumbers.some((e) => numbersMatchLoose(n, e)),
    );
    if (!allSupported) return false;
  }
  return content.every((t) =>
    /^\d/.test(t)
      ? evNumbers.some((e) => numbersMatchLoose(t, e))
      : windowText.includes(t),
  );
}

/**
 * 1記事だけがモデルの答えを裏づけるか。条件：
 * - その記事が証拠として使える記事であること（曖昧さ回避ページ・
 *   外国語主体のページは不可）。
 * - 答えの数字の全部と大事な言葉の全部が、同じ1つの「窓」（冒頭の
 *   リード、または本文のどれか1文）の中にあること。複数記事の
 *   寄せ集めも、1記事の中で散らばった語の寄せ集めも裏づけにしない。
 */
function resultSupportsAnswer(result, modelAnswer, content) {
  if (!isUsableEvidenceResult(result)) return false;
  return evidenceWindows(result).some((w) =>
    windowSupportsAnswer(w, modelAnswer, content),
  );
}

/**
 * その記事が、質問の対象についての記事か（話題アンカー）。
 * タイトルが対象・答えの主語・答えの値そのものと一致するか、
 * 記事の最初の文の主語が対象であること。検索で拾った別対象の記事
 * （高さの数字が偶然同じ別の建物など）が、数字と単位の一致だけで
 * 裏づけに紛れ込むのを防ぐ。対象が読み取れない質問では判定
 * できないため従来どおり通す（窓の条件はそのまま効く）。
 */
function resultAnchoredToQuestion(result, question, modelAnswer) {
  const subject = subjectOf(question);
  const topic = topicFromQuestion(question);
  const aSubject = answerSubjectOf(modelAnswer);
  const valueMatch = modelAnswer.match(
    /は「?([^。「」]{1,30}?)」?(?:です|である)[。]?$/,
  );
  const value = valueMatch?.[1]?.trim();
  const anchors = [subject, topic, aSubject, value].filter(Boolean);
  if (anchors.length === 0) return true;
  for (const a of anchors) {
    if (titleMatchesSubject(result.title, a)) return true;
    const nt = normalizeJa(result.title);
    const base = nt.replace(/[（(][^（）()]*[）)]\s*$/, "").trim();
    const na = normalizeJa(a);
    if (base.length >= 3 && na.startsWith(base)) return true;
    if (nt.length >= 2 && na === nt) return true;
  }
  const firstSentence =
    splitSentences(result.text)[0] ?? result.text.slice(0, 120);
  const headNorm = normalizeJa(firstSentence).slice(0, 48);
  for (const a of [subject, topic, aSubject].filter(Boolean)) {
    const na = normalizeJa(a);
    if (
      na.length >= 2 &&
      (headNorm.startsWith(na) || headNorm.startsWith(`『${na}`))
    ) {
      return true;
    }
  }
  return false;
}

/**
 * 照合：モデルの答えの主張を裏づける記事を1本返す。無ければ null。
 *
 * 以前の関門はキーワード全体の過半一致で通していたが、その大半は
 * 質問の言い換え（日本・総理大臣）と「です」のような言い回しで、
 * どの関連ページにも含まれてしまう。間違った答え（「総理大臣は
 * 太子です」）が3/4で「確認済み」になり、メモ帳にまで保存された。
 *
 * 今の関門：
 * - 答えの肝心な部分（質問の言い換えと言い回しを除き、語の芯だけに
 *   したもの）が2語以上あること。1語だけの答えは原則通さない——
 *   関係の薄いページにもたまたま含まれてしまうため（例外は下記：
 *   種類が一致し、主語か値そのものの記事で裏づけられる場合だけ）。
 * - その全部と、答えの数字の全部が、同じ1記事の同じ「窓」（冒頭の
 *   リード、または本文のどれか1文）の中にあること。さらにその記事は
 *   質問の対象についての記事でなければならない（話題アンカー）：
 *   別対象の記事や、語が記事の奥に散らばっているだけの記事、
 *   曖昧さ回避ページ・外国語主体のページは証拠にしない。
 * - 質問が人物・作者・本社・属性などのはっきりした種類で、答えの
 *   主張する種類が読めるのに食い違う場合は通さない（知事を聞いた
 *   のに名物の答えが名物の記事で裏づけられる、を防ぐ）。
 * 返した記事がそのまま出典になる——裏づけてもいない記事を出典に
 * 表示することはしない。
 */
function findCorroboratingResult(modelAnswer, results, question = "") {
  if (!modelAnswer) return null;
  if (keyTerms(normalizeJa(modelAnswer)).length === 0) return null;
  // 同語反復の文は裏づけの対象にしない（中身が無い文を「確認済み」
  // にはできない）。意味・定義の質問もこの緩い照合では通さない：
  // 「cubeの意味」では大事な言葉が「立方体」の1語だけになり、辞書
  // ページにその1語があれば通ってしまっていた。定義の答えは、
  // 対象記事から種類つき事実がきちんと取れたときだけ「確認済み」
  // ＋メモ帳保存にする（事実の道を通った場合のみ）。
  if (isTautologicalAnswer(modelAnswer)) return null;
  if (classifyQuestion(question) === "definition") return null;
  const STRICT_KINDS = ["person", "author", "hq", "operator", "attribute"];
  const qKind = classifyQuestion(question);
  const aKind = answerClaimKind(modelAnswer);
  if (qKind && aKind && qKind !== aKind && STRICT_KINDS.includes(qKind)) {
    return null;
  }
  const content = effectiveContentTerms(modelAnswer, question);
  if (content.length >= 2) {
    for (const r of results) {
      if (
        resultAnchoredToQuestion(r, question, modelAnswer) &&
        resultSupportsAnswer(r, modelAnswer, content)
      ) {
        return r;
      }
    }
    return null;
  }
  // 大事な言葉が1語だけの答え（「大阪の名物はたこ焼きです」の値は
  // たこ焼き1語）は、原則通さない。ただし例外として、質問と答えの
  // 種類が一致し（名物・属性・運営会社・場所など、人物と作者は除く）、
  // その1語が「主語の記事」か「値そのものの記事」の中で裏づけられる
  // 場合だけ通す。人物・作者の1語（「総理大臣は太子です」）は、
  // 関係記事のどこかにその語が出るだけであり得てしまうため、
  // 種類つき事実の道でしか確認済みにしない。
  if (content.length !== 1) return null;
  if (!qKind || !aKind || qKind !== aKind) return null;
  if (qKind === "person" || qKind === "author") return null;
  const term = content[0];
  if (/^\d/.test(term)) return null;
  const aSubject = answerSubjectOf(modelAnswer);
  const topic = topicFromQuestion(question);
  // 値そのもの（「Xの名物はYです」の Y）：たこ焼き のような漢字・
  // かな混じりの語は keyTerms では断片化するため、錨には文型から
  // 取り出した値の文字列を使う。
  const valueMatch = modelAnswer.match(
    /は「?([^。「」]{1,30}?)」?(?:です|である)[。]?$/,
  );
  const value = valueMatch?.[1]?.trim();
  for (const r of results) {
    const titleAnchored =
      (aSubject && titleMatchesSubject(r.title, aSubject)) ||
      (topic && titleMatchesSubject(r.title, topic)) ||
      (value && titleMatchesSubject(r.title, value)) ||
      titleMatchesSubject(r.title, term);
    if (titleAnchored && resultSupportsAnswer(r, modelAnswer, content)) {
      return r;
    }
  }
  return null;
}

function isCorroborated(modelAnswer, results, question = "") {
  return findCorroboratingResult(modelAnswer, results, question) !== null;
}

/**
 * モデルの答えが主張している事実の種類（文の形から判定する）。
 * このモデルの答えは「Xの名物はYです」「『Z』の作者はWです」のような
 * 決まった形で出てくるため、形から種類が読み取れる。形が読み取れない
 * 答え（地の文）は null。
 */
function answerClaimKind(answer) {
  const a = answer ?? "";
  if (/の作者は|の著者は/.test(a)) return "author";
  if (/の本社は|の本店は|本社所在地は/.test(a)) return "hq";
  if (/の名物は|が名物|の特産品?は|名物です/.test(a)) return "specialty";
  if (/の運営会社は|の運営元は|を運営しているのは/.test(a)) return "operator";
  if (/の国歌は|の首都は/.test(a)) return "attribute";
  if (/の(総理大臣|首相|知事|社長|会長|代表者?|創業者|創立者|市長|町長|村長|大統領)は/.test(a)) {
    return "person";
  }
  if (/の(高さ|標高|海抜|全長|長さ|重さ|重量|面積|人口)は/.test(a)) {
    return "measure";
  }
  if (/に設立されました|に創立されました|に創業しました|の設立は|の創業は|に完成しました|に開業しました/.test(a)) {
    return "date";
  }
  if (/は.{1,40}にあります|の所在地は/.test(a)) return "place";
  return null;
}

/** 決まった形の答えの主語（『…』か「Xの〜」の X）。取れなければ null。 */
function answerSubjectOf(answer) {
  const m =
    /^『(.+?)』/.exec(answer ?? "") ??
    /^(.{2,24}?)の(?:名物|特産品?|作者|著者|本社|本店|高さ|標高|海抜|全長|長さ|重さ|重量|面積|人口|国歌|首都|社長|会長|知事|総理大臣|首相|運営会社|運営元|設立|創業)/.exec(
      answer ?? "",
    );
  const s = m?.[1]?.trim();
  return s ? s : null;
}

/**
 * 未確認の答えを出す前の関門：質問が求める事実の種類と、モデルの答えが
 * 主張する種類が食い違っていたら、その答えは質問への答えではない
 * （知事を聞いたのに名物や作者の答えが出る、など）。主語が質問の対象と
 * 別物の場合も同じ。食い違う答えは出さず「確認できませんでした」に
 * する——種類の合わない答えを「未確認」と断っても読む側の助けに
 * ならないため。種類が読み取れない地の文の答えは従来どおり通す。
 */
function modelAnswerFitsQuestion(question, answer) {
  const claim = answerClaimKind(answer);
  if (!claim) return true;
  const expected = classifyQuestion(question);
  if (expected && claim !== expected) return false;
  if (!expected) return false;
  const qSubject = subjectOf(question);
  const aSubject = answerSubjectOf(answer);
  if (qSubject && aSubject) {
    return (
      titleMatchesSubject(aSubject, qSubject) ||
      titleMatchesSubject(qSubject, aSubject)
    );
  }
  if (!qSubject && aSubject) {
    const topic = topicFromQuestion(question);
    if (topic) {
      const nt = normalizeJa(topic);
      const na = normalizeJa(aSubject);
      if (!(nt.includes(na) || na.includes(nt))) return false;
    }
  }
  return true;
}

function splitSentences(text) {
  const out = [];
  const parts = text.split(/(?<=[。！？\n])/);
  for (const p of parts) {
    const s = p.trim();
    if (s.length >= 8 && s.length <= 400) out.push(s);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 種類つき事実抽出                                                    */
/* ------------------------------------------------------------------ */

/** この質問はどの種類の事実を聞いているか。対象外なら null。 */
function classifyQuestion(q) {
  if (/作者|著者|書いた|描いた/.test(q)) return "author";
  if (
    (/(本社|本店)/.test(q) &&
      /(どこ|所在地|場所|ある|教え|知りたい|聞きたい)/.test(q)) ||
    // 「〜の本社は？」のように場所の語が無くても、本社そのものを
    // 尋ねる文なら本社の質問として扱う（事実抽出の入口に立たせる）。
    /(本社|本店)は[？?]/.test(q)
  ) {
    return "hq";
  }
  // 現職者：「日本の総理大臣は誰ですか」は人物の質問だが、総理大臣は
  // 会社の役職の語に当たらず、以前は未分類のまま緩い照合に流れていた。
  if (
    /内閣総理大臣|総理大臣|首相/.test(q) &&
    /(誰|だれ|名前|現職|今|現在|教えて|教え|知りたい|聞きたい)/.test(q)
  ) {
    return "person";
  }
  // 知事も現職者の質問：以前は分類に無く、「福島県知事は誰ですか」が
  // 未分類のまま緩い照合に流れ、名物や作者の答えがそのまま出ていた。
  // 「福島県知事」という記事（知事一覧へ飛ぶ）が現職者欄を持つため、
  // 総理大臣と同じ現職者の仕組みで取り出す。
  if (
    /知事/.test(q) &&
    (/(誰|だれ|名前|現職|今|現在|教えて|教え|知りたい|聞きたい)/.test(q) ||
      /知事は[？?]/.test(q))
  ) {
    return "person";
  }
  if (
    (/社長|会長|創業者|創立者|代表/.test(q) &&
      /(誰|だれ|名前|教えて|教え|知りたい|聞きたい)/.test(q)) ||
    // 「〜の社長は？」のように「誰」が無くても、役職そのものを
    // 尋ねる文なら人物の質問として扱う（本社の裸形と同じ扱い）。
    /(社長|会長|創業者|創立者|代表)は[？?]/.test(q)
  ) {
    return "person";
  }
  // 運営会社：「モーリーファンタジーの運営会社はなんという社名ですか」
  // は、以前はどの種類にも当たらず事実を取りに行けなかった。
  if (
    /運営(会社|元|者|主体)/.test(q) ||
    (/(を|の)運営/.test(q) && /(会社|社名|何|なに|なん|誰)/.test(q))
  ) {
    return "operator";
  }
  // 「大阪の名物は？」「日本の国歌は？」には何/どこ/誰の印が無いため、
  // この2種類が無いと全パターンを素通りして事実が一切取れなかった。
  if (/名物|特産|名産|郷土料理|有名な食べ物/.test(q)) return "specialty";
  if (/国歌|首都/.test(q)) return "attribute";
  if (/いつ|何年|設立|創立|創業|結成|完成|竣工|開業|できた/.test(q)) {
    return "date";
  }
  if (
    /高さ|標高|海抜|全長|長さ|重さ|面積|人口|何メートル|何キロ|いくつ|何人|何円|何歳/.test(
      q,
    )
  ) {
    return "measure";
  }
  if (/(どこ|所在地|場所|ある？|ありますか)/.test(q)) return "place";
  if (
    /とは|何ですか|何が|何を|何に|何から|なに？|何？|どんなもの|意味|教えて|教え|知りたい|聞きたい/.test(
      q,
    )
  ) {
    return "definition";
  }
  return null;
}

/** 数字の整数部に3桁区切りを付ける。 */
function withCommas(num) {
  const [int, ...rest] = num.replace(/,/g, "").split(".");
  const grouped = (int ?? "").replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return rest.length > 0 ? `${grouped}.${rest.join(".")}` : grouped;
}

const UNIT_NORM = {
  m: "メートル",
  km: "キロメートル",
  cm: "センチメートル",
  mm: "ミリメートル",
  kg: "キログラム",
  g: "グラム",
  t: "トン",
  "km2": "平方キロメートル",
  "km²": "平方キロメートル",
};

/**
 * インフォボックスの場所の値を整える：空白区切りの自治体名は「、」でつなぎ、
 * 末尾の建物名は括弧に入れる（「…中瀬一丁目5番地1 イオンタワー7階」→
 * 「…中瀬一丁目5番地1（イオンタワー7階）」）。
 */
function formatPlace(raw) {
  const parts = raw.split(/\s+/).filter((p) => p.length > 0);
  if (parts.length <= 1) return raw.trim();
  const admin = /[県市町村区郡]$/;
  if (parts.every((p) => /[県市町村区郡]/.test(p))) {
    return parts.join("、");
  }
  const last = parts[parts.length - 1] ?? "";
  if (parts.length === 2 && !admin.test(last) && last.length <= 20) {
    return `${parts[0]}（${last}）`;
  }
  return parts.join(" ");
}

/**
 * 「イオンモール株式会社は、千葉県千葉市美浜区の幕張新都心」のような範囲から
 * 場所の部分だけを取り出す（最初の自治体名から）。
 */
function placeFromSpan(span) {
  const m =
    /(北海道|[^\s、,]{1,8}?(?:都|府|県)|[^\s、,]{2,10}?(?:市|区|町|村))/.exec(
      span,
    );
  if (!m || m.index === undefined) return null;
  const place = span
    .slice(m.index)
    .replace(/[、,\s]+$/, "")
    .trim();
  return place.length >= 3 ? place : null;
}

/** 条件を満たす最初のリード文。なければ null。 */
function findSentence(lead, pred, limit = 10) {
  const sentences = splitSentences(lead).slice(0, limit);
  for (const s of sentences) {
    if (pred(s)) return s;
  }
  return null;
}

function fieldEvidence(article, fieldName, value, probe) {
  const hit = findSentence(article.lead, (s) => s.includes(probe), 12);
  if (hit) return hit;
  return `出典記事のインフォボックス「${fieldName}」: ${value}`;
}

function makeFact(article, kind, value, answer, evidence) {
  return {
    kind,
    value,
    answer,
    evidence,
    sourceTitle: article.title,
    sourceUrl: article.url,
  };
}

function extractAuthor(article, label) {
  const fromField = article.fields["作者"] ?? article.fields["著者"];
  if (fromField && fromField.length <= 30 && !/[。]/.test(fromField)) {
    const fieldName = article.fields["作者"] ? "作者" : "著者";
    return makeFact(
      article,
      "author",
      fromField,
      `${label}の作者は${fromField}です。`,
      fieldEvidence(article, fieldName, fromField, fromField),
    );
  }
  // 順番が大事：「Xの小説」より先に「Xによる小説」を試さないと、
  // 『坊っちゃん』で「夏目漱石による日本」を作者として飲み込んでしまう。
  const patterns = [
    /作者は(.{2,20}?)(?:です|である|、|。)/,
    /著者は(.{2,20}?)(?:です|である|、|。)/,
    /(.{2,20}?)による(?:.{0,10}?(?:小説|作品|著書))/,
    /は、?(.{2,20}?)の(?:長編|短編|中編)?小説/,
  ];
  for (const re of patterns) {
    const hit = findSentence(article.lead, (s) => re.test(normalizeJa(s)));
    if (!hit) continue;
    const m = re.exec(normalizeJa(hit));
    // 「…の小説」の取り込みに「Xによる日本」が残ることがある：
    // 作者は による の前の部分。
    const value = ((m?.[1] ?? "").split("による")[0] ?? "")
      .replace(/^[、,\s]+|[、,\s]+$/g, "")
      .trim();
    if (
      value.length >= 2 &&
      value.length <= 20 &&
      // もっともらしい作者名に句読点や助詞は入らない（文の前半を
      // 飲み込んだ取り込みを弾く）。
      !/[。『』、（）()：:はがをにへとや]/.test(value)
    ) {
      return makeFact(
        article,
        "author",
        value,
        `${label}の作者は${value}です。`,
        hit,
      );
    }
  }
  return null;
}

function extractHq(article, label) {
  const fromField =
    article.fields["本社所在地"] ?? article.fields["本店所在地"];
  if (fromField) {
    const value = formatPlace(fromField);
    const fieldName = article.fields["本社所在地"] ? "本社所在地" : "本店所在地";
    const probe =
      value.match(/[^\s、（(]{2,}(?:市|区|町|村)/)?.[0] ?? value.slice(0, 6);
    return makeFact(
      article,
      "hq",
      value,
      `${label}の本社は${value}にあります。`,
      fieldEvidence(article, fieldName, fromField, probe),
    );
  }
  const patterns = [
    /(.+?)に本社を置く/,
    /(.+?)に本店を置く/,
    /本社所在地は、?(.{3,50}?)(?:である|です|にあたる)[。]?/,
    /本社は、?(.{3,40}?)(?:に所在|に位置|にあります)/,
    /本社が?、?(.{3,40}?)に所在/,
  ];
  for (const re of patterns) {
    const hit = findSentence(article.lead, (s) => re.test(s), 14);
    if (!hit) continue;
    const m = re.exec(hit);
    const place = m?.[1] ? placeFromSpan(m[1]) : null;
    if (place) {
      return makeFact(
        article,
        "hq",
        place,
        `${label}の本社は${place}にあります。`,
        hit,
      );
    }
  }
  return null;
}

/**
 * 県庁所在地（都庁・府庁・道庁所在地も）：聞かれているのは県庁のある
 * 市であり、所在地欄の番地ではない。所在地欄は「仙台市青葉区本町
 * 3丁目8番1号」の形で先頭がその市なので、先頭の自治体名だけを取り、
 * 「宮城県の県庁所在地は仙台市です。」と組み立てる。こうすると市の
 * 名前だけを答えるモデル（「福島市です」）とも食い違いにならない。
 * 明示の庁所在地欄があればそれを優先し、最後にリード文も試す。
 */
function extractPrefCapital(question, article, label) {
  const term = /([都道府県]庁所在地)/.exec(question)?.[1] ?? "県庁所在地";
  const cityOf = (raw) => {
    if (!raw) return null;
    let s = raw.replace(/[\s　]/g, "");
    if (article.title && s.startsWith(article.title)) {
      s = s.slice(article.title.length);
    }
    const m = /^(.{2,12}?[市区町村])/.exec(s);
    const city = m?.[1] ?? null;
    return city && city.length >= 2 ? city : null;
  };
  for (const name of ["県庁所在地", "都庁所在地", "府庁所在地", "道庁所在地", "庁所在地"]) {
    const fromField = article.fields[name];
    const city = cityOf(fromField);
    if (city) {
      return makeFact(
        article,
        "place",
        city,
        `${label}の${term}は${city}です。`,
        fieldEvidence(article, name, fromField, city),
      );
    }
  }
  const loc = article.fields["所在地"];
  const cityFromLoc = cityOf(loc);
  if (loc && cityFromLoc) {
    return makeFact(
      article,
      "place",
      cityFromLoc,
      `${label}の${term}は${cityFromLoc}です。`,
      fieldEvidence(article, "所在地", loc, cityFromLoc),
    );
  }
  const hit = findSentence(article.lead, (s) => /庁所在地は/.test(s), 12);
  if (hit) {
    const m = /庁所在地は、?(.{2,12}?[市区町村])/.exec(hit);
    if (m?.[1]) {
      return makeFact(
        article,
        "place",
        m[1],
        `${label}の${term}は${m[1]}です。`,
        hit,
      );
    }
  }
  return null;
}

function extractPlace(question, article, label) {
  // 逆向きの名物：「たこ焼きが有名な都市はどこ？」では題名は料理で、
  // 答えはその本場。料理自身の記事のリードが「…は〜の郷土料理／名物」
  // と自称している場合だけ、その土地を答えにする（自称していない
  // 記事からは推測しない）。
  if (/が有名な(都市|町|村|市|場所|地域)/.test(question)) {
    const hit = findSentence(
      article.lead,
      (s) => /の(郷土料理|名物|名産品?)/.test(s),
      6,
    );
    if (hit) {
      const re = /([一-鿿々]{2,8}?)の(?:郷土料理|名物|名産品?)/g;
      let place = null;
      for (const m of hit.matchAll(re)) {
        const cand = (m[1] ?? "").trim();
        if (
          cand &&
          cand !== article.title &&
          !/^(地元|全国|各地|郷土|日本各地)$/.test(cand)
        ) {
          place = cand;
        }
      }
      if (place) {
        return makeFact(
          article,
          "place",
          place,
          `${label}が有名なのは${place}です。`,
          hit,
        );
      }
    }
  }
  if (/庁所在地/.test(question)) {
    const capital = extractPrefCapital(question, article, label);
    if (capital) return capital;
  }
  const fieldNames = ["所在地", "場所", "位置"];
  for (const name of fieldNames) {
    const fromField = article.fields[name];
    if (fromField) {
      const value = formatPlace(fromField);
      return makeFact(
        article,
        "place",
        value,
        `${label}は${value}にあります。`,
        fieldEvidence(article, name, fromField, value.slice(0, 6)),
      );
    }
  }
  const hit = findSentence(
    article.lead,
    (s) => /は、?.+?に(?:ある|位置する|立地する)/.test(s),
    8,
  );
  if (hit) {
    const m = /は、?(.+?)に(?:ある|位置する|立地する)/.exec(hit);
    const place = m?.[1] ? placeFromSpan(m[1]) : null;
    if (place) {
      return makeFact(
        article,
        "place",
        place,
        `${label}は${place}にあります。`,
        hit,
      );
    }
  }
  return null;
}

function extractDate(question, article, label) {
  const founding = /設立|創立|創業|結成|発足/.test(question);
  const fieldNames = founding
    ? ["設立", "創立", "創業", "結成", "発足", "設立年", "創立年"]
    : ["完成", "竣工", "開業", "設立", "創立", "創業", "結成", "供用開始"];
  const dateRe = /(\d{3,4})年(?:(\d{1,2})月)?(?:(\d{1,2})日)?/;
  for (const name of fieldNames) {
    const fromField = article.fields[name];
    if (!fromField) continue;
    // 元号の括弧を先に落とす：「1911年（明治44年）11月12日」は
    // 「1911年」だけではなく日付全体を取りたい。
    const m = dateRe.exec(fromField.replace(/（[^）]*）/g, ""));
    if (!m) continue;
    const value = m[0];
    const verb = founding
      ? /創業/.test(question)
        ? "創業"
        : "設立"
      : /完成|竣工|できた/.test(question)
        ? "完成"
        : "設立";
    const answer = founding
      ? verb === "創業"
        ? `${label}の創業は${value}です。`
        : `${label}は${value}に設立されました。`
      : `${label}は${value}に${verb === "完成" ? "完成しました" : "できました"}。`;
    return makeFact(
      article,
      "date",
      value,
      answer,
      fieldEvidence(article, name, fromField, value),
    );
  }
  // リード文の保険：出来事を名指しし、日付を含む文。
  const kw = founding ? /設立|創立|創業|結成/ : /完成|竣工|開業|設立|創立/;
  const hit = findSentence(
    article.lead,
    (s) => kw.test(s) && dateRe.test(s),
    14,
  );
  if (hit) {
    const m = dateRe.exec(hit);
    if (m) {
      const value = m[0];
      const answer = founding
        ? `${label}は${value}に設立されました。`
        : `${label}は${value}にできました。`;
      return makeFact(article, "date", value, answer, hit);
    }
  }
  return null;
}

const MEASURE_SPECS = [
  {
    test: /高さ|標高|海抜|何メートル/,
    fields: ["標高", "高さ", "海抜"],
    name: "高さ",
    defaultUnit: "メートル",
    leadKeyword: /標高|高さ|海抜/,
  },
  {
    test: /全長|長さ/,
    fields: ["全長", "長さ", "全幅", "幅"],
    name: "長さ",
    defaultUnit: "メートル",
    leadKeyword: /全長|長さ/,
  },
  {
    test: /重さ|重量|何キロ/,
    fields: ["重量", "重さ", "質量"],
    name: "重さ",
    defaultUnit: "キログラム",
    leadKeyword: /重量|重さ/,
  },
  {
    test: /面積/,
    fields: ["面積"],
    name: "面積",
    defaultUnit: "平方キロメートル",
    leadKeyword: /面積/,
  },
  {
    test: /人口|何人/,
    fields: ["人口"],
    name: "人口",
    defaultUnit: "人",
    leadKeyword: /人口/,
  },
];

const MEASURE_RE =
  /([\d,]+(?:\.\d+)?)\s*(メートル|キロメートル|センチメートル|ミリメートル|平方キロメートル|キログラム|グラム|トン|km2|km²|km|cm|mm|kg|m|t|g|人|名|円|歳)?/;

function measureValue(raw, defaultUnit) {
  const m = MEASURE_RE.exec(raw.replace(/,/g, ","));
  if (!m || !m[1]) return null;
  const num = withCommas(m[1]);
  const unitRaw = m[2] ?? defaultUnit;
  const unit = UNIT_NORM[unitRaw] ?? unitRaw;
  return `${num}${unit}`;
}

function extractMeasure(question, article, label) {
  const spec = MEASURE_SPECS.find((s) => s.test.test(question));
  if (!spec) return null;
  for (const name of spec.fields) {
    const fromField = article.fields[name];
    if (!fromField) continue;
    const value = measureValue(fromField, spec.defaultUnit);
    if (value) {
      return makeFact(
        article,
        "measure",
        value,
        `${label}の${spec.name}は${value}です。`,
        fieldEvidence(article, name, fromField, value.replace(/[^\d]/g, "").slice(0, 4) || value),
      );
    }
  }
  const hit = findSentence(
    article.lead,
    (s) => spec.leadKeyword.test(s) && /\d/.test(s),
    10,
  );
  if (hit) {
    // キーワードの位置から探す（無関係な数字を拾わないため）。
    const idx = hit.search(spec.leadKeyword);
    const tail = idx >= 0 ? hit.slice(idx) : hit;
    const value = measureValue(tail, spec.defaultUnit);
    if (value) {
      return makeFact(
        article,
        "measure",
        value,
        `${label}の${spec.name}は${value}です。`,
        hit,
      );
    }
  }
  return null;
}

/**
 * 会社のインフォボックスは代表者欄に複数名が並ぶことがある：
 * 「岡田元也（取締役兼代表執行役会長） 吉田昭夫（取締役兼代表執行役社長）」。
 * 役職の注記が聞かれた役職と合う人の名前だけを返す。
 */
function personForRole(raw, role) {
  for (const seg of raw.split(/\s+/).filter((s) => s.length > 0)) {
    const m = /^(.{2,12}?)（([^）]*)）/.exec(seg);
    if (m && m[2]?.includes(role)) {
      const name = (m[1] ?? "").trim();
      if (name) return name;
    }
  }
  return null;
}

/** 先頭の役職名を落とす（「代表取締役社長 大野 惠司」→「大野 惠司」）。 */
function cleanPersonName(raw) {
  return raw
    .replace(
      /^(代表取締役社長|代表執行役社長|代表取締役|取締役社長|代表者|社長|会長)\s+/,
      "",
    )
    .trim();
}

/**
 * 現職の総理大臣：「日本の総理大臣」は（Wikipediaのリダイレクトで）
 * 内閣総理大臣の一覧に解決され、そのリードが現職をそのまま述べている
 * ——「現職は衆議院議員の高市早苗（2025年10月21日就任）で…」。名前は
 * 役職の接頭辞の後、続く括弧の前まで。インフォボックスに現職者欄が
 * あればそちらを優先する。
 */
function extractPrimeMinister(article, label) {
  const fromField = article.fields["現職者"] ?? article.fields["現職"];
  if (fromField && fromField.length <= 30 && !/[。]/.test(fromField)) {
    const value = cleanPersonName(fromField);
    if (value) {
      return makeFact(
        article,
        "person",
        value,
        `${label}の総理大臣は${value}です。`,
        fieldEvidence(article, "現職者", fromField, value.slice(0, 4)),
      );
    }
  }
  const hit = findSentence(article.lead, (s) => /現職(?:は|者)/.test(s), 10);
  if (!hit) return null;
  const m = /現職(?:は|者[はが]?)(.{2,24}?)(?:（|、|。)/.exec(hit);
  const captured = (m?.[1] ?? "").trim();
  // 「衆議院議員の高市早苗」——名前は役職の語の後の最後の「の」の後。
  const value = captured.includes("議員の")
    ? (captured.split("議員の").pop() ?? "").trim()
    : captured;
  if (
    value.length >= 2 &&
    value.length <= 12 &&
    !/[。『』、（）()：:はがをにへとやの]/.test(value)
  ) {
    return makeFact(
      article,
      "person",
      value,
      `${label}の総理大臣は${value}です。`,
      hit,
    );
  }
  return null;
}

/**
 * 現職の知事：「福島県知事」は（Wikipediaのリダイレクトで）福島県知事
 * 一覧に解決され、その記事は {{Infobox Political post}} に現職者欄を
 * 持つ（「現職者 = 内堀雅雄（第22代）」）。総理大臣と同じく現職者欄を
 * 優先し、無ければリードの「現職は…」の形を試す。都道府県名は質問文
 * から取る（記事タイトルは「…知事一覧」になるため）。
 */
function extractGovernor(question, article) {
  const pref = (
    /(.{2,8}?(?:都|道|府|県))知事/.exec(question)?.[1] ??
    article.title.replace(/(?:知事一覧|知事)$/, "")
  ).replace(/^(現在の|現|今の)/, "");
  const fieldNames = ["現職者", "現職", "知事", "都知事", "県知事", "府知事", "道知事"];
  for (const name of fieldNames) {
    const raw = article.fields[name];
    if (!raw || /[。]/.test(raw)) continue;
    // 「内堀雅雄（第22代）」の代数の注記は名前ではないので落とす。
    const value = cleanPersonName(raw.split(/[（(]/)[0]?.trim() ?? "");
    if (
      value.length >= 2 &&
      value.length <= 12 &&
      !/[。『』、（）()：:はがをにへとやの]/.test(value)
    ) {
      return makeFact(
        article,
        "person",
        value,
        `${pref}の知事は${value}です。`,
        fieldEvidence(article, name, raw, value.slice(0, 4)),
      );
    }
  }
  const hit = findSentence(article.lead, (s) => /現職(?:は|者)/.test(s), 10);
  if (hit) {
    const m = /現職(?:は|者[はが]?)(.{2,24}?)(?:（|、|。)/.exec(hit);
    const captured = (m?.[1] ?? "").trim();
    const value = captured.includes("議員の")
      ? (captured.split("議員の").pop() ?? "").trim()
      : captured;
    if (
      value.length >= 2 &&
      value.length <= 12 &&
      !/[。『』、（）()：:はがをにへとやの]/.test(value)
    ) {
      return makeFact(
        article,
        "person",
        value,
        `${pref}の知事は${value}です。`,
        hit,
      );
    }
  }
  return null;
}

function extractPerson(question, article, label) {
  if (/知事/.test(question)) {
    return extractGovernor(question, article);
  }
  if (/内閣総理大臣|総理大臣|首相/.test(question)) {
    const leaderLabel = question.includes("日本") ? "日本" : label;
    return extractPrimeMinister(article, leaderLabel);
  }
  let role = "";
  let fieldNames = [];
  if (/社長/.test(question)) {
    role = "社長";
    fieldNames = ["社長", "代表取締役社長", "代表取締役", "代表者"];
  } else if (/会長/.test(question)) {
    role = "会長";
    fieldNames = ["会長", "代表者"];
  } else if (/創業者|創立者/.test(question)) {
    role = "創業者";
    fieldNames = ["創業者", "創立者", "設立者"];
  } else if (/代表/.test(question)) {
    role = "代表者";
    fieldNames = ["代表者", "代表取締役", "社長"];
  }
  if (!role) return null;
  for (const name of fieldNames) {
    const fromField = article.fields[name];
    if (!fromField || /[。]/.test(fromField)) continue;
    const value =
      personForRole(fromField, role) ?? cleanPersonName(fromField);
    if (value && value.length <= 30) {
      return makeFact(
        article,
        "person",
        value,
        `${label}の${role}は${value}です。`,
        fieldEvidence(article, name, fromField, value.slice(0, 4)),
      );
    }
  }
  return null;
}

/**
 * 名物：その土地で一番知られた郷土料理。記事解決の側が、料理自身の
 * 記事が「その土地の郷土料理／名物」と自称している場合に名物欄として
 * 付けてくれる（根拠の文は 名物_根拠 に入る）。リード文のパターンも
 * 直接試す。
 */
function extractSpecialty(article, label) {
  const fromField =
    article.fields["名物"] ??
    article.fields["特産品"] ??
    article.fields["名産品"];
  if (fromField && fromField.length <= 30 && !/[。]/.test(fromField)) {
    const evidence =
      article.fields["名物_根拠"] ??
      fieldEvidence(article, "名物", fromField, fromField);
    return makeFact(
      article,
      "specialty",
      fromField,
      `${label}の名物は${fromField}です。`,
      evidence,
    );
  }
  const patterns = [
    /名物は、?『?(.{2,20}?)』?(?:です|である|として知られ|。)/,
    /名物といえば、?『?(.{2,20}?)』?(?:です|である|。)/,
    /『?(.{2,20}?)』?が名物(?:です|である)/,
  ];
  for (const re of patterns) {
    const hit = findSentence(article.lead, (s) => re.test(s), 14);
    if (!hit) continue;
    const m = re.exec(hit);
    const value = (m?.[1] ?? "").trim();
    if (
      value.length >= 2 &&
      value.length <= 20 &&
      !/[。『』、（）()：:はがをにへとや]/.test(value)
    ) {
      return makeFact(
        article,
        "specialty",
        value,
        `${label}の名物は${value}です。`,
        hit,
      );
    }
  }
  // 地元の料理を並べる一節——「…の発祥地としてや、たこ焼きや
  // お好み焼き、串カツといった…」——では一番知られたものが先に来る。
  const listHit = findSentence(
    article.lead,
    (s) => /(発祥|名物|郷土料理)/.test(s) && /としてや、/.test(s),
    30,
  );
  if (listHit) {
    const m = /としてや、(.{2,14}?)や/.exec(listHit);
    const value = (m?.[1] ?? "").trim();
    if (
      value.length >= 2 &&
      !/[。『』、（）()：:はがをにへとや]/.test(value)
    ) {
      return makeFact(
        article,
        "specialty",
        value,
        `${label}の名物は${value}です。`,
        listHit,
      );
    }
  }
  return null;
}

const ATTRIBUTE_SPECS = [
  { test: /国歌/, field: "国歌", name: "国歌", quote: true },
  { test: /首都/, field: "首都", name: "首都", quote: false },
];

/**
 * 属性欄の値を名前だけに整える：File: の残りを落とし、末尾の括弧を
 * 切る——日本の首都欄は「東京都（事実上…）」と読み、首都は東京都だけ。
 */
function cleanAttributeValue(raw) {
  return raw
    .replace(/File:[^\s]*/g, "")
    .split("（")[0]
    ?.split("(")[0]
    ?.replace(/[、。\s]+$/, "")
    .trim() ?? "";
}

/**
 * 属性（国歌／首都）：対象のインフォボックスの名前つき欄そのもの
 * ——日本の基礎情報 国 は国歌=君が代 と 首都=東京都 の両方を持つ。
 */
function extractAttribute(question, article, label) {
  const spec = ATTRIBUTE_SPECS.find((s) => s.test.test(question));
  if (!spec) return null;
  const fromField = article.fields[spec.field];
  if (fromField) {
    const value = cleanAttributeValue(fromField);
    if (value && value.length <= 30 && !/[。]/.test(value)) {
      const answer = spec.quote
        ? `${label}の${spec.name}は「${value}」です。`
        : `${label}の${spec.name}は${value}です。`;
      return makeFact(
        article,
        "attribute",
        value,
        answer,
        fieldEvidence(article, spec.field, fromField, value),
      );
    }
  }
  if (spec.field === "国歌") {
    const re = /国歌は?「(.{1,24}?)」/;
    const hit = findSentence(article.lead, (s) => re.test(s), 14);
    if (hit) {
      const m = re.exec(hit);
      const value = (m?.[1] ?? "").trim();
      if (value) {
        return makeFact(
          article,
          "attribute",
          value,
          `${label}の国歌は「${value}」です。`,
          hit,
        );
      }
    }
  }
  return null;
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractDefinition(article) {
  const titleNorm = normalizeJa(article.title);
  const baseNorm = titleNorm.replace(/[（(].*?[）)]/g, "").trim();
  const allSentences = splitSentences(article.lead);
  const namesSubject = (s) => {
    const ns = normalizeJa(s).replace(/^『/, "");
    const startsWithTitle =
      ns.startsWith(titleNorm) ||
      ns.startsWith(`${titleNorm}』`) ||
      ns.startsWith(`${titleNorm}（`);
    // 修飾つきタイトルは定義文の途中に現れることがある：
    // 球種 (野球) の定義は「野球における球種（きゅうしゅ）は、…である。」。
    // 基本名＋読みが文の前半にあり は が続けば定義とみなす。
    const head = ns.slice(0, 24);
    return (
      startsWithTitle ||
      (baseNorm.length >= 2 &&
        (head.includes(`${baseNorm}（`) || head.includes(`${baseNorm}は`)) &&
        /は、?/.test(ns.slice(0, 40)))
    );
  };
  for (const [idx, s] of allSentences.slice(0, 4).entries()) {
    // 案内文（「この項目では、〜について説明しています。」）や
    // 曖昧さ回避への誘導文は、対象を名指していても定義ではない。
    if (/^この項目では/.test(s.trim())) continue;
    if (/曖昧さ回避/.test(s)) continue;
    if (!namesSubject(s)) continue;
    // 同語反復の文（「立方体は立方体である」等）は定義として採用しない。
    if (isTautologicalAnswer(s)) continue;
    // 意味の列挙への導入文（「Xは、以下の…意味で用いられる」）自体は
    // 定義ではない。専用の処理が下にある。
    if (/以下の/.test(s)) continue;
    const trimmed = s.trim();
    const copula =
      /(である|です|をいう|とされる|のこと|にあたる)[。]?$/.test(trimmed);
    // 体言止め：リード文が名詞で終わる定義も多い
    // （「…は、福島県伊達市に開業予定のショッピングモール。」）。
    // この特例は記事の最初の一文だけ：慣例としてその文が定義であり、
    // 文末の漢字・カタカナは動詞・形容詞の語尾にはならない。
    const nounEnding =
      idx === 0 && /[一-鿿々ァ-ヶー][。]?$/.test(trimmed);
    if (!copula && !nounEnding) continue;
    if (s.length > 220) continue;
    return makeFact(article, "definition", s, s, s);
  }
  // 意味の列挙型リード：「地震（じしん）は、以下の2つの意味で用いられる。」
  // の後に「地震学における定義: …ずれ動くこと。」と続く場合、
  // 最初の意味の本文が定義そのものなので、直接の答えに組み立てる。
  const first = allSentences[0];
  if (
    first &&
    namesSubject(first) &&
    /以下/.test(first) &&
    /(意味で用いられる|を指す|とおり)/.test(first)
  ) {
    for (const s of allSentences.slice(1, 6)) {
      const m = /^[^:：]{1,24}[:：]\s*(.+)$/.exec(s.trim());
      const body = m?.[1]?.trim();
      if (!body || body.length < 15 || body.length > 200) continue;
      if (!/(こと|である|です)[。]?$/.test(body)) continue;
      const core = body.replace(/。$/, "").replace(/こと$/, "");
      const answer = `${article.title}とは、${core}ことです。`;
      // 意味の中身が記事タイトルそのもの（同語反復）なら採用しない。
      if (isTautologicalAnswer(answer)) continue;
      if (
        normalizeForCompare(core) === normalizeForCompare(titleNorm) ||
        normalizeForCompare(core) === normalizeForCompare(baseNorm)
      ) {
        continue;
      }
      return makeFact(article, "definition", answer, answer, s);
    }
  }
  // 逆向きの定義：定義文の末尾で対象を名指す
  // （「…力を解消する現象が地震である。」）。リードの少し奥まで探す。
  const names = [titleNorm, baseNorm].filter(
    (n, i, arr) => n.length >= 2 && arr.indexOf(n) === i,
  );
  for (const name of names) {
    const re = new RegExp(
      `(?:が|とは|は)${escapeRegExp(name)}(?:である|です)[。]?$`,
    );
    for (const s of allSentences.slice(0, 12)) {
      if (s.length > 220) continue;
      if (/^この項目では/.test(s.trim())) continue;
      if (/曖昧さ回避/.test(s)) continue;
      if (isTautologicalAnswer(s)) continue;
      if (re.test(normalizeJa(s).trim())) {
        return makeFact(article, "definition", s, s, s);
      }
    }
  }
  return null;
}

/**
 * 運営会社：聞かれているのは運営される側（ブランド・施設）で、答えは
 * 運営する会社の社名。置き場は3通りある。
 * 1) 対象記事の「運営会社」などの明示欄。
 * 2) 記事自体が運営会社の記事：モーリーファンタジーはイオンファン
 *    タジーの記事へ飛び、その会社記事は 社名 欄を持ち、リードで
 *    「『モーリーファンタジー』を…運営する」と対象を名指ししている。
 * 3) ブランド側の記事のリード：「X株式会社が運営する…」の形。
 * ラベルは記事タイトルではなく質問の題名（ブランド名）を使う。
 */
function extractOperator(question, article) {
  const topic = topicFromQuestion(question);
  const label = topic ?? article.title;
  for (const name of ["運営会社", "運営元", "運営者", "運営主体", "事業者"]) {
    const fromField = article.fields[name];
    if (fromField && fromField.length <= 40 && !/[。]/.test(fromField)) {
      return makeFact(
        article,
        "operator",
        fromField,
        `${label}の運営会社は${fromField}です。`,
        fieldEvidence(article, name, fromField, fromField.slice(0, 6)),
      );
    }
  }
  const company = article.fields["社名"];
  if (
    company &&
    topic &&
    company.length <= 40 &&
    !/[。]/.test(company) &&
    normalizeJa(article.lead).includes(normalizeJa(topic)) &&
    /運営/.test(article.lead)
  ) {
    return makeFact(
      article,
      "operator",
      company,
      `${label}の運営会社は${company}です。`,
      fieldEvidence(article, "社名", company, topic),
    );
  }
  const patterns = [
    /([一-鿿々ァ-ヶーA-Za-z0-9・]{2,30}?(?:株式会社|有限会社|合同会社|会社|グループ|財団法人|社団法人|公社|機構))が運営/,
    /([一-鿿々ァ-ヶーA-Za-z0-9・]{2,30}?(?:株式会社|有限会社|合同会社))の運営(?:する|による|で)/,
    /は、?([一-鿿々ァ-ヶーA-Za-z0-9・]{2,30}?(?:株式会社|有限会社|合同会社))により運営/,
  ];
  for (const re of patterns) {
    const hit = findSentence(article.lead, (s) => re.test(s), 14);
    if (!hit) continue;
    const value = (re.exec(hit)?.[1] ?? "").trim();
    if (
      value.length >= 2 &&
      value.length <= 30 &&
      !/[。『』、（）()：:はがをにへとや]/.test(value)
    ) {
      return makeFact(
        article,
        "operator",
        value,
        `${label}の運営会社は${value}です。`,
        hit,
      );
    }
  }
  return null;
}

/**
 * 対象記事から、質問の種類に合う事実を取り出す。
 * 取れなければ null（関係ありそうな文を答えとして貼らないための関門）。
 */
function extractFact(question, article) {
  if (!article) return null;
  const kind = classifyQuestion(question);
  if (!kind) return null;
  const subject = subjectOf(question);
  const label = subject ? `『${subject}』` : article.title;
  switch (kind) {
    case "author":
      return extractAuthor(article, label);
    case "hq":
      return extractHq(article, label);
    case "place":
      return extractPlace(question, article, label);
    case "date":
      return extractDate(question, article, label);
    case "measure":
      return extractMeasure(question, article, label);
    case "person":
      return extractPerson(question, article, label);
    case "operator":
      return extractOperator(question, article);
    case "specialty":
      return extractSpecialty(article, label);
    case "attribute":
      return extractAttribute(question, article, label);
    case "definition":
      return extractDefinition(article);
    default:
      return null;
  }
}

/* ------------------------------------------------------------------ */
/* 天気の質問                                                          */
/*                                                                    */
/* 天気は変わる情報であり、覚えられる事実ではない：聞かれた時点で      */
/* 天気APIから答え、知識メモ帳には決して入れない。parseWeatherQuery   */
/* は「天気の質問かどうか」「場所の指定（無ければ既定の場所を使う）」  */
/* 「何日のことか」だけを決める。                                      */
/* ------------------------------------------------------------------ */

function parseWeatherQuery(q) {
  if (!/天気|てんき/.test(q)) return null;
  let day = "today";
  if (/明後日|あさって/.test(q)) day = "dayafter";
  else if (/明日|あした/.test(q)) day = "tomorrow";
  else if (/週間|今週|週末/.test(q)) day = "week";
  // 日の語を先に落とす：「明日の郡山の天気」も「郡山の明日の天気」も
  // 天気 の前に場所だけが残るようにする。
  const s = q.replace(/今日|明日|明後日|あさって|あした|今週|週間|週末/g, " ");
  const m = /^(.{2,24}?)(?:の|は|って)?\s*天気/.exec(s.trim());
  let place = (m?.[1] ?? "")
    .replace(/[\s　]+/g, "")
    .replace(/^(の|は)+/, "")
    .replace(/(の|は|って|を)+$/, "")
    .trim();
  if (/^(今|現在|そこ|ここ|あそこ)$/.test(place)) place = "";
  // 1文字の地名（津など）も、場所として指定されたなら使う。
  return { place: place.length >= 1 ? place : null, day };
}

/* ================================================================== */
/* ブラウザ内Wikipedia索引（検索レーン：索引優先・ウェブ予備）         */
/*                                                                    */
/* 全記事の索引を Hugging Face から Range 取得で引く検索エンジン       */
/* （BrowserSearch）と、その転送＋アダプタ（CubeIndexLane）をこの節に  */
/* 取り込む。Wikipedia 情報の取得元だけを「索引優先・ウェブ予備」に    */
/* 切り替えるためのもので、回答ポリシーは一切変えない。索引で成功      */
/* したら WikiNet.via は "index" になる（ディスパッチャ側で記録）。    */
/* 自作モデル用の推論エンジンとは別物で、公開名は BrowserSearch /      */
/* CubeIndexLane のみ。                                               */
/* ================================================================== */

"use strict";
/* Browser/Node search engine replicating hybrid/search.py (INDEX_FIX2
 * final, md5 64fb744c3066ecf62887986420fc543c) over the format-2
 * Range-fetch shards in this directory (see ../BROWSER_INDEX_FULL.md).
 *
 * The ranking pipeline (title lanes -> FTS pool -> evidence lanes ->
 * best-paragraph re-rank) is a line-by-line port of search.py. The one
 * deliberate approximation is the FTS5 bm25 pool: postings are
 * doc-level, so per-term candidates are gathered by trigram
 * intersection, verified against the fetched title/lead text, and
 * scored with the Okapi bm25 formula (k1=1.2, b=0.75, column weights
 * 10/1, per-source avgdl from meta.json). search.py only uses bm25 to
 * *select* its pools (LIMIT 12/20/30/100 per source); final order is
 * the deterministic paragraph re-rank, which is ported exactly.
 *
 * String indexing follows Python (code points) where search.py slices
 * leads/bodies; a few hot paths use UTF-16 code units, which coincide
 * for BMP text (virtually all of jawiki).
 *
 * Transport is pluggable: any object with
 *   range(file, a, b) -> Promise<Uint8Array>   (b inclusive)
 *   whole(file)       -> Promise<Uint8Array>
 * where `file` is a shard name for index files, or a body file name
 * (wiki_full.txt / supplement_corpus.txt) for article bodies.
 */
(function (global) {
  /* Cube integration copy: fetch loops parallelized with bounded
   * concurrency (indexed mapLimit, insertion order preserved) and bulk
   * record priming with coalesced Range fetches. Ranking logic is
   * byte-identical to the master engine; only IO scheduling changed. */
  async function mapLimit(items, limit, fn) {
    const out = new Array(items.length);
    let next = 0;
    const n = Math.min(limit, items.length);
    const workers = [];
    for (let w = 0; w < n; w++) {
      workers.push((async () => {
        while (next < items.length) {
          const i = next++;
          out[i] = await fn(items[i], i);
        }
      })());
    }
    await Promise.all(workers);
    return out;
  }
  const te = new TextEncoder();
  const td = new TextDecoder("utf-8"); // non-fatal ~= errors="replace"

  // ------------------------------------------------------------ helpers
  const cps = (s) => [...s];
  const cpLen = (s) => [...s].length;
  const cpSlice = (s, a, b) => [...s].slice(a, b).join("");
  const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  function cmpBytes(a, b) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return a.length === b.length ? 0 : (a.length < b.length ? -1 : 1);
  }
  function readVarint(buf, pos) {
    let val = 0, shift = 0;
    while (pos < buf.length) {
      const b = buf[pos++];
      val += (b & 0x7f) * Math.pow(2, shift);
      if (!(b & 0x80)) break;
      shift += 7;
    }
    return [val, pos];
  }
  function countOverlap(s, sub) {
    if (!sub) return 0;
    let n = 0, i = 0;
    while ((i = s.indexOf(sub, i)) >= 0) { n++; i += 1; }
    return n;
  }
  function countNonOverlap(s, sub) {
    if (!sub) return 0;
    let n = 0, i = 0;
    while ((i = s.indexOf(sub, i)) >= 0) { n++; i += sub.length; }
    return n;
  }
  function trigramsOf(s) {
    const a = cps(s), out = [];
    for (let i = 0; i + 3 <= a.length; i++) out.push(a.slice(i, i + 3).join(""));
    return out;
  }

  // --------------------------------------- search.py constants (verbatim)
  const STOP_SPLIT = /[はがをにのへとやもだですますか？?、。・\s「」『』（）()]+/;
  const SYNONYMS = { "作者": ["作者", "著者"], "高さ": ["高さ", "標高"] };
  const KANA_MAP = { "ヰ": "ウィ", "ヱ": "エ", "ヲ": "オ", "ヶ": "ケ",
                     "ヵ": "カ", "ヂ": "ジ", "ヅ": "ズ" };
  const VOICED = { "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
    "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
    "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
    "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
    "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
    "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
    "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
    "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ" };
  const WORD_VARIANTS = { "珈琲": "コーヒー" };
  const CHAR_SWAPS = [["冨", "富"]];
  const GENERIC_TITLES = new Set(["高さ", "標高", "面積", "人口", "全長",
    "長さ", "深さ", "水深", "速さ", "速度", "名前", "名称", "国歌",
    "競技", "世界遺産", "都道府県", "監督", "化学式", "世界", "砂漠",
    "日本人", "首都"]);
  const NEVER_TITLES = new Set(["どこ", "誰", "何"]);
  const SCOPE_TITLES = new Set(["日本", "世界", "地球", "太陽系", "国内",
    "太陽"]);
  const FTS_JUNK = ["ですか", "ますか", "ください", "教えて", "でしょう",
    "は何", "の名前", "ありますか"];
  const FTS_STOP = new Set(["起きた", "始まった", "終わった", "あります",
    "います", "された", "なった", "アメリカ", "イギリス", "フランス",
    "ドイツ", "イタリア", "スペイン", "中国", "韓国", "ロシア",
    "カナダ", "オーストラリア"]);
  const SUBSTRING_STOP = new Set(["メートル", "キロメートル",
    "センチメートル", "ミリメートル", "キログラム", "グラム", "キロ",
    "メー", "リットル", "トン", "カロリー"]);
  const SUPER_RE = /一番|最も|最大|最小|最高|最古|最新|最初|世界一|日本一|第[0-9一二三四五六七八九十]+位|[0-9一二三四五六七八九十]+番目/;
  const SUPER_FRAGMENT_RE = /^(世界|日本|地球|国内|太陽系|全)?(最大|最小|最高|最低|最古|最新|最長|最短|最速|最深|世界一|日本一|一番|最も|最初|最後)$/;
  const ORDINAL_RE = /([0-9]+|[一二三四五六七八九十]+)番目|第([0-9]+|[一二三四五六七八九十]+)位/;
  const KANJI_NUM = { "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10 };
  const ATTR_SYNONYMS = [
    [["高い", "高さ"], ["標高", "高い", "最高峰", "高峰", "高さ"], "高い"],
    [["大きい", "大きな", "広い"], ["面積", "大きい", "最大", "広い"], "大きい"],
    [["長い", "最長"], ["全長", "長い", "最長"], "長い"],
    [["深い", "最深"], ["水深", "深い", "最深"], "深い"],
    [["近い"], ["近い", "最も近い"], "近い"],
    [["遠い"], ["遠い", "最も遠い"], "遠い"],
    [["古い", "最古"], ["最古", "古い"], "古い"],
    [["多い"], ["多い", "最大"], "多い"],
    [["重い"], ["質量", "重い", "最大"], "重い"],
    [["速い", "早い"], ["速さ", "速度", "速い"], "速い"],
    [["新しい"], ["新しい", "最新"], "新しい"],
    [["強い"], ["強い", "最強"], "強い"],
  ];
  const CATEGORY_SUFFIX = {
    "山": ["山", "岳", "峰"], "湖": ["湖", "沼", "池"],
    "川": ["川", "河"], "砂漠": ["砂漠"], "城": ["城"],
    "寺": ["寺"], "橋": ["橋"], "島": ["島"], "滝": ["滝"],
    "海": ["海"], "渓谷": ["渓谷"], "半島": ["半島"],
  };
  const CREATOR_RE = /作者|著者|原作者|描いた|描き|監督|作曲|作詞/;
  const ATTRIBUTION_RE = /『[^』]+』(（[^）]*）)?は、[^。]{0,60}?(による|の)(短編小説|長編小説|中編小説|小説|漫画|絵画|戯曲|詩集|楽曲)/;
  const SUPER_WORD_RE = /最も|一番|最大|最高|最古|最長|最小|最高峰|世界一|日本一|最初|第一/;
  const STEM_FAMILY = { "最古": "古", "最大": "大", "最高": "高",
    "最高峰": "高", "最長": "長", "最深": "深", "最速": "速",
    "最新": "新", "最初": "初", "第一": "位", "最小": "小" };
  const ATTR_FAMILY = { "高い": "高", "大きい": "大", "長い": "長",
    "深い": "深", "古い": "古", "速い": "速", "新しい": "新" };
  const ADJ_SET = new Set(["高い", "大きい", "長い", "深い", "近い",
    "遠い", "古い", "多い", "重い", "速い", "新しい", "強い"]);
  const CLAIM_SCOPES = { "世界": ["世界", "地球"], "地球": ["地球", "世界"],
    "日本": ["日本", "世界", "地球"], "太陽": ["太陽", "太陽系"],
    "太陽系": ["太陽系"] };

  // ------------------------------------- search.py pure functions (port)
  function normalizeTerm(s) {
    const out = [];
    for (const ch of s) {
      if (ch === "ゝ" || ch === "ヽ") out.push(out.length ? out[out.length - 1] : ch);
      else if (ch === "ゞ" || ch === "ヾ") {
        const prev = out.length ? out[out.length - 1] : ch;
        out.push(VOICED[prev] || prev);
      } else out.push(KANA_MAP[ch] || ch);
    }
    return out.join("");
  }
  function oldIterationForms(s) {
    const a = cps(s), forms = [];
    for (let i = 1; i < a.length; i++) {
      if (a[i] === a[i - 1] && "ぁ" <= a[i] && a[i] <= "ゖ") {
        forms.push(a.slice(0, i).join("") + "ゝ" + a.slice(i + 1).join(""));
      }
    }
    return forms;
  }
  function termVariants(t) {
    let vs = [t];
    const n = normalizeTerm(t);
    if (n !== t) vs.push(n);
    const w = WORD_VARIANTS[t] || WORD_VARIANTS[n];
    if (w && !vs.includes(w)) vs.push(w);
    for (const base of [...vs]) {
      if (base.includes("ヰ")) vs.push(base.split("ヰ").join("ウイ"));
    }
    for (const f of oldIterationForms(n)) if (!vs.includes(f)) vs.push(f);
    for (const [a, b] of CHAR_SWAPS) {
      for (const base of [...vs]) {
        if (base.includes(a)) vs.push(base.split(a).join(b));
        if (base.includes(b)) vs.push(base.split(b).join(a));
      }
    }
    return [...new Set(vs)].slice(0, 6);
  }
  function splitParts(q, minlen) {
    return q.split(STOP_SPLIT).filter((p) => cpLen(p) >= minlen);
  }
  function paraKeys(q) {
    let keys = [];
    for (const k of splitParts(q, 2)) {
      for (const v of termVariants(k)) keys = keys.concat(SYNONYMS[v] || [v]);
    }
    return [...new Set(keys)];
  }
  function bestParagraph(body, keys) {
    const paras = body.split("\n").map((p) => p.trim()).filter((p) => p);
    if (!paras.length) return [body.slice(0, 300), 0, 0];
    const score = (p) => keys.reduce((n, k) => n + (p.includes(k) ? 1 : 0), 0);
    let best = paras[0], bs = score(best);
    for (const p of paras) {
      const s = score(p);
      if (s > bs) { best = p; bs = s; }
    }
    const s = bs;
    const occ = Math.min(keys.reduce((n, k) => n + countNonOverlap(best, k), 0), 9);
    if (cpLen(best) > 420) {
      const present = keys.filter((k) => best.includes(k));
      let anchor = null, aKey = null;
      for (const k of present) {
        const cand = [countNonOverlap(best, k), best.indexOf(k)];
        if (aKey === null || cand[0] < aKey[0] ||
            (cand[0] === aKey[0] && cand[1] < aKey[1])) {
          aKey = cand; anchor = k;
        }
      }
      const pos = anchor ? best.indexOf(anchor) : 0;
      const st = best.slice(0, pos).lastIndexOf("。") + 1;
      best = cpSlice(best, st, st + 420);
    }
    return [best, s, occ];
  }
  function ordinalOf(q) {
    const m = q.match(ORDINAL_RE);
    if (!m) return null;
    const s = m[1] !== undefined ? m[1] : m[2];
    if (s === undefined || s === null) return null;
    if (/^[0-9]+$/.test(s)) return parseInt(s, 10);
    return KANJI_NUM[s] !== undefined ? KANJI_NUM[s] : null;
  }
  function scopeOf(q) {
    for (const w of ["太陽系", "世界", "日本", "地球", "国内", "太陽"]) {
      if (q.includes(w)) return w === "国内" ? "日本" : w;
    }
    return null;
  }
  function attrOf(q) {
    for (const [words, syns, base] of ATTR_SYNONYMS) {
      if (words.some((w) => q.includes(w))) return [syns, base];
    }
    return [[], null];
  }
  function categoryOf(q) {
    for (const cat of Object.keys(CATEGORY_SUFFIX)) {
      if (q.includes(cat)) return cat;
    }
    return null;
  }

  // ---------------------------------------------------------------- engine
  const MAX_POSTING_BYTES = 1 << 20; // per-trigram fetch cap
  const FTS_FETCH_CAP = 600;         // leads scored per (term, source)

  class SearchEngine {
    /** transport: {range(file,a,b), whole(file)}.
     * opts: {bodyFiles:[f0,f1]} overrides meta body file names
     * (transport resolves them). */
    constructor(transport, opts = {}) {
      this.tr = transport;
      this.opts = opts;
      this.meta = null;
      this.titleIdx = null;   // [{key, off, firstId, firstLeadOff}]
      this.triSuper = null;   // [{key, off}]
      this.titleTriSuper = null;
      this.redirIdx = null;   // [{key, off}]
      this.claims = null;     // [[gid, scope], ...]
      this.categories = null; // {cat: [gid, ...]}
      this._blocks = new Map();   // file@off -> Uint8Array
      this._leads = new Map();    // gid -> {title, lead}
      this._idRecs = new Map();   // gid -> rec|null
      this._termCache = new Map(); // src|term -> Map(gid -> score)
      this.stats = { requests: 0, bytes: 0 };
    }

    async _range(file, a, b) {
      const buf = await this.tr.range(file, a, b);
      this.stats.requests += 1; this.stats.bytes += buf.length;
      return buf;
    }
    async _whole(file) {
      const buf = await this.tr.whole(file);
      this.stats.requests += 1; this.stats.bytes += buf.length;
      return buf;
    }
    async _block(file, off, size) {
      const key = file + "@" + off;
      const hit = this._blocks.get(key);
      if (hit) return hit;
      const buf = await this._range(file, off, off + size - 1);
      if (this._blocks.size > 256) this._blocks.clear();
      this._blocks.set(key, buf);
      return buf;
    }

    async init() {
      if (this.meta) return this;
      this.meta = JSON.parse(td.decode(await this._whole("meta.json")));
      const parseIdx = (buf, withIds) => {
        const dv = new DataView(buf.buffer, buf.byteOffset);
        const out = [];
        let p = 0;
        while (p < buf.length) {
          const kl = dv.getUint16(p, true); p += 2;
          const key = buf.slice(p, p + kl); p += kl;
          const off = Number(dv.getBigUint64(p, true)); p += 8;
          if (withIds) {
            const firstId = dv.getUint32(p, true); p += 4;
            const firstLeadOff = Number(dv.getBigUint64(p, true)); p += 8;
            out.push({ key, off, firstId, firstLeadOff });
          } else out.push({ key, off });
        }
        return out;
      };
      const parseSuper = (buf) => {
        const dv = new DataView(buf.buffer, buf.byteOffset);
        const out = [];
        let p = 0;
        while (p < buf.length) {
          const kl = buf[p++];
          const key = buf.slice(p, p + kl); p += kl;
          const off = Number(dv.getBigUint64(p, true)); p += 8;
          out.push({ key, off });
        }
        return out;
      };
      this.titleIdx = parseIdx(await this._whole("titles.idx"), true);
      this.triSuper = parseSuper(await this._whole("tri_super.idx"));
      this.titleTriSuper = parseSuper(await this._whole("title_tri_super.idx"));
      this.redirIdx = parseIdx(await this._whole("redir.idx"), false);
      this.claims = JSON.parse(td.decode(await this._whole("claims.json")));
      this.categories = JSON.parse(td.decode(await this._whole("categories.json")));
      return this;
    }

    _findBlock(idx, targetBytes) {
      let lo = 0, hi = idx.length - 1, ans = 0;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (cmpBytes(idx[mid].key, targetBytes) <= 0) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
      }
      return ans;
    }
    _blockEnd(idx, i, fileSize) {
      return i + 1 < idx.length ? idx[i + 1].off : fileSize;
    }
    _scanTitleBlock(buf, firstLeadOff) {
      const dv = new DataView(buf.buffer, buf.byteOffset);
      const out = [];
      let p = 0, leadOff = firstLeadOff;
      while (p < buf.length) {
        const tl = dv.getUint16(p, true); p += 2;
        if (p + tl > buf.length) break;
        const title = td.decode(buf.slice(p, p + tl)); p += tl;
        let id, bodyOff, bodyLen, leadDelta, leadLen;
        [id, p] = readVarint(buf, p);
        [bodyOff, p] = readVarint(buf, p);
        [bodyLen, p] = readVarint(buf, p);
        [leadDelta, p] = readVarint(buf, p);
        [leadLen, p] = readVarint(buf, p);
        const bodyId = buf[p++];
        leadOff += leadDelta;
        out.push({ title, id, src: bodyId, bodyOff, bodyLen, bodyId,
                   leadOff, leadLen });
      }
      return out;
    }
    /** All title records whose title bytes lie in [loB, hiB). */
    async _titleRange(loB, hiB) {
      await this.init();
      const size = this.meta.files["titles.bin"];
      let bi = this._findBlock(this.titleIdx, loB);
      const startBi = bi;
      const out = [];
      for (; bi < this.titleIdx.length; bi++) {
        const blk = this.titleIdx[bi];
        if (bi > startBi && cmpBytes(blk.key, hiB) >= 0) break;
        const end = this._blockEnd(this.titleIdx, bi, size);
        const buf = await this._block("titles.bin", blk.off, end - blk.off);
        const recs = this._scanTitleBlock(buf, blk.firstLeadOff);
        if (!recs.length) break;
        let blockMax = null;
        for (const r of recs) {
          const kb = te.encode(r.title);
          blockMax = kb;
          if (cmpBytes(kb, loB) >= 0 && cmpBytes(kb, hiB) < 0) out.push(r);
        }
        if (blockMax && cmpBytes(blockMax, hiB) >= 0) break;
      }
      return out;
    }
    /** search.py exact_all: title == t, first 2 per source. */
    async exactAll(title) {
      const tb = te.encode(title);
      const hi = new Uint8Array(tb.length + 1); // tb + 0x00: strict bound
      hi.set(tb);
      const rows = await this._titleRange(tb, hi);
      const per = { 0: 0, 1: 0 };
      const out = [];
      for (const r of rows) {
        if (r.title !== title) continue;
        if (per[r.src] >= 2) continue;
        per[r.src]++; out.push(r);
      }
      return out;
    }
    /** search.py variant_all: 'title (...)' ranges, LIMIT 6 per source
     * per range. */
    async variantAll(title) {
      const out = [];
      for (const [loS, hiS] of [[title + " (", title + " )"],
                                [title + "（", title + "）"]]) {
        const rows = await this._titleRange(te.encode(loS), te.encode(hiS));
        const per = { 0: 0, 1: 0 };
        for (const r of rows) {
          if (per[r.src] >= 6) continue;
          per[r.src]++; out.push(r);
        }
      }
      return out;
    }
    async titleRows(title) {
      return (await this.exactAll(title)).concat(await this.variantAll(title));
    }
    /** search.py prefix_all: titles starting with prefix, LIMIT 6 per
     * source. hi bound = prefix with last code point incremented. */
    async prefixAll(prefix) {
      const a = cps(prefix);
      const last = a[a.length - 1];
      const hiS = a.slice(0, -1).join("") +
        String.fromCodePoint(last.codePointAt(0) + 1);
      const rows = await this._titleRange(te.encode(prefix), te.encode(hiS));
      const per = { 0: 0, 1: 0 };
      const out = [];
      for (const r of rows) {
        if (per[r.src] >= 6) continue;
        per[r.src]++; out.push(r);
      }
      return out;
    }
    async redirectTarget(title) {
      await this.init();
      const tb = te.encode(title);
      const size = this.meta.files["redir.bin"];
      const bi = this._findBlock(this.redirIdx, tb);
      const blk = this.redirIdx[bi];
      const end = this._blockEnd(this.redirIdx, bi, size);
      const buf = await this._block("redir.bin", blk.off, end - blk.off);
      const dv = new DataView(buf.buffer, buf.byteOffset);
      let p = 0;
      while (p < buf.length) {
        const fl = dv.getUint16(p, true); p += 2;
        if (p + fl > buf.length) break;
        const from = td.decode(buf.slice(p, p + fl)); p += fl;
        const tl = dv.getUint16(p, true); p += 2;
        const tgt = td.decode(buf.slice(p, p + tl)); p += tl;
        if (from === title) return tgt;
        if (cmpBytes(te.encode(from), tb) > 0) return null;
      }
      return null;
    }
    async redirectRows(title) {
      const tgt = await this.redirectTarget(title);
      return tgt ? this.titleRows(tgt) : [];
    }

    // ------------------------------------------------------------ records
    async idRecord(gid) {
      if (this._idRecs.has(gid)) return this._idRecs.get(gid);
      const stride = this.meta.id_stride;
      const buf = await this._range("ids.bin", gid * stride,
                                    gid * stride + stride - 1);
      const dv = new DataView(buf.buffer, buf.byteOffset);
      const leadLen = dv.getUint16(4, true);
      let rec = null;
      if (leadLen !== 0) {
        rec = { id: gid, src: buf[18],
                leadOff: dv.getUint32(0, true), leadLen,
                bodyOff: Number(dv.getBigUint64(6, true)),
                bodyLen: dv.getUint32(14, true), bodyId: buf[18] };
      }
      if (this._idRecs.size > 8192) this._idRecs.clear();
      this._idRecs.set(gid, rec);
      return rec;
    }
    async leadOf(rec) {
      const gid = rec.id;
      if (gid !== undefined && this._leads.has(gid)) return this._leads.get(gid);
      const buf = await this._range("leads.bin", rec.leadOff,
                                    rec.leadOff + rec.leadLen - 1);
      const text = td.decode(buf);
      const nl = text.indexOf("\n");
      const out = { title: nl >= 0 ? text.slice(0, nl) : text,
                    lead: nl >= 0 ? text.slice(nl + 1) : "" };
      if (gid !== undefined) {
        if (this._leads.size > 8192) this._leads.clear();
        this._leads.set(gid, out);
      }
      return out;
    }
    /** Batch-fill _idRecs/_leads for many gids with coalesced Range
     * fetches (ids.bin is fixed-stride; leads.bin offsets grow with
     * gid). Produces exactly the records leadOf/idRecord would. */
    async _primeRecords(gids) {
      if (!gids.length) return;
      const stride = this.meta.id_stride;
      const missing = [...new Set(gids)].filter((g) => !this._idRecs.has(g));
      if (missing.length) {
        missing.sort((a, b) => a - b);
        const spans = [];
        let s = missing[0], prev = missing[0];
        for (let i = 1; i < missing.length; i++) {
          if (missing[i] - s > 8192) { spans.push([s, prev]); s = missing[i]; }
          prev = missing[i];
        }
        spans.push([s, prev]);
        await mapLimit(spans, 4, async ([lo, hi]) => {
          const buf = await this._range("ids.bin", lo * stride,
                                        hi * stride + stride - 1);
          const dv = new DataView(buf.buffer, buf.byteOffset);
          for (let g = lo; g <= hi; g++) {
            const o = (g - lo) * stride;
            const leadLen = dv.getUint16(o + 4, true);
            let rec = null;
            if (leadLen !== 0) {
              rec = { id: g, src: buf[o + 18],
                      leadOff: dv.getUint32(o, true), leadLen,
                      bodyOff: Number(dv.getBigUint64(o + 6, true)),
                      bodyLen: dv.getUint32(o + 14, true),
                      bodyId: buf[o + 18] };
            }
            this._idRecs.set(g, rec);
          }
        });
      }
      const needLead = [];
      for (const g of new Set(gids)) {
        const rec = this._idRecs.get(g);
        if (rec && !this._leads.has(g)) needLead.push(rec);
      }
      if (!needLead.length) return;
      needLead.sort((a, b) => a.leadOff - b.leadOff);
      const spans = [];
      let cur = [needLead[0]];
      let curEnd = needLead[0].leadOff + needLead[0].leadLen;
      for (let i = 1; i < needLead.length; i++) {
        const r = needLead[i];
        const gap = r.leadOff - curEnd;
        const spanLen = (r.leadOff + r.leadLen) - cur[0].leadOff;
        if (gap <= 8192 && spanLen <= (1 << 20)) {
          cur.push(r);
          curEnd = Math.max(curEnd, r.leadOff + r.leadLen);
        } else { spans.push(cur); cur = [r]; curEnd = r.leadOff + r.leadLen; }
      }
      spans.push(cur);
      await mapLimit(spans, 6, async (recs) => {
        const lo = recs[0].leadOff;
        let hi = 0;
        for (const r of recs) hi = Math.max(hi, r.leadOff + r.leadLen);
        const buf = await this._range("leads.bin", lo, hi - 1);
        for (const r of recs) {
          const text = td.decode(buf.subarray(
            r.leadOff - lo, r.leadOff - lo + r.leadLen));
          const nl = text.indexOf("\n");
          this._leads.set(r.id, {
            title: nl >= 0 ? text.slice(0, nl) : text,
            lead: nl >= 0 ? text.slice(nl + 1) : "" });
        }
      });
    }
    _bodyFile(rec) {
      const files = this.opts.bodyFiles || this.meta.body_files;
      return files[rec.bodyId];
    }
    /** search.py read_article: full stored block -> [title, body]. */
    async readArticle(rec) {
      const buf = await this._range(this._bodyFile(rec), rec.bodyOff,
                                    rec.bodyOff + rec.bodyLen - 1);
      const text = td.decode(buf).split("&nbsp;").join(" ");
      const nl = text.indexOf("\n");
      return [nl >= 0 ? text.slice(0, nl).trim() : text.trim(),
              nl >= 0 ? text.slice(nl + 1) : ""];
    }
    /** Body head: enough bytes for the first `chars` body characters
     * (used by the ordinal/descriptive lanes, which read body heads). */
    async readBodyHead(rec, chars) {
      const want = Math.min(rec.bodyLen, chars * 3 + 1024);
      const buf = await this._range(this._bodyFile(rec), rec.bodyOff,
                                    rec.bodyOff + want - 1);
      const text = td.decode(buf).split("&nbsp;").join(" ");
      const nl = text.indexOf("\n");
      return nl >= 0 ? text.slice(nl + 1) : "";
    }

    // ------------------------------------------------------------ FTS pool
    async _trigramEntry(tg, titleOnly) {
      const superIdx = titleOnly ? this.titleTriSuper : this.triSuper;
      const dirFile = titleOnly ? "title_tri_dir.bin" : "tri_dir.bin";
      const kb = te.encode(tg);
      const bi = this._findBlock(superIdx, kb);
      const blk = superIdx[bi];
      const end = this._blockEnd(superIdx, bi, this.meta.files[dirFile]);
      const buf = await this._block(dirFile, blk.off, end - blk.off);
      let p = 0;
      while (p < buf.length) {
        const kl = buf[p++];
        if (p + kl > buf.length) break;
        const key = buf.slice(p, p + kl); p += kl;
        let off, len, df;
        [off, p] = readVarint(buf, p);
        [len, p] = readVarint(buf, p);
        [df, p] = readVarint(buf, p);
        const c = cmpBytes(key, kb);
        if (c === 0) return { off, len, df };
        if (c > 0) return null;
      }
      return null;
    }
    async _postings(entry, titleOnly) {
      const file = titleOnly ? "title_postings.bin" : "postings.bin";
      const len = Math.min(entry.len, MAX_POSTING_BYTES);
      const buf = await this._range(file, entry.off, entry.off + len - 1);
      const ids = [];
      let p = 0, cur = 0;
      while (p < buf.length) {
        let d;
        [d, p] = readVarint(buf, p);
        if (p > buf.length || (buf[p - 1] & 0x80)) break; // truncated tail
        cur += d;
        ids.push(cur);
      }
      return ids;
    }
    async _intersect(term, titleOnly) {
      const tris = [...new Set(trigramsOf(term))];
      if (!tris.length) return [];
      const entries = await mapLimit(tris, 12,
        (tg) => this._trigramEntry(tg, titleOnly));
      if (entries.some((e) => !e)) return [];
      const lists = await mapLimit(entries, 12,
        (e) => this._postings(e, titleOnly));
      lists.sort((a, b) => a.length - b.length);
      let cur = lists[0];
      for (let i = 1; i < lists.length && cur.length; i++) {
        const other = new Set(lists[i]);
        cur = cur.filter((id) => other.has(id));
      }
      return cur;
    }
    srcOf(gid) { return gid < this.meta.main_total ? 0 : 1; }

    /** Approximate FTS5 bm25 for one term in one source:
     * Map(gid -> score) over verified docs (see header note). */
    async termScores(src, term) {
      const ck = src + "|" + term;
      if (this._termCache.has(ck)) return this._termCache.get(ck);
      const out = new Map();
      const all = await this._intersect(term, false);
      const inSrc = all.filter((g) => this.srcOf(g) === src);
      if (inSrc.length) {
        const titleHits = new Set(
          (await this._intersect(term, true)).filter(
            (g) => this.srcOf(g) === src));
        const order = [...inSrc].sort((a, b) =>
          (titleHits.has(b) ? 1 : 0) - (titleHits.has(a) ? 1 : 0) || a - b);
        const pick = order.slice(0, FTS_FETCH_CAP);
        const N = this.meta.sources[src].n;
        const df = Math.max(inSrc.length, 1);
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        const avgT = Math.max(this.meta.sources[src].avg_title_tokens, 1e-9);
        const avgL = Math.max(this.meta.sources[src].avg_lead_tokens, 1e-9);
        const k1 = 1.2, b = 0.75;
        const col = (tf, dl, avg) =>
          tf <= 0 ? 0 : (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avg));
        await this._primeRecords(pick);
        await mapLimit(pick, 16, async (gid) => {
          const rec = await this.idRecord(gid);
          if (!rec) return;
          const { title, lead } = await this.leadOf(rec);
          const tfT = countOverlap(title, term);
          const tfL = countOverlap(lead, term);
          if (!tfT && !tfL) return;
          const dlT = Math.max(cpLen(title) - 2, 0);
          const dlL = Math.max(cpLen(lead) - 2, 0);
          const sc = idf * (10 * col(tfT, dlT, avgT) + col(tfL, dlL, avgL));
          if (sc > 0) out.set(gid, sc);
        });
      }
      if (this._termCache.size > 512) this._termCache.clear();
      this._termCache.set(ck, out);
      return out;
    }
    /** search.py's per-source MATCH ... ORDER BY bm25 LIMIT n, as a
     * ranked gid list (OR over terms, scores summed). */
    async ftsQuery(src, terms, limit) {
      const total = new Map();
      for (const term of terms) {
        const m = await this.termScores(src, term);
        for (const [gid, sc] of m) total.set(gid, (total.get(gid) || 0) + sc);
      }
      return [...total.entries()]
        .sort((a, b) => b[1] - a[1] || a[0] - b[0])
        .slice(0, limit).map(([gid]) => gid);
    }
    async _recWithLead(gid) {
      const rec = await this.idRecord(gid);
      if (!rec) return null;
      const { title, lead } = await this.leadOf(rec);
      return { rec, title, lead };
    }

    // ---------------------------------------------------- search.py search()
    async search(q, n = 5, opts = {}) {
      await this.init();
      const keys0 = paraKeys(q);
      let keys = keys0;
      const cand = new Map(); // "src:gid" -> {prio, rec, compound, evidence}
      const add = (rows, prio, compound = 0, evidence = 0) => {
        for (const rec of rows) {
          const k = rec.src + ":" + rec.id;
          const cur = cand.get(k);
          if (!cur || prio < cur.prio) {
            cand.set(k, { prio, rec, compound, evidence });
          } else if (prio === cur.prio && evidence > cur.evidence) {
            cur.evidence = evidence;
          }
        }
      };
      const titleRows = (t) => this.titleRows(t);
      const redirectRows = (t) => this.redirectRows(t);

      const quoted = [...q.matchAll(/『([^』]+)』/g)].map((m) => m[1]);
      const quotedNorms = quoted.map(normalizeTerm);
      const isCreatorQ = CREATOR_RE.test(q);

      // 1) quoted titles and their variants
      for (const title of quoted) {
        let hit = false;
        const stubRows = [];
        for (const v of termVariants(title)) {
          const rows = await titleRows(v);
          if (rows.length) {
            hit = true;
            for (const r of rows) if (r.bodyLen <= 3000) stubRows.push(r);
          }
          add(rows, 0);
        }
        if (!hit) {
          for (const v of termVariants(title)) {
            const rows = await redirectRows(v);
            if (rows.length) { hit = true; add(rows, 0); break; }
          }
        }
        if (isCreatorQ) {
          const prio0 = [...cand.values()].filter((c) => c.prio === 0);
          await mapLimit(prio0, 8, async (c) => {
            const [, body] = await this.readArticle(c.rec);
            const first = body.split("。")[0];
            if (ATTRIBUTION_RE.test(first)) {
              c.evidence = Math.max(c.evidence, 1);
            }
          });
        }
        const gn = normalizeTerm(title);
        for (const r of stubRows.slice(0, 2)) {
          const [, stubBody] = await this.readArticle(r);
          for (const line of stubBody.split("\n").slice(0, 60)) {
            const m = line.trim().match(/^『([^』]+)』\s*[-–—]/);
            if (!m) continue;
            const candTitle = m[1];
            const cn = normalizeTerm(candTitle);
            if (gn.includes(cn) || cn.includes(gn)) {
              add(await titleRows(candTitle), 0);
            }
          }
        }
        if (!hit) {
          if (cpLen(gn) >= 3) {
            for (let src = 0; src < 2; src++) {
              const gids = await this.ftsQuery(src, [gn], 20);
              for (const gid of gids) {
                const got = await this._recWithLead(gid);
                if (!got) continue;
                const prio = normalizeTerm(got.title).includes(gn) ? 0 : 2;
                add([got.rec], prio);
              }
            }
          }
        }
      }

      // 2) bare question terms that are exact article titles
      const parts = splitParts(q, 2);
      const genericRows = [];
      let entityFound = false;
      const entityTitles = new Set();
      const bareTitles = new Set();
      for (const t of parts) {
        if (NEVER_TITLES.has(t)) continue;
        if ((t === "名前" || t === "名称") &&
            /(名前|名称)は(何|なん)/.test(q)) continue;
        const isGeneric =
          GENERIC_TITLES.has(t) || SUPER_FRAGMENT_RE.test(t) ||
          new RegExp("の" + escRe(t) + "(は|が|を|に|の|です|ですか)").test(q) ||
          GENERIC_TITLES.has(normalizeTerm(t));
        let termHit = false;
        for (const v of termVariants(t)) {
          const rows = await this.exactAll(v);
          if (!rows.length) continue;
          termHit = true;
          bareTitles.add(t);
          if (isGeneric) genericRows.push(...rows);
          else {
            entityFound = true;
            entityTitles.add(t);
            add(rows, v === t ? 0 : 1);
          }
        }
        if (!termHit && !isGeneric) {
          for (const v of termVariants(t)) {
            const rows = await redirectRows(v);
            if (rows.length) {
              entityFound = true;
              entityTitles.add(t);
              add(rows, 0);
              break;
            }
          }
        }
      }
      // 2b) 'AのB' compound titles
      const parts1 = splitParts(q, 1);
      for (let i = 0; i + 1 < parts1.length; i++) {
        const a = parts1[i], b = parts1[i + 1];
        if (cpLen(a) + cpLen(b) < 3 || GENERIC_TITLES.has(b) ||
            GENERIC_TITLES.has(normalizeTerm(b))) continue;
        const comp = a + "の" + b;
        let hit = false;
        for (const v of termVariants(comp)) {
          const rows = await this.exactAll(v);
          if (rows.length) {
            hit = true;
            entityFound = true;
            entityTitles.add(comp);
            add(rows, 0, 1);
          }
        }
        if (!hit) {
          for (const v of termVariants(comp)) {
            const rows = await redirectRows(v);
            if (rows.length) {
              entityFound = true;
              entityTitles.add(comp);
              add(rows, 0, 1);
              break;
            }
          }
        }
      }
      // 2d) year siblings
      const wantsFirst = /最初|初めて|第1回|第一回/.test(q);
      const wantsLast = /最新|最近|最後|前回/.test(q);
      if ((wantsFirst || wantsLast) &&
          (entityTitles.size || bareTitles.size)) {
        for (const t of [...entityTitles, ...bareTitles]) {
          if (t.includes("(") || t.includes("（") || t.includes("の")) continue;
          const yearList = [];
          for (let y = 1800; y <= 2030; y++) yearList.push(y);
          const yearHits = await mapLimit(yearList, 12, async (y) => {
            const prefix = `${y}年${t}`;
            const plen = cpLen(prefix);
            const rows = (await this.prefixAll(prefix)).filter((r) =>
              r.title === prefix ||
              "・(（".includes(cpSlice(r.title, plen, plen + 1)));
            return rows.length ? [y, rows] : null;
          });
          const years = yearHits.filter((x) => x !== null);
          if (years.length >= 2) {
            years.forEach(([_y, rows], i) => {
              let ev = 1;
              if (wantsFirst && i === 0) ev = 3;
              if (wantsLast && i === years.length - 1) ev = 3;
              add(rows, ev === 3 ? 0 : 1, 0, ev);
            });
          }
        }
      }
      // 2c) titles hidden inside the particle-stripped question
      let substringFound = false;
      const foundTitles = [];
      if (!entityFound && !quoted.length) {
        const parts1Set = splitParts(q, 1);
        const titleParts = new Set();
        const partHits = await mapLimit(parts1Set, 12,
          async (p) => (await this.exactAll(p)).length > 0);
        parts1Set.forEach((p, i) => { if (partHits[i]) titleParts.add(p); });
        const stripped = normalizeTerm(q.replace(
          /[はがをにのへとやもだですますか？?、。・\s「」『』（）()]+/g, ""));
        const sa = cps(stripped);
        const subs = new Set();
        for (let ln = 1; ln <= 8; ln++) {
          for (let i = 0; i + ln <= sa.length; i++) {
            const s = sa.slice(i, i + ln).join("");
            if (!GENERIC_TITLES.has(s) && !NEVER_TITLES.has(s) &&
                !SUBSTRING_STOP.has(s) &&
                !FTS_JUNK.some((j) => s.includes(j))) subs.add(s);
          }
        }
        const sortedSubs = [...subs].sort((x, y) => cpLen(y) - cpLen(x));
        const subList = sortedSubs.slice(0, 200);
        const subRows = await mapLimit(subList, 12,
          (s) => this.exactAll(s));
        for (let si = 0; si < subList.length; si++) {
          const s = subList[si];
          const rows = subRows[si];
          if (!rows.length) continue;
          const counted = (cpLen(s) === 1 && parts1Set.includes(s)) ||
            (cpLen(s) >= 2 && ![...titleParts].some(
              (p) => s !== p && p.includes(s)));
          if (counted) {
            substringFound = true;
            foundTitles.push(s);
            add(rows, 1);
          } else add(rows, 2);
        }
      }
      // 3) FTS pool
      const nBeforeFts = cand.size;
      let terms = [];
      for (const srcQ of [q, normalizeTerm(q)]) {
        for (const t of splitParts(srcQ, 3)) {
          if (NEVER_TITLES.has(t) || FTS_STOP.has(t) ||
              FTS_JUNK.some((j) => t.includes(j))) continue;
          terms.push(t);
        }
      }
      terms = [...new Set(terms)].slice(0, 6);
      if (terms.length &&
          !(terms.length === 1 && FTS_STOP.has(terms[0]))) {
        for (let src = 0; src < 2; src++) {
          const gids = await this.ftsQuery(src, terms, 12);
          const gots = await mapLimit(gids, 12,
            (gid) => this._recWithLead(gid));
          for (let gi = 0; gi < gids.length; gi++) {
            const got = gots[gi];
            if (!got) continue;
            let prio = 2;
            const tnorm = normalizeTerm(got.title);
            if (quotedNorms.some((gn2) => gn2 && cpLen(gn2) >= 3 &&
                (tnorm.includes(gn2) || gn2.includes(tnorm)))) prio = 0;
            add([got.rec], prio);
          }
        }
      }
      const ftsAdded = cand.size > nBeforeFts;
      add(genericRows,
          (entityFound || quoted.length || substringFound || ftsAdded)
            ? 3 : 0);

      // 4) evidence lanes
      const laneOk = !quoted.length &&
        (!entityFound || [...entityTitles].every((t) => SCOPE_TITLES.has(t)));
      if (laneOk) await this._evidenceLanes(q, add, terms);

      if (foundTitles.length) {
        keys = [...new Set(keys.concat(foundTitles))];
      }
      const candArr = [...cand.values()];
      const scored = await mapLimit(candArr, 12, async (c) => {
        const [t, body] = await this.readArticle(c.rec);
        const [para, s, occ] = bestParagraph(body, keys);
        return { prio: c.prio, compound: c.compound, ev: c.evidence,
                 s, occ, len: c.rec.bodyLen, title: t, para, body };
      });
      scored.sort((a, b) => a.prio - b.prio || b.compound - a.compound ||
                    b.ev - a.ev || b.s - a.s || b.occ - a.occ ||
                    b.len - a.len);
      return scored.slice(0, n).map((x) => opts.withBody
        ? { title: x.title, para: x.para, body: x.body }
        : { title: x.title, para: x.para });
    }

    // ------------------------------------------- search.py _evidence_lanes()
    async _evidenceLanes(q, add, ftsTerms) {
      const isSuper = SUPER_RE.test(q);
      const ordinal = ordinalOf(q);
      const scope = scopeOf(q);
      const [attrSyns, attrBase] = attrOf(q);
      const category = categoryOf(q);
      const pool = new Map(); // "src:gid" -> {rec, title, lead}
      const poolSet = async (gid) => {
        const src = this.srcOf(gid);
        const k = src + ":" + gid;
        if (pool.has(k)) return;
        const got = await this._recWithLead(gid);
        if (got) pool.set(k, { rec: got.rec, title: got.title, lead: got.lead });
      };

      let laneTerms = [];
      if (scope && attrBase) {
        laneTerms.push(scope + "に最も" + attrBase, scope + "で最も" + attrBase);
      }
      if (scope) {
        laneTerms.push(...attrSyns
          .filter((s) => cpLen(s) >= 3 && s !== attrBase &&
                  !s.startsWith("最も"))
          .map((s) => scope + s));
      }
      laneTerms.push(...attrSyns.filter((s) => cpLen(s) >= 3));
      laneTerms.push(...ftsTerms);
      const segs = splitParts(q, 1);
      const crossTerms = [];
      for (let i = 0; i + 1 < segs.length; i++) {
        const a = segs[i], b = segs[i + 1];
        if (a && b) {
          crossTerms.push(cpSlice(a, cpLen(a) - 1) + "を" + cpSlice(b, 0, 1));
        }
      }
      laneTerms.push(...crossTerms);
      if (attrBase) laneTerms.push("最も" + attrBase);
      if (scope && isSuper) laneTerms.push(scope + "最大", scope + "最高");
      laneTerms = [...new Set(laneTerms.filter((t) => cpLen(t) >= 3 &&
        !GENERIC_TITLES.has(t) && !SCOPE_TITLES.has(t)))].slice(0, 12);
      for (const term of laneTerms) {
        const lim = crossTerms.includes(term) ? 100 : 30;
        for (let src = 0; src < 2; src++) {
          const gids = await this.ftsQuery(src, [term], lim);
          await this._primeRecords(gids);
          for (const gid of gids) await poolSet(gid);
        }
      }
      // claim cache pool
      if (isSuper && ordinal === null) {
        const cm2 = q.match(/(?:一番|最も|最大|最高|最古|最長|最小|第一|最初)の([^\s、。がはをにへと]{2,8})/);
        let cnoun = cm2 ? cm2[1] : null;
        if (cnoun) cnoun = cnoun.split(/[のはがをにへと]/)[0] || null;
        const okScopes = CLAIM_SCOPES[scope] || (scope ? [scope] : []);
        await this._primeRecords(this.claims
          .filter(([, cs]) => okScopes.includes(cs)).map(([g]) => g));
        for (const [gid, cscope] of this.claims) {
          if (!okScopes.includes(cscope)) continue;
          if (category) {
            const got = await this._recWithLead(gid);
            if (!got) continue;
            const lead200 = cpSlice(got.lead || "", 0, 200);
            if (!got.title.includes(category) &&
                !lead200.includes(category) &&
                !(cnoun && lead200.includes(cnoun))) continue;
          }
          await poolSet(gid);
        }
      }
      // category cache pool
      if (category) {
        await this._primeRecords(this.categories[category] || []);
        for (const gid of this.categories[category] || []) {
          const got = await this._recWithLead(gid);
          if (!got) continue;
          const lead = got.lead || "";
          if ((scope && lead.includes(scope)) ||
              attrSyns.some((s) => lead.includes(s)) ||
              /最も|最大|最高|最古|一番|最高峰/.test(lead)) {
            const k = got.rec.src + ":" + gid;
            if (!pool.has(k)) {
              pool.set(k, { rec: got.rec, title: got.title, lead });
            }
          }
        }
      }
      if (!pool.size) return;

      const focusWords = [...GENERIC_TITLES].filter((g) => q.includes(g) &&
        !SCOPE_TITLES.has(g) && !["名前", "名称", "日本人"].includes(g));
      const scopes = CLAIM_SCOPES[scope] || (scope ? [scope] : []);
      let cm = q.match(/(?:一番|最も|最大|最高|最古|最長|最小|第一|最初)の([^\s、。がはをにへと]{2,8})/);
      let claimNoun = cm ? cm[1] : null;
      if (!claimNoun) {
        cm = q.match(/(?:高い|大きい|大きな|長い|深い|近い|遠い|古い|多い|重い|速い|新しい|強い)([^\s、。がはをにへと]{1,6})/);
        claimNoun = cm ? cm[1] : null;
      }
      if (claimNoun) {
        claimNoun = claimNoun.split(/[のはがをにへと]/)[0] || null;
      }
      let qFamily = null;
      for (const [stem, fam] of Object.entries(STEM_FAMILY)) {
        if (q.includes(stem)) { qFamily = fam; break; }
      }
      if (qFamily === null && ATTR_FAMILY[attrBase] !== undefined) {
        qFamily = ATTR_FAMILY[attrBase];
      }

      const superEvidence = (title, lead) => {
        let ev = 0;
        let claimM = null;
        let weakClaim = false;
        const claimRe = (cs) => new RegExp(escRe(cs) +
          ".{0,3}(最も|一番|最大|最高峰|最高|最古|最長|最小|第一|最初)");
        for (const cs of scopes) {
          const m = lead.match(claimRe(cs));
          if (m) {
            const stem = m[1];
            if (stem === "最も" || stem === "一番" || qFamily === null ||
                STEM_FAMILY[stem] === qFamily) {
              claimM = m; ev += 3; weakClaim = false; break;
            }
            if (!weakClaim) { weakClaim = true; ev += 1; }
          }
        }
        if (!claimM && !weakClaim) {
          for (const cs of scopes) {
            if (claimRe(cs).test(lead)) { ev += 1; break; }
          }
        }
        if (claimM) {
          const endCh = cpLen(lead.slice(0, claimM.index)) +
            cpLen(claimM[0]);
          const tail = cpSlice(lead, endCh, endCh + 12);
          const headCtx = cpSlice(lead,
            Math.max(0, cpLen(lead.slice(0, claimM.index)) - 12),
            cpLen(lead.slice(0, claimM.index)));
          const bounded = [...attrSyns];
          if (category && !(claimNoun && claimNoun !== category)) {
            bounded.push(category);
          }
          const stem = claimM[1];
          const ADJ_FORMS = attrSyns.filter((s) => ADJ_SET.has(s));
          const tail8 = cpSlice(tail, 0, 8);
          const cond1 = claimNoun && (
            tail.startsWith("の" + claimNoun) ||
            (stem !== "最も" && stem !== "一番" &&
             tail8.includes(claimNoun)) ||
            ADJ_FORMS.some((a) => tail8.includes(a + claimNoun)) ||
            new RegExp(escRe(claimNoun) + "の中|のうち").test(headCtx));
          const cond2 = bounded.some((t) => t && new RegExp(
            "^の?" + escRe(t) + "(の|は|が|を|に|と|、|。|で|$)").test(tail));
          if (cond1 || cond2) ev += 2;
        }
        if (scope && lead.includes(scope)) ev += 1;
        if (SUPER_WORD_RE.test(lead)) ev += 1;
        if (attrSyns.some((s) => lead.includes(s))) ev += 1;
        if (focusWords.some((f) => lead.includes(f))) {
          ev += SUPER_WORD_RE.test(lead) ? 3 : 2;
        }
        if (category) {
          if (CATEGORY_SUFFIX[category].some((s) => title.endsWith(s))) {
            ev += 2;
          } else if (title.includes(category) || lead.includes(category)) {
            ev += 1;
          }
        }
        return ev;
      };

      let promoted = [];
      if (isSuper && ordinal === null) {
        const ranked = [];
        for (const p of pool.values()) {
          if (SCOPE_TITLES.has(p.title)) continue;
          if (attrSyns.includes(p.title) ||
              GENERIC_TITLES.has(p.title)) continue;
          const ev = superEvidence(p.title, p.lead || "");
          if (ev >= 5) ranked.push({ ev, ...p });
        }
        ranked.sort((a, b) => b.ev - a.ev || b.rec.bodyLen - a.rec.bodyLen);
        promoted = ranked.slice(0, 2);
        if (ranked.length > 2 && ranked[2].ev === ranked[0].ev) {
          promoted = ranked.slice(0, 3);
        }
        promoted = promoted.map((p) => ({ ...p, prio: 0 }));
      } else if (isSuper && ordinal !== null && ordinal >= 2) {
        const num = String(ordinal);
        const pat = new RegExp("は[^。]{0,40}?(第" + num + "位|" + num +
                               "番目|" + num + "番)");
        const stage = [];
        for (const p of pool.values()) {
          if (SCOPE_TITLES.has(p.title)) continue;
          const lead = p.lead || "";
          if ((scope && lead.includes(scope)) ||
              attrSyns.some((s) => lead.includes(s))) {
            const leadOrd = /第[0-9一二三四五六七八九十]+位|[0-9一二三四五六七八九十]+番目/.test(lead) ? 1 : 0;
            const leadHit = new RegExp("第" + num + "位|" + num +
                                        "番目|" + num + "番").test(lead) ? 1 : 0;
            stage.push({ leadHit, leadOrd,
                         ev: superEvidence(p.title, lead), ...p });
          }
        }
        stage.sort((a, b) => b.leadHit - a.leadHit ||
                     b.leadOrd - a.leadOrd || b.ev - a.ev ||
                     b.rec.bodyLen - a.rec.bodyLen);
        const NOUN_SYNS = attrSyns.filter((s) => !ADJ_SET.has(s));
        const ADJ_SYNS = attrSyns.filter((s) => ADJ_SET.has(s));
        const checkOrdinal = (e, body) => {
          const head = cpSlice(body, 0, 2500);
          const m = pat.exec(head);
          let unitOk = false;
          if (m) {
            const segRaw = cpSlice(head,
              cpLen(head.slice(0, m.index)) + cpLen(m[0]),
              cpLen(head.slice(0, m.index)) + cpLen(m[0]) + 12);
            const seg = segRaw.replace(/^[にのは、]+/, "");
            if (NOUN_SYNS.some((s) => seg.startsWith(s))) unitOk = true;
            else {
              for (const a of ADJ_SYNS) {
                if (seg.startsWith(a)) {
                  const rest = seg.slice(a.length);
                  if ((claimNoun && rest.startsWith(claimNoun)) ||
                      (category && rest.startsWith(category))) {
                    unitOk = true;
                  }
                  break;
                }
              }
            }
          }
          const scopeOk = !scope || (m && new RegExp(
            "(?<![東西南北])" + escRe(scope)).test(cpSlice(head,
              Math.max(0, cpLen(head.slice(0, m.index)) - 80),
              cpLen(head.slice(0, m.index)) + cpLen(m[0]) + 40)));
          return !!(m && unitOk &&
            (cpSlice(head, Math.max(0, cpLen(head.slice(0, m.index)) - 45),
                     cpLen(head.slice(0, m.index)) + 2).includes(e.title) ||
             head.split("。")[0].includes(e.title)) && scopeOk);
        };
        const stageTop = stage.slice(0, 160);
        for (let ci = 0; ci < stageTop.length && !promoted.length;
             ci += 12) {
          const chunk = stageTop.slice(ci, ci + 12);
          const bodies = await mapLimit(chunk, 12,
            (e) => this.readBodyHead(e.rec, 2600));
          for (let j = 0; j < chunk.length; j++) {
            if (checkOrdinal(chunk[j], bodies[j])) {
              promoted = [{ ev: 6, ...chunk[j], prio: 0 }];
              break;
            }
          }
        }
      }
      if (!isSuper) {
        const DK_PREFIX = ["一番", "最も", "大きな", "大きい", "伝統的な",
          "有名な", "新しい", "古い", "多い", "少ない", "高い", "低い",
          "長い", "短い", "深い", "浅い", "強い", "速い", "持つ"];
        const DK_SUFFIX = ["を持つことで知られる", "ことで知られる",
          "として知られる", "で知られる", "を持つ", "がある", "とされる",
          "である"];
        const families = [];
        for (const p of splitParts(q, 1)) {
          if (NEVER_TITLES.has(p) || GENERIC_TITLES.has(p) ||
              SCOPE_TITLES.has(p) || p === "名前" || p === "名称" ||
              FTS_STOP.has(p) || p.startsWith("何") ||
              p.startsWith("どこ") || p.startsWith("誰")) continue;
          const forms = [p];
          let cur = p;
          for (let i = 0; i < 3; i++) {
            let nxt = cur;
            for (const pre of DK_PREFIX) {
              if (nxt.startsWith(pre) && cpLen(nxt) > cpLen(pre)) {
                nxt = nxt.slice(pre.length); break;
              }
            }
            for (const suf of DK_SUFFIX) {
              if (nxt.endsWith(suf) && cpLen(nxt) > cpLen(suf)) {
                nxt = nxt.slice(0, nxt.length - suf.length); break;
              }
            }
            for (const sep of ["を持つことで知られる", "ことで知られる",
                               "として知られる", "で知られる"]) {
              if (nxt.includes(sep)) { nxt = nxt.split(sep).pop(); break; }
            }
            if (nxt === cur) break;
            cur = nxt;
            if (cur && !forms.includes(cur)) forms.push(cur);
          }
          families.push(forms);
        }
        const dkeys = families.flat();
        if (families.length >= 2 && dkeys.length >= 3) {
          const staged = [];
          for (const p of pool.values()) {
            if (SCOPE_TITLES.has(p.title)) continue;
            const leadHits = dkeys.reduce(
              (acc, k) => acc + ((p.lead || "").includes(k) ? 1 : 0), 0);
            staged.push({ leadHits, ...p });
          }
          staged.sort((a, b) => b.leadHits - a.leadHits);
          const stagedTop = staged.slice(0, 80);
          const stagedBodies = await mapLimit(stagedTop, 12,
            (e) => this.readBodyHead(e.rec, 1300));
          let best = null;
          for (let ei = 0; ei < stagedTop.length; ei++) {
            const e = stagedTop[ei];
            const body = stagedBodies[ei];
            const text = (e.lead || "") + cpSlice(body, 0, 1200);
            const cov = families.filter(
              (fam) => fam.some((k) => text.includes(k))).length /
              families.length;
            if (!best || cov > best.cov) best = { cov, ...e };
          }
          if (best && best.cov >= 0.55) {
            const dprio = best.cov >= 0.8 ? 0 : 1;
            promoted = [{ ev: Math.trunc(best.cov * 10), ...best,
                          prio: dprio }];
          }
        }
      }
      for (const e of promoted) {
        add([e.rec], e.prio !== undefined ? e.prio : 0, 0, e.ev);
      }
    }
  }

  // ------------------------------------------------------------ transports
  function httpTransport(indexBase, bodyBase) {
    const fetchRange = async (url, range) => {
      const headers = {};
      if (range) headers["Range"] = `bytes=${range[0]}-${range[1]}`;
      const res = await fetch(url, { headers });
      if (!res.ok && res.status !== 206) {
        throw new Error(`fetch ${url} -> ${res.status}`);
      }
      return new Uint8Array(await res.arrayBuffer());
    };
    return {
      range: (file, a, b) => fetchRange(
        (file.endsWith(".txt") ? bodyBase : indexBase) + file, [a, b]),
      whole: (file) => fetchRange(indexBase + file, null),
    };
  }

  const api = { SearchEngine, httpTransport, normalizeTerm, termVariants,
                splitParts, bestParagraph };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.BrowserSearch = api;
})(typeof window !== "undefined" ? window : globalThis);

"use strict";
/* Cube index lane: managed transport + Cube-shaped adapter over the
 * parallelized browser Wikipedia index engine
 * (browser_search_cube.js, a fetch-parallelized copy of
 * hybrid/browser_index_full/browser_search.js; ranking unchanged).
 *
 * The lane is the ONLY new networking surface: it talks to the public
 * Hugging Face datasets FMA03/cube-index (shards) and
 * FMA03/hybrid-ai-data (article bodies) with single Range GETs and no
 * custom headers (CORS-simple), exactly like the design contract in
 * BROWSER_INDEX_FULL.md.
 *
 * Guarantees for the host app:
 * - bounded concurrency (default 12 in-flight requests),
 * - a timeout on every request (default 20s, AbortController),
 * - an in-session cache for fetched ranges (repeat questions are
 *   fast; byte-capped, wholesale eviction like the engine's own),
 * - a per-question budget (max requests / max wall time): when it is
 *   exceeded the lane aborts with IndexBudgetError and the host falls
 *   back to its existing web path, so no question can hang.
 * Every adapter function resolves to null / [] / an empty Map on any
 * failure instead of throwing.
 *
 * Host wiring (both Cube versions):
 *   const lane = CubeIndexLane.create({
 *     SearchEngine: globalThis.BrowserSearch.SearchEngine,
 *     wikiUrl: (title) => ...,
 *     disambiguationCandidates: (page, subject, extended) => [...],
 *   });
 *   lane.beginQuestion();            // at each question's search start
 *   await lane.fetchExtract(title);  // {title,url,snippet,text} | null
 *   await lane.searchTitles(q, n);   // [title, ...]
 *   await lane.fetchExtractHeads(titles); // Map<title, headText>
 *   await lane.articleSource(title); // {title,url,lead,fields:{},disambiguation,candidates} | null
 *   await lane.searchResults(q, n);  // [{title,url,snippet,text}]
 */
(function (global) {
  const INDEX_BASE =
    "https://huggingface.co/datasets/FMA03/cube-index/resolve/main/";
  const BODY_BASE =
    "https://huggingface.co/datasets/FMA03/hybrid-ai-data/resolve/main/";

  class IndexBudgetError extends Error {
    constructor(msg) {
      super(msg);
      this.name = "IndexBudgetError";
    }
  }

  async function mapLimit(items, limit, fn) {
    const out = new Array(items.length);
    let next = 0;
    const n = Math.min(limit, items.length);
    const workers = [];
    for (let w = 0; w < n; w++) {
      workers.push((async () => {
        while (next < items.length) {
          const i = next++;
          out[i] = await fn(items[i], i);
        }
      })());
    }
    await Promise.all(workers);
    return out;
  }

  function makeSemaphore(max) {
    let active = 0;
    const queue = [];
    return {
      acquire() {
        if (active < max) {
          active += 1;
          return Promise.resolve();
        }
        return new Promise((res) => queue.push(res));
      },
      release() {
        const next = queue.shift();
        if (next) next();
        else active -= 1;
      },
    };
  }

  function createIndexTransport(indexBase, bodyBase, opts = {}) {
    const maxConc = opts.concurrency ?? 12;
    const timeoutMs = opts.timeoutMs ?? 20000;
    const cacheCapBytes = opts.cacheBytes ?? 96 * 1024 * 1024;
    const sem = makeSemaphore(maxConc);
    const cache = new Map(); // key -> {p, bytes}
    let cacheBytes = 0;
    const stats = { requests: 0, bytes: 0, cacheHits: 0 };
    let budget = null;

    function beginQuestion(qopts = {}) {
      budget = {
        count: 0,
        maxRequests: qopts.maxRequests ?? 20000,
        deadline: Date.now() + (qopts.maxMs ?? 120000),
        aborted: false,
      };
    }
    function budgetActive() {
      return !!budget && !budget.aborted && Date.now() <= budget.deadline;
    }
    function autoBegin() {
      if (!budgetActive()) beginQuestion();
    }
    function checkBudget() {
      if (!budget) return;
      if (budget.aborted) throw new IndexBudgetError("index-budget-aborted");
      if (budget.count > budget.maxRequests || Date.now() > budget.deadline) {
        budget.aborted = true;
        throw new IndexBudgetError("index-budget-exceeded");
      }
    }

    async function fetchRaw(url, range) {
      checkBudget();
      await sem.acquire();
      try {
        checkBudget();
        if (budget) budget.count += 1;
        const ctrl =
          typeof AbortController !== "undefined"
            ? new AbortController()
            : null;
        const timer = ctrl
          ? setTimeout(() => ctrl.abort(), timeoutMs)
          : null;
        try {
          const headers = {};
          if (range) headers["Range"] = `bytes=${range[0]}-${range[1]}`;
          const res = await fetch(
            url,
            ctrl ? { headers, signal: ctrl.signal } : { headers },
          );
          if (!res.ok && res.status !== 206) {
            throw new Error(`index fetch HTTP ${res.status}`);
          }
          const buf = new Uint8Array(await res.arrayBuffer());
          stats.requests += 1;
          stats.bytes += buf.length;
          return buf;
        } finally {
          if (timer) clearTimeout(timer);
        }
      } finally {
        sem.release();
      }
    }

    function cached(key, make) {
      const hit = cache.get(key);
      if (hit) {
        stats.cacheHits += 1;
        return hit.p;
      }
      const entry = { p: null, bytes: 0 };
      entry.p = make();
      cache.set(key, entry);
      entry.p.then(
        (buf) => {
          entry.bytes = buf.length;
          cacheBytes += buf.length;
          if (cacheBytes > cacheCapBytes) {
            for (const [k, e] of cache) {
              if (e !== entry && e.bytes > 0) {
                cache.delete(k);
                cacheBytes -= e.bytes;
              }
            }
          }
        },
        () => {
          if (cache.get(key) === entry) cache.delete(key);
        },
      );
      return entry.p;
    }

    const transport = {
      range(file, a, b) {
        const url = (file.endsWith(".txt") ? bodyBase : indexBase) + file;
        return cached(`${file}|${a}|${b}`, () => fetchRaw(url, [a, b]));
      },
      whole(file) {
        const url = (file.endsWith(".txt") ? bodyBase : indexBase) + file;
        return cached(`${file}|whole`, () => fetchRaw(url, null));
      },
    };
    return {
      transport,
      stats,
      beginQuestion,
      budgetActive,
      autoBegin,
      isBudgetError: (e) => e instanceof IndexBudgetError,
    };
  }

  function create(deps) {
    const T = createIndexTransport(INDEX_BASE, BODY_BASE, deps.transportOpts);
    let engine = null;
    let enginePromise = null;
    const lane = {
      transport: T,
      lastError: "",
      beginQuestion: (o) => T.beginQuestion(o),
      stats: () => ({ ...T.stats }),
    };

    async function ensureEngine() {
      if (engine) return engine;
      if (!enginePromise) {
        enginePromise = (async () => {
          const e = new deps.SearchEngine(T.transport);
          await e.init();
          engine = e;
          return e;
        })();
        enginePromise.catch(() => {
          enginePromise = null;
        });
      }
      return enginePromise;
    }

    async function rowsForTitle(e, title) {
      let rows = await e.titleRows(title);
      if (!rows.length) rows = await e.redirectRows(title);
      return rows;
    }

    /** Same shape as the host's wikiFetchExtract result. */
    lane.fetchExtract = async (title) => {
      try {
        T.autoBegin();
        const e = await ensureEngine();
        const rows = await rowsForTitle(e, title);
        if (!rows.length) return null;
        const [t, body] = await e.readArticle(rows[0]);
        if (!body) return null;
        const text = body.slice(0, 6000);
        return {
          title: t,
          url: deps.wikiUrl(t),
          snippet: text.slice(0, 240),
          text,
        };
      } catch (err) {
        lane.lastError = err instanceof Error ? err.message : String(err);
        return null;
      }
    };

    /** Ranked article titles for a query (host's wikiSearchTitles). */
    lane.searchTitles = async (q, limit) => {
      try {
        T.autoBegin();
        const e = await ensureEngine();
        const hits = await e.search(q, limit);
        return hits.map((h) => h.title);
      } catch (err) {
        lane.lastError = err instanceof Error ? err.message : String(err);
        return [];
      }
    };

    /** Corroboration-ready results: ranked articles with full text. */
    lane.searchResults = async (q, limit) => {
      try {
        T.autoBegin();
        const e = await ensureEngine();
        const hits = await e.search(q, limit, { withBody: true });
        return hits.map((h) => ({
          title: h.title,
          url: deps.wikiUrl(h.title),
          snippet: (h.para ?? "").slice(0, 240),
          text: (h.body ?? "").slice(0, 6000),
        }));
      } catch (err) {
        lane.lastError = err instanceof Error ? err.message : String(err);
        return [];
      }
    };

    /** Map of title -> lead head (~1200 chars), host's
     * wikiFetchExtractHeads. Keyed by the requested title. */
    lane.fetchExtractHeads = async (titles) => {
      const out = new Map();
      if (!titles.length) return out;
      try {
        T.autoBegin();
        const e = await ensureEngine();
        const heads = await mapLimit(titles, 8, async (title) => {
          const rows = await rowsForTitle(e, title);
          if (!rows.length) return null;
          const head = await e.readBodyHead(rows[0], 1200);
          return head ? head.slice(0, 1200) : null;
        });
        titles.forEach((t, i) => {
          if (heads[i]) out.set(t, heads[i]);
        });
      } catch (err) {
        lane.lastError = err instanceof Error ? err.message : String(err);
      }
      return out;
    };

    /** Same shape as the host's wikiArticleSource result. The index
     * stores plain-text bodies (no wikitext), so infobox `fields` is
     * empty and the host's lead-based extractors do the work — the
     * same fallbacks used today for articles whose infobox lacks the
     * field. Disambiguation is detected structurally (title suffix or
     * a list-shaped lead), mirroring the host's own rule minus the
     * wikitext template mark. */
    lane.articleSource = async (title) => {
      try {
        T.autoBegin();
        const e = await ensureEngine();
        const rows = await rowsForTitle(e, title);
        if (!rows.length) return null;
        const [t, body] = await e.readArticle(rows[0]);
        if (body == null) return null;
        const candidates = deps.disambiguationCandidates
          ? deps.disambiguationCandidates({ title: t, text: body }, title, true)
          : [];
        const firstLines = body
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l.length > 0)
          .slice(0, 6);
        const listy =
          firstLines.filter((l) => /^.+?\s-\s/.test(l)).length >= 2;
        const templateMark = /[（(]曖昧さ回避[）)]$/.test(t);
        return {
          title: t,
          url: deps.wikiUrl(t),
          lead: body.slice(0, 5000),
          fields: {},
          disambiguation:
            templateMark || (candidates.length > 0 && listy),
          candidates,
        };
      } catch (err) {
        lane.lastError = err instanceof Error ? err.message : String(err);
        return null;
      }
    };

    return lane;
  }

  const api = { create, createIndexTransport, IndexBudgetError, mapLimit };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.CubeIndexLane = api;
})(typeof window !== "undefined" ? window : globalThis);

/* 索引レーンはグローバルに1つだけ作る。wikiUrl と
   disambiguationCandidates は既存の関数をそのまま渡す（どちらも
   関数宣言なので、この時点でも参照できる）。 */
const IndexLane = CubeIndexLane.create({
  SearchEngine: globalThis.BrowserSearch.SearchEngine,
  wikiUrl,
  disambiguationCandidates,
});

/* ================================================================== */
/* Wikipedia クライアント（ブラウザから直接呼ぶ）                       */
/*                                                                    */
/* 日本語Wikipedia API は origin=* を付けると CORS でブラウザから      */
/* 呼べる。ただし プレビュー画面（iframe）のように直接の      */
/* fetch がブロックされる環境があるため、fetch が失敗したら予備の      */
/* 経路として JSONP（<script> タグ経由。CORS の対象外で、iframe の     */
/* 接続制限にも引っかかりにくい）に切り替える。一度 fetch が失敗した    */
/* 環境では、以降は JSONP を先に試す。失敗したときは理由を WikiNet.    */
/* lastError に残し、画面に表示する（黙って「確認できませんでした」に   */
/* しない）。                                                          */
/*                                                                    */
/* 事実の出どころは「対象記事1本」に定める：タイトル完全一致を先に試し、*/
/* 曖昧さ回避ページはそれ自体を事実源にせず、指し先の記事の中から質問に */
/* 答えられるものを選ぶ。                                              */
/* ================================================================== */

const WIKI_API = "https://ja.wikipedia.org/w/api.php";

/** 検索の接続状態：最後に成功した経路と、最後の失敗理由（画面表示用）。 */
const WikiNet = { via: "", lastError: "" };
/** fetch が一度でも失敗した環境では JSONP を優先する。 */
let wikiFetchBroken = false;

/** 通信エラーを、画面に出せる短い日本語の理由に変える。 */
function netErrorReason(e) {
  const msg = e instanceof Error ? e.message : String(e);
  if (/HTTP \d+/.test(msg)) return msg; // 例: Wikipedia API HTTP 403
  if (msg === "jsonp-timeout" || (e && e.name === "AbortError")) {
    return "時間内に答えが返ってきませんでした";
  }
  if (msg === "jsonp-error" || msg === "jsonp-unavailable") {
    return "予備の経路でも読み込めませんでした";
  }
  return "ブラウザからWikipediaへ接続できませんでした（画面の制限でブロックされた可能性があります）";
}

/** 本命の経路：fetch（origin=* で CORS 許可）。タイムアウト付き。
 *  429／503（Wikipedia側のレート制限）は少し待って再試行する：
 *  質問を続けて聞くと制限に当たることがあり、そのまま失敗にすると
 *  検索できたはずの質問まで「確認できませんでした」になってしまう。 */
async function wikiApiViaFetch(params) {
  const qs = new URLSearchParams({ ...params, origin: "*", format: "json", formatversion: "2" });
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), 12000) : null;
    try {
      const res = await fetch(`${WIKI_API}?${qs}`, ctrl ? { signal: ctrl.signal } : undefined);
      if ((res.status === 429 || res.status === 503) && attempt < 2) {
        const ra = Number(res.headers?.get?.("retry-after"));
        const wait =
          Number.isFinite(ra) && ra > 0
            ? Math.min(ra * 1000, 4000)
            : 900 * (attempt + 1);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }
      if (!res.ok) throw new Error(`Wikipedia API HTTP ${res.status}`);
      return await res.json();
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
  throw new Error("Wikipedia API HTTP 429");
}

let wikiJsonpSeq = 0;
/** 予備の経路：JSONP。<script> で読み込むため CORS の対象外。 */
function wikiApiViaJsonp(params) {
  return new Promise((resolve, reject) => {
    if (typeof document === "undefined") {
      reject(new Error("jsonp-unavailable"));
      return;
    }
    wikiJsonpSeq += 1;
    const cbName = `__wikiJsonp${wikiJsonpSeq}`;
    const qs = new URLSearchParams({ ...params, format: "json", formatversion: "2", callback: cbName });
    const script = document.createElement("script");
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("jsonp-timeout"));
    }, 15000);
    const cleanup = () => {
      clearTimeout(timer);
      try { delete window[cbName]; } catch { window[cbName] = undefined; }
      script.remove();
    };
    window[cbName] = (data) => {
      cleanup();
      resolve(data);
    };
    script.onerror = () => {
      cleanup();
      reject(new Error("jsonp-error"));
    };
    script.src = `${WIKI_API}?${qs}`;
    document.head.appendChild(script);
  });
}

/**
 * Wikipedia API を呼ぶ。本命（fetch）が失敗したら予備（JSONP）に切り替え、
 * 両方とも失敗したら理由を WikiNet.lastError に残して例外を投げる。
 */
async function wikiApi(params) {
  const order = wikiFetchBroken ? ["jsonp", "fetch"] : ["fetch", "jsonp"];
  let lastErr = null;
  for (const how of order) {
    try {
      const data = how === "fetch" ? await wikiApiViaFetch(params) : await wikiApiViaJsonp(params);
      WikiNet.via = how;
      WikiNet.lastError = "";
      if (how === "jsonp") wikiFetchBroken = true;
      return data;
    } catch (e) {
      if (how === "fetch") wikiFetchBroken = true;
      lastErr = e;
    }
  }
  WikiNet.lastError = netErrorReason(lastErr);
  throw lastErr instanceof Error ? lastErr : new Error(WikiNet.lastError);
}

function wikiUrl(title) {
  return `https://ja.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`;
}

/**
 * 1記事の平文エキストラクトをタイトル完全指定で取る（索引優先・
 * ウェブ予備）。ブラウザ内Wikipedia索引（IndexLane）から取れたら
 * それを使い、取れなければ従来のウェブ版に倒れる。索引で成功したら
 * WikiNet.via を "index" にし、前の質問の失敗理由を持ち越さないよう
 * lastError を空に戻す。索引が失敗してウェブに倒れた場合の WikiNet
 * の記録はウェブ側に任せる。例外は外に漏らさない。
 */
async function wikiFetchExtract(title) {
  try {
    const hit = await IndexLane.fetchExtract(title);
    if (hit) {
      WikiNet.via = "index";
      WikiNet.lastError = "";
      return hit;
    }
  } catch {
    /* 索引の例外は漏らさずウェブ版へ倒れる */
  }
  return wikiFetchExtractWeb(title);
}

/** 1記事の平文エキストラクトをタイトル完全指定で取る。失敗時は null。 */
async function wikiFetchExtractWeb(title) {
  try {
    const data = await wikiApi({
      action: "query",
      prop: "extracts",
      explaintext: "1",
      exsectionformat: "plain",
      redirects: "1",
      titles: title,
    });
    const page = data.query?.pages?.[0];
    if (!page || page.missing || typeof page.title !== "string") return null;
    const text = (page.extract ?? "").slice(0, 6000);
    if (!text) return null;
    return {
      title: page.title,
      url: wikiUrl(page.title),
      snippet: text.slice(0, 240),
      text,
    };
  } catch {
    return null;
  }
}

/**
 * Wikipedia検索のタイトル一覧（索引優先・ウェブ予備）。索引から
 * 取れたらそれを使い、空なら従来のウェブ版に倒れる。WikiNet の
 * 扱いは wikiFetchExtract と同じ（索引で成功したら via を "index"
 * にし lastError を空に戻す）。例外は外に漏らさない。
 */
async function wikiSearchTitles(q, limit) {
  try {
    const hits = await IndexLane.searchTitles(q, limit);
    if (hits && hits.length > 0) {
      WikiNet.via = "index";
      WikiNet.lastError = "";
      return hits;
    }
  } catch {
    /* 索引の例外は漏らさずウェブ版へ倒れる */
  }
  return wikiSearchTitlesWeb(q, limit);
}

/** Wikipedia検索のタイトル一覧（順位順）。失敗時は空。 */
async function wikiSearchTitlesWeb(q, limit) {
  try {
    const data = await wikiApi({
      action: "query",
      generator: "search",
      gsrsearch: q,
      gsrnamespace: "0",
      gsrlimit: String(limit),
    });
    return [...(data.query?.pages ?? [])]
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
      .map((p) => p.title)
      .filter((t) => typeof t === "string" && t.length > 0);
  } catch {
    return [];
  }
}

/**
 * 曖昧さ回避ページが指す記事タイトルの候補。
 * Tier 1 は従来の文学寄りの偏りを保つ（作品・作者に触れる行で対象を名指す見出し）。
 * Tier 2（extended のとき）は全見出しを対象一致優先で並べる：企業・概念の
 * 曖昧さ回避（イオン → イオン (企業)）は作者らしい語なしに項目を説明するため、
 * Tier 1 だけでは何も見つからなかった。呼び出し側は「実際に質問へ答えられるか」
 * で候補をふるいにかけるので、網を広げても安全。
 */
function disambiguationCandidates(page, subject, extended = false) {
  const lines = page.text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  // 参照型の曖昧さ回避（球種 →「野球の球種については球種 (野球)を参照。」）は
  // 「タイトル - 説明」の一覧ではなく を参照 で指す。そのタイトルも集める。
  const refTitles = [];
  for (const line of lines) {
    const m = line.match(/については\s*(.+?)\s*を参照/);
    const t = m?.[1]?.trim();
    if (t && t.length <= 40 && t !== page.title && !refTitles.includes(t)) {
      refTitles.push(t);
    }
  }
  const listLines = lines.filter((l) => /^.+?\s-\s/.test(l));
  if (listLines.length < 5) return refTitles.slice(0, 3);
  const normSubject = normalizeJa(subject);
  const headOf = (line) => (line.split(/\s-\s/)[0] ?? "").trim();
  const usableHead = (head) =>
    head.length > 0 && head.length <= 40 && head !== page.title;
  const relWords = ["小説", "作者", "著者", "著", "による", "作品"];
  // Tier 1：作品関係の行で対象と完全一致する見出し（こゝろ のような旧仮名含む）。
  // 同名の派生作品（こころ (漫画) など）が作品自体より上位に来てはいけない。
  const exact = [];
  const loose = [];
  for (const line of listLines) {
    const head = headOf(line);
    if (!usableHead(head)) continue;
    if (!relWords.some((w) => line.includes(w))) continue;
    const normHead = normalizeJa(head);
    if (normHead === normSubject) {
      if (!exact.includes(head)) exact.push(head);
    } else if (normHead.startsWith(normSubject)) {
      if (!loose.includes(head)) loose.push(head);
    }
  }
  const tier1 = exact.length > 0 ? exact : loose;
  if (!extended) {
    return [...tier1, ...refTitles.filter((t) => !tier1.includes(t))].slice(0, 3);
  }
  // Tier 2：全見出し。対象一致を先に、残りはページ順。
  const tier2match = [];
  const tier2rest = [];
  for (const line of listLines) {
    const head = headOf(line);
    if (!usableHead(head)) continue;
    const normHead = normalizeJa(head);
    if (normHead === normSubject || normHead.startsWith(normSubject)) {
      if (!tier2match.includes(head)) tier2match.push(head);
    } else if (!tier2rest.includes(head)) {
      tier2rest.push(head);
    }
  }
  const listed = [
    ...tier1,
    ...tier2match.filter((t) => !tier1.includes(t)),
    ...tier2rest.filter((t) => !tier1.includes(t)),
  ];
  return [...listed, ...refTitles.filter((t) => !listed.includes(t))].slice(0, 6);
}

/**
 * 質問の種類に合う検索語（本社・作者・運営会社など）。題名だけの検索では
 * 同名の別対象の記事が上位に来ることがあるため、種類の語を足した検索も
 * 併せて行い、対象記事を取りこぼさないようにする。種類が読めない質問や
 * 汎用の種類では空文字（追加検索をしない）。
 */
function kindSearchPhrase(q) {
  const kind = classifyQuestion(q);
  switch (kind) {
    case "author":
      return "作者";
    case "hq":
      return "本社";
    case "operator":
      return "運営会社";
    case "specialty":
      return "名物";
    case "attribute":
      return /国歌/.test(q) ? "国歌" : "首都";
    case "person": {
      const role = /(知事|総理大臣|首相|社長|会長|創業者|創立者)/.exec(q);
      return role?.[1] ?? "";
    }
    case "date":
      return /創業/.test(q) ? "創業" : "設立";
    case "place":
      return "所在地";
    default:
      return "";
  }
}

/**
 * 照合用のWikipedia証拠。『…』の質問ではそのタイトルの記事（と曖昧さ回避の
 * 指し先）を先頭に置く。質問文そのままで検索すると単語記事（高さ 等）が
 * 混ざるため、題名だけの検索も併せて行う。
 */
async function searchWikipedia(q) {
  const ordered = [];
  const seen = new Set();
  const push = (r) => {
    if (r && !seen.has(r.title)) {
      seen.add(r.title);
      ordered.push(r);
    }
  };
  const subject = q.match(/『(.+?)』/)?.[1]?.trim();
  const topic = topicFromQuestion(q);
  try {
    if (subject) {
      const page = await wikiFetchExtract(subject);
      if (page) {
        // 派生記事を先に：曖昧さ回避ページの1行項目と同点でも本記事が勝つ。
        for (const cand of disambiguationCandidates(page, subject)) {
          push(await wikiFetchExtract(cand));
        }
        push(page);
      }
    }
    // 題名そのものの記事は、検索順位に頼らず必ず先頭候補に入れる：
    // 検索の上位に来なかったという理由だけで、本命の記事が証拠から
    // 漏れて正しい答えまで「未確認」になるのを防ぐ。
    if (topic && (!subject || normalizeJa(topic) !== normalizeJa(subject))) {
      const page = await wikiFetchExtract(topic);
      if (page) {
        for (const cand of disambiguationCandidates(page, topic)) {
          push(await wikiFetchExtract(cand));
        }
        push(page);
      }
    }
    if (topic && normalizeJa(topic) !== normalizeJa(q)) {
      const topicTitles = await wikiSearchTitles(topic, 4);
      const fetchedTopic = await Promise.all(topicTitles.map(wikiFetchExtract));
      for (const r of fetchedTopic) push(r);
    }
    // タイトル内検索（intitle:）も併せる：題名がタイトルに含まれる記事を
    // 優先的に拾い、本文にたまたま題名が出るだけの記事の混入を減らす。
    if (topic && !/\s/.test(topic)) {
      const intitleTitles = await wikiSearchTitles(`intitle:${topic}`, 4);
      const fetchedIntitle = await Promise.all(
        intitleTitles.map(wikiFetchExtract),
      );
      for (const r of fetchedIntitle) push(r);
    }
    // 種類の語を足した検索と、引用符つきの完全一致検索も併せる。
    // 題名だけでは同名の別対象（化学のイオン等）が上位に来て本命が
    // 埋もれることがある。集めた候補は後のスコアと厳しい照合で
    // ふるいにかけるため、網を広げても裏づけの精度は下がらない。
    if (topic) {
      const phrase = kindSearchPhrase(q);
      if (phrase && !topic.includes(phrase)) {
        const kindTitles = await wikiSearchTitles(`${topic} ${phrase}`, 4);
        const fetchedKind = await Promise.all(
          kindTitles.map(wikiFetchExtract),
        );
        for (const r of fetchedKind) push(r);
      }
      if (!/\s/.test(topic) && topic.length >= 2) {
        const quotedTitles = await wikiSearchTitles(`"${topic}"`, 3);
        const fetchedQuoted = await Promise.all(
          quotedTitles.map(wikiFetchExtract),
        );
        for (const r of fetchedQuoted) push(r);
      }
    }
    const titles = await wikiSearchTitles(q, 4);
    const fetched = await Promise.all(titles.map(wikiFetchExtract));
    for (const r of fetched) push(r);
    if (subject) {
      const extra = await wikiSearchTitles(subject, 3);
      const fetchedExtra = await Promise.all(extra.map(wikiFetchExtract));
      for (const r of fetchedExtra) push(r);
    }
  } catch {
    /* 集まった分だけ返す */
  }
  // 関連度スコア（安定ソート）：タイトルが題名や『…』の対象そのもの
  // （イオン (企業) のような基本名一致含む）の記事を最優先にし、
  // 次にタイトルに題名を含む記事、質問の種類に合う語（本社・作者など）を
  // 冒頭に含む記事を上位にする。同点は集めた順序を保つ。
  const normSubject = subject ? normalizeJa(subject) : "";
  const normTopic = topic ? normalizeJa(topic) : "";
  const kindRes = {
    author: /作者|著者|による/,
    hq: /本社/,
    person: /社長|会長|代表|創業者|総理大臣|首相|知事|現職/,
    operator: /運営/,
    specialty: /名物|郷土料理|特産/,
    attribute: /国歌|首都/,
    date: /設立|創立|創業|竣工|開業|完成/,
    measure: /\d/,
    place: /所在地|位置する|にある/,
  };
  const kindRe = kindRes[classifyQuestion(q)] ?? null;
  const scoreOf = (r) => {
    const nt = normalizeJa(r.title);
    const base = nt.replace(/[（(][^（）()]*[）)]\s*$/, "").trim();
    let s = 0;
    if (
      (normSubject && (nt === normSubject || base === normSubject)) ||
      (normTopic && (nt === normTopic || base === normTopic))
    ) {
      s += 100;
    } else if (
      (normTopic && (nt.includes(normTopic) || base.includes(normTopic))) ||
      (normSubject && nt.includes(normSubject))
    ) {
      s += 40;
    }
    if (kindRe && kindRe.test(r.text.slice(0, 1500))) s += 15;
    if (normTopic && normalizeJa(r.text).includes(normTopic)) s += 10;
    // 冒頭文の主語が題名そのものである記事は、その対象についての
    // 記事である可能性が高いので押し上げる（話題アンカーの軽い版）。
    if (normTopic) {
      const firstSentence = splitSentences(r.text)[0] ?? "";
      const headNorm = normalizeJa(firstSentence).slice(0, 48);
      if (
        headNorm.startsWith(normTopic) ||
        headNorm.startsWith(`『${normTopic}`)
      ) {
        s += 12;
      }
      // リダイレクト先の記事（立方体 → 正六面体）はタイトルが題名と
      // 一致しないが、冒頭文が「または立方体は…」と別名として題名を
      // 名指す。その場合はその対象の記事として扱い、派生記事
      // （変形立方体など、タイトルに題名を含むだけの記事）より上に置く。
      const headNormWide = normalizeJa(firstSentence).slice(0, 60);
      if (
        headNormWide.includes(`または${normTopic}`) ||
        headNormWide.includes(`とも呼ばれる${normTopic}`) ||
        headNormWide.includes(`ともいう${normTopic}`)
      ) {
        s += 45;
      }
    }
    // 曖昧さ回避ページと一覧ページは事実の証拠になりにくいので
    // 後回しにする（知事・総理大臣の一覧は事実の道で別に扱う）。
    // 外国語の項目（【KR】等）も証拠にしない方針と揃えて強く下げる。
    if (/曖昧さ回避/.test(r.title)) s -= 100;
    if (/^【/.test(r.title)) s -= 100;
    if (/曖昧さ回避のためのページ/.test(r.text.slice(0, 600))) s -= 80;
    if (/一覧$/.test(nt) && !/知事|総理大臣|首相/.test(q)) s -= 30;
    return s;
  };
  return ordered
    .map((r, i) => ({ r, i, s: scoreOf(r) }))
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .map((x) => x.r)
    .slice(0, 10);
}

/* ------------------------------------------------------------------ */
/* 事実源の記事解決＋インフォボックス解析                              */
/* ------------------------------------------------------------------ */

/**
 * 質問から題名（記事タイトルの候補）を導く：『…』があればそれ、
 * なければ質問の末尾（本社はどこ／の高さは／とは何／は誰…）を落とした名詞句。
 */
function topicFromQuestion(q) {
  const subj = q.match(/『(.+?)』/)?.[1]?.trim();
  if (subj) return subj;
  let t = q.trim().replace(/[？?！!。]+$/, "");
  // 「光合成では…何が放出される？」型：題名は では/には の前。
  // 後段の疑問詞カットより先に切り取る（「では何が」の は+何 に
  // 食われて「光合成で」のような中途半端な題名になるのを防ぐ）。
  const leadCut = t.match(/^(.{2,12}?)(では|には|とは)/);
  if (leadCut?.[1] && /何が|何を|何に|何から/.test(t)) {
    t = leadCut[1];
  }
  // 「たこ焼きが有名な都市はどこ？」型：題名は「有名な」の前にある
  // 名物・対象そのもの。「たこ焼きが有名な都市」のまま検索すると
  // 文の断片への全文検索になり、関係ない記事しか返ってこない。
  const famousCut = t.match(/^(.{2,16}?)(?:が|で)有名/);
  if (famousCut?.[1]) {
    t = famousCut[1];
  }
  t = t.replace(
    /(の)?(本社|本店|所在地|住所|場所)(は|が|って|を)?(どこ|どちら|なに|何|教えて|教え).*$/,
    "",
  );
  t = t.replace(/について.*$/, "");
  t = t.replace(/(を)?(教えて|教え|知りたい|聞きたい).*$/, "");
  // 運営会社の質問：題名は運営される側（ブランド・施設）そのもの。
  // 「モーリーファンタジーの運営会社は…」の尾を残すと記事が見つからない。
  if (/運営/.test(t)) {
    const cut = t.replace(/(の|を)運営.*$/, "");
    if (cut.length >= 2) t = cut;
  }
  // 属性の末尾：「日本の国歌は？」は日本（のインフォボックスに国歌が
  // ある）の質問、「大阪の名物は？」は大阪の質問。これを落とさないと
  // 題名が「日本の国歌」「大阪の名物」になって記事が見つからない。
  t = t.replace(
    /(の|で)(名物|特産品?|名産品?|郷土料理|国歌|首都|有名な食べ物)(は|が|って)?.*$/,
    "",
  );
  t = t.replace(
    // 「人口とは何ですか」「人口の意味は？」のような定義質問では、
    // 数値の名詞の後ろの とは／の意味 を数値カットで食わない
    // （題名そのものが消えてしまうため）。
    /(の)?(高さ|標高|海抜|全長|長さ|重さ|面積|人口|値段|価格|年齢)(?!(?:とは|の意味|の定義))(は|が|って)?.*$/,
    "",
  );
  t = t.replace(/(とは|って)(何|なに|どんな).*$/, "");
  t = t.replace(/(は|が)(誰|だれ|いつ|どこ|何|なに|いくつ|どんな|どう).*$/, "");
  t = t.replace(/(ですか|ますか|でしょうか|だろうか)$/, "");
  // 末尾に残った助詞と質問の名詞を落とす：「イオン株式会社の本社は？」
  // のような裸の「〜は？」型では上の置換が効かず、題名に「の本社」が
  // くっついて記事が見つからなくなる（タイトル完全一致が外れる）ため。
  // ただし題名そのもの（「人口」等）を消してしまわないよう、
  // 落とした結果が2文字未満になる場合は落とさない。
  const tailNouns =
    /(の)?(本社所在地|本店所在地|庁所在地|本社|本店|所在地|住所|場所|社長|会長|代表者|創業者|創立者|作者|著者|高さ|標高|海抜|全長|長さ|重さ|重量|面積|人口|値段|価格|年齢|設立|創立|創業|社名|意味|定義|由来|読み方)$/;
  for (let i = 0; i < 3; i += 1) {
    const before = t;
    t = t.replace(/(は|が|の|って|について)$/, "");
    const stripped = t.replace(tailNouns, "");
    if (stripped.length >= 2) t = stripped;
    if (t === before) break;
  }
  // 漠然とした尾：「大谷翔平のすごいところは？」の題名は大谷翔平。
  // 「すごいところ」まで含めて検索すると本人の記事が候補にすら
  // 上がらず、正しい答えでも裏づけようがなかった。
  const vagueStripped = t.replace(
    /(の)?(すごいところ|すごい点|特徴|魅力|強み|弱み|すごさ|凄さ|人気|様子)$/,
    "",
  );
  if (vagueStripped.length >= 2) t = vagueStripped;
  // 庁所在地の質問では上の置換で「宮城県の県庁」までしか落ちない：
  // 題名は都道府県そのものなので、尾の「の県庁」も落とす。
  const hallStripped = t.replace(/(の)?[都道府県]庁$/, "");
  if (hallStripped.length >= 2) t = hallStripped;
  // 現職者の質問の先頭の「現／現在の」：「現福島県知事は誰ですか」の
  // 題名は「福島県知事」で、「現福島県知事」という記事は無い。
  if (/(知事|総理大臣|首相|大統領|社長|会長|市長|町長|村長)/.test(q)) {
    const stripped = t.replace(/^(現在の|現|今の)/, "");
    if (stripped.length >= 2) t = stripped;
  }
  t = t.trim();
  return t.length >= 2 ? t : null;
}

/** 1記事のウィキテキスト（インフォボックス解析用）。失敗時は null。 */
async function wikiFetchWikitext(title) {
  try {
    const data = await wikiApi({
      action: "query",
      prop: "revisions",
      rvprop: "content",
      rvslots: "main",
      redirects: "1",
      titles: title,
    });
    const page = data.query?.pages?.[0];
    const content = page?.revisions?.[0]?.slots?.main?.content;
    if (!page || page.missing || typeof page.title !== "string" || !content) {
      return null;
    }
    return { title: page.title, content };
  } catch {
    return null;
  }
}

/** 一番内側の {{template|a|b}} を引数を並べた形に置換する。
 *  名前つき引数（key=value / key:value）は値そのものではないので落とす：
 *  都道府県の所在地欄の {{Coord|format=dms|type:adm1st_region:JP-07|
 *  display=inline,title}} が「format=dms type:adm1st_region:JP-07
 *  display=inline,title」というゴミとして答えに混入していた。 */
function unwrapTemplates(s) {
  let cur = s;
  for (let guard = 0; guard < 12; guard += 1) {
    const next = cur.replace(/\{\{([^{}]*)\}\}/g, (_m, inner) => {
      const parts = String(inner).split("|");
      return parts
        .slice(1)
        .filter((p) => !/^\s*[A-Za-z_][\w-]*\s*[=:]/.test(p))
        .join(" ");
    });
    if (next === cur) break;
    cur = next;
  }
  return cur;
}

/** ウィキテキストの欄の値を平文に整える。 */
function cleanFieldValue(v) {
  let s = v;
  s = s.replace(/<ref[^>]*>[\s\S]*?<\/ref>/g, "");
  s = s.replace(/<ref[^>]*\/>/g, "");
  s = s.replace(/<!--[\s\S]*?-->/g, "");
  // ファイルリンクは事実の値にはならない：日本の国歌欄は
  // 「[[君が代]] {{center|[[File:Kimi ga Yo instrumental.ogg]]}}」で、
  // この除去が無いと値に「File:…ogg」の尾が残って国歌の名前に見えない。
  s = s.replace(/\[\[(?:File|ファイル|Image|画像):[^\]]*\]\]/gi, "");
  s = s.replace(/<br\s*\/?>/gi, " ");
  s = unwrapTemplates(s);
  // 座標テンプレートの引数ゴミの残り（名前つき引数を落としても形を変えて
  // 残った場合の保険）：「format=dms」「type:adm1st_region:JP-07」
  // 「display=inline,title」のような語は住所の一部ではありえない。
  s = s.replace(
    /\b(?:format|type|display|scale|region|globe|dim)\s*[=:][^\s]*/g,
    " ",
  );
  s = s.replace(/\[\[(?:File|ファイル|Image|画像):[^\]]*\]\]/gi, "");
  s = s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2");
  s = s.replace(/\[\[([^\]]+)\]\]/g, "$1");
  s = s.replace(/\[https?:\/\/\S+\s+([^\]]+)\]/g, "$1");
  s = s.replace(/\[https?:\/\/\S+\]/g, "");
  s = s.replace(/<[^>]+>/g, "");
  s = s.replace(/'''?/g, "");
  s = s.replace(/[[\]]/g, "");
  s = s.replace(/;\s*/g, " ");
  s = s.replace(/(^|\s)\*\s*/g, " ");
  s = s.replace(/\s+/g, " ").trim();
  s = s.replace(/[{}]+$/, "").trim();
  return s.slice(0, 500);
}

/** テンプレート本文をトップレベルの `key = value` に割る。 */
function parseTemplateParams(body) {
  const parts = [];
  let depth = 0;
  let linkDepth = 0;
  let cur = "";
  for (let i = 0; i < body.length; i += 1) {
    const two = body.slice(i, i + 2);
    if (two === "{{") {
      depth += 1;
      cur += two;
      i += 1;
      continue;
    }
    if (two === "}}") {
      depth = Math.max(0, depth - 1);
      cur += two;
      i += 1;
      continue;
    }
    if (two === "[[") {
      linkDepth += 1;
      cur += two;
      i += 1;
      continue;
    }
    if (two === "]]") {
      linkDepth = Math.max(0, linkDepth - 1);
      cur += two;
      i += 1;
      continue;
    }
    const ch = body.charAt(i);
    if (ch === "|" && depth === 0 && linkDepth === 0) {
      parts.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  parts.push(cur);
  const fields = {};
  for (const part of parts.slice(1)) {
    const eq = part.indexOf("=");
    if (eq <= 0) continue;
    const key = part.slice(0, eq).trim();
    const value = cleanFieldValue(part.slice(eq + 1));
    if (key && value && !(key in fields)) fields[key] = value;
  }
  return fields;
}

/** ウィキテキストをトップレベルの {{…}} テンプレート本文に割る。 */
function topLevelTemplateBodies(content, maxBlocks) {
  const bodies = [];
  let depth = 0;
  let start = -1;
  for (let i = 0; i < content.length - 1 && bodies.length < maxBlocks; i += 1) {
    const two = content.slice(i, i + 2);
    if (two === "{{") {
      if (depth === 0) start = i;
      depth += 1;
      i += 1;
    } else if (two === "}}") {
      depth -= 1;
      i += 1;
      if (depth === 0 && start >= 0) {
        bodies.push(content.slice(start + 2, i - 1));
        start = -1;
      }
      if (depth < 0) depth = 0;
    }
  }
  return bodies;
}

/**
 * 回答コアが事実にできる欄の名前。名前つきインフォボックスでない
 * テンプレート（{{建築物}}, {{駅情報}} 等）からもこれらのキーだけは拾う
 * （東京スカイツリーの {{建築物}} は高さ/所在地/竣工を持っている）。
 */
const FACT_FIELD_KEYS = new Set([
  "作者", "著者",
  "本社所在地", "本店所在地", "所在地", "場所", "位置",
  "設立", "創立", "創業", "結成", "発足", "設立年", "創立年",
  "完成", "竣工", "開業", "供用開始",
  "標高", "高さ", "海抜", "全長", "長さ", "全幅", "幅",
  "重量", "重さ", "質量", "面積", "人口",
  "社長", "会長", "創業者", "創立者", "設立者",
  "代表者", "代表取締役", "代表取締役社長",
  "現職者", "現職", "知事",
  "運営会社", "運営元", "運営者", "社名",
  "国歌", "首都", "名物", "特産品", "名産品",
]);

/**
 * ページのトップレベルテンプレート群からインフォボックスの欄を解析する。
 * 名前つきインフォボックス（{{Infobox …}}, {{基礎情報 …}}）は全欄を、
 * それ以外は事実キーのみを、隙間を埋める形で取り込む（会社の事実は
 * リードより下の2つ目のインフォボックスにあることが多いため全部走査する）。
 */
function parseInfoboxFields(content) {
  const fields = {};
  const generic = {};
  for (const body of topLevelTemplateBodies(content, 80)) {
    const name = (body.split("|")[0] ?? "").trim();
    const found = parseTemplateParams(body);
    if (/^(?:Infobox|基礎情報)/.test(name)) {
      for (const [k, v] of Object.entries(found)) {
        if (!(k in fields)) fields[k] = v;
      }
    } else {
      for (const [k, v] of Object.entries(found)) {
        if (!FACT_FIELD_KEYS.has(k) || k in generic) continue;
        // {{coord}} を展開しただけの 位置 は座標そのもので、場所名ではない。
        if (/display=|_type:/.test(v)) continue;
        generic[k] = v;
      }
    }
  }
  for (const [k, v] of Object.entries(generic)) {
    if (!(k in fields)) fields[k] = v;
  }
  return fields;
}

/**
 * タイトル完全指定で1記事分の事実源データ（索引優先・ウェブ予備）。
 * 索引から取れたらそれを使い、取れなければ従来のウェブ版に倒れる。
 * WikiNet の扱いは wikiFetchExtract と同じ。索引版はウィキテキストを
 * 持たないため fields は空で、事実はリード文からの抽出で担う
 * （ウェブ版でもインフォボックスに欄が無い記事は同じ経路になる）。
 * 例外は外に漏らさない。
 */
async function wikiArticleSource(title) {
  try {
    const hit = await IndexLane.articleSource(title);
    if (hit) {
      WikiNet.via = "index";
      WikiNet.lastError = "";
      return hit;
    }
  } catch {
    /* 索引の例外は漏らさずウェブ版へ倒れる */
  }
  return wikiArticleSourceWeb(title);
}

/** タイトル完全指定で1記事分の事実源データ。失敗時は null。 */
async function wikiArticleSourceWeb(title) {
  const [wt, extract] = await Promise.all([
    wikiFetchWikitext(title),
    wikiFetchExtract(title),
  ]);
  if (!wt) return null;
  const fields = parseInfoboxFields(wt.content);
  const candidates = extract
    ? disambiguationCandidates(extract, title, true)
    : [];
  // 曖昧さ回避の判定は構造で：本物の記事も曖昧さ回避ページへ言及する
  // （富士山のウィキテキストには Otheruses の中に「富士山 (曖昧さ回避)」が
  // ある）ため、部分一致では誤判定する。{{曖昧さ回避}} の印、タイトルの
  // 接尾、またはインフォボックス無しで冒頭から一覧、のいずれかで判定。
  const firstLines = (extract?.text ?? "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .slice(0, 6);
  const listy =
    firstLines.filter((l) => /^.+?\s-\s/.test(l)).length >= 2;
  const templateMark =
    /\{\{\s*曖昧さ回避\s*(\||\}\})/.test(wt.content) ||
    /[（(]曖昧さ回避[）)]$/.test(wt.title);
  return {
    title: wt.title,
    url: wikiUrl(wt.title),
    lead: (extract?.text ?? "").slice(0, 5000),
    fields,
    disambiguation:
      templateMark ||
      (candidates.length > 0 && Object.keys(fields).length === 0 && listy),
    candidates,
  };
}

/** この記事源に、回答コアが使える事実があるか。 */
function hasUsableFact(s) {
  const keys = [
    "作者", "著者", "本社所在地", "本店所在地", "所在地", "標高", "高さ",
    "設立", "創立", "創業", "代表者", "社長", "会長", "創業者",
    "現職者", "現職", "運営会社", "運営元", "社名",
    "竣工", "開業", "全長", "面積", "人口", "国歌", "首都", "名物",
  ];
  if (keys.some((k) => s.fields[k])) return true;
  return /の(長編|短編|中編)?小説|による(小説|作品)|に本社を置く|標高は/.test(
    s.lead,
  );
}

/** この質問に答えるのに要る欄のキー（候補記事どうしの優先づけ用）。 */
function questionFieldKeys(q) {
  if (/作者|著者|書いた|描いた/.test(q)) return ["作者", "著者"];
  if (/知事/.test(q)) return ["現職者", "現職", "知事"];
  if (/運営/.test(q)) return ["運営会社", "運営元", "運営者", "社名"];
  if (/国歌/.test(q)) return ["国歌"];
  if (/首都/.test(q)) return ["首都"];
  if (/名物|特産|名産|郷土料理|有名な食べ物/.test(q)) {
    return ["名物", "特産品", "名産品"];
  }
  if (/社長/.test(q)) {
    return ["代表者", "社長", "代表取締役", "代表取締役社長"];
  }
  if (/会長/.test(q)) return ["会長", "代表者"];
  if (/創業者|創立者/.test(q)) {
    return ["創業者", "創立者", "設立者", "代表者"];
  }
  if (/代表/.test(q)) return ["代表者", "代表取締役", "社長"];
  if (/本社|本店/.test(q)) return ["本社所在地", "本店所在地", "所在地"];
  if (/設立|創立|創業|結成|いつ/.test(q)) {
    return ["設立", "創立", "創業", "結成", "竣工", "完成", "開業"];
  }
  if (/高さ|標高|海抜/.test(q)) return ["標高", "高さ", "海抜"];
  if (/全長|長さ/.test(q)) return ["全長", "長さ"];
  if (/重さ|重量/.test(q)) return ["重量", "重さ", "質量"];
  if (/面積/.test(q)) return ["面積"];
  if (/人口/.test(q)) return ["人口"];
  if (/どこ|所在地|場所/.test(q)) return ["所在地", "本社所在地", "場所"];
  return [];
}

function hasQuestionField(s, keys) {
  return keys.length > 0 && keys.some((k) => !!s.fields[k]);
}

/**
 * 質問に対する対象記事1本を解決する：タイトル完全一致を先に試し、
 * 曖昧さ回避ページ自体は事実源にせず指し先から使える記事を選ぶ。
 * 最後は題名のタイトル検索で、質問に名指しされたタイトルの記事だけを拾う。
 */
async function resolveArticleSourceInner(q) {
  const topic = topicFromQuestion(q);
  if (!topic) return null;
  const wantKeys = questionFieldKeys(q);
  const pick = async (titles, allowPlain) => {
    const sources = [];
    for (const t of titles) {
      const src = await wikiArticleSource(t);
      if (src && !src.disambiguation) sources.push(src);
    }
    return (
      // 実際に質問へ答えられる（種類つき事実が抜ける）記事を最優先する：
      // 欄の名前が合うだけの記事より、答えまで作れる記事の方が確実。
      sources.find((s) => extractFact(q, s)) ??
      sources.find((s) => hasQuestionField(s, wantKeys)) ??
      sources.find((s) => hasUsableFact(s)) ??
      (allowPlain
        ? (sources.find((s) => s.lead.length > 0) ?? sources[0] ?? null)
        : null)
    );
  };
  try {
    // 題名のタイトル検索から答えられる記事を探す（下の主記事チェックと
    // フォールバックの両方で使う）。
    const searchPick = async () => {
      // 候補タイトルは複数の検索から集める：題名そのもの、タイトル内
      // 検索、種類の語を足した検索。題名だけでは同名の別対象が上位に
      // 来て本命が6件の枠から漏れることがあるため。
      const titles = [];
      const addTitles = (list) => {
        for (const t of list) {
          if (!titles.includes(t)) titles.push(t);
        }
      };
      addTitles(await wikiSearchTitles(topic, 6));
      if (!/\s/.test(topic)) {
        addTitles(await wikiSearchTitles(`intitle:${topic}`, 4));
      }
      const phrase = kindSearchPhrase(q);
      if (phrase && !topic.includes(phrase)) {
        addTitles(await wikiSearchTitles(`${topic} ${phrase}`, 4));
      }
      const normTopic = normalizeJa(topic);
      const normQ = normalizeJa(q);
      // 検索タイトルの順位づけ：厳密一致（タイトルが題名そのもの、質問の
      // 先頭にある、球種 (野球) のように全要素を質問が名指す修飾つき、
      // イオン (企業) のように基本名が題名）を先に、単に質問に現れるだけの
      // タイトルは後にする（光合成の質問で通りすがりの酸素が勝ってはいけない）。
      const matched = [];
      for (const t of titles) {
        const nt = normalizeJa(t);
        const base = nt.replace(/[（(][^（）()]*[）)]\s*$/, "").trim();
        const parts = nt
          .replace(/[（）()]/g, " ")
          .split(/[\s　]+/)
          .filter((p) => p.length > 0);
        const coveredByQuestion =
          parts.length >= 2 && parts.every((p) => normQ.includes(p));
        const strictHit =
          nt === normTopic ||
          normQ.startsWith(nt) ||
          coveredByQuestion ||
          (base.length >= 2 && (base === normTopic || normQ.startsWith(base)));
        if (strictHit) matched.push({ title: t, rank: 0 });
        else if (normQ.includes(nt)) matched.push({ title: t, rank: 1 });
      }
      matched.sort((a, b) => a.rank - b.rank);
      const fromMatched = await pick(
        matched.map((m) => m.title),
        true,
      );
      if (fromMatched) return fromMatched;
      // 一致候補で決まらなかった場合の予備探索：タイトルが質問文に
      // 文字どおり現れない記事でも（例：「日本で一番高い山」→富士山）、
      // 質問の種類に合う事実が実際に抜けて、かつ答えが題名に触れて
      // いる記事だけを採用する。事実が抜けない記事や、答えが題名と
      // 無関係な記事は採用しない（精度を落とさないための関門）。
      // 定義の質問はこの予備探索の対象外（定義文は対象記事のもので
      // なければ意味がないため、従来どおり厳密一致だけに頼る）。
      const kind = classifyQuestion(q);
      if (!kind || kind === "definition") return null;
      const rest = titles.filter(
        (t) => !matched.some((m) => m.title === t),
      );
      for (const t of rest.slice(0, 4)) {
        const src = await wikiArticleSource(t);
        if (!src || src.disambiguation) continue;
        const fact = extractFact(q, src);
        if (!fact) continue;
        if (
          titleMatchesSubject(src.title, topic) ||
          normalizeJa(fact.answer).includes(normTopic)
        ) {
          return src;
        }
      }
      return null;
    };
    const primary = await wikiArticleSource(topic);
    if (primary && !primary.disambiguation) {
      // 種類つきの事実質問なのに主記事から事実が取れないときは、同名の
      // 別記事（例：企業のイオンではなく化学のイオン）を掴んでいる
      // 可能性がある。検索候補の中に実際に答えられる記事があれば、
      // そちらを優先し、無ければ主記事に戻る。
      const kind = classifyQuestion(q);
      if (!kind || extractFact(q, primary)) return primary;
      const better = await searchPick();
      if (better && extractFact(q, better)) return better;
      return primary;
    }
    if (primary && primary.disambiguation) {
      const resolved = await pick(primary.candidates.slice(0, 5), true);
      if (resolved) return resolved;
      // 候補で答えられなければタイトル検索へ進む。
    }
    return await searchPick();
  } catch {
    return null;
  }
}

/**
 * 複数記事の冒頭（約1200字）を1回のリクエストで取る。候補記事の
 * ふるい分け用（名物探し）。失敗時は空のMap。
 */
/**
 * 複数記事の冒頭（索引優先・ウェブ予備）。索引で埋まったタイトルは
 * 索引から取り、埋まらなかったタイトルだけ従来のウェブ版（引数は
 * 不足分だけ）で補ってマージする。索引が全部埋めたらウェブ版は
 * 呼ばない。索引から1件でも取れたら WikiNet.via を "index" にし
 * lastError を空に戻す。例外は外に漏らさない。
 */
async function wikiFetchExtractHeads(titles) {
  let fromIndex = null;
  try {
    fromIndex = await IndexLane.fetchExtractHeads(titles);
  } catch {
    fromIndex = null;
  }
  if (!fromIndex || fromIndex.size === 0) {
    return wikiFetchExtractHeadsWeb(titles);
  }
  const missing = titles.filter((t) => !fromIndex.has(t));
  if (missing.length > 0) {
    try {
      const extra = await wikiFetchExtractHeadsWeb(missing);
      for (const [k, v] of extra) {
        if (!fromIndex.has(k)) fromIndex.set(k, v);
      }
    } catch {
      /* 索引で取れた分だけ返す */
    }
  }
  WikiNet.via = "index";
  WikiNet.lastError = "";
  return fromIndex;
}

async function wikiFetchExtractHeadsWeb(titles) {
  const out = new Map();
  if (titles.length === 0) return out;
  try {
    const data = await wikiApi({
      action: "query",
      prop: "extracts",
      explaintext: "1",
      exsectionformat: "plain",
      exintro: "1",
      exchars: "1200",
      exlimit: "20",
      redirects: "1",
      titles: titles.join("|"),
    });
    for (const page of data.query?.pages ?? []) {
      if (page.title && page.extract) out.set(page.title, page.extract);
    }
  } catch {
    /* 集まった分だけ返す */
  }
  return out;
}

/**
 * その土地の名物料理を、料理自身の記事から探す。候補として認めるのは、
 * 冒頭の文がその土地の郷土料理／名物であると自称しているものだけ
 * （「…焼き上げた大阪の郷土料理。」＝たこ焼き、「宮城県仙台市の名物で
 * ある牛タン焼きについて扱う」＝仙台牛タン焼き)。候補が複数あれば、
 * 土地の記事自身のリードに名前が出るものを優先する（どて焼きも大阪の
 * 郷土料理を自称するが、大阪の食の節には出てこない）。
 */
async function findSpecialtyDish(topic, topicSrc) {
  const queries = [
    `"${topic}の郷土料理"`,
    `"${topic}の名物"`,
    `${topic} 名物`,
  ];
  const titles = [];
  for (const qy of queries) {
    for (const t of await wikiSearchTitles(qy, 5)) {
      if (!titles.includes(t)) titles.push(t);
    }
  }
  const wanted = titles.slice(0, 8).filter((t) => t !== topicSrc.title);
  // 候補の冒頭は1回のリクエストでまとめて取る（1件ずつ取ると
  // Wikipediaの429に当たりやすいため）。
  const heads = await wikiFetchExtractHeads(wanted);
  const cands = [];
  let order = 0;
  for (const t of wanted) {
    order += 1;
    const head = heads.get(t);
    if (!head) continue;
    const sentences = head
      .slice(0, 700)
      .split(/(?<=[。！？\n])/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    let name = null;
    let evidence = "";
    for (const s of sentences) {
      if (s.includes(`${topic}の郷土料理`) || s.includes(`${topic}の名物`)) {
        name = t;
        evidence = s;
        break;
      }
      const m = s.match(/名物である(.{2,16}?)(?:について|です|である|。|、)/);
      if (m?.[1] && s.includes(topic)) {
        name = m[1].trim();
        evidence = s;
        break;
      }
    }
    if (!name) continue;
    const mentioned =
      topicSrc.lead.includes(name) || topicSrc.lead.includes(t);
    cands.push({ name, evidence, mentioned, order });
  }
  cands.sort(
    (a, b) => Number(b.mentioned) - Number(a.mentioned) || a.order - b.order,
  );
  const top = cands[0];
  return top ? { name: top.name, evidence: top.evidence } : null;
}

/**
 * 対象記事を解決し、名物系の質問で記事自身に名物欄が無ければ、
 * 料理の記事から見つけた名物を 名物 欄として付ける（根拠の文は
 * 名物_根拠 に保持）。種類つき抽出が他のインフォボックス事実と
 * まったく同じ形で答えを組み立てられるようにするため。
 */
async function resolveArticleSource(q) {
  const src = await resolveArticleSourceInner(q);
  if (!src) return null;
  if (!/名物|特産|名産|郷土料理|有名な食べ物/.test(q)) return src;
  if (src.fields["名物"] || src.fields["特産品"] || src.fields["名産品"]) {
    return src;
  }
  const topic = topicFromQuestion(q);
  if (!topic) return src;
  try {
    const dish = await findSpecialtyDish(topic, src);
    if (dish) {
      src.fields["名物"] = dish.name;
      src.fields["名物_根拠"] = dish.evidence;
    }
  } catch {
    /* 対象記事だけでも立つ */
  }
  return src;
}

/* ================================================================== */
/* 天気クライアント（気象庁＋Open-Meteo：無料・キー不要、直接取得）    */
/*                                                                    */
/* 予報（今日・明日・明後日・週間）は気象庁の公式JSONから取り、        */
/* 現在の気温・現在の天気のような細かい情報は Open-Meteo から取る。    */
/* 出典は分けて表示する（予報：気象庁／現在の状況：Open-Meteo）。      */
/* 気象庁が取れないときは Open-Meteo のみにフォールバックし、出典      */
/* 表示もそれに合わせる。聞かれた時点の実データを読むだけで、何も      */
/* 保存しない——天気は変わるので、昨日の答えを今日の答えとして          */
/* 出してはいけない。                                                  */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* 天気取得の時間制限と並列化                                          */
/*                                                                    */
/* 以前はジオコーディング→Open-Meteo予報→気象庁area.json→気象庁予報   */
/* の4リクエストを直列に、しかもタイムアウトなしで待っていたため、    */
/* 遅い回線では回答まで10秒以上かかり、どれか1つの応答が止まると      */
/* 「検索で確認中…」のまま返ってこなかった。対策として、すべての      */
/* 天気リクエストに制限時間を付け、独立した取得は並列に走らせ、       */
/* 場所の解決結果（座標は変わらない）はセッション内で覚える。         */
/* ------------------------------------------------------------------ */

const WEATHER_TIMEOUT_MS = 8000;

/** 天気系のJSON取得。制限時間を過ぎたら諦めて null を返す（例外にしない）。 */
async function fetchWeatherJson(url, timeoutMs = WEATHER_TIMEOUT_MS) {
  const ctrl =
    typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  try {
    const res = await fetch(url, ctrl ? { signal: ctrl.signal } : undefined);
    if (!res || !res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** 解決できた場所（座標）はセッション内で覚え、2回目以降の天気質問では
 *  ジオコーディング（実測で一番遅いリクエスト）を飛ばす。 */
const geocodeCache = new Map();

/**
 * Open-Meteoの無料ジオコーディングで場所を探す。日本語の入力は
 * 都道府県と市区町村がくっついている（「福島県郡山市」）ことが多く、
 * そのままでは一致しないため、少しずつ短くした形も試す
 * （そのまま／都道府県なし／市区町村の接尾なし／両方なし）。
 * 逆に「東京」「横浜」「津」のような短い形は、データ側の正式名
 * （「東京都」「横浜市」「津市」）でないと一致しないことがあるため、
 * 接尾を足した形も試す。同名の小さい地名（青森県の横浜町など）が
 * 本命の都市より先に返ることがあるため、候補の中では人口の多い
 * ＝代表的な場所を優先する（人口10万人以上の都市が見つかれば即決）。
 * 日本の結果だけを使う。失敗時は null。
 */
async function geocodePlace(place) {
  const cached = geocodeCache.get(place);
  if (cached) return cached;
  const noPref = place.replace(/^.+?(?:都|道|府|県)/, "");
  const noCity = place.replace(/(?:市|区|町|村)$/, "");
  const bare = noPref.replace(/(?:市|区|町|村)$/, "");
  const withCity = (s) =>
    !s || /(?:市|区|町|村|都|道|府|県)$/.test(s) ? null : `${s}市`;
  const prefDone = /(?:都|府|県)$/.test(place);
  const variants = [
    ...new Set(
      [
        place,
        withCity(place),
        noPref,
        withCity(noPref),
        noCity,
        bare,
        withCity(bare),
        prefDone ? null : `${place}都`,
        prefDone ? null : `${place}府`,
        prefDone ? null : `${place}県`,
      ].filter((v) => v && v.length >= 1),
    ),
  ];
  const popOf = (r) => (typeof r.population === "number" ? r.population : 0);
  const toGeo = (r, v) => ({
    name: r.name ?? v,
    admin1: r.admin1 ?? "",
    latitude: r.latitude,
    longitude: r.longitude,
    population: popOf(r),
  });
  let fallback = null;
  for (const v of variants) {
    try {
      const data = await fetchWeatherJson(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(v)}&count=10&language=ja&format=json`,
      );
      // 通信自体が失敗・時間切れのときは、同じサーバーへの別の形の
      // 試行も同じように失敗するだけなので、ここで打ち切る
      // （結果が空だった場合は次の形を試す）。
      if (!data) break;
      const list = (data.results ?? []).filter(
        (r) =>
          typeof r.latitude === "number" &&
          typeof r.longitude === "number" &&
          r.country_code === "JP",
      );
      if (list.length === 0) continue;
      const best = list.reduce(
        (a, b) => (popOf(b) > popOf(a) ? b : a),
        list[0],
      );
      if (!fallback || popOf(best) > popOf(fallback)) fallback = best;
      if (popOf(best) >= 100000) {
        const g = toGeo(best, v);
        geocodeCache.set(place, g);
        return g;
      }
    } catch {
      /* 次の形を試す */
    }
  }
  if (!fallback) return null;
  const g = toGeo(fallback, place);
  geocodeCache.set(place, g);
  return g;
}

/** WMO天気コード（Open-Meteo）→ 平易な日本語。 */
function weatherCodeText(code) {
  const map = {
    0: "晴れ",
    1: "ほぼ晴れ",
    2: "晴れ時々くもり",
    3: "くもり",
    45: "霧",
    48: "霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "強い霧雨",
    56: "凍る霧雨",
    57: "凍る霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    66: "凍る雨",
    67: "凍る雨",
    71: "弱い雪",
    73: "雪",
    75: "強い雪",
    77: "雪",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "激しいにわか雨",
    85: "にわか雪",
    86: "にわか雪",
    95: "雷雨",
    96: "雷を伴う雨",
    99: "雷を伴う雨",
  };
  return map[code] ?? "不明";
}

const WEEKDAYS = ["日", "月", "火", "水", "木", "金", "土"];

/** "2026-11-09" → "11月9日（月）"（曜日は日付から計算する）。 */
function formatForecastDate(iso, index) {
  const [y = 0, mo = 1, d = 1] = iso.split("-").map(Number);
  const wd = WEEKDAYS[new Date(y, mo - 1, d).getDay()] ?? "";
  const base = `${mo}月${d}日（${wd}）`;
  if (index === 0) return `今日 ${base}`;
  if (index === 1) return `明日 ${base}`;
  if (index === 2) return `明後日 ${base}`;
  return base;
}

/**
 * Open-Meteo から現在の状況と日ごとの予報を取る（従来の実装）。
 * 成功時は { current_temp, current_text, days }、失敗時は null。
 * days の形は呼び出し側で気象庁の予報と差し替えられるよう共通にする。
 */
async function fetchOpenMeteoWeather(geo, day) {
  try {
    const params = new URLSearchParams({
      latitude: String(geo.latitude),
      longitude: String(geo.longitude),
      current: "temperature_2m,weather_code",
      daily:
        "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
      timezone: "auto",
      forecast_days: "7",
    });
    const data = await fetchWeatherJson(
      `https://api.open-meteo.com/v1/forecast?${params}`,
    );
    if (!data) return null;
    const daily = data.daily;
    const times = daily?.time ?? [];
    if (times.length === 0) return null;
    const indices =
      day === "week"
        ? times.map((_, i) => i)
        : [day === "today" ? 0 : day === "tomorrow" ? 1 : 2].filter(
            (i) => i < times.length,
          );
    if (indices.length === 0) return null;
    const days = indices.map((i) => ({
      date_label: formatForecastDate(times[i] ?? "", i),
      text: weatherCodeText(daily?.weather_code?.[i] ?? -1),
      temp_max: daily?.temperature_2m_max?.[i] ?? null,
      temp_min: daily?.temperature_2m_min?.[i] ?? null,
      precip_prob: daily?.precipitation_probability_max?.[i] ?? null,
    }));
    return {
      current_temp:
        typeof data.current?.temperature_2m === "number"
          ? data.current.temperature_2m
          : null,
      current_text:
        typeof data.current?.weather_code === "number"
          ? weatherCodeText(data.current.weather_code)
          : "",
      days,
    };
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* 気象庁の予報（公式JSON：無料・キー不要、CORS許可あり）               */
/*                                                                    */
/* 予報JSONは都道府県コード（office）単位で公開される：                */
/*   https://www.jma.go.jp/bosai/forecast/data/forecast/{office}.json */
/* 地点の解決には地域階層 area.json を使う：                           */
/*   https://www.jma.go.jp/bosai/common/const/area.json               */
/* 市区町村（class20）→ 細分区域（class15）→ 予報区（class10）→       */
/* 都道府県（office）とたどる（例：郡山市→中通り→福島県 070000）。     */
/* area.json は大きいので、セッション内で1回だけ取りに行く。           */
/* ------------------------------------------------------------------ */

const JMA_AREA_URL = "https://www.jma.go.jp/bosai/common/const/area.json";
const JMA_FORECAST_BASE =
  "https://www.jma.go.jp/bosai/forecast/data/forecast/";
const JMA_FORECAST_PAGE = "https://www.jma.go.jp/bosai/forecast/";

let jmaAreasPromise = null;

/** area.json をセッション内で1回だけ取得する。失敗したら次回再試行できるよう忘れる。 */
function loadJmaAreas() {
  if (!jmaAreasPromise) {
    jmaAreasPromise = (async () => {
      const data = await fetchWeatherJson(JMA_AREA_URL, 12000);
      if (!data) throw new Error("area.json を取得できませんでした");
      return data;
    })().catch((e) => {
      jmaAreasPromise = null;
      throw e;
    });
  }
  return jmaAreasPromise;
}

/**
 * 場所の文字列とジオコーディング結果から、気象庁の
 * { officeCode, officeName, class10Code, class10Name } を解決する。
 * 解決できなければ null（呼び出し側は Open-Meteo にフォールバックする）。
 */
async function resolveJmaArea(place, geo) {
  let areas;
  try {
    areas = await loadJmaAreas();
  } catch {
    return null;
  }
  const { offices, class10s, class15s, class20s } = areas ?? {};
  if (!offices || !class10s || !class15s || !class20s) return null;
  const names = [];
  const push = (s) => {
    const v = (s ?? "").trim();
    if (v && !names.includes(v)) names.push(v);
  };
  push(geo?.name);
  push(place);
  for (const s of [...names]) push(s.replace(/^.+?(?:都|道|府|県)/, ""));
  // 「東京」「津」のような短い形は、正式名（「東京都」「津市」）に
  // 接尾を足した形でも探す（area.json の名前は正式名で載っている）。
  for (const s of [...names]) {
    if (/(?:市|区|町|村|都|道|府|県)$/.test(s)) continue;
    push(`${s}市`);
    push(`${s}都`);
    push(`${s}府`);
    push(`${s}県`);
  }
  const fromClass20 = (entry) => {
    const c15 = class15s[entry.parent];
    const c10 = c15 ? class10s[c15.parent] : null;
    const off = c10 ? offices[c10.parent] : null;
    if (!c10 || !off || !c10.parent) return null;
    return {
      officeCode: c10.parent,
      officeName: off.name,
      class10Code: c15.parent,
      class10Name: c10.name,
    };
  };
  // 市区町村の完全一致を最優先する（大和郡山市と郡山市を取り違えない）。
  let hit = null;
  for (const n of names) {
    for (const [code, v] of Object.entries(class20s)) {
      if (v.name === n) {
        hit = { code, ...v };
        break;
      }
    }
    if (hit) break;
  }
  // 前方一致（「札幌市」→「札幌市中央区」など、区に分かれる政令市）。
  // 名前の優先順（ジオコーディング名→入力の順）に探し、最初に当たった
  // 名前の中で一番短いものを採る。全名前を通して一番短いものを採ると、
  // 「横浜」で青森県の横浜町のような同名の小さい地名を拾ってしまう。
  if (!hit) {
    for (const n of names) {
      if (n.length < 2) continue;
      let best = null;
      for (const [code, v] of Object.entries(class20s)) {
        if (v.name.startsWith(n) && (!best || v.name.length < best.name.length)) {
          best = { code, ...v };
        }
      }
      if (best) {
        hit = best;
        break;
      }
    }
  }
  if (hit) {
    const resolved = fromClass20(hit);
    if (resolved) return resolved;
  }
  // 都道府県だけ分かった場合：その office の予報区から、地名に名前が
  // 含まれるものを優先し、無ければ先頭の予報区を使う。
  const prefName =
    (geo?.admin1 ?? "").trim() ||
    (place.match(/^.+?(?:都|道|府|県)/)?.[0] ?? "");
  for (const [code, off] of Object.entries(offices)) {
    if (off.name !== prefName && !names.includes(off.name)) continue;
    const children = off.children ?? [];
    if (children.length === 0) return null;
    let c10code = children[0];
    for (const ch of children) {
      const nm = class10s[ch]?.name ?? "";
      if (nm && (place.includes(nm) || names.some((n) => n.includes(nm)))) {
        c10code = ch;
        break;
      }
    }
    const c10 = class10s[c10code];
    if (!c10) return null;
    return {
      officeCode: code,
      officeName: off.name,
      class10Code: c10code,
      class10Name: c10.name,
    };
  }
  return null;
}

/** 気象庁の天気コード（週間予報用）→ 日本語の天気文。 */
const JMA_WEATHER_TEXT = {
  100: "晴れ", 101: "晴れ時々くもり", 102: "晴れ一時雨", 103: "晴れ時々雨",
  104: "晴れ一時雪", 105: "晴れ時々雪", 106: "晴れ一時雨か雪",
  107: "晴れ時々雨か雪", 108: "晴れ一時雨か雷雨",
  110: "晴れ後時々くもり", 111: "晴れ後くもり", 112: "晴れ後一時雨",
  113: "晴れ後時々雨", 114: "晴れ後雨", 115: "晴れ後一時雪",
  116: "晴れ後時々雪", 117: "晴れ後雪", 118: "晴れ後雨か雪",
  119: "晴れ後雨か雷雨", 120: "晴れ朝夕一時雨", 121: "晴れ朝の内一時雨",
  122: "晴れ夕方一時雨", 123: "晴れ山沿い雷雨", 124: "晴れ山沿い雪",
  125: "晴れ午後は雷雨", 126: "晴れ昼頃から雨", 127: "晴れ夕方から雨",
  128: "晴れ夜は雨", 130: "朝の内霧後晴れ", 131: "晴れ明け方霧",
  132: "晴れ朝夕くもり", 140: "晴れ時々雨で雷を伴う",
  160: "晴れ一時雨か雪", 170: "晴れ時々雨か雪", 181: "晴れ後雪か雨",
  200: "くもり", 201: "くもり時々晴れ", 202: "くもり一時雨",
  203: "くもり時々雨", 204: "くもり一時雪", 205: "くもり時々雪",
  206: "くもり一時雨か雪", 207: "くもり時々雨か雪",
  208: "くもり一時雨か雷雨", 209: "霧", 210: "くもり後時々晴れ",
  211: "くもり後晴れ", 212: "くもり後一時雨", 213: "くもり後時々雨",
  214: "くもり後雨", 215: "くもり後一時雪", 216: "くもり後時々雪",
  217: "くもり後雪", 218: "くもり後雨か雪", 219: "くもり後雨か雷雨",
  220: "くもり朝夕一時雨", 221: "くもり朝の内一時雨",
  222: "くもり夕方一時雨", 223: "くもり日中時々晴れ",
  224: "くもり昼頃から雨", 225: "くもり夕方から雨", 226: "くもり夜は雨",
  228: "くもり昼頃から雪", 229: "くもり夕方から雪", 230: "くもり夜は雪",
  231: "くもり海・海岸は霧か霧雨のち晴れ",
  240: "くもり時々雨で雷を伴う", 250: "くもり時々雪で雷を伴う",
  260: "くもり一時雨か雪", 270: "くもり時々雨か雪", 281: "くもり後雪か雨",
  300: "雨", 301: "雨時々晴れ", 302: "雨時々止む", 303: "雨時々雪",
  304: "雨か雪", 306: "大雨", 308: "雨で暴風を伴う", 309: "雨一時雪",
  311: "雨後晴れ", 313: "雨後くもり", 314: "雨後時々雪", 315: "雨後雪",
  316: "雨か雪後晴れ", 317: "雨か雪後くもり", 320: "朝の内雨後晴れ",
  321: "朝の内雨後くもり", 322: "雨朝晩一時雪", 323: "雨昼頃から晴れ",
  324: "雨夕方から晴れ", 325: "雨夜は晴れ", 326: "雨夕方から雪",
  327: "雨夜は雪", 328: "雨一時強く降る", 329: "雨一時みぞれ",
  340: "雪か雨", 350: "雨で雷を伴う", 361: "雨か雪後晴れ",
  371: "雨か雪後くもり",
  400: "雪", 401: "雪時々晴れ", 402: "雪時々止む", 403: "雪時々雨",
  405: "大雪", 406: "風雪強い", 407: "暴風雪", 409: "雪一時雨",
  411: "雪後晴れ", 413: "雪後くもり", 414: "雪後雨",
  420: "朝の内雪後晴れ", 421: "朝の内雪後くもり", 422: "雪昼頃から雨",
  423: "雪夕方から雨", 424: "雪夜は雨", 425: "雪一時強く降る",
  426: "雪後みぞれ", 427: "雪一時みぞれ", 430: "みぞれ",
  450: "雪で雷を伴う",
};

function jmaWeatherText(code) {
  const n = Number(code);
  if (Number.isFinite(n) && JMA_WEATHER_TEXT[n]) return JMA_WEATHER_TEXT[n];
  const head = { 1: "晴れ", 2: "くもり", 3: "雨", 4: "雪" }[
    Math.floor(n / 100)
  ];
  return head ?? "不明";
}

/** 気象庁JSONの数値文字列（"" は欠測）→ 数値 or null。 */
function jmaNum(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** "2026-11-10" 同士の日数差（今日=0、明日=1、明後日=2）。 */
function jmaDayDiff(iso, todayIso) {
  const p = (s) => s.split("-").map(Number);
  const [y1, m1, d1] = p(iso);
  const [y2, m2, d2] = p(todayIso);
  return Math.round(
    (Date.UTC(y1, m1 - 1, d1) - Date.UTC(y2, m2 - 1, d2)) / 86400000,
  );
}

function localTodayIso() {
  const now = new Date();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${mm}-${dd}`;
}

/**
 * 気象庁の予報を取る。成功時は
 * { officeName, areaName, detailed: Map<日付, day>, weekly: [day] }、
 * 失敗時は null（呼び出し側は Open-Meteo のみにフォールバックする）。
 * day は { date, text, temp_max, temp_min, precip_prob }（date は ISO日付）。
 * 詳細予報（3日間）は天気文・降水確率・気温（地点別）を、週間予報は
 * 天気コード・降水確率・気温（代表地点）を使う。
 */
async function fetchJmaForecast(place, geo) {
  const area = await resolveJmaArea(place, geo);
  if (!area) return null;
  const data = await fetchWeatherJson(
    `${JMA_FORECAST_BASE}${area.officeCode}.json`,
  );
  if (!data) return null;
  if (!Array.isArray(data) || data.length === 0) return null;
  // 気温地点の名前候補：市区町村名から接尾（市など）を外した形を優先する
  // （例：郡山市→「郡山」、仙台市→「仙台」）。
  const cityName = (geo?.name ?? "").trim();
  const stem = cityName.replace(/(?:市|区|町|村)$/, "");
  const pickPoint = (areasList) => {
    if (!Array.isArray(areasList) || areasList.length === 0) return null;
    return (
      areasList.find((a) => a.area?.name === stem) ??
      areasList.find((a) => a.area?.name === cityName) ??
      null
    );
  };
  const det = data[0];
  const detailed = new Map();
  const weeklyTemps = new Map();
  const wk = data[1];
  if (wk) {
    const wkTemps = wk.timeSeries?.[1];
    const pt = wkTemps
      ? (pickPoint(wkTemps.areas) ?? wkTemps.areas?.[0] ?? null)
      : null;
    if (pt && wkTemps) {
      (wkTemps.timeDefines ?? []).forEach((td, i) => {
        weeklyTemps.set(td.slice(0, 10), {
          temp_max: jmaNum(pt.tempsMax?.[i]),
          temp_min: jmaNum(pt.tempsMin?.[i]),
        });
      });
    }
  }
  if (det) {
    const tsW = det.timeSeries?.[0];
    const tsP = det.timeSeries?.[1];
    const tsT = det.timeSeries?.[2];
    const wArea =
      tsW?.areas?.find((a) => a.area?.code === area.class10Code) ??
      tsW?.areas?.find((a) => a.area?.name === area.class10Name);
    if (tsW && wArea) {
      // 降水確率：6時間ごとの値を日ごとにまとめ、その日の最大値を使う。
      const popsByDate = new Map();
      const pArea =
        tsP?.areas?.find((a) => a.area?.code === area.class10Code) ??
        tsP?.areas?.find((a) => a.area?.name === area.class10Name);
      if (tsP && pArea) {
        (tsP.timeDefines ?? []).forEach((td, i) => {
          const date = td.slice(0, 10);
          const v = jmaNum(pArea.pops?.[i]);
          if (v === null) return;
          const cur = popsByDate.get(date);
          if (cur === undefined || v > cur) popsByDate.set(date, v);
        });
      }
      // 気温：地点別の予報。9時の値がその日の最高、0時の値が最低。
      const tempsByDate = new Map();
      const tPt = tsT ? pickPoint(tsT.areas) : null;
      if (tsT && tPt) {
        (tsT.timeDefines ?? []).forEach((td, i) => {
          const date = td.slice(0, 10);
          const hour = td.slice(11, 13);
          const v = jmaNum(tPt.temps?.[i]);
          if (v === null) return;
          const cur = tempsByDate.get(date) ?? {};
          if (hour === "09") cur.temp_max = v;
          else if (hour === "00") cur.temp_min = v;
          tempsByDate.set(date, cur);
        });
      }
      (tsW.timeDefines ?? []).forEach((td, i) => {
        const date = td.slice(0, 10);
        const text = (wArea.weathers?.[i] ?? "")
          .replace(/[\s　]+/g, " ")
          .trim();
        if (!text) return;
        const dt = tempsByDate.get(date) ?? {};
        const wt = weeklyTemps.get(date) ?? {};
        detailed.set(date, {
          date,
          text,
          temp_max: dt.temp_max ?? wt.temp_max ?? null,
          temp_min: dt.temp_min ?? wt.temp_min ?? null,
          precip_prob: popsByDate.get(date) ?? null,
        });
      });
    }
  }
  const weekly = [];
  if (wk) {
    const tsW = wk.timeSeries?.[0];
    // 週間予報の地域は詳細予報とコードが違うことがある（例：福島県の
    // 「中通り・浜通り」070100）。コード完全一致→名前の含み合い→先頭、で選ぶ。
    const wArea =
      tsW?.areas?.find((a) => a.area?.code === area.class10Code) ??
      tsW?.areas?.find(
        (a) =>
          a.area?.name &&
          (a.area.name.includes(area.class10Name) ||
            area.class10Name.includes(a.area.name)),
      ) ??
      tsW?.areas?.[0];
    if (tsW && wArea) {
      (tsW.timeDefines ?? []).forEach((td, i) => {
        const date = td.slice(0, 10);
        const wt = weeklyTemps.get(date) ?? {};
        weekly.push({
          date,
          text: jmaWeatherText(wArea.weatherCodes?.[i]),
          temp_max: wt.temp_max ?? null,
          temp_min: wt.temp_min ?? null,
          precip_prob: jmaNum(wArea.pops?.[i]),
        });
      });
    }
  }
  if (detailed.size === 0 && weekly.length === 0) return null;
  return {
    officeName: area.officeName,
    areaName: area.class10Name,
    detailed,
    weekly,
  };
}

/** 気象庁の予報から、聞かれた日に使う分だけを date_label つきで返す。 */
function selectJmaDays(jma, day) {
  const todayIso = localTodayIso();
  const toLabeled = (d) => ({
    date_label: formatForecastDate(d.date, jmaDayDiff(d.date, todayIso)),
    text: d.text,
    temp_max: d.temp_max,
    temp_min: d.temp_min,
    precip_prob: d.precip_prob,
  });
  if (day === "week") {
    // 週間予報は明日から始まることが多く、そのままだと今日が抜ける。
    // 詳細予報（今日〜明後日）と重ねて、日付順に最大7日分を返す
    // （同じ日は詳細予報の文を優先する）。
    const byDate = new Map();
    for (const d of jma.weekly) byDate.set(d.date, d);
    for (const d of jma.detailed.values()) byDate.set(d.date, d);
    return [...byDate.values()]
      .sort((a, b) => (a.date < b.date ? -1 : 1))
      .slice(0, 7)
      .map(toLabeled);
  }
  const offset = day === "today" ? 0 : day === "tomorrow" ? 1 : 2;
  const target = new Date();
  target.setDate(target.getDate() + offset);
  const mm = String(target.getMonth() + 1).padStart(2, "0");
  const dd = String(target.getDate()).padStart(2, "0");
  const targetIso = `${target.getFullYear()}-${mm}-${dd}`;
  const found =
    jma.detailed.get(targetIso) ??
    jma.weekly.find((d) => d.date === targetIso);
  return found ? [toLabeled(found)] : [];
}

/**
 * 天気を取る。予報は気象庁、現在の状況は Open-Meteo から取り、
 * { ok, error, place_label, current_temp, current_text, days,
 *   forecast_source, jma_area_label, source_url } を返す。
 * 気象庁が取れない場合は Open-Meteo のみ（forecast_source は
 * "open-meteo"）にフォールバックする。失敗は ok:false ＋理由で返し、
 * 天気をでっち上げない。
 */
async function fetchWeather(place, defaultPlace, day) {
  const empty = {
    ok: false,
    error: "",
    place_label: "",
    current_temp: null,
    current_text: "",
    days: [],
    forecast_source: "open-meteo",
    jma_area_label: "",
    source_url: "https://open-meteo.com/",
  };
  const wanted = (place || "").trim() || (defaultPlace || "").trim();
  // 気象庁の地域階層（area.json、約260KBで一番大きい）は、場所探しと
  // 並行して先に読み始める（読み込み結果はセッション内で共有される）。
  loadJmaAreas().catch(() => {});
  const geo = await geocodePlace(wanted);
  if (!geo) {
    // Open-Meteo のジオコーディングで見つからなくても、気象庁の地域
    // 階層（area.json）だけで市区町村→予報区が解決できれば、予報だけは
    // 答える（現在の状況は Open-Meteo が必要なため出ない）。
    let jmaOnly = null;
    try {
      jmaOnly = await fetchJmaForecast(wanted, null);
    } catch {
      jmaOnly = null;
    }
    if (jmaOnly) {
      const days = selectJmaDays(jmaOnly, day);
      if (days.length > 0) {
        return {
          ok: true,
          error: "",
          place_label: `${jmaOnly.officeName} ${wanted}`,
          current_temp: null,
          current_text: "",
          days,
          forecast_source: "jma",
          jma_area_label: `${jmaOnly.officeName}・${jmaOnly.areaName}`,
          source_url: JMA_FORECAST_PAGE,
        };
      }
    }
    return { ...empty, error: "place_not_found" };
  }
  const placeLabel =
    geo.admin1 && geo.admin1 !== geo.name
      ? `${geo.admin1} ${geo.name}`
      : geo.name;
  // Open-Meteo の予報と気象庁の予報は互いに独立なので並列に取る
  // （以前は直列で、両方の待ち時間がそのまま積み上がっていた）。
  const [om, jma] = await Promise.all([
    fetchOpenMeteoWeather(geo, day),
    fetchJmaForecast(wanted, geo).catch(() => null),
  ]);
  if (jma) {
    // 人口情報のない弱いジオコーディング（同名の小さい地名など）と
    // 気象庁の解決した都道府県が食い違うときは、混ぜて答えず
    // 気象庁の場所で統一する（例：「富山」で千葉県の富山地区の
    // 現在気温と富山県の予報が混ざる事故の防止）。
    const prefForm = (s) => /(?:都|府|県)$/.test(s) || s === "北海道";
    if (
      geo.admin1 &&
      prefForm(geo.admin1) &&
      prefForm(jma.officeName) &&
      geo.admin1 !== jma.officeName &&
      !(geo.population > 0)
    ) {
      const days = selectJmaDays(jma, day);
      if (days.length > 0) {
        return {
          ok: true,
          error: "",
          place_label: `${jma.officeName} ${wanted}`,
          current_temp: null,
          current_text: "",
          days,
          forecast_source: "jma",
          jma_area_label: `${jma.officeName}・${jma.areaName}`,
          source_url: JMA_FORECAST_PAGE,
        };
      }
    }
  }
  if (jma) {
    const days = selectJmaDays(jma, day);
    if (days.length > 0) {
      return {
        ok: true,
        error: "",
        place_label: placeLabel,
        current_temp: om?.current_temp ?? null,
        current_text: om?.current_text ?? "",
        days,
        forecast_source: "jma",
        jma_area_label: `${jma.officeName}・${jma.areaName}`,
        source_url: JMA_FORECAST_PAGE,
      };
    }
  }
  if (om) {
    return {
      ok: true,
      error: "",
      place_label: placeLabel,
      current_temp: om.current_temp,
      current_text: om.current_text,
      days: om.days,
      forecast_source: "open-meteo",
      jma_area_label: "",
      source_url: "https://open-meteo.com/",
    };
  }
  return { ...empty, error: "fetch_failed" };
}

/** article の形（{title,url,lead,fields}）だけを返す。 */
async function fetchArticle(q) {
  const src = await resolveArticleSource(q);
  if (!src) return null;
  return { title: src.title, url: src.url, lead: src.lead, fields: src.fields };
}

/* ================================================================== */
/* 知識メモ帳（この端末のブラウザ内に保存）                             */
/* ================================================================== */

const NB_KEY = "jisaku-ai-codepen-notebook-v1";
const URL_KEY = "jisaku-ai-model-base-url";
const WEATHER_PLACE_KEY = "jisaku-ai-weather-place";
const DEFAULT_WEATHER_PLACE = "福島県郡山市";
const DEFAULT_BASE_URL = "https://ydf65ypmtj-boop.github.io/llm-chat/";

const store = {
  get(key) {
    try { return window.localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { window.localStorage.setItem(key, value); } catch { /* 保存できない環境では諦める */ }
  },
};

const Notebook = {
  items: [],
  load() {
    try {
      const raw = store.get(NB_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      this.items = Array.isArray(parsed) ? parsed : [];
    } catch {
      this.items = [];
    }
  },
  save() {
    store.set(NB_KEY, JSON.stringify(this.items));
  },
  /* 同じ質問は重複させず上書き。事実が変わったら未学習に戻る。 */
  add(entry) {
    const now = new Date().toISOString();
    const existing = this.items.find((it) => it.question === entry.question);
    if (existing) {
      const keepTrained = existing.fact === entry.fact ? existing.trained : false;
      existing.answer = entry.answer;
      existing.source_title = entry.source_title;
      existing.source_url = entry.source_url;
      existing.question_type = entry.question_type;
      existing.fact = entry.fact;
      existing.trained = keepTrained;
      existing.created_at = now;
    } else {
      const maxId = this.items.reduce((a, it) => Math.max(a, it.id || 0), 0);
      this.items.unshift({
        id: maxId + 1,
        question: entry.question,
        answer: entry.answer,
        source_title: entry.source_title || "",
        source_url: entry.source_url || "",
        question_type: entry.question_type || "",
        fact: entry.fact || "",
        trained: false,
        created_at: now,
      });
    }
    this.save();
  },
  remove(id) {
    this.items = this.items.filter((it) => it.id !== id);
    this.save();
  },
  markTrained(ids) {
    for (const it of this.items) {
      if (!ids || ids.includes(it.id)) it.trained = true;
    }
    this.save();
  },
};

/* ================================================================== */
/* 画面まわり                                                          */
/* ================================================================== */

const BADGES = {
  notebook: { label: "知識メモ帳の答え", className: "badge-notebook" },
  "notebook-stale": {
    label: "知識メモ帳の答え（今回ネットで確認できなかった）",
    className: "badge-unverified",
  },
  calc: { label: "検算の答え", className: "badge-calc" },
  self: {
    label: "Cube自身のこと（アプリ固定の情報）",
    className: "badge-self",
  },
  weather: {
    label: "天気情報（気象庁・Open-Meteo）",
    className: "badge-weather",
  },
  "weather-om": { label: "天気情報（Open-Meteo）", className: "badge-weather" },
  model: {
    label: "Cubeの答え（ネットで確認済み）",
    className: "badge-model",
  },
  net: { label: "ネット検索の答え", className: "badge-net" },
  unverified: {
    label: "Cubeの答え（ネットで確認できなかった）",
    className: "badge-unverified",
  },
  unknown: {
    label: "ネットで確認できなかった（未確認）",
    className: "badge-unverified",
  },
  none: { label: "", className: "" },
};

const KIND_LABELS = {
  author: "作者",
  hq: "本社",
  place: "場所",
  date: "日付",
  measure: "数値",
  person: "人物",
  operator: "運営会社",
  specialty: "名物",
  attribute: "国歌・首都",
  definition: "定義",
};

function kindLabel(kind) {
  if (kind === "calc") return "計算";
  return KIND_LABELS[kind] ?? kind;
}

function formatBytes(n) {
  if (n <= 0) return "0 MB";
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatParams(n) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

const el = (id) => document.getElementById(id);

const App = {
  phase: "idle", // idle | thinking | searching | saving
  modelStatus: { kind: "not_loaded" },
  modelId: "v1", // 選択中のモデルID（localStorage に保存。初期値は v1）
  modelBusy: false, // モデル読み込み中は切り替えボタンを無効化する
  ultraLoadSeq: 0, // Ultra 読み込みの世代。通常版の読み込み中に軽量エンジン（GGUF版）へ引き継いだとき、古い読み込みの後処理（状態表示の上書き）を無効化するために使う
  baseUrl: DEFAULT_BASE_URL,
  weatherPlace: DEFAULT_WEATHER_PLACE,
  confirmMarkAll: false,

  init() {
    Notebook.load();
    // 選択中のモデルを復元する（初期値は v1）。ベースURLはモデルごとの
    // 上書きとして読み込む（v1 は従来キー、v2 / Twin / Pro / Max は専用キー）。
    const savedId = store.get(MODEL_ID_KEY);
    this.modelId = MODELS[savedId] ? savedId : "v1";
    const def = modelDef(this.modelId);
    // Ultra にはベースURLが無い（Hugging Face から読み込む）。
    this.baseUrl = def.engine === "ultra"
      ? ""
      : store.get(def.urlKey) || def.defaultBaseUrl;
    el("base-url").value = this.baseUrl;
    el("current-url").textContent = this.baseUrl;
    for (const btn of document.querySelectorAll(".model-switch-btn")) {
      btn.addEventListener("click", () => void this.switchModel(btn.dataset.model));
    }
    // 軽量エンジン（GGUF版）の開始ボタン：通常版の読み込みがブラウザの
    // 再読み込みで中断されたときにだけ通知エリアへ出す。自動では
    // 始めず、押されたときだけ GGUF 版の読み込みを始める。
    const ggufBtn = el("ultra-gguf-btn");
    if (ggufBtn) {
      ggufBtn.addEventListener("click", () => void this.loadUltraGguf());
    }
    // エンジン選択ブロックの2ボタン：Ultra を選んだら最初から出し、
    // 押されたエンジンだけを開始する（どちらも自動では開始しない）。
    const onnxChoiceBtn = el("ultra-load-onnx-btn");
    if (onnxChoiceBtn) {
      onnxChoiceBtn.addEventListener("click", () => void this.loadUltraOnnx());
    }
    const ggufChoiceBtn = el("ultra-load-gguf-btn");
    if (ggufChoiceBtn) {
      ggufChoiceBtn.addEventListener("click", () => void this.loadUltraGguf());
    }
    const imacChoiceBtn = el("ultra-load-imac-btn");
    if (imacChoiceBtn) {
      imacChoiceBtn.addEventListener("click", () => void this.loadUltraImac());
    }
    this.renderModelSwitch();
    this.weatherPlace = store.get(WEATHER_PLACE_KEY) || DEFAULT_WEATHER_PLACE;
    const wpInput = el("weather-place");
    if (wpInput) wpInput.value = this.weatherPlace;
    const wpSave = el("weather-place-save");
    if (wpSave) {
      wpSave.addEventListener("click", () => {
        const v = el("weather-place").value.trim();
        if (!v) return;
        this.weatherPlace = v;
        store.set(WEATHER_PLACE_KEY, v);
        this.renderSettings();
      });
    }

    for (const btn of document.querySelectorAll(".tab-btn")) {
      btn.addEventListener("click", () => this.showTab(btn.dataset.tab));
    }
    el("ask-form").addEventListener("submit", (e) => {
      e.preventDefault();
      void this.ask(el("question-input").value);
    });
    el("question-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void this.ask(el("question-input").value);
      }
    });
    el("question-input").addEventListener("input", () => this.syncSend());
    for (const btn of document.querySelectorAll(".suggestion-btn")) {
      btn.addEventListener("click", () => void this.ask(btn.dataset.q));
    }
    el("load-model-btn").addEventListener("click", () => {
      const url = el("base-url").value.trim();
      if (!url) return;
      this.baseUrl = url;
      el("current-url").textContent = url;
      void this.loadModel(url);
    });
    const searchTestBtn = el("search-test-btn");
    if (searchTestBtn) {
      searchTestBtn.addEventListener("click", () => void this.testSearch());
    }
    el("export-download").addEventListener("click", () => this.downloadTrainingData());
    el("export-copy").addEventListener("click", () => void this.copyTrainingData());
    el("mark-all-btn").addEventListener("click", () => {
      this.confirmMarkAll = true;
      this.renderSettings();
    });
    el("mark-all-yes").addEventListener("click", () => {
      Notebook.markTrained(null);
      this.confirmMarkAll = false;
      this.renderSettings();
      this.renderNotebook();
      el("mark-all-done").hidden = false;
    });
    el("mark-all-no").addEventListener("click", () => {
      this.confirmMarkAll = false;
      this.renderSettings();
    });

    this.renderNotebook();
    this.renderSettings();
    this.setModelStatus({ kind: "not_loaded" });
    void this.loadModel(this.baseUrl);
  },

  showTab(name) {
    for (const btn of document.querySelectorAll(".tab-btn")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.tab === name));
    }
    el("view-chat").hidden = name !== "chat";
    el("view-notebook").hidden = name !== "notebook";
    el("view-settings").hidden = name !== "settings";
    if (name === "notebook") this.renderNotebook();
    if (name === "settings") this.renderSettings();
  },

  /* 切り替えボタンの見た目：選択中のモデルを押し込み表示にし、
     読み込み中はすべて無効化する。 */
  renderModelSwitch() {
    for (const btn of document.querySelectorAll(".model-switch-btn")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.model === this.modelId));
      btn.disabled = this.modelBusy;
    }
  },

  /* モデルの切り替え：選択を保存し、そのモデルのベースURL（保存済みの
     上書きがあればそれ、無ければ既定値）で読み込み直す。v2 は最初に
     選択されたときにだけ読み込む（初期表示では先読みしない）。 */
  async switchModel(id) {
    if (!MODELS[id] || id === this.modelId || this.modelBusy) return;
    // ユーザーが自分でモデルを選び直したときは、Ultra 読み込みの
    // 試行の印を消す（起動中の再読み込みで自動再試行が止まって
    // いても、選び直せば最初から試せるようにする）。起動時の
    // 自動読み込み（init）はこの関数を通らないので印は残る。
    writeUltraStage(null);
    this.modelId = id;
    store.set(MODEL_ID_KEY, id);
    const def = modelDef(id);
    this.baseUrl = def.engine === "ultra"
      ? ""
      : store.get(def.urlKey) || def.defaultBaseUrl;
    el("base-url").value = this.baseUrl;
    el("current-url").textContent = this.baseUrl;
    this.renderModelSwitch();
    await this.loadModel(this.baseUrl);
  },

  setModelStatus(st) {
    this.modelStatus = st;
    const label = modelDef(this.modelId).label;
    const box = el("model-status");
    const dot = box.querySelector(".dot");
    const text = el("model-status-text");
    box.className = "model-status mono " +
      (st.kind === "ready" ? "ready" : st.kind === "error" ? "error" : st.kind === "loading" ? "loading" : "");
    dot.className = "dot";
    if (st.kind === "not_loaded") text.textContent = "モデル未配置";
    // standby：未読み込みだが、案内文（st.message）をそのまま出す。
    // 見た目は not_loaded と同じ（ドット・色・メモ欄の扱いも同じ）。
    if (st.kind === "standby") text.textContent = st.message;
    if (st.kind === "loading") {
      text.textContent = `${label} 読み込み中… ${formatBytes(st.loadedBytes)}${st.totalBytes > 0 ? ` / ${formatBytes(st.totalBytes)}` : ""}（${st.label}）${this.modelId === "ultra" ? " ※Cube Ultraの初回読み込みは約1.2GBのダウンロードがあります" : ""}`;
    }
    if (st.kind === "ready") {
      text.textContent = `${label} 読み込み完了（パラメータ数 ${formatParams(st.paramCount)}）`;
    }
    if (st.kind === "error") text.textContent = `モデル未配置（${st.message}）`;
    el("model-note").hidden = st.kind === "ready" || st.kind === "loading";
    el("nb-tab-count").textContent = Notebook.items.length > 0 ? ` (${Notebook.items.length})` : "";
    const s = el("settings-model-status");
    if (s) {
      s.textContent = "状態: " + (
        st.kind === "ready"
          ? `${label} 読み込み完了（パラメータ数 ${formatParams(st.paramCount)}）`
          : st.kind === "loading"
            ? `${label} 読み込み中…`
            : st.kind === "error"
              ? `エラー: ${st.message}`
              : "モデル未配置"
      );
    }
    // Ultra 選択中は、ベースURLの上書き欄（自作モデル専用）を無効化し、
    // 現在のベースURL欄には Ultra には無い旨を表示する。
    const isUltra = this.modelId === "ultra";
    const baseInput = el("base-url");
    if (baseInput) baseInput.disabled = isUltra;
    const loadBtn = el("load-model-btn");
    if (loadBtn) loadBtn.disabled = isUltra;
    if (isUltra) {
      const cur = el("current-url");
      if (cur) {
        cur.textContent =
          "（Cube Ultraにはありません。Hugging Face の FMA03/cube-ultra から読み込みます）";
      }
    }
  },

  /* エンジン選択ブロックの表示／非表示。show のときは、前回読み込みに
     成功したエンジンが保存されていれば、その案内を添える。Ultra の
     エンジンはどちらも自動では開始せず、このブロックのボタンが
     押されたときだけ開始する。 */
  setUltraEngineChoice(show) {
    const box = el("ultra-engine-choice");
    if (!box) return;
    if (show) {
      const note = el("ultra-engine-saved-note");
      if (note) {
        const saved = store.get(ULTRA_ENGINE_KEY);
        note.textContent =
          saved === "onnx"
            ? "前回は通常エンジンで読み込めました"
            : saved === "gguf"
              ? "前回は軽量エンジンで読み込めました"
              : saved === "imac"
                ? "前回はiMacサーバーで読み込めました"
                : "";
      }
    }
    box.hidden = !show;
  },

  /* 軽量エンジン（GGUF版）への切り替え案内（通知エリア）の表示。
     mode が null なら隠す。通常版の試行がタブ落下で中断された状態
     （"onnx-interrupted"：再試行の案内／"onnx-stopped"：2回落ちて
     停止）と、GGUF 版の試行が中断・失敗した状態（"gguf-retry"／
     "gguf-failed"）、GGUF 版の読み込み完了後に落ちた状態
     （"gguf-gen-crash"：生成中／"gguf-ready-crash"：回答前）で出す。
     "gguf-retry" と落下通知の文言は、フェーズ記録を見て「どこで
     落ちたか」を具体的に書く。自動でダウンロードは始めず、ボタンが
     押されたときだけ loadUltraGguf が走る。 */
  setUltraGgufNotice(mode) {
    const box = el("ultra-gguf-notice");
    if (!box) return;
    if (!mode) {
      box.hidden = true;
      return;
    }
    const text = el("ultra-gguf-notice-text");
    if (text) {
      if (mode === "gguf-retry") {
        // フェーズ記録から、前回 GGUF がどこで落ちたかを具体的に書く。
        const rec = readGgufPhase();
        text.textContent =
          rec && (rec.phase === "load" || (typeof rec.pct === "number" && rec.pct >= 100))
            ? "前回は軽量エンジン（GGUF版）のダウンロードは完了しましたが、モデルの読み込み中にブラウザが再読み込みされました。キャッシュから再試行できます。"
            : `前回は軽量エンジン（GGUF版）のダウンロード中（約${rec && typeof rec.pct === "number" ? rec.pct : 0}%）にブラウザが再読み込みされました。再試行すると続きからダウンロードします。`;
      } else if (mode === "gguf-gen-crash") {
        const rec = readGgufPhase();
        const answered = rec && typeof rec.answered === "number" ? rec.answered : 0;
        text.textContent =
          answered <= 0
            ? "前回は軽量エンジン（GGUF版）の読み込みは完了しましたが、最初の回答生成の途中でブラウザが再読み込みされました。この端末のメモリ上限に達した可能性があります。"
            : `前回は軽量エンジン（GGUF版）で回答を生成している途中でブラウザが再読み込みされました（その前に${answered}回は回答できていました）。この端末のメモリ上限に達した可能性があります。`;
      } else if (mode === "gguf-ready-crash") {
        text.textContent =
          "前回は軽量エンジン（GGUF版）の読み込みは完了しましたが、回答する前にブラウザが再読み込みされました。この端末のメモリ上限に達した可能性があります。";
      } else {
        text.textContent =
          mode === "onnx-interrupted"
            ? "前回は Cube Ultra の起動の途中でブラウザが再読み込みされました。通常版の再試行を続けていますが、読み込みの軽い軽量エンジン（GGUF版）に切り替えることもできます。"
            : mode === "onnx-stopped"
              ? "通常版の Cube Ultra は、この端末では起動できませんでした。読み込みの軽い軽量エンジン（GGUF版）なら開ける可能性があります。"
              : "軽量エンジン（GGUF版）の読み込みに失敗しました。もう一度試せます。";
      }
    }
    box.hidden = false;
  },

  async loadModel(url) {
    const def = modelDef(this.modelId);
    if (def.engine === "ultra") {
      // Cube Ultra：エンジンは transformers.js の pipeline（通常版）か
      // wllama（軽量エンジン・GGUF版）。ベースURL・manifest 照合・
      // IndexedDB キャッシュは使わない。ここは「待機ディスパッチャ」で、
      // Ultra が選ばれてもどちらのエンジンも自動では開始しない。
      // 前回の試行の痕跡（試行の印・GGUF フェーズ記録）から状態表示と
      // 通知を出し、エンジン選択ブロックを表示して待つ。読み込み完了の
      // 記録は各エンジン側にあり、失敗時は ready にならないので、他の
      // モデルが Ultra の名前で回答することは絶対にない。
      // (a) どちらかのエンジンが読み込み済みなら ready 表示にする。
      if (UltraModel.ready || UltraGgufModel.ready || UltraImacModel.ready) {
        this.setUltraEngineChoice(false);
        this.setUltraGgufNotice(null);
        this.setModelStatus({ kind: "ready", paramCount: ULTRA_PARAM_COUNT });
        return;
      }
      const stage = readUltraStage();
      // (b) GGUF の落下が合計2回に達していた：停止理由を表示し、
      // 選択ブロックを出す（通常版は選べるようにする）。
      if (stage === "gguf2") {
        this.setUltraGgufNotice(null);
        this.setModelStatus({
          kind: "error",
          message:
            "Cube Ultra（軽量エンジン）の起動中にブラウザが繰り返し再読み込みされたため、自動の再試行を止めました。" +
            "この端末ではメモリが足りず Ultra を起動できない可能性があります。" +
            "他のタブやアプリを閉じるか、他のモデルをお使いください。" +
            "他のモデルを選んでから Ultra を選び直すと、もう一度試せます",
        });
        this.setUltraEngineChoice(true);
        return;
      }
      // (c) GGUF の試行が前回のタブ落下で中断されていた：割り込みの
      // 通知（フェーズ記録つき）と選択ブロックを出す。
      if (stage === "gguf") {
        this.setModelStatus({
          kind: "error",
          message: "Cube Ultra（軽量エンジン）の起動の途中でブラウザが再読み込みされました",
        });
        this.setUltraGgufNotice("gguf-retry");
        this.setUltraEngineChoice(true);
        return;
      }
      // (d) 通常版の落下が合計2回に達していた：UltraModel.load が投げる
      // のと同じ停止理由を表示し、選択ブロックを出す。
      if (stage === "build2") {
        this.setModelStatus({
          kind: "error",
          message:
            "Cube Ultra の起動中にブラウザが繰り返し再読み込みされたため、自動の再試行を止めました。" +
            "この端末ではメモリが足りず Ultra を起動できない可能性があります。" +
            "他のタブやアプリを閉じるか、他のモデルをお使いください。" +
            "他のモデルを選んでから Ultra を選び直すと、もう一度試せます",
        });
        this.setUltraGgufNotice("onnx-stopped");
        this.setUltraEngineChoice(true);
        return;
      }
      // (e) 通常版の試行が前回のタブ落下で中断されていた：通知と
      // 選択ブロックを出す（自動では再試行しない）。
      if (stage === "build") {
        this.setModelStatus({
          kind: "error",
          message: "Cube Ultra の起動の途中でブラウザが再読み込みされました",
        });
        this.setUltraGgufNotice("onnx-interrupted");
        this.setUltraEngineChoice(true);
        return;
      }
      // (f) 試行の印が無い場合：GGUF フェーズ記録が「読み込み完了後に
      // 落ちた」ことを示していたら、その落下通知を出して報告済みに
      // する。"answered" の記録は静かに消す。"download" / "load" の
      // 記録だけが残っている場合（タブを閉じた等）は通知を出さずに
      // 記録を消す。
      const phaseRec = readGgufPhase();
      if (phaseRec) {
        if (!phaseRec.reported && phaseRec.phase === "generating") {
          this.setUltraGgufNotice("gguf-gen-crash");
          writeGgufPhase({ ...phaseRec, reported: true });
        } else if (!phaseRec.reported && phaseRec.phase === "ready") {
          this.setUltraGgufNotice("gguf-ready-crash");
          writeGgufPhase({ ...phaseRec, reported: true });
        } else {
          writeGgufPhase(null);
        }
      }
      // (g) 待機表示にして、エンジン選択ブロックを出す。
      this.setModelStatus({
        kind: "standby",
        message: "Cube Ultra は未読み込みです。エンジンを選んでください",
      });
      this.setUltraEngineChoice(true);
      return;
    }
    // 自作モデルが選択された：Ultra のエンジン（通常版・軽量エンジン
    // とも）があれば破棄し、GGUF の案内も隠す（アクティブなエンジンは
    // 常に1つだけ）。進行中の Ultra 読み込みの後処理も無効化する。
    this.ultraLoadSeq++;
    this.setUltraGgufNotice(null);
    this.setUltraEngineChoice(false);
    disposeUltra();
    // ベースURLの上書きは選択中モデルごとのキーに保存する
    // （Nano は従来の jisaku-ai-model-base-url、Flash / Twin / Pro / Max は専用キー）。
    store.set(def.urlKey, url);
    this.modelBusy = true;
    this.renderModelSwitch();
    this.setModelStatus({ kind: "loading", loadedBytes: 0, totalBytes: 0, label: "manifest.json" });
    try {
      await LocalModel.load(url, def.id, (loaded, total, label) => {
        this.setModelStatus({ kind: "loading", loadedBytes: loaded, totalBytes: total, label });
      });
      this.setModelStatus({ kind: "ready", paramCount: LocalModel.paramCount });
    } catch (e) {
      let message = e instanceof Error ? e.message : String(e);
      // Flash / Twin / Pro / Max の重みがまだアップロードされていない
      // 場合の案内を添える（黙って Nano で回答することは絶対にしない）。
      if ((def.id === "v2" || def.id === "twin" || def.id === "pro" || def.id === "max") && message.includes("manifest.json")) {
        message += `。${def.label} の重みはまだ GitHub にアップロードされていない可能性があります。Cube Nano に戻すと今まで通り使えます`;
      }
      this.setModelStatus({ kind: "error", message });
    } finally {
      this.modelBusy = false;
      this.renderModelSwitch();
    }
  },

  /* 通常エンジン（ONNX版）の読み込み：エンジン選択ブロックのボタンが
     押されたときだけ始める。試行の印・世代番号・通知の出し分けは、
     以前 loadModel の Ultra 分岐で自動開始していた試行本体そのまま。
     成功したら、読み込めたエンジンを保存し（次回の選択ブロックの
     案内用）、GGUF のフェーズ記録は消す（古い生成落下の誤通知防止）。 */
  async loadUltraOnnx() {
    if (this.modelId !== "ultra" || UltraModel.ready) return;
    UltraImacModel.dispose();
    this.setUltraEngineChoice(false);
    const seq = ++this.ultraLoadSeq;
    this.modelBusy = true;
    this.renderModelSwitch();
    const crashedBefore = UltraModel.previousAttemptCrashed();
    this.setModelStatus({
      kind: "loading",
      loadedBytes: 0,
      totalBytes: 0,
      label: crashedBefore
        ? "Cube Ultra（前回は起動の途中でブラウザが再読み込みされました。キャッシュから再試行しています）"
        : "Cube Ultra（初回は約1.2GBのダウンロードがあります）",
    });
    // 前回の試行がタブ落下で中断されていたら、通知エリアに軽量
    // エンジン（GGUF版）への切り替えボタンも出す。
    this.setUltraGgufNotice(crashedBefore ? "onnx-interrupted" : null);
    try {
      await loadUltra((loaded, total, label) => {
        if (seq !== this.ultraLoadSeq) return;
        this.setModelStatus({ kind: "loading", loadedBytes: loaded, totalBytes: total, label });
      });
      // 読み込み中に軽量エンジンへ引き継がれていたら、状態表示は
      // そちらに任せる（ここで ready に上書きしない）。
      if (seq !== this.ultraLoadSeq) return;
      this.setUltraGgufNotice(null);
      store.set(ULTRA_ENGINE_KEY, "onnx");
      writeGgufPhase(null);
      this.setModelStatus({ kind: "ready", paramCount: ULTRA_PARAM_COUNT });
    } catch (e) {
      if (seq !== this.ultraLoadSeq) return;
      const message = e instanceof Error ? e.message : String(e);
      this.setModelStatus({ kind: "error", message });
      // 2回続けて落ちて自動の再試行が止まった状態（印が "build2"）でも、
      // 軽量エンジンのボタンは出したままにする。再試行中の通常の
      // 失敗でもボタンは残し、中断の無かった通常の失敗では隠す。
      if (readUltraStage() === "build2") this.setUltraGgufNotice("onnx-stopped");
      else if (!crashedBefore) this.setUltraGgufNotice(null);
    } finally {
      if (seq === this.ultraLoadSeq) {
        this.modelBusy = false;
        this.renderModelSwitch();
      }
    }
  },

  /* 軽量エンジン（GGUF版）の読み込み：通知エリアのボタンか、エンジン
     選択ブロックのボタンが押されたときだけ始める。開始時に通常版（ONNX）のモデルは読み込まない
     （読み込み中だった通常版は破棄し、その後処理は世代番号で無効化
     する）。進捗表示は通常版のローダーと同じ形（読み込み済み／全体
     バイト数とファイル名）。読み込みが成功したら、そのセッション中は
     Ultra の回答に GGUF 版を使い続ける（generateUltra 側の振り分け）。 */
  async loadUltraGguf() {
    if (this.modelId !== "ultra" || UltraGgufModel.ready) return;
    const seq = ++this.ultraLoadSeq;
    UltraModel.dispose();
    UltraImacModel.dispose();
    this.setUltraEngineChoice(false);
    this.modelBusy = true;
    this.renderModelSwitch();
    this.setUltraGgufNotice(null);
    const crashedBefore = UltraGgufModel.previousAttemptCrashed();
    this.setModelStatus({
      kind: "loading",
      loadedBytes: 0,
      totalBytes: 0,
      label: crashedBefore
        ? "Cube Ultra（軽量エンジン・GGUF版）（前回は起動の途中でブラウザが再読み込みされました。キャッシュから再試行しています）"
        : "Cube Ultra（軽量エンジン・GGUF版）（初回は約986MBのダウンロードがあります）",
    });
    try {
      await loadUltraGguf((loaded, total, label) => {
        if (seq !== this.ultraLoadSeq) return;
        this.setModelStatus({ kind: "loading", loadedBytes: loaded, totalBytes: total, label });
      });
      if (seq !== this.ultraLoadSeq) return;
      // 読み込めたエンジンを保存する（次回の選択ブロックの案内用）。
      store.set(ULTRA_ENGINE_KEY, "gguf");
      this.setModelStatus({ kind: "ready", paramCount: ULTRA_PARAM_COUNT });
    } catch (e) {
      if (seq !== this.ultraLoadSeq) return;
      const message = e instanceof Error ? e.message : String(e);
      this.setModelStatus({ kind: "error", message });
      // 通常の例外で失敗したときは、もう一度押せるようボタンを出し直す。
      // タブ落下で2回止まった場合（UltraGgufModel.load の例外）は印が
      // "gguf2" のまま残るので、ボタンは出さない。
      if (readUltraStage() !== "gguf2") this.setUltraGgufNotice("gguf-failed");
    } finally {
      if (seq === this.ultraLoadSeq) {
        this.modelBusy = false;
        this.renderModelSwitch();
      }
    }
  },

  /* iMacサーバーエンジンの開始：エンジン選択ブロックのボタンが押された
     ときだけ始める。端末へのダウンロードは無く、iMac への到達確認だけ。
     開始時に他のエンジン（通常版・軽量エンジン）は破棄する（アクティブな
     エンジンは常に1つだけ）。成功したら読み込めたエンジンを保存する。
     届かなかったら素直な文面をエラー表示にし、エンジン選択ブロックを
     出し直して、他のエンジンや他のモデルを選べるようにする。 */
  async loadUltraImac() {
    if (this.modelId !== "ultra" || UltraImacModel.ready) return;
    const seq = ++this.ultraLoadSeq;
    UltraModel.dispose();
    UltraGgufModel.dispose();
    this.setUltraEngineChoice(false);
    this.setUltraGgufNotice(null);
    this.modelBusy = true;
    this.renderModelSwitch();
    this.setModelStatus({
      kind: "loading",
      loadedBytes: 0,
      totalBytes: 0,
      label: "Cube Ultra（iMacサーバーに接続しています）",
    });
    try {
      await loadUltraImac((loaded, total, label) => {
        if (seq !== this.ultraLoadSeq) return;
        this.setModelStatus({ kind: "loading", loadedBytes: loaded, totalBytes: total, label });
      });
      if (seq !== this.ultraLoadSeq) return;
      // 読み込めたエンジンを保存する（次回の選択ブロックの案内用）。
      store.set(ULTRA_ENGINE_KEY, "imac");
      this.setModelStatus({ kind: "ready", paramCount: ULTRA_PARAM_COUNT });
    } catch (e) {
      if (seq !== this.ultraLoadSeq) return;
      const message = e instanceof Error ? e.message : String(e);
      this.setModelStatus({ kind: "error", message });
      // もう一度選べるよう、エンジン選択ブロックを出し直す。
      this.setUltraEngineChoice(true);
    } finally {
      if (seq === this.ultraLoadSeq) {
        this.modelBusy = false;
        this.renderModelSwitch();
      }
    }
  },

  setPhase(p) {
    this.phase = p;
    const row = el("phase-row");
    if (p === "idle") {
      row.hidden = true;
    } else {
      row.hidden = false;
      el("phase-text").textContent =
        p === "thinking" ? "考え中…" : p === "searching" ? "検索で確認中…" : "知識メモ帳に保存中…";
    }
    this.syncSend();
    this.scrollChat();
  },

  syncSend() {
    el("send-btn").disabled = !el("question-input").value.trim() || this.phase !== "idle";
  },

  scrollChat() {
    const sc = el("chat-scroll");
    sc.scrollTo({ top: sc.scrollHeight, behavior: "smooth" });
  },

  pushUser(text) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);
    li.appendChild(bubble);
    el("msg-list").appendChild(li);
    this.scrollChat();
  },

  pushAssistant(msg) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-ai";
    if (msg.badge && msg.badge !== "none" && BADGES[msg.badge]) {
      const b = document.createElement("span");
      b.className = `badge mono ${BADGES[msg.badge].className}`;
      b.textContent = BADGES[msg.badge].label;
      li.appendChild(b);
    }
    const p = document.createElement("p");
    p.className = "answer";
    p.textContent = msg.text;
    li.appendChild(p);
    if (msg.evidence) {
      const ev = document.createElement("p");
      ev.className = "evidence";
      ev.textContent = `根拠: ${msg.evidence}`;
      li.appendChild(ev);
    }
    if (msg.modelSaid) {
      const ms = document.createElement("p");
      ms.className = "model-said";
      ms.textContent = `Cubeはこう答えた：${msg.modelSaid}`;
      li.appendChild(ms);
    }
    if (msg.sourceUrl) {
      const sp = document.createElement("p");
      sp.className = "source";
      sp.append("出典: ");
      const a = document.createElement("a");
      a.href = msg.sourceUrl;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = msg.sourceTitle || msg.sourceUrl;
      sp.appendChild(a);
      li.appendChild(sp);
    }
    if (msg.note) {
      const n = document.createElement("p");
      n.className = "note";
      n.textContent = msg.note;
      li.appendChild(n);
    }
    el("msg-list").appendChild(li);
    this.scrollChat();
  },

  saveNotebook(entry) {
    // 同語反復の文は知識として固定しない（最後の関門）。
    if (isTautologicalAnswer(entry.answer)) return;
    Notebook.add(entry);
    el("nb-tab-count").textContent = ` (${Notebook.items.length})`;
  },

  /* 質問への回答の流れ：
     0) 天気 → 気象庁の予報＋Open-Meteoの現在の状況（実データ）
     0.5) Cube自身の質問 → 固定情報
     0.75) 計算レーン → 正確な計算結果（過程つき）
     1) 知識メモ帳 → 即答＋裏で再確認
     2) 計算 → 検算 3) モデル生成 → 対象記事の事実で照合 */
  async ask(rawQuestion) {
    const question = (rawQuestion || "").trim();
    if (!question || this.phase !== "idle") return;
    this.pushUser(question);
    el("question-input").value = "";
    this.syncSend();
    // 「回答可能」は、読み込み済みのモデルIDと選択中のIDが一致する
    // ときだけ真。切り替え先の読み込みに失敗した状態では、前のモデル
    // のエンジンで（新しいモデルの名前で）回答することはない。
    const modelReady = modelReadyFor(this.modelId);
    const normalized = normalizeQuestion(question);
    // 今回の検索の失敗理由はここから集める（前の質問の理由を持ち越さない）。
    WikiNet.lastError = "";

    // 0) 天気：聞かれた時点の実データで答える。モデル・事実コア・
    // 知識メモ帳をすべて素通りする——天気は変わる情報なので、保存した
    // 答えや生成した答えは古くなり、モデルは今日の空を知らない。
    // 場所が見つからない・取得に失敗した場合は、その旨を正直に言う。
    const wq = parseWeatherQuery(question);
    if (wq) {
      this.setPhase("searching");
      let w = null;
      try {
        w = await fetchWeather(wq.place ?? "", this.weatherPlace, wq.day);
      } catch {
        w = null;
      }
      this.setPhase("idle");
      if (w && w.ok) {
        const fromJma = w.forecast_source === "jma";
        const showCurrent = wq.day === "today" && w.current_temp !== null;
        const lines = [`${w.place_label}の天気`];
        if (showCurrent) {
          lines.push(
            `現在${fromJma ? "（Open-Meteo）" : ""}: ${w.current_text ? `${w.current_text}、` : ""}気温 ${w.current_temp}°C`,
          );
        }
        for (const d of w.days) {
          const parts = [d.text];
          if (d.temp_max !== null && d.temp_min !== null) {
            parts.push(`最高 ${d.temp_max}°C / 最低 ${d.temp_min}°C`);
          } else if (d.temp_max !== null) {
            parts.push(`最高 ${d.temp_max}°C`);
          } else if (d.temp_min !== null) {
            parts.push(`最低 ${d.temp_min}°C`);
          }
          if (d.precip_prob !== null) {
            parts.push(`降水確率 ${d.precip_prob}%`);
          }
          lines.push(`${d.date_label}: ${parts.join("、")}`);
        }
        // 出典は分けて書く：予報は気象庁、現在の状況は Open-Meteo。
        // 気象庁が取れず Open-Meteo だけのときは、その旨だけを書く。
        const noteParts = [];
        if (fromJma) {
          noteParts.push(`予報：気象庁（${w.jma_area_label}）`);
          if (showCurrent) noteParts.push("現在の状況：Open-Meteo");
        }
        noteParts.push(
          "天気はその時点の予報です（変わる情報なので知識メモ帳には保存しません）",
        );
        this.pushAssistant({
          text: lines.join("\n"),
          badge: fromJma ? "weather" : "weather-om",
          sourceTitle: fromJma ? "予報：気象庁" : "Open-Meteo",
          sourceUrl: w.source_url,
          note: `${noteParts.join("。")}。`,
        });
      } else if (w && w.error === "place_not_found") {
        const asked = wq.place ?? this.weatherPlace;
        this.pushAssistant({
          text: "天気を確認できませんでした。",
          badge: "unknown",
          note: `「${asked}」という場所が天気データの中で見つかりませんでした。地名を変えて（例: 郡山、仙台、東京）もう一度聞いてください。`,
        });
      } else {
        this.pushAssistant({
          text: "天気を確認できませんでした。",
          badge: "unknown",
          note: "天気データの取得に失敗しました。通信状態を確認して、もう一度試してください。",
        });
      }
      return;
    }

    // 0.5) Cube自身についての質問（作った人・名前・名前の意味・綴り）：
    // ウェブでは確かめられない事実なので、モデル生成・ネット検索・
    // 知識メモ帳をすべて素通りし、アプリに固定した正しい情報から答える。
    // メモ帳より前に置くのは、以前の誤った答えがメモ帳に残っていても
    // それを繰り返さないため。固定情報なのでメモ帳にも保存しない。
    const selfAnswer = cubeSelfAnswer(question);
    if (selfAnswer) {
      this.pushAssistant({
        text: selfAnswer,
        badge: "self",
        note: "Cube自身についての質問には、モデルやネット検索ではなく、アプリに固定した正しい情報から答えています。この種別の答えは知識メモ帳に保存しません。",
      });
      return;
    }

    // 0.75) 計算レーン：文章題を含む計算のパターンに一致したら、
    // モデル・メモ帳を素通りして、正確な計算結果（過程つき）を
    // そのまま答える。結果は従来の計算経路と同じ形でメモ帳に残す。
    const lane = CalcLane.solve(question);
    if (lane) {
      this.pushAssistant({
        text: lane.steps,
        badge: "calc",
      });
      this.saveNotebook({
        question,
        answer: lane.steps,
        source_title: "",
        source_url: "",
        question_type: "calc",
        fact: String(lane.answer),
      });
      return;
    }

    // 1) 知識メモ帳：保存済みの答えは即答する——ネットワークを待たせない。
    // 事実レベルの再確認（対象記事から種類つき事実を取り直して比べる）は
    // 裏で走らせ、記事の事実が実際に変わっていた時だけ訂正の返答を出し、
    // 保存エントリも更新する。古い答えが「確認済み」として永遠に
    // 繰り返されることもない。
    const hit = Notebook.items.find(
      (it) => normalizeQuestion(it.question) === normalized,
    );
    if (hit) {
      this.pushAssistant({
        text: hit.answer,
        badge: "notebook",
        sourceTitle: hit.source_title || undefined,
        sourceUrl: hit.source_url || undefined,
      });
      void (async () => {
        IndexLane.beginQuestion();
        let fact = null;
        try {
          const article = await fetchArticle(question);
          fact = extractFact(question, article);
        } catch {
          fact = null;
        }
        if (!fact || fact.answer === hit.answer) return;
        this.saveNotebook({
          question,
          answer: fact.answer,
          source_title: fact.sourceTitle,
          source_url: fact.sourceUrl,
          question_type: fact.kind,
          fact: fact.value,
        });
        // 別の言い方で保存されていた古い行は消す（古い答えが残らないように）。
        if (hit.question !== question) {
          Notebook.remove(hit.id);
          this.renderNotebook();
        }
        this.pushAssistant({
          text: fact.answer,
          badge: "net",
          sourceTitle: fact.sourceTitle,
          sourceUrl: fact.sourceUrl,
          evidence: fact.evidence,
          note: "知識メモ帳の答えをネットで確認し直したところ、内容が変わっていたため、新しい答えに更新しました。",
        });
      })();
      return;
    }

    // 2) 計算は実際に計算して検算する。
    const calc = evalArithmetic(question);
    if (calc !== null) {
      let modelSaid;
      let calcModelError = "";
      if (modelReady) {
        this.setPhase("thinking");
        await new Promise((r) => setTimeout(r, 30)); // 「考え中…」を先に描く
        try {
          // 検算経路のモデル生成にも、選択中モデルを使う。Ultra の
          // 生成は非同期（transformers.js）で、上限は Ultra 固定の128。
          const modelAns = this.modelId === "ultra"
            ? await generateUltra(question)
            : LocalModel.generate(question, modelDef(this.modelId).maxNew);
          const modelNums = numbersIn(modelAns);
          if (
            modelAns &&
            modelNums.length > 0 &&
            !modelNums.some((n) => Number(n) === calc)
          ) {
            modelSaid = modelAns;
          }
        } catch (e) {
          // モデルの失敗で検算の答えは止めないが、エラー自体は注記に出す。
          calcModelError = e instanceof Error ? e.message : String(e);
        }
      }
      // 検算の答えを先に表示する。メモ帳への吸収（種類=計算、事実=計算
      // 結果）は表示の後——保存が表示を遅らせることはない。
      this.setPhase("idle");
      this.pushAssistant({
        text: `答えは ${calc} です。`,
        badge: "calc",
        modelSaid,
        note: modelSaid
          ? "Cubeの答えと食い違ったため、検算の答えを表示しています。"
          : calcModelError
            ? `Cubeの生成でエラーが起きましたが、検算の答えを表示しています。（エラー内容: ${calcModelError}）`
            : undefined,
      });
      this.saveNotebook({
        question,
        answer: `答えは ${calc} です。`,
        source_title: "",
        source_url: "",
        question_type: "calc",
        fact: String(calc),
      });
      return;
    }

    // 3) モデル生成＋ウェブ検索での照合。確認のリクエストは生成の前に
    // 開始する：生成はメインスレッドの重い処理だが、ネットワークは
    // その裏で進むため、Wikipediaの往復がモデルの思考の後ろに
    // 並ばずに済む。
    IndexLane.beginQuestion();
    const articlePromise = fetchArticle(question).catch(() => null);
    const searchPromise = searchWikipedia(question).catch(() => []);
    // モデルが何も生まなかった理由を追う——未読み込み・空の生成・
    // 例外は以前は見分けがつかず、すべて黙って「確認できませんでした」
    // や理由のないネット検索ラベルに潰れていた。
    let modelAnswer = "";
    let modelIssue = null; // null | "not_ready" | "empty" | "error"
    // 実際のエラーの文面も保持し、注記に出す（「生成でエラー」だけでは
    // 利用者が報告も対処もできないため）。
    let modelError = "";
    if (!modelReady) modelIssue = "not_ready";
    if (modelReady) {
      this.setPhase("thinking");
      await new Promise((r) => setTimeout(r, 30));
      try {
        // 通常の質問応答の生成にも、選択中モデルの生成上限
        // （Nano・Pro: 48 / Flash・Twin: 96）を明示的に渡す。Ultra の生成は
        // 非同期（transformers.js）で、上限は Ultra 固定の128。
        modelAnswer = this.modelId === "ultra"
          ? await generateUltra(question)
          : LocalModel.generate(question, modelDef(this.modelId).maxNew);
        if (!modelAnswer) modelIssue = "empty";
      } catch (e) {
        modelAnswer = "";
        modelIssue = "error";
        modelError = e instanceof Error ? e.message : String(e);
      }
    }

    this.setPhase("searching");
    let results = [];
    let fact = null;
    try {
      const [article, searchResults] = await Promise.all([
        articlePromise,
        searchPromise,
      ]);
      results = searchResults;
      fact = extractFact(question, article);
    } catch {
      results = [];
      fact = null;
    }
    // 検索への接続そのものが失敗していたら、その理由を覚えておく
    // （下の「確認できませんでした」で、理由なしにごまかさないため）。
    const searchError = WikiNet.lastError;
    // 『…』の質問では、その作品自身の証拠だけを数える。
    results = scopeToSubject(question, results);

    // 4) 事実の道（本命）：対象記事から抜いた種類つき事実で組み立てた答え。
    if (fact) {
      // メモ帳への吸収は答えが画面に出た後に行う——保存は表示を遅らせない。
      const saveFact = () => {
        this.saveNotebook({
          question,
          answer: fact.answer,
          source_title: fact.sourceTitle,
          source_url: fact.sourceUrl,
          question_type: fact.kind,
          fact: fact.value,
        });
      };
      this.setPhase("idle");
      // モデル優先：モデルが答えを出したなら、それが主表示になる。
      // ウェブの事実は裏づけの確認——表記のゆれ（空白・カンマ・単位・
      // 小数の丸め）を許して一致すれば「ネットで確認済み」のラベルを
      // 付け、食い違えば従来の透明性ルール（ネットの事実を正として
      // 表示し、モデルの元の答えも消さずに併記する）。
      if (modelAnswer && modelAgreesWithFact(modelAnswer, fact)) {
        this.pushAssistant({
          text: modelAnswer,
          badge: "model",
          sourceTitle: fact.sourceTitle,
          sourceUrl: fact.sourceUrl,
          evidence: fact.evidence,
          note: "Cubeの答えを主表示しています。対象記事の事実と一致し、ネットで裏づけが取れました。事実ベースの答えを知識メモ帳に保存しました。",
        });
        saveFact();
        return;
      }
      if (modelAnswer) {
        this.pushAssistant({
          text: fact.answer,
          badge: "net",
          sourceTitle: fact.sourceTitle,
          sourceUrl: fact.sourceUrl,
          evidence: fact.evidence,
          modelSaid: modelAnswer,
          note: "Cubeの答えとネットの事実が食い違ったため、ネットの事実を正として表示しています。事実ベースの答えを知識メモ帳に保存しました。",
        });
        saveFact();
        return;
      }
      // モデルの答えが無い時だけ、ウェブの事実が主表示になる——そして
      // 注記はモデルが答えなかった理由（未読み込み／空の生成／生成
      // エラー）を言う。
      const noModelNote =
        modelIssue === "not_ready"
          ? "Cubeが読み込まれていないため、ネット検索の答えを表示しています。"
          : modelIssue === "empty"
            ? "Cubeに聞きましたが、この聞き方では答えが生成されなかったため、ネット検索の答えを表示しています。"
            : modelIssue === "error"
              ? `Cubeの生成でエラーが起きたため、ネット検索の答えを表示しています。（エラー内容: ${modelError}）`
              : "対象記事から抜いた事実で組み立てた答えを表示しています。";
      this.pushAssistant({
        text: fact.answer,
        badge: "net",
        sourceTitle: fact.sourceTitle,
        sourceUrl: fact.sourceUrl,
        evidence: fact.evidence,
        note: `${noModelNote}事実ベースの答えを知識メモ帳に保存しました。`,
      });
      saveFact();
      return;
    }

    // 出典は、実際に裏づけが成立した記事そのものだけにする。文頭の
    // 一致や先頭の記事で代替すると、裏づけてもいない記事が出典に
    // 表示されてしまう（辞書ページが出典になった cube の件）。
    const corrobSrc = modelAnswer
      ? findCorroboratingResult(modelAnswer, results, question)
      : null;

    if (modelAnswer && corrobSrc) {
      const src = corrobSrc;
      // 先に表示する。照合できた答えも構造化して吸収する。
      this.setPhase("idle");
      this.pushAssistant({
        text: modelAnswer,
        badge: "model",
        sourceTitle: src?.title,
        sourceUrl: src?.url,
        note: "Cubeの答えを、ネット検索の結果と照合しました。知識メモ帳に保存しました。",
      });
      this.saveNotebook({
        question,
        answer: modelAnswer,
        source_title: src?.title ?? "",
        source_url: src?.url ?? "",
        question_type: classifyQuestion(question) ?? "",
        fact: modelAnswer,
      });
      return;
    }

    // 5) 種類つき事実が取れなかった：モデルの答えは未確認として出すか、
    // 正直に「確認できませんでした」と言う（関係ありそうな文は貼らない）。
    // ただし未確認で出す前に、答えの種類が質問と合っているか確かめる：
    // 知事を聞いたのに名物や作者の答えが出るような、種類の食い違う
    // 答えは出さない（「未確認」と断っても質問への答えになっていない）。
    // モデルが何も出さなかった場合は、その理由も注記に言う。
    // 検索に接続できなかった場合は、その理由を必ず添える。
    this.setPhase("idle");
    if (modelAnswer && !modelAnswerFitsQuestion(question, modelAnswer)) {
      this.pushAssistant({
        text: "確認できませんでした。",
        badge: "unknown",
        note: searchError
          ? `検索に接続できなかったため、記事の事実では確認できませんでした（理由：${searchError}）。Cubeの答えは質問の種類と合っていなかったため、表示しませんでした。設定の「検索テスト」で接続を確かめられます。`
          : "記事の事実では確認できませんでした。Cubeの答えは質問の種類と合っていなかったため、表示しませんでした。",
      });
    } else if (modelAnswer) {
      this.pushAssistant({
        text: modelAnswer,
        badge: "unverified",
        note: searchError
          ? `検索に接続できなかったため、記事の事実では確認できませんでした（理由：${searchError}）。モデルの回答をそのまま表示しています。設定の「検索テスト」で接続を確かめられます。`
          : "記事の事実では確認できませんでした。モデルの回答をそのまま表示しています。",
      });
    } else {
      this.pushAssistant({
        text: "確認できませんでした。",
        badge: "unknown",
        note: searchError
          ? `検索に接続できませんでした（理由：${searchError}）。設定の「検索テスト」で接続を確かめられます。`
          : modelIssue === "not_ready"
            ? "対象記事から、質問に対応する事実を抜き出せませんでした。Cubeは読み込まれていないため、モデルの答えもありません。関係ありそうな文を貼ってごまかすことはしません。"
            : modelIssue === "empty"
              ? "対象記事から、質問に対応する事実を抜き出せませんでした。Cubeにも聞きましたが、この聞き方では答えが生成されませんでした。聞き方を変えると答えが出ることがあります。関係ありそうな文を貼ってごまかすことはしません。"
              : modelIssue === "error"
                ? `対象記事から、質問に対応する事実を抜き出せず、Cubeの生成でもエラーが起きました。（エラー内容: ${modelError}）関係ありそうな文を貼ってごまかすことはしません。`
                : "対象記事から、質問に対応する事実を抜き出せませんでした。関係ありそうな文を貼ってごまかすことはしません。",
      });
    }
  },

  /* ---- 知識メモ帳タブ ---- */
  renderNotebook() {
    const items = Notebook.items;
    const untrained = items.filter((it) => !it.trained).length;
    el("nb-counts").textContent = items.length > 0
      ? `全${items.length}件 ／ 未学習 ${untrained}件 ／ 学習済み ${items.length - untrained}件`
      : "";
    el("nb-counts").hidden = items.length === 0;
    el("nb-empty").hidden = items.length > 0;
    el("nb-tab-count").textContent = items.length > 0 ? ` (${items.length})` : "";
    const list = el("nb-list");
    list.textContent = "";
    for (const it of items) {
      const li = document.createElement("li");
      li.className = "nb-item";
      const flex = document.createElement("div");
      flex.className = "nb-flex";
      const main = document.createElement("div");
      main.className = "nb-main";
      const q = document.createElement("p");
      q.className = "nb-q";
      q.textContent = it.question;
      main.appendChild(q);
      const tags = document.createElement("p");
      tags.className = "nb-tags";
      if (it.question_type) {
        const t = document.createElement("span");
        t.className = "tag mono";
        t.textContent = kindLabel(it.question_type);
        tags.appendChild(t);
      }
      const tr = document.createElement("span");
      tr.className = `tag mono ${it.trained ? "tag-trained" : "tag-untrained"}`;
      tr.textContent = it.trained ? "学習済み" : "未学習";
      tags.appendChild(tr);
      main.appendChild(tags);
      if (it.fact) {
        const f = document.createElement("p");
        f.className = "nb-fact";
        const k = document.createElement("span");
        k.className = "k";
        k.textContent = "事実：";
        f.append(k, it.fact);
        main.appendChild(f);
      }
      const a = document.createElement("p");
      a.className = "nb-answer";
      a.textContent = it.answer;
      main.appendChild(a);
      if (it.source_url) {
        const sp = document.createElement("p");
        sp.className = "nb-source";
        sp.append("出典: ");
        const link = document.createElement("a");
        link.href = it.source_url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = it.source_title || it.source_url;
        sp.appendChild(link);
        main.appendChild(sp);
      }
      const d = document.createElement("p");
      d.className = "nb-date mono";
      d.textContent = new Date(it.created_at).toLocaleString("ja-JP");
      main.appendChild(d);
      flex.appendChild(main);
      const actions = document.createElement("div");
      actions.className = "nb-actions";
      if (!it.trained) {
        const mb = document.createElement("button");
        mb.type = "button";
        mb.className = "small-btn";
        mb.textContent = "学習済みにする";
        mb.setAttribute("aria-label", `「${it.question}」を学習済みにする`);
        mb.addEventListener("click", () => {
          Notebook.markTrained([it.id]);
          this.renderNotebook();
          this.renderSettings();
        });
        actions.appendChild(mb);
      }
      const db = document.createElement("button");
      db.type = "button";
      db.className = "small-btn";
      db.textContent = "削除";
      db.setAttribute("aria-label", `「${it.question}」を削除`);
      db.addEventListener("click", () => {
        Notebook.remove(it.id);
        this.renderNotebook();
        this.renderSettings();
      });
      actions.appendChild(db);
      flex.appendChild(actions);
      li.appendChild(flex);
      list.appendChild(li);
    }
  },

  /* 設定の「検索テスト」：実際にWikipedia検索を1回行い、届くかどうかと
     使えた経路（直接／予備）、失敗した場合はその理由を表示する。 */
  async testSearch() {
    const out = el("search-test-result");
    const btn = el("search-test-btn");
    if (!out || !btn) return;
    btn.disabled = true;
    out.textContent = "テスト中…（Wikipediaに接続しています）";
    // 経路の判定をやり直すため、状態を初期化してから試す。
    wikiFetchBroken = false;
    WikiNet.via = "";
    WikiNet.lastError = "";
    const titles = await wikiSearchTitles("富士山", 3);
    // タイトルが引けるだけでは不十分：照合に使う記事本文まで取得
    // できるかも確かめる（本文が取れなければ「確認済み」は出ない）。
    let bodyOk = false;
    if (titles.length > 0) {
      const page = await wikiFetchExtract(titles[0]);
      bodyOk = !!(page && page.text.length > 0);
    }
    btn.disabled = false;
    if (titles.length > 0 && bodyOk) {
      out.textContent =
        `検索できました（経路: ${WikiNet.via === "jsonp" ? "予備の経路" : "直接接続"}）。` +
        `見つかった記事: ${titles.join("、")}。記事本文も取得できました。`;
    } else if (titles.length > 0) {
      out.textContent =
        `記事のタイトルは見つかりましたが、本文を取得できませんでした。` +
        `見つかった記事: ${titles.join("、")}`;
    } else if (WikiNet.lastError) {
      out.textContent = `検索できませんでした。理由: ${WikiNet.lastError}。`;
    } else {
      out.textContent = "接続はできましたが、検索結果が0件でした。";
    }
  },

  /* ---- 設定タブ ---- */
  trainingJson() {
    return JSON.stringify(
      {
        format: "jisaku-ai-training-data",
        version: 1,
        exported_at: new Date().toISOString(),
        entries: Notebook.items.map((it) => ({
          id: it.id,
          question: it.question,
          question_type: it.question_type,
          fact: it.fact,
          answer: it.answer,
          source_title: it.source_title,
          source_url: it.source_url,
          trained: it.trained,
          created_at: it.created_at,
        })),
      },
      null,
      2,
    );
  },

  downloadTrainingData() {
    try {
      const blob = new Blob([this.trainingJson()], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "jisaku-ai-training-data.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      el("export-note").textContent = "学習用データ（JSON）をダウンロードしました。";
    } catch {
      el("export-note").textContent = "ダウンロードできませんでした。「コピー」を試してください。";
    }
  },

  async copyTrainingData() {
    try {
      await navigator.clipboard.writeText(this.trainingJson());
      el("export-note").textContent = "学習用データ（JSON）をコピーしました。";
    } catch {
      el("export-note").textContent = "コピーできませんでした。「ダウンロード」を試してください。";
    }
  },

  renderSettings() {
    const wpCurrent = el("weather-place-current");
    if (wpCurrent) wpCurrent.textContent = this.weatherPlace;
    const items = Notebook.items;
    const untrained = items.filter((it) => !it.trained).length;
    el("training-counts").textContent =
      `未学習 ${untrained}件 ／ 学習済み ${items.length - untrained}件 ／ 全${items.length}件`;
    el("export-download").disabled = items.length === 0;
    el("export-copy").disabled = items.length === 0;
    el("mark-all-btn").disabled = untrained === 0;
    el("mark-all-confirm").hidden = !this.confirmMarkAll;
    el("mark-all-btn").hidden = this.confirmMarkAll;
    el("mark-all-text").textContent =
      `未学習 ${untrained}件をすべて学習済みにします。いいですか？`;
    this.setModelStatus(this.modelStatus);
  },
};

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => App.init());
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BPE, GPT, QTensor, Int4Tensor, fp16ToFloat32, tensorElementCount,
    buildEngine, LocalModel,
    MODELS, modelDef, validateManifest,
    UltraModel, loadUltra, generateUltra, disposeUltra, modelReadyFor,
    UltraGgufModel, loadUltraGguf,
    UltraImacModel, loadUltraImac, ULTRA_IMAC_BASE_URL, ULTRA_IMAC_UNREACHABLE,
    readUltraStage, writeUltraStage,
    ULTRA_REPO, ULTRA_PARAM_COUNT, ULTRA_MAX_NEW_TOKENS,
    ULTRA_GGUF_URL, WLLAMA_ESM_URL, WLLAMA_WASM_FROM_CDN_URL,
    evalArithmetic, CalcLane, extractFact, classifyQuestion,
    resolveArticleSource, searchWikipedia, topicFromQuestion, kindSearchPhrase,
    canonicalTensorName, expectedTensors, modelAgreesWithFact,
    isCorroborated, findCorroboratingResult,
    answerContentTerms, numbersMatchLoose,
    cubeSelfAnswer, isTautologicalAnswer,
    isUsableEvidenceResult, resultAnchoredToQuestion, resultSupportsAnswer,
    modelAnswerFitsQuestion, answerClaimKind,
    parseWeatherQuery, fetchWeather, fetchJmaForecast, resolveJmaArea,
    jmaWeatherText, normalizeQuestion, IndexLane,
  };
}
