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
/*                                                                    */
/* 2026-11-15 のサーバー固定化（本人決定）：Twin / Pro / Max は端末に  */
/* 読み込まず、自宅の iMac サーバー（ServerModel）で動かす。engine が */
/* "server" のモデルは LocalModel を一切通らず、ベースURL・manifest   */
/* 照合・端末キャッシュも使わない。Ultra も同じ iMac サーバー固定に   */
/* なり、エンジン選択（ONNX / GGUF）は廃止した（エンジン本体のコード  */
/* は互換のため残置してあるが、UI からは到達できない）。              */
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
    engine: "server",
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
    engine: "server",
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
    engine: "server",
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
  // Cube 会話：雑談専門モデル v1（2026-11-20 本人指示で通常チャットに
  // 配線）。土台は Qwen3-4B-Instruct-2507 ＋ 会話専用 LoRA（318件で
  // 学習）。iMac の cube-server（8096 番・オンデマンド起動）で動かし、
  // ページは /chat に会話メッセージ列を渡す。システムプロンプトは
  // 学習時固定のためサーバー側が差し込む（ページは送らない）。
  // 事実の質問には答えず橋渡しする設計で、回答フロー側でも事実の
  // 質問は従来の証拠＋検証ゲート経路に倒す（ask の雑談分岐参照）。
  // 時間制メーターは他のサーバー側モデルと同じ扱い。
  chat: {
    id: "chat",
    label: "Cube 会話",
    engine: "chat",
    defaultBaseUrl: "",
    urlKey: "",
    maxNew: 256,
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

/* Cube 記憶（cube_memory.js）の Ultra 向け注入分。質問ごとに App.ask の
   入口で CubeMem.systemSuffix(question) の結果に差し替える。記憶が
   無い・モジュールが読み込まれていない環境では常に空文字で、Ultra の
   システムプロンプトは従来と完全同一になる。記憶はシステム側にだけ
   入れ、読解形式の「文：…」（記事本文＝証拠）には決して混ぜない。 */
let cubeMemorySystemSuffix = "";

/* 記憶の保存先に iMac の cube-server（/memory）を使うかどうか。
   false の間はこの端末の localStorage だけが保存先（公開版の基本形）。
   サーバー保存を有効化するときは、memory_endpoint.js を cube-server
   へ配置し、疎通を確認してから true にする（今回はデプロイしない）。 */
const MEMORY_SERVER_ENABLED = false;

/* Cube アカウント（Supabase）の設定。Cube 用の Supabase プロジェクトを
   作ったあと、Project URL と Publishable key をここへ入れる
   （SETUP_GUIDE.md の手順参照）。空の間はアカウント機能の全体
   （Google ログイン・チャット履歴・端末間同期）が無効で、ページの
   挙動は従来と完全同一になる。
   ※ ここへ入れてよいのは Publishable key（公開前提のキー）だけ。
   Secret key は絶対に入れないこと。 */
const CUBE_ACCOUNT_CONFIG =
  (typeof window !== "undefined" && window.__CUBE_ACCOUNT_CONFIG__) || {
    url: "https://glvlbgylnxxkyjvcaeym.supabase.co",
    publishableKey: "sb_publishable_8VCMVdBvYivEFVNA841N8g_EE7c8H3T",
  };

/* Cubeポイント（グループ共通ポイント）。残高と台帳は Cube の
   Supabase（points_balance / points_ledger）にあり、書き込みは
   RPC（earn_points / spend_points）経由のみ。金額・1日上限・冪等は
   サーバー側の関数が決め、ページは呼び出しと表示だけを行う。
   ポイントは付加機能で、失敗してもチャットは一切妨げない。
   未ログインでは貯まらない（表示も出さない）。 */
const POINTS_TOPUP_MENU = [
  { pct: 10, price: 25 },
  { pct: 50, price: 100 },
  { pct: 100, price: 180 },
];
/* トップアップで得た「メーター追加 %」の保持先（端末内）。
   時間制メーター（cube_meter.js）がこの値を読み、基準単位
   （600秒）に対する割合として秒に換算して残量へ足す。
   ポイント側はこの累積 % を所有し、メーター側は読み取り専用。 */
const POINTS_BONUS_KEY = "cube-points-meter-bonus-v1";

/* JST（Asia/Tokyo）の日付キー "YYYY-MM-DD"。RPC 側の1日判定
   （Asia/Tokyo の日付）とそろえるため、端末のタイムゾーンでは
   なく JST で数える。 */
function pointsJstDateKey(date) {
  const d = date || new Date();
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Tokyo",
      year: "numeric", month: "2-digit", day: "2-digit",
    }).formatToParts(d);
    const pick = (t) => (parts.find((p) => p.type === t) || {}).value || "";
    const y = pick("year"), m = pick("month"), dd = pick("day");
    if (y && m && dd) return `${y}-${m}-${dd}`;
  } catch { /* Intl が使えない環境では端末ローカルの日付で代替 */ }
  return localDateKey(d);
}

/* ログインボーナスの冪等キー（RPC 側の想定形式と一致）。 */
function pointsDailyLoginKey(date) {
  return `daily_login:${pointsJstDateKey(date)}`;
}

let pointsTopupSeq = 0;
/* トップアップの冪等キー。購入操作ごとに一意に発行する。 */
function pointsNewTopupKey() {
  pointsTopupSeq += 1;
  return `meter_topup:${Date.now().toString(36)}_${pointsTopupSeq.toString(36)}` +
    Math.floor(Math.random() * 1296).toString(36);
}

/* Cube 時間制メーター（cube_meter.js）。サーバー側モデル
   （Twin / Pro / Max / Ultra）の利用時間を数える。設定の上書きは
   window.__CUBE_METER_CONFIG__（{plan, plans, congestion,
   bonusUnitSec}）で渡せる。課金・学生判定は未決のため、既定は
   全員 free プランで、プラン切替はこの設定でのみ行う。モジュール
   が読み込まれていない環境ではメーターなし（従来挙動）になる。 */
const METER_CONFIG_OVERRIDE =
  (typeof window !== "undefined" && window.__CUBE_METER_CONFIG__) || null;

let cubeMeterInstance = null;
/* メーターのシングルトン。残量の保存先は端末の localStorage
   （cube_meter.js 側のキー）。ポイント追加分（%）はポイント側の
   保存キーからそのまま読む（書き換えない）。 */
function getCubeMeter() {
  if (cubeMeterInstance) return cubeMeterInstance;
  if (typeof CubeMeter === "undefined") return null;
  try {
    cubeMeterInstance = CubeMeter.createMeter({
      storage: {
        getItem: (k) => {
          try { return window.localStorage.getItem(k); } catch { return null; }
        },
        setItem: (k, v) => {
          try { window.localStorage.setItem(k, v); } catch { /* 続行 */ }
        },
      },
      config: CubeMeter.resolveConfig(METER_CONFIG_OVERRIDE),
      onChange: () => {
        try { App.renderMeter(); } catch { /* 表示前でも計量は続ける */ }
      },
    });
  } catch {
    cubeMeterInstance = null;
  }
  return cubeMeterInstance;
}

/* このモデルIDがメーター対象（サーバー側）かどうか。Ultra は
   iMac サーバー固定、Twin / Pro / Max は engine "server"、
   Cube 会話は engine "chat"（iMac のオンデマンド起動）。
   端末内モデル（Nano / Flash）は対象外。 */
function meterAppliesTo(id) {
  return id === "ultra" || modelDef(id).engine === "server" || modelDef(id).engine === "chat";
}

/* メーター切れの例外。生成経路が残量0を検知したときに投げる。 */
function meterEmptyError() {
  const e = new Error(
    "サーバーモデルの時間メーターが0になりました。Cube Nano / Cube Flash はこの端末で動くため、メーターなしで使えます。");
  e.name = "MeterEmptyError";
  e.code = "METER_EMPTY";
  return e;
}

/* 記憶ビューの表示ラベル（モジュールの KIND_LABEL と同じ文言）。 */
const MEM_KIND_LABELS = {
  user: "ユーザー", preference: "好み", decision: "決めたこと",
  person: "人物", daily: "記録",
};
const MEM_SOURCE_LABELS = {
  explicit: "あなたの明示", verified: "検証済み", conversation: "会話から",
};

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
  /* アプリがモデルへ渡すプロンプト文字列から、Ultra の答えを生成する。
     historyMessages（再開した会話の直近ターン）は system の後ろ・
     今回の user の前に入れる。無い場合は従来と完全同一。 */
  async generate(prompt, historyMessages) {
    if (!this.generator) throw new Error("Cube Ultra が読み込まれていません");
    const messages = [
      { role: "system", content: ULTRA_SYSTEM_PROMPT + cubeMemorySystemSuffix },
      ...(Array.isArray(historyMessages) ? historyMessages : []),
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
  async generate(prompt, historyMessages) {
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
    const historyChatml = (Array.isArray(historyMessages) ? historyMessages : [])
      .map(
        (m) =>
          `<|im_start|>${m.role === "assistant" ? "assistant" : "user"}\n${m.content}<|im_end|>\n`,
      )
      .join("");
    const chatml =
      `<|im_start|>system\n${ULTRA_SYSTEM_PROMPT + cubeMemorySystemSuffix}<|im_end|>\n` +
      historyChatml +
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
  async generate(prompt, historyMessages) {
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
              { role: "system", content: ULTRA_SYSTEM_PROMPT + cubeMemorySystemSuffix },
              ...(Array.isArray(historyMessages) ? historyMessages : []),
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

/* ================================================================== */
/* 自作モデル iMacサーバーエンジン（Cube Twin / Pro / Max）            */
/*                                                                    */
/* Twin（約121MB）・Pro（約156MB）・Max（約583MB）の int4 重みは、     */
/* 端末（iPad 等）のブラウザのメモリ上限に当たって読み込めないため、   */
/* 2026-11-15 の本人決定で、自宅の iMac 上の Node サーバーで動かす    */
/* 方式に固定した（Ultra の iMac エンジンと同じ考え方）。サーバーは   */
/* 公開版と同一の int4 パッケージと同一のエンジンで生成するため、     */
/* 答えはブラウザ版とバイト単位で一致する（サーバー側で検証済み）。   */
/* ページ側は質問文をそのまま渡し、学習時のQA形式（質問：…\n答え：）  */
/* の組み立て・greedy 生成・末尾処理はサーバー側が LocalModel.generate */
/* と同一に行う。エンドポイントは Tailscale Serve の HTTPS            */
/* （tailnet 内のみ、専用ポート 8443）。iMac が起動していない等で     */
/* 届かない場合は例外にして素直な文面で知らせ、他のモデルで黙って     */
/* 代答することは絶対にしない。                                       */
/* ================================================================== */

const CUBE_SERVER_BASE_URL = "https://imac.tail9ed76a.ts.net:8443";
const CUBE_SERVER_HEALTH_TIMEOUT_MS = 15000;
const CUBE_SERVER_GENERATE_TIMEOUT_MS = 300000;
const CUBE_SERVER_POLL_INTERVAL_MS = 3000;
const CUBE_SERVER_POLL_DEADLINE_MS = 120000;
const CUBE_SERVER_UNREACHABLE =
  "iMacに届きませんでした。iMacが起動しているか確認してください";

const ServerModel = {
  /* モデルIDごとの接続状態 { ready, paramCount }。サーバー側はモデルを
     常駐させているので、ここで保持するのは「このページのセッションで
     到達と ready を確認済みか」だけ。他のモデルが選択されたら破棄し、
     選び直したときに health を引き直す。 */
  states: {},
  readyFor(id) { return this.states[id]?.ready === true; },
  paramCountFor(id) { return this.states[id]?.paramCount || 0; },
  /* GET /health を1回引く。到達不能・形式不正は「届かない」例外にする。 */
  async fetchHealth() {
    let res;
    try {
      res = await fetchWithTimeout(
        `${CUBE_SERVER_BASE_URL}/health`,
        { method: "GET" },
        CUBE_SERVER_HEALTH_TIMEOUT_MS,
      );
    } catch {
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    if (!res || !res.ok) throw new Error(CUBE_SERVER_UNREACHABLE);
    const data = await res.json().catch(() => null);
    if (!data || data.status !== "ok" || !data.models) {
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    return data.models;
  },
  /**
   * 接続：health を引き、対象モデルが ready になるまでポーリングする
   * （サーバー起動直後はモデル読み込み中のことがある）。「iMac側で
   * 読み込み中」が分かったら onPolling を1回呼ぶ（表示の切り替え用）。
   * 届かない・サーバー側エラー・時間切れは例外にする。
   */
  async connect(id, onPolling) {
    const deadline = Date.now() + CUBE_SERVER_POLL_DEADLINE_MS;
    let notified = false;
    for (;;) {
      const models = await this.fetchHealth();
      const st = models[id];
      if (!st) throw new Error(CUBE_SERVER_UNREACHABLE);
      if (st.state === "ready") {
        this.states[id] = { ready: true, paramCount: st.paramCount || 0 };
        return;
      }
      if (st.state === "error") {
        throw new Error(
          "iMac側でモデルの読み込みに失敗しました。iMacのサーバーを確認してください",
        );
      }
      // state === "loading"：少し待って引き直す。
      if (!notified) {
        notified = true;
        if (onPolling) onPolling();
      }
      if (Date.now() >= deadline) {
        throw new Error(
          "iMac側のモデルの読み込みが時間内に終わりませんでした。少し待ってから接続し直してください",
        );
      }
      await new Promise((r) => setTimeout(r, CUBE_SERVER_POLL_INTERVAL_MS));
    }
  },
  /**
   * 生成：アプリがモデルへ渡す質問文をそのまま prompt として渡す。
   * 失敗したら接続状態を忘れ（次に選び直したとき health から確認
   * し直す）、素直な文面の例外にする（Ultra の iMac エンジンと同じ）。
   */
  async generate(id, question) {
    if (!this.readyFor(id)) {
      throw new Error(`${modelDef(id).label}（iMacサーバー）が接続されていません`);
    }
    let res;
    try {
      res = await fetchWithTimeout(
        `${CUBE_SERVER_BASE_URL}/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: id,
            prompt: question,
            max_new_tokens: modelDef(id).maxNew,
          }),
        },
        CUBE_SERVER_GENERATE_TIMEOUT_MS,
      );
    } catch {
      this.states[id] = { ready: false, paramCount: 0 };
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    const data = res ? await res.json().catch(() => null) : null;
    if (!res || !res.ok || !data || typeof data.text !== "string") {
      this.states[id] = { ready: false, paramCount: 0 };
      const detail = data && typeof data.error === "string" ? data.error : "";
      throw new Error(
        detail ? `${CUBE_SERVER_UNREACHABLE}（サーバーの応答: ${detail}）` : CUBE_SERVER_UNREACHABLE,
      );
    }
    return data.text.trim();
  },
  /* 接続状態を忘れる（他のモデルが選択されたとき）。引数なしで全部、
     ID を渡せばそのモデルだけ。サーバー側の常駐には触れない。 */
  dispose(id) {
    if (id) delete this.states[id];
    else this.states = {};
  },
};

/* ================================================================== */
/* 雑談専門モデルエンジン（Cube 会話）                                  */
/*                                                                    */
/* 雑談専門 v1（Qwen3-4B-Instruct-2507 ＋ 会話 LoRA）は iMac の       */
/* llama-server（8096 番）で動かし、cube-server の /chat が窓口。      */
/* サーバー側モデルはオンデマンド起動（初回はモデルの読み込みで       */
/* 数十秒かかる）で、10分無通信で自動停止する。システムプロンプトは   */
/* 学習時固定のためサーバー側が差し込む——ページは user / assistant    */
/* の会話列だけを送り、system は絶対に送らない。事実の質問への        */
/* 橋渡しはモデル自身も行うが、回答フロー側でも事実の質問はこの       */
/* エンジンに渡さない（ask の雑談分岐参照）。                         */
/* ================================================================== */

const CHAT_CONNECT_TIMEOUT_MS = 240000;
const CHAT_GENERATE_TIMEOUT_MS = 300000;
const CHAT_PARAM_COUNT = 4022691840;

const ChatModel = {
  connected: false,
  get ready() { return this.connected; },
  readyFor(id) { return id === "chat" && this.connected; },
  paramCountFor() { return CHAT_PARAM_COUNT; },
  /* 接続：サーバー側の /chat/start がモデルの起動完了まで待ってから
     返す（オンデマンド起動）。起動中であることが分かったら onPolling
     を1回呼ぶ（表示の切り替え用）。 */
  async connect(onPolling) {
    if (this.connected) return;
    if (onPolling) onPolling();
    let res;
    try {
      res = await fetchWithTimeout(
        `${CUBE_SERVER_BASE_URL}/chat/start`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
        CHAT_CONNECT_TIMEOUT_MS,
      );
    } catch {
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    const data = res ? await res.json().catch(() => null) : null;
    if (!res || !res.ok || !data || data.state !== "ready") {
      const detail = data && typeof data.error === "string" ? data.error : "";
      throw new Error(
        detail ? `${CUBE_SERVER_UNREACHABLE}（サーバーの応答: ${detail}）` : CUBE_SERVER_UNREACHABLE,
      );
    }
    this.connected = true;
  },
  /* 生成：会話列（履歴＋今回の発話）をそのまま渡す。失敗したら接続
     状態を忘れ、素直な文面の例外にする（他のサーバーエンジンと同
     じ流儀）。 */
  async generate(question, historyMessages) {
    if (!this.connected) throw new Error("Cube 会話（iMacサーバー）が接続されていません");
    const messages = [
      ...(Array.isArray(historyMessages) ? historyMessages : []),
      { role: "user", content: question },
    ];
    let res;
    try {
      res = await fetchWithTimeout(
        `${CUBE_SERVER_BASE_URL}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages, max_tokens: modelDef("chat").maxNew }),
        },
        CHAT_GENERATE_TIMEOUT_MS,
      );
    } catch {
      this.connected = false;
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    const data = res ? await res.json().catch(() => null) : null;
    if (!res || !res.ok || !data || typeof data.text !== "string") {
      this.connected = false;
      const detail = data && typeof data.error === "string" ? data.error : "";
      throw new Error(
        detail ? `${CUBE_SERVER_UNREACHABLE}（サーバーの応答: ${detail}）` : CUBE_SERVER_UNREACHABLE,
      );
    }
    return data.text.trim();
  },
  /* 接続状態を忘れる（他のモデルが選択されたとき）。サーバー側の
     常駐には触れない（無通信10分で自動停止する）。 */
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

/* Ultra の生成：iMacサーバーエンジンが接続済みならそれを使う。
   2026-11-15 のサーバー固定化で、UI から開始できるのは iMac
   エンジンだけになった。通常版（ONNX）・軽量エンジン（GGUF版）の
   振り分けは互換のため残してあるが、どちらも UI からは到達
   できない。回答フロー・バッジ・出典表示はどのエンジンでも同一。 */
async function generateUltra(prompt, historyMessages) {
  if (UltraImacModel.ready) {
    /* iMac サーバーでの生成は時間制メーターの対象。残量0なら
       生成を始めず、専用の例外で穏やかに止める。消費は実時間で
       数える（失敗してもサーバーが動いた時間は消費する）。
       端末内エンジン（ONNX / GGUF 版）へ落ちる経路はメーター
       対象外（端末で動くため）。 */
    const meter = getCubeMeter();
    if (meter) {
      if (!meter.canUse()) throw meterEmptyError();
      return meter.consumeAround(
        () => UltraImacModel.generate(prompt, historyMessages));
    }
    return UltraImacModel.generate(prompt, historyMessages);
  }
  return UltraGgufModel.ready
    ? UltraGgufModel.generate(prompt, historyMessages)
    : UltraModel.generate(prompt, historyMessages);
}

/* Ultra のエンジンをすべて破棄する（他のモデルが選択されたとき）。 */
function disposeUltra() {
  UltraModel.dispose();
  UltraGgufModel.dispose();
  UltraImacModel.dispose();
}

/* ================================================================== */
/* Ultra 画像質問（添付画像）                                           */
/*                                                                    */
/* 画像を見られるのは Ultra（Qwen2.5-VL）だけ。ページで選んだ画像は   */
/* 長辺1024pxまでに縮小して JPEG の data URI にし、送信まではこの     */
/* 変数の中だけに保持する。送信時は cube-server の /ultra/image       */
/* レーンを通し、サーバー側で検証してから llama-server（8082）へ      */
/* image_url として転送する。画像はサーバーに保存されない。生成条件・ */
/* システムプロンプトはテキスト経路（UltraImacModel）と同一にそろえ、 */
/* 時間制メーターの扱いもテキストの Ultra 生成と同じにする。          */
/* ================================================================== */

let pendingChatImage = null;

const ULTRA_IMAGE_GENERATE_TIMEOUT_MS = 300000;

const UltraImageModel = {
  async generate(question, imageDataUrl, historyMessages) {
    let res;
    try {
      res = await fetchWithTimeout(
        `${CUBE_SERVER_BASE_URL}/ultra/image`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            system: ULTRA_SYSTEM_PROMPT + cubeMemorySystemSuffix,
            history: Array.isArray(historyMessages) ? historyMessages : [],
            question,
            image: imageDataUrl,
            max_tokens: ULTRA_MAX_NEW_TOKENS,
          }),
        },
        ULTRA_IMAGE_GENERATE_TIMEOUT_MS,
      );
    } catch {
      throw new Error(CUBE_SERVER_UNREACHABLE);
    }
    const data = res ? await res.json().catch(() => null) : null;
    if (!res || !res.ok || !data || typeof data.text !== "string") {
      const detail = data && typeof data.error === "string" ? data.error : "";
      throw new Error(
        detail ? `${CUBE_SERVER_UNREACHABLE}（サーバーの応答: ${detail}）` : CUBE_SERVER_UNREACHABLE,
      );
    }
    return data.text.trim();
  },
};

/* Ultra の画像生成：テキスト生成と同じく時間制メーターの対象
   （残量0なら生成を始めず、専用の例外で止める）。 */
async function generateUltraImage(question, imageDataUrl, historyMessages) {
  const meter = getCubeMeter();
  if (meter) {
    if (!meter.canUse()) throw meterEmptyError();
    return meter.consumeAround(
      () => UltraImageModel.generate(question, imageDataUrl, historyMessages));
  }
  return UltraImageModel.generate(question, imageDataUrl, historyMessages);
}

/* 添付用に選ばれた画像ファイルを、長辺1024pxまでの JPEG data URI に
   変換する。透過は白で埋める。読み込めない形式（壊れたファイル等）
   は例外にして、呼び出し側が正直な文面で知らせる。 */
function fileToResizedJpegDataUrl(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      try {
        const MAX_EDGE = 1024;
        const scale = Math.min(1, MAX_EDGE / Math.max(img.naturalWidth, img.naturalHeight));
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
        URL.revokeObjectURL(url);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.88);
        if (!dataUrl || !dataUrl.startsWith("data:image/jpeg")) {
          throw new Error("convert failed");
        }
        resolve(dataUrl);
      } catch (e) {
        URL.revokeObjectURL(url);
        reject(e);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("image load failed"));
    };
    img.src = url;
  });
}

/* 選択中モデルの生成ディスパッチャ：Ultra は generateUltra、自作の
   サーバーモデル（Twin / Pro / Max）は iMac サーバーの ServerModel、
   それ以外（Nano / Flash）は端末内の LocalModel。LocalModel.generate
   は同期だが、呼び出し側は常に await するので形をそろえてある。 */
async function generateForModel(id, question, opts) {
  if (id === "ultra") return generateUltra(question, opts && opts.historyMessages);
  if (modelDef(id).engine === "chat") {
    /* 雑談専門（Cube 会話）は時間制メーターの対象（サーバー側モデルと
       同じ扱い）。会話列は opts.historyMessages で受け取る。 */
    const meter = getCubeMeter();
    if (meter) {
      if (!meter.canUse()) throw meterEmptyError();
      return meter.consumeAround(
        () => ChatModel.generate(question, opts && opts.historyMessages));
    }
    return ChatModel.generate(question, opts && opts.historyMessages);
  }
  if (modelDef(id).engine === "server") {
    /* サーバーモデル（Twin / Pro / Max）は時間制メーターの対象。
       残量0なら生成を始めず、専用の例外で穏やかに止める。 */
    const meter = getCubeMeter();
    if (meter) {
      if (!meter.canUse()) throw meterEmptyError();
      return meter.consumeAround(() => ServerModel.generate(id, question));
    }
    return ServerModel.generate(id, question);
  }
  return LocalModel.generate(question, modelDef(id).maxNew);
}

/* ================================================================== */
/* Ultra 統合：索引レーンの事実を Ultra が自分の言葉で書く              */
/*                                                                    */
/* 「ネット検索の答え」の提示に限った統合（2026-11-15 本人決定）。      */
/* 一般ウェブ検索は静的版では CORS で不可能なため、検索は従来どおり    */
/* 索引レーン一本に統一し、Ultra が選択されているときにだけ、見つ      */
/* かった記事を Ultra の訓練済み読解形式で読ませて、提示文を Ultra     */
/* の言葉で書かせる。他モデル・確認済み経路・根拠・出典・メモ帳        */
/* 保存はすべて従来どおり。棄権・失敗・空の生成では、従来どおり        */
/* 記事から組み立てた事実文をそのまま出す（フェイルソフト）。          */
/* （2026-11-19 追記：サーバー側の一般Web検索レーン（/gsearch）が
   実動化したため、索引で答えられない場合の受け皿として、一般Web
   検索フォールバックを後段（「一般Web検索レーン」節）に組み込んだ。
   検索結果を読むのは引き続き Ultra のみ。） */
/* ================================================================== */

/* 読ませる本文の上限（区切りの改行込み）。Muse版ウェブレーンと       */
/* 同じ 1,800 字。本文が上限を超えるページは丸ごと含めない。           */
const ULTRA_PASSAGE_CAP = 1800;

/* ページ群 [{title, url, text}] から読解用の本文を組み立てる。        */
/* 各ページは「改行＋本文」で寄与し、合計が上限を超えるページは        */
/* 本文からも出典からも外す（後続の短いページは入ることがある）。      */
function assembleUltraPassage(pages) {
  let passage = "";
  const included = [];
  for (const page of pages || []) {
    const text = typeof page?.text === "string" ? page.text : "";
    if (!text) continue;
    const piece = `\n${text}`;
    if (passage.length + piece.length > ULTRA_PASSAGE_CAP) continue;
    passage += piece;
    included.push({ title: page.title, url: page.url });
  }
  return { passage, sources: included };
}

/* Ultra の訓練済み棄権文。これらで始まる生成は「文には書かれて        */
/* いない」という回答として扱い、事実文の提示に落とす。                */
const ULTRA_ABSTENTION_PREFIXES = [
  "文には書かれていません",
  "文に書かれていません",
  "文には記載されていません",
  "文に記載されていません",
  "文には載っていません",
  "文に載っていません",
  "本文には書かれていません",
  "本文に書かれていません",
];

function isUltraAbstention(generation) {
  const t = String(generation || "").trim();
  return ULTRA_ABSTENTION_PREFIXES.some((p) => t.startsWith(p));
}

/* Ultra が書いた提示文の文末だけを、です・ます調にそろえる表示側の
 * 整形。Ultra は記事の文体（である調）を写しやすく、せっかく自分の
 * 言葉で書いても文末が「である。」になってしまうため、文末（「。」の
 * 直前と、文の末尾）だけを変換する。文中の「である」や「…」『…』の
 * 引用の中は触らない。対象の文末が無ければ無変換（そのまま返す）。
 * 適用先は Ultra が書いた提示文だけ：事実文のフォールバック・根拠・
 * メモ帳・他モデルの表示には一切かからない。 */
function normalizeUltraWrittenStyle(text) {
  const parts = String(text || "").split(/(「[^」]*」|『[^』]*』)/g);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part; // 引用の中は触らない
      let out = part.replace(/であった。/g, "でした。");
      out = out.replace(/である。/g, "です。");
      out = out.replace(/だ。/g, "です。");
      if (i === parts.length - 1) {
        // 文の末尾（「。」で終わっていない文末）だけを対象にする。
        out = out.replace(/である$/, "です");
        out = out.replace(/だ$/, "です");
      }
      return out;
    })
    .join("");
}

/* 対象記事の本文を Ultra に読ませ、「ネット検索の答え」の提示文を     */
/* 書かせる。読ませるのはこのフローが既に持っている記事本文（冒頭      */
/* 最大5,000字）の先頭で、上限に収まる長さに切ってから組み立て        */
/* 規則を通す。書けなければ（本文なし・生成失敗・空・棄権）null を     */
/* 返し、呼び出し側は従来の事実文をそのまま提示する。                  */
async function ultraWriteFactAnswer(question, article, fact) {
  try {
    const lead = article?.lead || fact?.evidence || "";
    if (!lead) return null;
    const text = lead.slice(0, ULTRA_PASSAGE_CAP - 1);
    const { passage } = assembleUltraPassage([
      {
        title: article?.title ?? fact.sourceTitle,
        url: article?.url ?? fact.sourceUrl,
        text,
      },
    ]);
    if (!passage) return null;
    const out = await generateUltra(`文：${passage}\n質問：${question}`);
    const written = (out || "").trim();
    if (!written || isUltraAbstention(written)) return null;
    // 提示文としてだけ使う戻り値なので、ここで文末をです・ます調に
    // そろえる（棄権判定は変換前の文で済ませてある）。
    return normalizeUltraWrittenStyle(written);
  } catch {
    return null;
  }
}

/* ================================================================== */
/* 一般Web検索レーン：/gsearch の本文を Ultra が読む（2026-11-19）     */
/*                                                                    */
/* 索引レーン（Wikipedia）で答えられない質問の受け皿。サーバー側の   */
/* 一般Web検索（cube-server の /gsearch、SearXNG 等）が集めた本文     */
/* つきの結果を、索引レーンと同じ読解形式で Ultra に読ませる。       */
/* 検索結果を読むのは Ultra のみ（自作モデルは読まない）。書けた     */
/* 答えも、索引レーンと同じ文単位の検証ゲートを通し、裏づけが        */
/* 取れたものだけを「ネット検索の答え」として出す。検索の失敗・      */
/* 空・Ultra の棄権・ゲート不通過では何も出さず、呼び出し側の        */
/* 従来の棄権表示に落とす（フェイルソフト）。出典は、実際に裏づけ    */
/* が成立したページだけを併記する。知識メモ帳には保存しない          */
/* （確認済み保存は従来経路のまま）。                                 */
/* ================================================================== */

/* /gsearch のページ側タイムアウト。サーバー側のハードキャップ        */
/* （20秒）にネットワークの往復を足した上限。                          */
const GSEARCH_TIMEOUT_MS = 30000;

/* 1ページから Ultra に読ませる冒頭の長さ。組み立て上限（1,800字）の  */
/* 中で上位2ページが入る大きさにする。検証ゲートには全文を使う。      */
const GSEARCH_LEAD_CHARS = 800;

/* サーバー側の一般Web検索に質問を投げ、本文を持つ結果だけを索引      */
/* レーンの結果と同じ形（{title, url, snippet, text}）で返す。         */
/* 失敗・混雑（busy）・0件は null。                                    */
async function fetchWebSearchPages(question) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/gsearch`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: question }),
      },
      GSEARCH_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || !Array.isArray(data.results)) return null;
  const pages = [];
  for (const r of data.results) {
    if (!r || typeof r.text !== "string" || !r.text) continue;
    pages.push({
      title: typeof r.title === "string" ? r.title : "",
      url: typeof r.url === "string" ? r.url : "",
      snippet: typeof r.snippet === "string" ? r.snippet : "",
      text: r.text,
    });
  }
  return pages.length ? pages : null;
}

/* 一般Web検索の本文を Ultra に読ませて答えを書かせ、検証ゲートを      */
/* 通った場合だけ表示用の材料（{text, gate, pages, sourceTitle,        */
/* sourceUrl, evidence}）を返す。通らなければ null。                   */
async function webSearchAnswer(question) {
  const pages = await fetchWebSearchPages(question);
  if (!pages) return null;
  const leads = pages.map((p) => ({
    title: p.title,
    url: p.url,
    text: p.text.slice(0, GSEARCH_LEAD_CHARS),
  }));
  const { passage, sources } = assembleUltraPassage(leads);
  if (!passage || !sources.length) return null;
  let written = "";
  try {
    written = (
      (await generateUltra(`文：${passage}
質問：${question}`)) || ""
    ).trim();
  } catch {
    return null;
  }
  if (!written || isUltraAbstention(written)) return null;
  const styled = normalizeUltraWrittenStyle(written);
  const gate = verifyModelAnswer(styled, question, {
    fact: null,
    article: null,
    results: pages,
  });
  if (gate.status !== "all") return null;
  const source =
    gate.source || pages.find((p) => p.url === sources[0].url) || pages[0];
  const evidence =
    source.snippet || (source.text ? source.text.slice(0, 200) : "");
  return {
    text: styled,
    gate,
    pages,
    sourceTitle: source.title || source.url,
    sourceUrl: source.url,
    evidence,
  };
}

/* ================================================================== */
/* 朝のブリーフィング                                                  */
/*                                                                    */
/* 朝（05:00–11:00）にその日初めて Cube を開いたとき、iMac サーバー   */
/* の /briefing（天気・学校の行事・近くのイベントを集めた材料）を     */
/* 取り、Ultra（iMac の llama-server）に読解形式で読ませて挨拶文を   */
/* 書かせ、会話の先頭に出す。材料集めはサーバー側（briefing.js）が   */
/* 担当し、ページ側は表示だけ。どの段階で失敗しても何も出さず、      */
/* 通常の利用を妨げない（フェイルソフト）。Ultra が書けなければ      */
/* 材料のデータだけの文面で出す。選択中のモデルが Ultra でなくても   */
/* ブリーフィングは Ultra が担当する（バッジで明示する）。           */
/* ================================================================== */

const BRIEFING_FETCH_TIMEOUT_MS = 20000;
const BRIEFING_START_HOUR = 5;
const BRIEFING_END_HOUR = 11;

/* ローカル日付を YYYY-MM-DD で返す（ブリーフィングの当日判定用）。 */
function localDateKey(d) {
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

/* 履歴リストの日付表示（ISO → "11/17"）。解釈できなければ空。 */
function formatHistoryDate(iso) {
  const d = new Date(iso || "");
  if (Number.isNaN(d.getTime())) return "";
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/* "2026-11-19" -> "11月19日"（材料の文章用）。 */
function briefingDateLabel(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso || ""));
  if (!m) return String(iso || "");
  return `${Number(m[2])}月${Number(m[3])}日`;
}

/* /briefing の JSON から、Ultra に読ませる材料を組み立てる。
   材料は天気・学校・イベントの節（セクション）単位に分ける：
   Ultra は1つの材料につき1つの話題を書くのが得意で、全部を
   1回で書かせると最初の話題だけで終わってしまうため、節ごとに
   書かせて後でつなぐ。各節は {key, passage（読ませる文）,
   prose（書けなかった節の代替文）}、全体の plain は全節が
   書けなかったときのデータのみ表示文。伝えることが何も
   無ければ null。 */
function buildBriefingMaterial(data) {
  if (!data || typeof data !== "object") return null;
  const sections = [];
  const plain = [];
  const w = data.weather;
  if (w && typeof w === "object") {
    const parts = [];
    if (w.text) parts.push(String(w.text));
    const hasMax = w.temp_max !== null && w.temp_max !== undefined;
    const hasMin = w.temp_min !== null && w.temp_min !== undefined;
    if (hasMax && hasMin) parts.push(`最高気温${w.temp_max}°C・最低気温${w.temp_min}°C`);
    else if (hasMax) parts.push(`最高気温${w.temp_max}°C`);
    else if (hasMin) parts.push(`最低気温${w.temp_min}°C`);
    if (w.precip_prob !== null && w.precip_prob !== undefined) {
      parts.push(`降水確率${w.precip_prob}%`);
    }
    if (w.current_temp !== null && w.current_temp !== undefined) {
      parts.push(`現在の気温${w.current_temp}°C`);
    }
    if (parts.length) {
      const prose = `今日の郡山市の天気は、${parts.join("、")}です。`;
      sections.push({ key: "weather", passage: prose, prose });
      plain.push(`天気：${parts.join("、")}`);
    }
  }
  const school = Array.isArray(data.school) ? data.school : [];
  const schoolProse = [];
  for (const s of school.slice(0, 5)) {
    if (!s || !s.event) continue;
    if (s.date === data.date) {
      schoolProse.push(`今日の学校の行事は「${s.event}」です。`);
      plain.push(`学校：今日は ${s.event}`);
    } else {
      schoolProse.push(`学校の次の行事は${briefingDateLabel(s.date)}の「${s.event}」です。`);
      plain.push(`学校：${briefingDateLabel(s.date)} ${s.event}`);
    }
  }
  if (schoolProse.length) {
    sections.push({
      key: "school",
      passage: schoolProse.join("\n"),
      prose: schoolProse.join(""),
    });
  }
  const events = Array.isArray(data.events) ? data.events : [];
  // イベントは3件まで：4件あると Ultra の生成上限（128トークン）に
  // 当たって途中で切れ、節ごと材料文に落ちてしまうため。
  const evs = events.filter((e) => e && e.title).slice(0, 3);
  if (evs.length) {
    const names = evs.map(
      (e) => `「${e.title}」${e.venue ? `（会場：${e.venue}）` : ""}`,
    );
    const prose = `今日の近くのイベントには、${names.join("、")}があります。`;
    sections.push({ key: "events", passage: prose, prose });
    for (const e of evs) {
      plain.push(`イベント：${e.title}${e.venue ? `（${e.venue}）` : ""}`);
    }
  }
  if (!sections.length) return null;
  return {
    passage: sections.map((s) => s.passage).join("\n"),
    sections,
    plain: plain.join("\n"),
  };
}

/* ブリーフィングの節ごとに Ultra へ渡す質問文。1節＝1話題に
   絞る（全部まとめて書かせると最初の話題だけで終わるため）。 */
const BRIEFING_SECTION_QUESTIONS = {
  weather:
    "この内容をもとに、今日の天気を簡潔に伝えてください。です・ます調で書いてください。",
  school:
    "この内容をもとに、学校の行事を簡潔に伝えてください。今日の行事と、次の行事の日付も入れてください。です・ます調で書いてください。",
  events:
    "この内容をもとに、今日の近くのイベントを簡潔に紹介してください。です・ます調で書いてください。",
};

/* Ultra の生成がトークン上限で途中に切れていたら、最後の文の
   終わりまでで整える。文の終わりが1つも無ければ失敗扱い（空文字）。 */
function trimBriefingTail(text) {
  const s = String(text || "").trim();
  if (!s) return "";
  if (/[。！？」』]$/.test(s)) return s;
  const idx = s.lastIndexOf("。");
  return idx >= 20 ? s.slice(0, idx + 1) : "";
}

/**
 * 「回答可能」の判定：準備完了したエンジンが選択中モデルと一致する
 * ときだけ真。端末内モデル（Nano / Flash）は LocalModel、サーバー
 * モデル（Twin / Pro / Max）は ServerModel の接続確認、Ultra は
 * いずれかのエンジン（サーバー固定化後は iMac エンジンのみ到達
 * 可能）の読み込み済みで判定する。切り替え先の準備に失敗した状態で
 * 前のモデルが新しいモデルの名前で回答することは絶対にない。
 */
function modelReadyFor(id) {
  if (id === "ultra") {
    return UltraModel.readyFor(id) || UltraGgufModel.readyFor(id) || UltraImacModel.readyFor(id);
  }
  if (modelDef(id).engine === "chat") return ChatModel.readyFor(id);
  if (modelDef(id).engine === "server") return ServerModel.readyFor(id);
  return LocalModel.readyFor(id);
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
 *   a recursive-descent evaluator (no eval): * and / bind tighter than
 *   + and -, left-associative, exactly like Python, with decimals and
 *   parentheses supported. Division by zero -> null (Python raises ->
 *   None). A numeric token with a leading zero such as "007" -> null
 *   (Python eval raises SyntaxError on such literals).
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

  // Recursive-descent evaluator for expressions with decimals and
  // parentheses, with Python eval semantics: a numeric literal with a
  // leading zero ("007") -> null, division by zero -> null. Non-integer
  // results are rounded to 10 decimal places to hide float noise.
  function evalExpr(expr) {
    const tokens = expr.match(/\d+(?:\.\d+)?|\.\d+|[+\-×÷*/()]/g);
    if (!tokens || tokens.join("") !== expr) return null;
    let pos = 0;
    function parseExpr() {
      let v = parseTerm();
      if (v === null) return null;
      while (tokens[pos] === "+" || tokens[pos] === "-") {
        const op = tokens[pos++];
        const rhs = parseTerm();
        if (rhs === null) return null;
        v = op === "+" ? v + rhs : v - rhs;
      }
      return v;
    }
    function parseTerm() {
      let v = parseFactor();
      if (v === null) return null;
      while (tokens[pos] === "×" || tokens[pos] === "÷" ||
             tokens[pos] === "*" || tokens[pos] === "/") {
        const op = tokens[pos++];
        const rhs = parseFactor();
        if (rhs === null) return null;
        if (op === "×" || op === "*") v = v * rhs;
        else {
          if (rhs === 0) return null; // Python: ZeroDivisionError
          v = v / rhs;
        }
      }
      return v;
    }
    function parseFactor() {
      const t = tokens[pos++];
      if (t === undefined) return null;
      if (t === "(") {
        const v = parseExpr();
        if (v === null || tokens[pos++] !== ")") return null;
        return v;
      }
      if (t === "-") {
        const v = parseFactor();
        return v === null ? null : -v;
      }
      if (t === "+") return parseFactor();
      if (/^0\d/.test(t)) return null; // Python: SyntaxError on "007"
      const n = Number(t);
      return Number.isFinite(n) ? n : null;
    }
    const val = parseExpr();
    if (val === null || pos !== tokens.length || !Number.isFinite(val)) {
      return null;
    }
    if (Number.isInteger(val) || Math.abs(val) >= 1e15) return val;
    return Math.round(val * 1e10) / 1e10;
  }

  function fmtVal(val) {
    // Python: float with integral value -> int before str()
    return Number.isInteger(val) ? String(Math.trunc(val)) : String(val);
  }

  function solveExpression(text) {
    const t = norm(text);
    // Maximal runs of expression characters. Decimals and parentheses
    // belong to the run, so "12.5×4" and "(3+5)×7" are evaluated as
    // written instead of being truncated to an integer sub-expression
    // (the old integer-only pattern answered 20 and 8 for those).
    const runs = t.match(/[0-9.+\-×÷*/()]+/g) || [];
    for (const raw of runs) {
      const expr = raw.replace(/^[+×÷*/.]+/, "").replace(/[+×÷*/.]+$/, "");
      if (!expr) continue;
      const operands = expr.match(/\d+(?:\.\d+)?|\.\d+/g) || [];
      const operators = expr.match(/[+\-×÷*/]/g) || [];
      if (operands.length < 2 || operators.length < 1) continue;
      const val = evalExpr(expr);
      if (val === null || !Number.isFinite(val)) continue;
      const shown = expr.replace(/\*/g, "×").replace(/\//g, "÷");
      const v = Number.isInteger(val) ? Math.trunc(val) : val;
      return { answer: v, steps: `${shown}${FEQ}${fmtVal(val)}`, kind: "expr", expr: shown };
    }
    return null;
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

/**
 * 事実の出典記事が、質問の対象（題名か『…』の対象）と関係するか。
 * 食い違い時に「ネットの事実」を正としてモデルの答えを上書きする
 * 前の安全弁：タイトルが対象を指すか、記事の冒頭か根拠文が対象を
 * 名指していれば関係ありとする。リダイレクト先の記事（モーリー
 * ファンタジー → イオンファンタジー）はタイトルが対象と一致しない
 * が、冒頭で元の名前を名指すため、本文側の名指しでも拾う。
 */
function factSourceRelatedToEntity(fact, article, entity) {
  if (!fact || !entity) return false;
  if (titleMatchesSubject(fact.sourceTitle, entity)) return true;
  const normEntity = normalizeJa(entity);
  if (normEntity.length < 2) {
    // 1文字の対象は通常は証拠にならないが、「音」→「音速」のように
    // 対象＋属性の語幹でできた複合タイトルは、その属性を聞く質問の
    // 記事そのものなので認める（語幹の集合は属性の略称だけに限る）。
    const nTitle1 = normalizeJa(fact.sourceTitle);
    if (
      nTitle1.startsWith(normEntity) &&
      nTitle1.length === normEntity.length + 1 &&
      /[速高長重深広低熱圧]/.test(nTitle1.slice(normEntity.length))
    ) {
      return true;
    }
    return false;
  }
  if (normalizeJa(fact.sourceTitle).includes(normEntity)) return true;
  // 題名が文節ごと落ちた質問（「徳川家康が江戸幕府を開いた」→題名は
  // 「江戸幕府」）では、対象文字列の中に記事タイトルがそのまま
  // 含まれていることが、記事と質問の関係の証拠になる。ただし
  // タイトルの直前が助詞や文の区切りである場合に限る：「東京
  // スカイタワー」の中の「スカイタワー」（実在する別物の塔）の
  // ように、複合名の一部として現れる場合は別の対象なので認めない。
  {
    const nTitle = normalizeJa(fact.sourceTitle);
    if (normEntity.length >= 4 && nTitle.length >= 2) {
      const idx = normEntity.indexOf(nTitle);
      if (
        idx === 0 ||
        (idx > 0 && /[がはをにの、・「]/.test(normEntity[idx - 1]))
      ) {
        return true;
      }
    }
  }
  const lead = article?.lead ?? "";
  if (lead && normalizeJa(lead).includes(normEntity)) return true;
  if (fact.evidence && normalizeJa(fact.evidence).includes(normEntity)) {
    return true;
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

/* ------------------------------------------------------------------ */
/* 会話レーン（あいさつ・お礼・別れ・定番の雑談への即答）               */
/*                                                                    */
/* あいさつやお礼は事実の主張を含まない会話であり、裏づけの対象では    */
/* ない。以前は通常の質問と同じ経路（モデル生成＋証拠集め＋文単位の    */
/* 検証ゲート）に入り、自然な返事の文が「裏づけの無い主張」として      */
/* 落とされ、「こんにちは」にすら「確認できませんでした」と返ることが  */
/* あった。そこで、発話全体が定番の会話パターンに一致する場合だけ、    */
/* 生成も検索も検証もせず、その場で丁寧な返事を返す専用レーンにする。  */
/*                                                                    */
/* 境界は保守的に引く：判定は正規化した発話全体への完全一致のみで、    */
/* 事実の質問らしい語（何・誰・いつ・どこ・なぜ等）を含むものや、      */
/* あいさつに事実の質問が続く複合文は一切拾わず、従来の経路に倒す。    */
/* 事実についての原則（裏づけの無い主張は出さない）は緩めない：        */
/* 緩めるのは「事実でない会話」だけである。返事は知識メモ帳にも        */
/* 保存しない（Cube自身レーンと同じ扱い）。                            */
/* ------------------------------------------------------------------ */

/* 会話判定専用の正規化：前後の空白を除き、英字は小文字にし、空白を
   詰め、末尾の句読点・感嘆符・波線と、末尾の「ね」「よ」を取り除く。 */
function conversationKey(question) {
  let t = String(question ?? "").trim().toLowerCase();
  t = t.replace(/[\s　]+/g, "");
  t = t.replace(/[！!？?。、・…〜~]+$/u, "");
  t = t.replace(/(ね|よ)$/u, "");
  t = t.replace(/[！!？?。、・…〜~]+$/u, "");
  return t;
}

/* 事実の質問らしい語。完全一致の判定に対する二重の安全網として、
   これらを含む発話は会話レーンでは拾わない。 */
const CONVERSATION_FACT_GUARDS =
  /何|誰|いつ|どこ|なぜ|どうして|どれ|どの|いくつ|いくら|教えて|とは/;

/* 会話パターン（正規化した発話全体への完全一致）。種類 → 正規表現。 */
const CONVERSATION_PATTERNS = [
  ["greeting_morning", /^(おはようございます|おはよう|おはよー|おはよ|グッドモーニング|goodmorning)$/],
  ["greeting_day", /^(こんにちは|こんにちわ|グッドアフタヌーン|goodafternoon)$/],
  ["greeting_evening", /^(こんばんは|こんばんわ|グッドイーブニング|goodevening)$/],
  ["greeting_neutral", /^(やあ|やっほー|やっほ|ハロー|ハーイ|hello|hi|hey|ども|どうも|おっす)$/],
  ["first_meet", /^(はじめまして|初めまして)$/],
  ["long_time", /^(お久しぶりです|お久しぶり|久しぶりです|久しぶり|ひさしぶり)$/],
  ["otsukare", /^(おつかれさまです|おつかれさま|お疲れ様です|お疲れ様|おつかれ|ご苦労様です|ご苦労様|ご苦労さま)$/],
  ["thanks", /^((ありがとうございます|ありがとうございました|ありがとう|ありがと|どうもありがとうございます|どうもありがとう|サンキュー|thankyou|thanks|感謝します)([。、](助かりました|たすかりました|助かった|たすかった))?|助かりました|たすかりました|助かった|たすかった)$/],
  ["farewell", /^(さようなら|さよなら|サヨナラ|さよーなら|バイバイ|ばいばい|じゃあね|じゃね|またね|また明日|またあした|また今度|bye|seeyou|グッバイ)$/],
  ["goodnight", /^(おやすみなさい|おやすみ|グッドナイト|goodnight)$/],
  ["wellbeing", /^(元気|元気ですか|元気です|お元気ですか|元気にしてる|元気にしてますか|元気にしてます|調子はどう|調子はどうですか|調子どう|最近どう|最近どうですか|お変わりないですか|変わりないですか)$/],
  ["apology", /^(ごめんなさい|ごめん|すみません|すいません|申し訳ありません|申し訳ないです|申し訳ない)$/],
  ["ack", /^(わかりました|分かりました|わかった|分かった|了解です|了解しました|了解|なるほどです|なるほど|そうなんだ|そうなんですね|そうですね|オッケー|おっけー|オーケー|ok)$/],
  ["ai_yes", /^(あなたはaiですか|aiですか|あなたはロボットですか|ロボットですか|あなたはaiなのですか|aiなのですか|あなたはaiなの|aiなの)$/],
  ["human", /^(あなたは人間ですか|人間ですか|あなたは人間なのですか|人間なのですか|あなたは人間なの|人間なの)$/],
];

/* 種類ごとの返事（です・ます調）。同じ種類では言い回しを回し、
   丸写しの固定文に見えないようにする。 */
const CONVERSATION_REPLIES = {
  greeting_morning: [
    "おはようございます！今日もよろしくお願いします。",
    "おはようございます。何かお手伝いできることはありますか。",
    "おはようございます！いい一日になりますように。",
  ],
  greeting_day: [
    "こんにちは！何かお手伝いできることはありますか。",
    "こんにちは。今日もよろしくお願いします。",
    "こんにちは！気になることがあれば、気軽に聞いてください。",
  ],
  greeting_evening: [
    "こんばんは！今日もおつかれさまです。",
    "こんばんは。何かお手伝いできることはありますか。",
    "こんばんは！気になることがあれば、気軽に聞いてください。",
  ],
  first_meet: [
    "はじめまして！私はCubeです。よろしくお願いします。",
    "はじめまして。私はCubeといいます。よろしくお願いします。",
  ],
  long_time: [
    "お久しぶりです！また話せてうれしいです。",
    "お久しぶりです。お元気でしたか。",
  ],
  otsukare: [
    "おつかれさまです！ゆっくり休んでください。",
    "おつかれさまです。今日もがんばりましたね。",
  ],
  thanks: [
    "どういたしまして。お役に立ててうれしいです。",
    "どういたしまして！またいつでも聞いてください。",
    "いえいえ、どういたしまして。",
  ],
  farewell: [
    "さようなら。またいつでも話しかけてください。",
    "さようなら！お元気でいてください。",
    "またお話しできるのを楽しみにしています。さようなら。",
  ],
  goodnight: [
    "おやすみなさい。ゆっくり休んでください。",
    "おやすみなさい。また明日もよろしくお願いします。",
  ],
  wellbeing: [
    "元気ですよ。聞いてくれてありがとうございます。あなたも元気ですか。",
    "おかげさまで元気です。あなたは元気ですか。",
    "元気です！今日もよろしくお願いします。",
  ],
  apology: [
    "いえいえ、気にしないでください。",
    "大丈夫ですよ。気にしないでください。",
  ],
  ack: [
    "よかったです。他にも気になることがあれば、気軽に聞いてください。",
    "はい！また何かあれば声をかけてください。",
  ],
  ai_yes: [
    "はい、私はCubeというAIです。佑眞が作った自作LLMです。",
    "はい、CubeというAI（自作LLM）です。人間ではありません。",
  ],
  human: [
    "いいえ、私は人間ではなく、CubeというAI（自作LLM）です。",
  ],
};

let conversationReplySeq = 0;
function conversationReply(kind) {
  const list = CONVERSATION_REPLIES[kind];
  if (!list || list.length === 0) return null;
  const text = list[conversationReplySeq % list.length];
  conversationReplySeq += 1;
  return text;
}

function conversationKindOfKey(key) {
  for (const [kind, re] of CONVERSATION_PATTERNS) {
    if (re.test(key)) return kind;
  }
  return null;
}

/* 時刻だけから決まるあいさつの種類（中立な呼びかけへの返し用）。 */
function conversationTimeKind(hour) {
  if (hour >= 5 && hour < 11) return "greeting_morning";
  if (hour >= 11 && hour < 17) return "greeting_day";
  return "greeting_evening";
}

/**
 * 発話があいさつ・お礼・別れ・定番の雑談なら、その場の返事を返す。
 * そうでなければ null（通常の流れに任せる）。now は時刻による
 * 出し分けのテスト用に差し替え可能（通常は現在時刻）。
 */
function conversationAnswer(question, now) {
  let key = conversationKey(question);
  if (!key || key.length > 24) return null;
  // 『…』「…」で対象を名指しする質問は会話ではない。
  if (/[『「]/.test(key)) return null;
  // 事実の質問らしい語を含むものは拾わない（安全側に倒す）。
  if (CONVERSATION_FACT_GUARDS.test(key)) return null;
  // 呼びかけ（「Cube、…」「…、キューブ」）は判定から外す。
  key = key.replace(/^(cube|キューブ)(さん)?[、,]/, "");
  key = key.replace(/[、,](cube|キューブ)(さん)?$/, "");
  const date = now instanceof Date ? now : new Date();
  const resolve = (kind) =>
    conversationReply(
      kind === "greeting_neutral"
        ? conversationTimeKind(date.getHours())
        : kind,
    );
  // 先行するあいさつ（「こんにちは、ありがとう」等）：あいさつ部分を
  // 外して残りを判定する。残りが会話パターンでなければ拾わない
  // （「こんにちは、富士山の高さは？」は事実の経路に倒す）。
  const lead = key.match(
    /^(おはようございます|おはよう|こんにちは|こんばんは|やあ|どうも|ハロー|hello)[、,！!]/,
  );
  if (lead) {
    const rest = key.slice(lead[0].length);
    if (rest) {
      const kind = conversationKindOfKey(rest);
      return kind ? resolve(kind) : null;
    }
    // 区切りだけが末尾に残った形（「こんにちは、」等）は単体として判定。
    key = key.slice(0, key.length - 1);
  }
  const kind = conversationKindOfKey(key);
  return kind ? resolve(kind) : null;
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

/* ================================================================== */
/* 場所・話題・買い物レーン（cube-server の /places・/topics・/shop）  */
/*                                                                    */
/* 事実質問の証拠レーンとは別に、cube-server の実データ API で答える  */
/* 3つの担当レーン。どのレーンもモデル生成・検証ゲートを通さず、     */
/* API が返した値をそのまま構造化して出す（数字は API フィールドの   */
/* 転記だけ。モデルに言い換えさせない）。発火は保守的に絞る：        */
/*   話題：定型の聞き方（「今の話題は？」等）に一致したときだけ。    */
/*   場所：ジャンル語＋場所の手がかり（近さ・地名）があるときだけ。  */
/*   買い物：購入相談のシグナルが明確なときだけ（事実の価格質問や    */
/*   商品の知識質問は従来レーンのまま。迷ったら従来レーン）。        */
/* サーバーに届かない・失敗した場合は何も出さず、呼び出し側の従来    */
/* 経路に倒れる（フェイルソフト）。結果は変わる情報なので、どの      */
/* レーンも知識メモ帳には保存しない。                               */
/* ================================================================== */

const PLACES_TIMEOUT_MS = 20000;
const TOPICS_TIMEOUT_MS = 15000;
const SHOP_TIMEOUT_MS = 20000;
const PC_POST_TIMEOUT_MS = 30000;

/* ---- PC操作レーン（cube-server の /pc/） ----
   チャットが入口になり、実行は橋渡し（pc_bridge）の先の仮想PCで
   PC専門モデル v1 が行う。成功の表示は橋渡し側の外部検証が pass
   のときだけ（computeResult が決める）。ページ側は進行状況と
   最終結果をそのまま出す。失敗（届かない・体が落ちている等）は
   成功のふりをせず、その旨を出す。 */

/* 事実・方法の質問らしい語。PCレーンでは拾わないための安全網
   （「フォルダの作り方は？」のような聞き方を操作依頼にしない）。 */
const PC_FACT_GUARDS =
  /とは|の意味|って何|作り方|使い方|やり方|方法を|なぜ|どうやって|歴史|本社/;

/**
 * PC操作の依頼かどうか。cube_pc.js の classifyPreview（対応
 * カテゴリの判定）を使い、チャットでは誤発火を避けるため
 * カテゴリごとに絞る：
 *   - ブラウザ検索は通常のWeb検索と競合するため、ブラウザ／PC
 *     への明示がある場合だけ。
 *   - ページを開くは、開く操作の明示がある場合だけ。
 *   - メモ帳入力・保存／フォルダ作成／端末実行は、チャット側に
 *     対応する別レーンが無いため、そのまま発火する。
 * CubePc が読み込まれていない環境では発火しない。
 */
function detectPcQuery(question) {
  const q = String(question || "");
  if (!q) return false;
  if (PC_FACT_GUARDS.test(q)) return false;
  if (typeof CubePc === "undefined" || !CubePc.classifyPreview) return false;
  const v = CubePc.classifyPreview(q);
  if (!v.supported) return false;
  if (v.category === "browser_search") {
    return /(ブラウザ|Chrome|クローム|PC|パソコン|仮想PC)/i.test(q);
  }
  if (v.category === "open_page") {
    return /(開いて|開けて|表示して|アクセスして)/.test(q) ||
      /(ブラウザ|Chrome|クローム|PC|パソコン|仮想PC)/i.test(q);
  }
  return true;
}

/* ログインの実行依頼かどうか。ログインはチャットからは実行せず、
   PC操作パネルの専用フォーム（金庫方式）へ誘導する。チャットに
   パスワードを書かせないための入口で、事実・方法の質問
   （「ログインの方法は？」等）は PC_FACT_GUARDS で弾いて
   通常の回答経路に残す。 */
function detectLoginRequest(question) {
  const q = String(question || "");
  if (!q) return false;
  if (PC_FACT_GUARDS.test(q)) return false;
  return /(ログイン|サインイン)(して|したい|する|し直し)/.test(q) ||
    /(に|へ)(ログイン|サインイン)/.test(q);
}

/* PC操作の結果メッセージへの注記（ステップ数と検証の状態）。 */
function pcResultNote(result) {
  const parts = [];
  if (result && typeof result.steps === "number" && result.steps > 0) {
    parts.push(`仮想PCで${result.steps}ステップ実行しました。`);
  }
  if (result && result.status === "verified_success") {
    parts.push("成功の表示は、こちらで実際に確認できた場合だけ出ます。");
  } else if (result && result.status === "unverified") {
    parts.push("エージェントの報告だけでは成功にできないため、未確認として表示しています。");
  }
  parts.push("画面つきの経過は「おためし」タブのPC操作パネルでも見られます。");
  return parts.join("");
}

/**
 * PC操作を橋渡しへ依頼し、SSE で経過を受けながら完了を待つ。
 * onProgress({type:"started"|"step"|"screenshot"|"control"|"verification"|"confirm"|"confirm_resolved", ...})
 * を随時呼ぶ。screenshot はライブ画面（step:"live"）を含む画面更新、
 * control は操作モード（AI／ユーザー乗っ取り）の確定通知、
 * confirm は削除・送信の実行前確認の保留通知（confirm_resolved が確定）。
 * 戻り値：
 *   {kind:"finished", result}    橋渡しが決めた最終結果
 *   {kind:"unsupported", detail} 橋渡しが受け付けなかった（422）
 *   {kind:"busy", detail}        別のタスクが実行中（409）
 *   {kind:"unreachable"}         サーバーに届かなかった
 */
async function executePcTask(question, onProgress) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/pc/tasks`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: question }),
      },
      PC_POST_TIMEOUT_MS,
    );
  } catch {
    return { kind: "unreachable" };
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 422) {
    return { kind: "unsupported", detail: data.detail || "その操作はまだできません。" };
  }
  if (res.status === 409) {
    return { kind: "busy", detail: data.detail || "別のタスクを実行中です。" };
  }
  if (!res.ok || !data.taskId) return { kind: "unreachable" };
  const taskId = data.taskId;
  if (onProgress) {
    try { onProgress({ type: "started", taskId }); } catch { /* noop */ }
  }
  return await new Promise((resolve) => {
    let settled = false;
    let es = null;
    const done = (v) => {
      if (settled) return;
      settled = true;
      if (es) { try { es.close(); } catch { /* noop */ } }
      resolve(v);
    };
    try {
      es = new EventSource(`${CUBE_SERVER_BASE_URL}/pc/tasks/${taskId}/events`);
    } catch {
      resolve({ kind: "unreachable" });
      return;
    }
    es.addEventListener("step", (ev) => {
      try {
        if (onProgress) onProgress({ type: "step", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("screenshot", (ev) => {
      try {
        if (onProgress) onProgress({ type: "screenshot", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("control", (ev) => {
      try {
        if (onProgress) onProgress({ type: "control", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("confirm", (ev) => {
      try {
        if (onProgress) onProgress({ type: "confirm", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("confirm_resolved", (ev) => {
      try {
        if (onProgress) onProgress({ type: "confirm_resolved", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("verification", (ev) => {
      try {
        if (onProgress) onProgress({ type: "verification", ...JSON.parse(ev.data) });
      } catch { /* noop */ }
    });
    es.addEventListener("finished", (ev) => {
      try {
        const d = JSON.parse(ev.data);
        done({ kind: "finished", result: d.result || null });
      } catch {
        done({ kind: "unreachable" });
      }
    });
    es.addEventListener("error", () => {
      // SSE の一時切断（EventSource が自動再接続する）。サーバー側で
      // 終了済みなら再接続時の snapshot 経由で finished が届くため
      // 即座には諦めず、状態を1回確認しに行く。
      void (async () => {
        await new Promise((r) => setTimeout(r, 3000));
        if (settled) return;
        try {
          const st = await fetchWithTimeout(
            `${CUBE_SERVER_BASE_URL}/pc/tasks/${taskId}`,
            {}, PC_POST_TIMEOUT_MS);
          const sj = await st.json().catch(() => ({}));
          if (st.ok && sj.state === "finished" && sj.result) {
            done({ kind: "finished", result: sj.result });
          }
        } catch { /* まだ実行中とみなして待つ */ }
      })();
    });
  });
}

/* ISO 日時を「M月D日 H:MM」で返す（解釈できなければ空文字）。 */
function formatTimestampLabel(iso) {
  if (typeof iso !== "string" || !iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${d.getMonth() + 1}月${d.getDate()}日 ${d.getHours()}:${mm}`;
}

/* 距離（メートル）を表示用に整える。 */
function formatDistanceLabel(m) {
  if (typeof m !== "number" || !Number.isFinite(m) || m < 0) return "";
  if (m < 1000) return `約${Math.round(m)}m`;
  return `約${(m / 1000).toFixed(1)}km`;
}

/* http(s) の URL だけをリンクとして採用する。 */
function httpUrlOrEmpty(u) {
  return typeof u === "string" && /^https?:\/\//i.test(u) ? u : "";
}

/* ================================================================== */
/* YouTube要約レーン（cube-server の /yt、2026-11-20 本人決定）        */
/*                                                                    */
/* チャットに YouTube の URL が貼られたら、サーバー側（/yt）が動画   */
/* の字幕トラック（日本語の手動→自動、英語の順）を取得し、その字幕   */
/* を選択中のモデルに読解形式で読ませて要約させる。仮想PCで再生      */
/* する方式ではない（字幕テキストだけを使う）。URL が明示された      */
/* 依頼なので、このレーンは必ずここで完結させる：字幕なし・取得      */
/* 失敗・生成失敗でも通常経路には倒さず、正直な文面を出す（検索      */
/* レーン等に化けると「要約して」と頼んだのに無関係な答えが返る）。  */
/* 要約は検証ゲートを通さない「モデルの読み取り」であり、バッジと    */
/* 注記で字幕由来であることを明示する。知識メモ帳には保存しない。    */
/* ================================================================== */

/* /yt のページ側タイムアウト。サーバー側は字幕一覧の取得＋字幕      */
/* 本文の取得で数秒が通常だが、混雑時の余裕を見て 30 秒。             */
const YT_TIMEOUT_MS = 30000;

/* 要約のためにモデルへ渡す字幕の上限（文字数）。Ultra（iMac の       */
/* llama-server、コンテキスト 2048 トークン）に渡せる大きさと、       */
/* 端末・サーバーの自作モデル（ブロック長が短く、プロンプトは末尾      */
/* から切り取られる）に渡せる大きさで分ける。字幕が上限を超えたら    */
/* 先頭から切り、切り詰めた旨を注記する。                             */
const YT_SUMMARY_CAP_ULTRA = 1500;
const YT_SUMMARY_CAP_DEFAULT = 500;

function youtubeSummaryCap(modelId) {
  return modelId === "ultra" ? YT_SUMMARY_CAP_ULTRA : YT_SUMMARY_CAP_DEFAULT;
}

/* 質問文から YouTube の動画 ID（11桁）を取り出す。対応する形：      */
/* youtube.com/watch?v=…、youtu.be/…、shorts/、live/、embed/。       */
/* どれにも当たらなければ null。                                      */
function extractYoutubeVideoId(question) {
  const q = String(question || "");
  const m = q.match(
    /(?:youtube\.com\/(?:watch\?[^ \t\r\n]*v=|shorts\/|live\/|embed\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/,
  );
  return m ? m[1] : null;
}

/* サーバー側（/yt）に字幕取得を依頼する。戻り値はサーバーの応答      */
/* そのまま（{ok:true, transcript...} / {ok:true, hasCaptions:false}  */
/* / {ok:false, error}）。届かない・解釈できない場合は null。         */
async function fetchYoutubeCaptions(videoId) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/yt`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoId }),
      },
      YT_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!data || typeof data.ok !== "boolean") return null;
  return data;
}

/* 字幕の表示用ラベル（「日本語字幕（自動生成）」など）。 */
function ytLangLabel(data) {
  const lang = (data && data.lang) || "";
  const base = lang.split("-")[0];
  const name = base === "ja" ? "日本語" : base === "en" ? "英語" : (lang || "不明な言語");
  return `${name}字幕${data && data.kind === "asr" ? "（自動生成）" : ""}`;
}

/* YouTube要約レーン本体：字幕を取り、選択中モデルに要約させる。      */
/* 戻り値は表示分岐用の {kind, videoId, data?, text?, truncated?,     */
/* cap?}。kind: summary / no_captions / unreachable / failed /        */
/* meter_empty / generate_failed。URL が無ければ null（レーン自体     */
/* が発火していない）。                                               */
async function youtubeSummary(question, modelId) {
  const videoId = extractYoutubeVideoId(question);
  if (!videoId) return null;
  const data = await fetchYoutubeCaptions(videoId);
  if (!data) return { kind: "unreachable", videoId };
  if (!data.ok) return { kind: "failed", videoId, data };
  if (typeof data.transcript !== "string" || !data.transcript) {
    return { kind: "no_captions", videoId, data };
  }
  const cap = youtubeSummaryCap(modelId);
  const truncated = data.transcript.length > cap;
  const passage = truncated ? data.transcript.slice(0, cap) : data.transcript;
  let written = "";
  try {
    written = (
      (await generateForModel(
        modelId,
        `文：${passage}\n質問：この動画の内容を分かりやすく要約してください。`,
      )) || ""
    ).trim();
  } catch (e) {
    if (e && e.code === "METER_EMPTY") {
      return { kind: "meter_empty", videoId, data };
    }
    return { kind: "generate_failed", videoId, data };
  }
  if (!written) return { kind: "generate_failed", videoId, data };
  if (modelId === "ultra") {
    if (isUltraAbstention(written)) {
      return { kind: "generate_failed", videoId, data };
    }
    // 公開版Cubeの Ultra の文章は「です・ます」調にそろえる
    // （既存の提示文と同じ整形。適用先は Ultra の生成だけ）。
    written = normalizeUltraWrittenStyle(written);
  }
  return { kind: "summary", videoId, data, text: written, truncated, cap };
}

/* ---- 場所レーン（POST /places） ---- */

/* ジャンル語はサーバー側（places_endpoint.js の GENRES）と揃える。
   ページ側は「場所の質問らしいか」の判定だけに使い、実際の検索
   キーワードや地名の抽出はサーバーが行う（質問文をそのまま渡す）。 */
const PLACES_GENRE_WORDS = [
  "ドラッグストア", "薬局", "調剤", "コンビニ", "公園", "駅", "病院",
  "クリニック", "医院", "診療所", "歯医者", "歯科", "カフェ", "喫茶店",
  "コーヒー", "レストラン", "食堂", "飲食店", "ファストフード",
  "ファーストフード", "郵便局", "銀行", "スーパー", "スーパーマーケット",
  "ガソリンスタンド", "ガソリン", "駐車場", "パーキング", "ホテル",
  "旅館", "図書館", "学校", "小学校", "中学校", "高校", "神社", "寺",
  "寺院", "お寺", "交番", "警察署", "警察", "市役所", "区役所", "町役場",
  "村役場", "美容室", "美容院", "理容室", "床屋", "パン屋", "ベーカリー",
  "本屋", "書店", "ホームセンター", "温泉", "銭湯", "映画館", "シネマ",
  "ジム", "フィットネス", "コインランドリー", "ランドリー", "水族館",
  "動物園", "博物館", "美術館", "ゲームセンター", "ゲーセン",
  // 飲食の細分
  "ラーメン", "らーめん", "ラーメン屋", "ラーメン店", "寿司", "すし",
  "寿司屋", "回転寿司", "焼肉", "焼き肉", "居酒屋", "うどん", "そば",
  "蕎麦", "ピザ", "ハンバーガー", "バーガー", "定食", "定食屋",
  "スイーツ", "デザート", "ケーキ", "ケーキ屋", "たこ焼き", "たこやき",
  // 遊び・レジャー
  "カラオケ", "ボウリング", "ボーリング", "バッティングセンター",
  "バッティング", "プール", "スイミング", "ゴルフ練習場", "ゴルフ",
  "スケート", "アイススケート", "スケートリンク", "ダーツ",
  "ビリヤード", "釣り堀", "釣堀", "テーマパーク", "遊園地",
  // 買い物の細分
  "100円ショップ", "100均", "百円ショップ", "100円均一", "家電量販店",
  "家電店", "電気屋", "家電", "スポーツ用品", "スポーツショップ",
  "スポーツ店", "アウトドア用品", "アウトドア", "キャンプ用品",
  "ペットショップ", "ペット用品", "花屋", "フラワーショップ", "眼鏡",
  "メガネ", "めがね", "眼鏡店", "コンタクトレンズ", "コンタクト",
  "携帯ショップ", "携帯電話", "スマホショップ", "ドコモショップ",
  "auショップ", "ソフトバンクショップ", "ゲームショップ", "ゲーム店",
  "CDショップ", "レコード店", "レコードショップ", "レコード",
  "楽器店", "楽器屋", "楽器", "文房具", "文房具店", "文具店", "文具",
  "雑貨", "雑貨屋", "雑貨店", "洋服", "洋服店", "服屋", "衣料品",
  "アパレル", "靴", "靴屋", "靴店", "シューズ",
  // 医療の細分
  "内科", "外科", "眼科", "耳鼻咽喉科", "耳鼻科", "皮膚科", "小児科",
  "整形外科", "整骨院", "接骨院",
  // 公共・交通・生活
  "ATM", "現金自動", "コインロッカー", "ロッカー", "トイレ",
  "お手洗い", "公衆トイレ", "バス停", "タクシー", "タクシー乗り場",
  "レンタカー", "クリーニング", "クリーニング店", "写真館",
  "写真スタジオ", "フォトスタジオ", "証明写真", "証明写真機",
  "コピー", "印刷", "プリント", "レンタルビデオ", "ビデオレンタル",
  "レンタルショップ", "学習塾", "進学塾", "予備校", "塾", "保育園",
  "保育所", "幼稚園",
];

/* 事実の質問らしい語。場所レーンでは拾わないための安全網
   （本社・歴史などの会社事実、件数を聞く質問、行き方・所要時間、
   人口・面積・天気などの地名にくっつく属性の質問）。 */
const PLACES_FACT_GUARDS =
  /本社|創業|設立|歴史|とは|の意味|って何|日本一|世界一|日本で一番|世界で一番|いくつ|何軒|何件|何か所|までの時間|所要時間|行き方|人口|面積|標高|高さ|天気|気温|市長|知事|特産|名物|有名|誰|いつ|なぜ|何県|何市|何線/;

/* 場所探しの言い回し（「は？」「教えて」「ありますか」等）。 */
const PLACES_LOOKUP_FORMS =
  /(は？|は\?|を教えて|教えて|ありますか|ある？|ある\?|探して|検索して|見つけて|どこですか|どこに|一覧)/;

/**
 * 場所系の質問かどうか。ジャンル語があり、かつ近さの表現・地名・
 * 探す言い回しのいずれかがあるときだけ true。事実の質問
 * （「法隆寺の高さは？」「コーヒーの淹れ方は？」等）は拾わない。
 * ジャンル辞書に無いカテゴリでも、「地名＋カテゴリ語＋言い回し」
 * の形なら場所探しとみなしてサーバーに委ねる（2026-11-19 実機
 * 報告：「郡山市のゲームセンターは？」が発火しなかった）。
 */
function detectPlacesQuery(question) {
  const q = String(question || "");
  if (!q) return false;
  if (PLACES_FACT_GUARDS.test(q)) return false;
  if (PLACES_GENRE_WORDS.some((w) => q.includes(w))) {
    if (/近く|近所|周辺|近辺|付近|あたり|最寄|近い/.test(q)) return true;
    if (/[^\s、。・]{1,20}(駅|市|区|町|村|県|都|府)/.test(q)) return true;
    if (/(探して|検索して|見つけて|ありますか|ある？|ある\?)/.test(q)) return true;
    return false;
  }
  return detectPlacesByShape(q);
}

/* 辞書に無いカテゴリの形判定：「地名アンカー（〜駅・市・区・町・
   村・県・都・府）＋カテゴリ語らしい残り＋場所探しの言い回し」が
   そろったときだけ true。地名と言い回しを除いた残りが2文字未満
   ならカテゴリ語が無い（「郡山市はどこですか？」等）ので拾わない。 */
function detectPlacesByShape(q) {
  const anchor = q.match(/[^\s、。・]{1,20}(駅|市|区|町|村|県|都|府)/);
  if (!anchor) return false;
  if (!PLACES_LOOKUP_FORMS.test(q)) return false;
  const rest = q
    .replace(anchor[0], " ")
    .replace(/(は？|は\?|を教えて|教えて|ありますか|ある？|ある\?|探して|検索して|見つけて|どこですか|どこに|どこ|一覧|ですか|です|ます|ください)/g, " ")
    .replace(/[のではをにがはもへとや、。・！？!?「」『』\s]/g, "");
  return rest.length >= 2;
}

/**
 * /places に質問文をそのまま投げ、検証済みの形に整えて返す。
 * 通信失敗・形が違う場合は null（呼び出し側は従来経路に倒れる）。
 * サーバーが ok:false（両源とも失敗）を返した場合はそのまま返し、
 * 呼び出し側が失敗として扱う。ok:true で0件は「本当に無い」。
 * 端末位置は既存の仕組みが無いため送らない（地名はサーバーが
 * 質問文から解決する）。
 */
async function fetchPlaces(question) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/places`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: question, limit: 8 }),
      },
      PLACES_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (
    !res || !res.ok || !data ||
    typeof data.ok !== "boolean" || !Array.isArray(data.places)
  ) {
    return null;
  }
  const places = [];
  for (const p of data.places) {
    if (!p || typeof p.name !== "string" || !p.name) continue;
    places.push({
      name: p.name,
      address: typeof p.address === "string" ? p.address : "",
      genre: Array.isArray(p.genre)
        ? p.genre.filter((s) => typeof s === "string") : [],
      tel: typeof p.tel === "string" ? p.tel : "",
      hours: typeof p.hours === "string" ? p.hours : "",
      distance_m: typeof p.distance_m === "number" ? p.distance_m : null,
      nearest_station:
        typeof p.nearest_station === "string" ? p.nearest_station : "",
      sources: Array.isArray(p.sources)
        ? p.sources.filter((s) => typeof s === "string") : [],
      link: Array.isArray(p.links) ? httpUrlOrEmpty(p.links[0]) : "",
    });
  }
  const query = data.query && typeof data.query === "object" ? data.query : {};
  return {
    ok: data.ok,
    places,
    keyword: typeof query.keyword === "string" ? query.keyword : "",
    placeLabel: typeof query.place === "string" ? query.place : "",
    centerLabel:
      query.center && typeof query.center.label === "string"
        ? query.center.label : "",
    attribution: Array.isArray(data.attribution)
      ? data.attribution.filter((s) => typeof s === "string") : [],
  };
}

/* 場所1件を表示用カードにする。 */
function placeCardOf(p) {
  return {
    kind: "place",
    name: p.name,
    address: p.address || "",
    genre: Array.isArray(p.genre) ? p.genre : [],
    tel: p.tel || "",
    hours: p.hours || "",
    distance_m: typeof p.distance_m === "number" ? p.distance_m : null,
    nearest_station: p.nearest_station || "",
    sources: Array.isArray(p.sources) ? p.sources : [],
    link: httpUrlOrEmpty(p.link),
  };
}

/* ---- 話題レーン（GET /topics） ---- */

const TOPIC_SOURCE_LABELS = {
  google_trends: "Googleトレンド",
  yahoo_news: "Yahoo!ニュース",
  nhk_news: "NHKニュース",
  hatena_hot: "はてなブックマーク",
};

function topicSourceLabel(key) {
  return TOPIC_SOURCE_LABELS[key] || (typeof key === "string" ? key : "");
}

/**
 * 「今の話題・ニュースを教えて」系の定型の聞き方かどうか。
 * 末尾の句読点を落としてから、文全体への一致で判定する。具体的な
 * 事実を聞く質問（「今日のニュースで発表された〜の価格は？」等）は
 * 一致しないので従来経路に倒れる。
 */
function detectTopicsQuery(question) {
  const q = String(question || "").trim()
    .replace(/[！!？?。、…〜~\s]+$/u, "");
  if (!q) return false;
  if (/とは|の意味|って何|なぜ|本当ですか|デマ|嘘/.test(q)) return false;
  const patterns = [
    /^(今|今日|最近|昨日|最新)の?(話題|ニュース|トレンド)(は|って|を|が|について)?(何|なに|教えて|見せて|知りたい|ある|ありますか|です|だ)?$/,
    /^(話題|ニュース|トレンド)(は|って|を|が|について)?(何|なに|教えて|見せて|知りたい|ある|ありますか)?$/,
    /^(今|今日|最近)?(何|なに)が話題(になって(い)?る|ですか|なの|だ)?$/,
    /^世間(の|で)(話題|ニュース)(は|を)?(何|なに|教えて)?$/,
    /^急上昇(ワード|検索|トレンド|キーワード)?(は|を|について)?(何|なに|教えて)?$/,
    /^はてブ(の)?(人気|話題|ホットエントリー|エントリー)?(は|を)?(何|なに|教えて)?$/,
  ];
  return patterns.some((re) => re.test(q));
}

/**
 * /topics から今の話題を取り、検証済みの形に整えて返す。
 * 失敗・形が違う場合は null。stale（古いキャッシュ）のときは
 * stale:true をそのまま返し、表示側で「少し前の話題」と出す。
 */
async function fetchTopics(limit, timeoutMs) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/topics?limit=${limit || 10}`,
      { method: "GET" },
      timeoutMs || TOPICS_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || !Array.isArray(data.topics)) return null;
  const topics = [];
  for (const t of data.topics) {
    if (!t || typeof t.title !== "string" || !t.title) continue;
    topics.push({
      title: t.title,
      link: typeof t.link === "string" ? t.link : "",
      snippet: typeof t.snippet === "string" ? t.snippet : "",
      published: typeof t.published === "string" ? t.published : "",
      sources: Array.isArray(t.sources)
        ? t.sources.filter((s) => typeof s === "string")
        : (typeof t.source === "string" ? [t.source] : []),
    });
  }
  return {
    topics,
    stale: data.stale === true,
    fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
  };
}

/* 話題1件を表示用カードにする。出典ラベルは必ず付ける。 */
function topicCardOf(t) {
  const labels = (Array.isArray(t.sources) ? t.sources : [])
    .map(topicSourceLabel)
    .filter(Boolean);
  return {
    kind: "topic",
    title: t.title,
    link: httpUrlOrEmpty(t.link),
    snippet: t.snippet || "",
    published: t.published || "",
    sourceLabel: labels.join("・"),
  };
}

/* ---- 買い物レーン（POST /shop） ---- */

/* 事実の質問らしい語。買い物レーンでは拾わない（定価・発売当時の
   価格・相場・知識質問は従来の事実レーンの担当）。 */
const SHOP_FACT_GUARDS =
  /定価|希望小売価格|発売日|発売当時|発売価格|発売された|当時の価格|当時の値段|の相場|とは|の意味|って何|の違い|仕組み/;

/* 購入相談のシグナル（設計 §5.1）。これが明確なときだけ発火する。
   価格条件の指定もシグナルとして扱う（extractShopConditions 側）。 */
const SHOP_SIGNALS = [
  /買いたい/, /買おう/, /欲しい/, /ほしい/, /購入したい/,
  /探して(いる|います|る|ください)?/, /探し中/, /買い替え/, /買い換え/,
  /プレゼント/, /贈(りたい|り物)/, /おすすめ/,
  /安い[^。]{0,24}(どれ|は？|は\?|ありますか|ある？|ある\?)/,
  /安く(買える|買いたい|手に入る)/, /買える/,
  /人気[^。]{0,24}(どれ|は？|は\?|ありますか)/,
  /(人気|おすすめ)の[^。]{1,30}(教えて|欲しい|買いたい|探して|見せて|ください)/,
];

/* キーワードとして意味を成さない語（これだけなら聞き返す）。 */
const SHOP_EMPTY_KEYWORDS = new Set([
  "何か", "なにか", "何", "なんか", "もの", "商品", "アイテム", "品物",
]);

/* 商品ではなく場所・体験を指す語（「人気の観光地は？」等）。
   これだけがキーワードなら買い物レーンには入らない。 */
const SHOP_NON_PRODUCT_KEYWORDS = new Set([
  "観光地", "観光スポット", "観光", "スポット", "名所", "旅行先",
  "店", "お店", "レストラン", "飲食店", "食堂",
]);

/**
 * 質問文から検索条件（価格帯・送料無料・並び）をルールで抽出する。
 * 数値は質問文に実際にあるものだけを使う。
 */
function extractShopConditions(question) {
  const q = String(question || "");
  const cond = {
    price_min: null, price_max: null, free_shipping: false, sort: "score",
  };
  const man = (s) => Math.round(parseFloat(s) * 10000);
  const yen = (s) => parseInt(String(s).replace(/,/g, ""), 10);
  let m;
  // 「○円から○円」「○万円から○万円」の範囲指定を先に拾う。
  if ((m = q.match(/(\d[\d,]*)\s*円\s*から\s*(\d[\d,]*)\s*円/))) {
    cond.price_min = yen(m[1]);
    cond.price_max = yen(m[2]);
  } else if (
    (m = q.match(/(\d+(?:\.\d+)?)\s*万円?\s*から\s*(\d+(?:\.\d+)?)\s*万円?/))
  ) {
    cond.price_min = man(m[1]);
    cond.price_max = man(m[2]);
  }
  if ((m = q.match(/予算\s*(は|が|約|だいたい)?\s*(\d+(?:\.\d+)?)\s*万円/))) {
    cond.price_max = man(m[2]);
  } else if ((m = q.match(/予算\s*(は|が|約|だいたい)?\s*(\d[\d,]*)\s*円/))) {
    cond.price_max = yen(m[2]);
  }
  if ((m = q.match(/(\d+(?:\.\d+)?)\s*万円?\s*(以下|以内|まで|くらい|程度|前後)/))) {
    cond.price_max = man(m[1]);
  }
  if ((m = q.match(/(\d+(?:\.\d+)?)\s*万円?\s*(以上|から)/))) {
    cond.price_min = man(m[1]);
  }
  if (cond.price_max === null &&
      (m = q.match(/(\d[\d,]*)\s*円\s*(以下|以内|まで|くらい|程度|前後)/))) {
    cond.price_max = yen(m[1]);
  }
  if (cond.price_min === null &&
      (m = q.match(/(\d[\d,]*)\s*円\s*(以上|から)/))) {
    cond.price_min = yen(m[1]);
  }
  if (/送料\s*(無料|込み|が無料)/.test(q)) cond.free_shipping = true;
  if (/安い順|安いものから|安い方から|価格の安い順|値段の安い順/.test(q)) {
    cond.sort = "price_asc";
  } else if (/高い順|高いものから|価格の高い順/.test(q)) {
    cond.sort = "price_desc";
  } else if (/人気順|人気の|評価の高い順|評価が高い|レビューの多い順|レビューが多い/.test(q)) {
    cond.sort = "review_count";
  }
  return cond;
}

/* キーワードに残る口語の接続表現（文の途中・末尾の残渣）。
   定型句（買いたい・欲しい等）を消した後に「んだけど」だけが
   残ると、そのまま検索語になって無関係な商品が返る（実例：
   「ミニPCを買いたいんだけどおすすめある？」→「ミニPC んだけど」）。
   商品名の一部になり得ない接続表現だけを対象にし、平仮名の
   商品名（はちみつ・すもも・たから等）は削らない。 */
const SHOP_RESIDUE_ONLY =
  /^(んだけど|んですが|んですけど|んだけれど|だけど|けれど|けど|ので|って|とか|のに|んだ|んです)$/;
const SHOP_RESIDUE_PREFIX =
  /^(んだけど|んですが|んですけど|んだけれど|だけど|けれど|けど|ので)+/;
const SHOP_RESIDUE_SUFFIX =
  /(んだけど|んですが|んですけど|んだけれど|だけど|けれど|けど|ので)$/;
/* から・って・とか は、漢字・カタカナ・英数字の直後のときだけ
   落とす（「たから」のような平仮名の語を守るため）。 */
const SHOP_RESIDUE_SUFFIX_GUARDED =
  /([一-龯ァ-ヶーA-Za-z0-9])(から|って|とか|のに)$/;

/* キーワードの1語から接続表現の残渣を取り除く。 */
function cleanShopKeywordToken(tok) {
  let t = String(tok || "");
  if (!t || SHOP_RESIDUE_ONLY.test(t)) return "";
  t = t.replace(SHOP_RESIDUE_PREFIX, "");
  let prev = null;
  while (t && t !== prev) {
    prev = t;
    t = t.replace(SHOP_RESIDUE_SUFFIX, "");
    t = t.replace(SHOP_RESIDUE_SUFFIX_GUARDED, "$1");
  }
  return t;
}

/**
 * 質問文から検索キーワードを取り出す。条件表現と相談の定型句を
 * 取り除いた残りだけを使うので、キーワードは必ず質問文に実在
 * する語になる（モデルに商品名を作らせない）。
 */
function extractShopKeyword(question) {
  let t = String(question || "");
  // 条件表現を取り除く（金額・送料・並び）。金額の直後の「の」
  // （「5000円の本」）も一緒に取り除く。
  t = t.replace(/予算\s*(は|が|約|だいたい)?\s*\d[\d,]*(?:\.\d+)?\s*万?円?/g, " ");
  t = t.replace(/\d[\d,]*(?:\.\d+)?\s*万\s*円?\s*(以下|以内|まで|以上|から|くらい|程度|前後|の)?/g, " ");
  t = t.replace(/\d[\d,]*\s*円\s*(以下|以内|まで|以上|から|くらい|程度|前後|の)?/g, " ");
  t = t.replace(/送料\s*(無料|込み|が無料)/g, " ");
  t = t.replace(/人気順|評価の高い順|レビュー順|新着順|価格順|安い順|高い順|順(で|に)/g, " ");
  // 相談の定型句を取り除く。
  t = t.replace(/が欲しい|を買いたい|買いたい|欲しい|ほしい|を購入したい|購入したい/g, " ");
  t = t.replace(/を探して(いる|います|る)?|探して(いる|います|る)?|探し中/g, " ");
  t = t.replace(/のおすすめ|おすすめの|おすすめは|おすすめを|おすすめ/g, " ");
  t = t.replace(/プレゼントに|プレゼント用|プレゼント/g, " ");
  t = t.replace(/買い替え(たい|る|の)?|買い換え(たい|る)?/g, " ");
  t = t.replace(/安く買える|安く買いたい|買える|安い|安く|高い|高く/g, " ");
  t = t.replace(/人気の|人気が(ある|高い)|評価の高い|レビューの多い|よく売れている/g, " ");
  t = t.replace(/教えて(ください)?|ありますか|ある？|ある\?|はどれ(ですか)?|どれが(いい|良い|おすすめ)|がいい(ですか)?|ください|を探す|を検索/g, " ");
  t = t.replace(/もっと|だけ(で|にして)|にして|に変えて/g, " ");
  t = t.replace(/[、。・！？!?「」『』（）()]/g, " ");
  // 助詞だけの語を落とし、前後の助詞を整える。先頭の助詞は
  // 漢字・カタカナの語が続くときだけ落とす（「はちみつ」の「は」
  // のような商品名の一部を削らないため）。末尾も同様に、
  // 漢字・カタカナの直後の助詞だけを落とす（「すもも」保護）。
  const PARTICLE_ONLY = /^(の|が|を|は|に|で|と|も|へ|から|まで|や)$/;
  let kw = t.split(/\s+/)
    .map(cleanShopKeywordToken)
    .filter((tok) => tok && !PARTICLE_ONLY.test(tok))
    .join(" ")
    .trim();
  kw = kw.replace(/^(の|が|を|は|に|で|と|も|へ)+(?=[一-龯ァ-ヶ])/, "");
  kw = kw.replace(/([一-龯ァ-ヶー])(の|が|を|は|に|で|と|も|へ)$/, "$1");
  if (/^(の|が|を|は|に|で|と|も|へ)+$/.test(kw)) kw = "";
  return kw;
}

/**
 * 質問が買い物相談なら {keyword, conditions} を返す。購入相談だが
 * キーワードが取れなければ {askBack:true}。買い物相談でなければ
 * null（従来レーンに任せる）。prev（前回の {keyword, conditions}）
 * がある場合、「もっと安いの」「送料無料だけにして」等の条件変更
 * の続きも拾う（買い物レーンが直前に動いた会話に限る）。
 */
function parseShopQuery(question, prev) {
  const q = String(question || "").trim();
  if (!q) return null;
  if (SHOP_FACT_GUARDS.test(q)) return null;
  const cond = extractShopConditions(q);
  const keyword = extractShopKeyword(q);
  // 続きの条件変更：前回の商品のまま、条件だけを差し替える。
  if (prev && prev.keyword && (!keyword || keyword === prev.keyword)) {
    const changed =
      /もっと|にして|に変えて|だけ(で|にして)|絞って|緩めて/.test(q) ||
      cond.price_max !== null || cond.price_min !== null ||
      /送料/.test(q) || cond.sort !== "score";
    if (changed) {
      const merged = { ...(prev.conditions || {}) };
      if (cond.price_max !== null) merged.price_max = cond.price_max;
      if (cond.price_min !== null) merged.price_min = cond.price_min;
      if (/送料\s*(無料|込み)/.test(q)) merged.free_shipping = true;
      if (/送料は気にしない|送料ありでも/.test(q)) merged.free_shipping = false;
      if (/もっと安い|もっと安く|安くして|安いのにして/.test(q)) {
        merged.sort = "price_asc";
      } else if (/もっと高い|高いのにして/.test(q)) {
        merged.sort = "price_desc";
      } else if (cond.sort !== "score") {
        merged.sort = cond.sort;
      }
      return { keyword: prev.keyword, conditions: merged, continuation: true };
    }
  }
  const hasSignal =
    SHOP_SIGNALS.some((re) => re.test(q)) ||
    cond.price_max !== null || cond.price_min !== null;
  if (!hasSignal) return null;
  if (!keyword || SHOP_EMPTY_KEYWORDS.has(keyword)) {
    return { askBack: true, conditions: cond };
  }
  if (SHOP_NON_PRODUCT_KEYWORDS.has(keyword)) return null;
  return { keyword, conditions: cond, continuation: false };
}

/**
 * /shop に正規化した条件を投げ、検証済みの形に整えて返す。
 * 失敗・形が違う場合は null（呼び出し側は従来経路に倒れる）。
 * ゼロ件は total:0・candidates:[] で返る（失敗とは区別する）。
 */
async function fetchShop(keyword, conditions) {
  const cond = conditions || {};
  const body = {
    query: keyword,
    sort: cond.sort || "score",
    limit: 5,
  };
  if (typeof cond.price_min === "number") body.price_min = cond.price_min;
  if (typeof cond.price_max === "number") body.price_max = cond.price_max;
  if (cond.free_shipping) body.free_shipping = true;
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/shop`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      SHOP_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (
    !res || !res.ok || !data || data.ok !== true ||
    !Array.isArray(data.candidates)
  ) {
    return null;
  }
  const candidates = [];
  for (const c of data.candidates) {
    if (!c || typeof c.name !== "string" || !c.name) continue;
    candidates.push(shopCardOf(c));
  }
  return {
    total: typeof data.total === "number" ? data.total : candidates.length,
    candidates,
    fetchedAt: typeof data.fetched_at === "string" ? data.fetched_at : "",
    attribution: Array.isArray(data.attribution)
      ? data.attribution.filter((s) => typeof s === "string") : [],
    guardExcluded:
      typeof data.guard_excluded === "number" ? data.guard_excluded : 0,
  };
}

/* 送料ラベルの表示整形。Yahoo が送料名として返す「設定なし」は
   利用者には不具合に見えるため、カード化の段階で「無し」扱いに
   して、表示側の既定文（「送料は商品ページでご確認ください」）に
   落とす。サーバーが返した値そのものは変えない。 */
function shopShippingLabelOf(v) {
  if (typeof v !== "string") return "";
  const s = v.trim();
  return s === "設定なし" ? "" : s;
}

/* 商品1件を表示用カードにする。数字は API フィールドの転記だけ。 */
function shopCardOf(c) {
  return {
    kind: "shop",
    name: c.name,
    price: typeof c.price === "number" ? c.price : null,
    tax_included: c.tax_included !== false,
    url: httpUrlOrEmpty(c.url),
    image: httpUrlOrEmpty(c.image),
    in_stock: c.in_stock !== false,
    shipping_label: shopShippingLabelOf(c.shipping_label),
    review_rate: typeof c.review_rate === "number" ? c.review_rate : null,
    review_count: typeof c.review_count === "number" ? c.review_count : 0,
    shop_name: typeof c.shop_name === "string" ? c.shop_name : "",
    genre_name: typeof c.genre_name === "string" ? c.genre_name : "",
    rank_reason: typeof c.rank_reason === "string" ? c.rank_reason : "",
  };
}

/* 買い物レーンの導入文。ユーザーが言った条件の再掲だけで作り、
   候補の数字（価格・評価など）は文章に入れない。 */
function shopIntroText(keyword, conditions) {
  const cond = conditions || {};
  const parts = [];
  const yenLabel = (n) => `${Number(n).toLocaleString("ja-JP")}円`;
  if (typeof cond.price_min === "number" &&
      typeof cond.price_max === "number") {
    parts.push(`${yenLabel(cond.price_min)}〜${yenLabel(cond.price_max)}`);
  } else if (typeof cond.price_max === "number") {
    parts.push(`${yenLabel(cond.price_max)}以下`);
  } else if (typeof cond.price_min === "number") {
    parts.push(`${yenLabel(cond.price_min)}以上`);
  }
  if (cond.free_shipping) parts.push("送料無料");
  const sortNote = {
    price_asc: "価格の安い順",
    price_desc: "価格の高い順",
    review_count: "レビュー件数の多い順",
  }[cond.sort || "score"] || "";
  let text = parts.length
    ? `${parts.join("・")}の条件で「${keyword}」を探しました。`
    : `「${keyword}」を探しました。`;
  if (sortNote) text += `（${sortNote}に並べています）`;
  return text;
}

/* ポイント付与の冪等キー用の条件ハッシュ（FNV-1a）。 */
function shopConditionsHash(keyword, conditions) {
  const cond = conditions || {};
  const s = JSON.stringify({
    q: keyword || "",
    min: typeof cond.price_min === "number" ? cond.price_min : null,
    max: typeof cond.price_max === "number" ? cond.price_max : null,
    fs: !!cond.free_shipping,
    sort: cond.sort || "score",
  });
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/* ================================================================== */
/* 住所レーン（cube-server の /addr、2026-11-20 本人決定）             */
/*                                                                    */
/* 住所・郵便番号・位置そのものを聞く質問に、サーバー側（/addr）が   */
/* 公開 API（zipcloud / 国土地理院）で引いた実データだけで答える     */
/* 専用レーン。モデル生成・検証ゲートは通さず、API の値をそのまま    */
/* 構造化して出す。引けなかった住所・座標を推測で作ることはしない。  */
/* 郵便番号の形式は決定的なので YouTube レーンと同じ段で先に確定     */
/* させ、キーワード形は場所レーンの直前で拾う（近くの店探しは        */
/* 場所レーンの担当で、住所そのものとは切り分ける）。                */
/* 結果は変わりうる情報なので知識メモ帳には保存しない。              */
/* ================================================================== */

const ADDR_TIMEOUT_MS = 20000;

/* 郵便番号の抽出用に、全角数字と全角ハイフンを半角にそろえる。 */
function zipNormText(s) {
  return String(s || "")
    .replace(/[０-９]/g, (c) =>
      String.fromCharCode(c.charCodeAt(0) - 0xfee0))
    .replace(/－/g, "-");
}

/**
 * 質問文から郵便番号（7桁の数字列）を取り出す。当たる形：
 *   - 〒963-8001 / 〒9638001（〒つき）
 *   - 963-8001（長い番号列や電話番号の一部でないもの）
 *   - 9638001（単独、または「郵便番号」と一緒に出てきた場合だけ）
 * どれにも当たらなければ null。金額・年号・電話番号は拾わない。
 */
function extractZipcode(question) {
  const q = zipNormText(question);
  if (!q) return null;
  let m = q.match(/〒\s*(\d{3})-(\d{4})/);
  if (m) return m[1] + m[2];
  m = q.match(/〒\s*(\d{7})(?!\d)/);
  if (m) return m[1];
  m = q.match(/(^|[^\d-])(\d{3})-(\d{4})(?![\d-])/);
  if (m) {
    const after = q.charAt(m.index + m[0].length);
    if (after !== "円" && after !== "年") return m[2] + m[3];
  }
  m = q.match(/(^|[^\d])(\d{7})(?!\d)/);
  if (m) {
    const bare = q.replace(/[\s、。・！？!?「」『』]/g, "");
    if (bare === m[2] || q.includes("郵便番号")) return m[2];
  }
  return null;
}

/**
 * 郵便番号の形＋場所の文脈を含む質問かどうか（「963-8001は
 * どこ？」「〒963-8001の住所は？」等）。この形は計算レーンから
 * 見ると引き算（963-8001）に化けるため、ask() の計算レーン
 * （0.75 の CalcLane と手順2の evalArithmetic）の両方で対象外
 * にし、住所レーン（パターンA）に委ねるための判定。場所の
 * 文脈が無いもの（「963-8001を計算して」「963-8001は？」）は
 * false を返し、従来どおり計算で答える。
 */
function isZipcodePlaceQuestion(question) {
  const q = String(question || "");
  if (!extractZipcode(q)) return false;
  return /はどこ|どこ[？?]|どこです|どこに|どこの|住所|郵便番号|の場所/.test(q);
}

/* 住所キーワードの対象（地名・施設名）を整える。 */
function cleanAddressTarget(s) {
  const t = String(s || "")
    .trim()
    .replace(/^[「『]/, "")
    .replace(/[」』]$/, "")
    .replace(/(は|が|って|の)$/, "")
    .trim();
  return t;
}

/* 対象がジャンル語そのものだけ（「病院はどこにある？」等）なら
   場所探しの言い方なので、住所レーンでは拾わない。 */
function isGenreOnlyTarget(t) {
  if (!t) return true;
  return PLACES_GENRE_WORDS.some((w) => t === w);
}

/**
 * 住所キーワード形の質問かどうか。住所・郵便番号・座標そのものを
 * 聞いているときだけ {target, wantsZip} を返し、それ以外は null。
 * 郵便番号の数字が含まれる質問はパターンA（extractZipcode）の
 * 担当なのでここでは扱わない。「近くの〜」は場所レーン優先、
 * 行き方・ルートと、作り方・意味などの事実質問は対象外。
 */
function parseAddressQuery(question) {
  const q = String(question || "");
  if (!q) return null;
  if (extractZipcode(q)) return null;
  if (/近く|近所|周辺|近辺|付近/.test(q)) return null;
  if (/行き方|どうやって行く|ルート|経路/.test(q)) return null;
  if (/作り方|書き方|意味|とは/.test(q)) return null;
  const finish = (raw, wantsZip) => {
    const t = cleanAddressTarget(raw);
    if (!t || t.length > 40) return null;
    if (/[、。？！?!]/.test(t)) return null;
    if (/^(それ|これ|あれ|ここ|そこ|何|なに)/.test(t)) return null;
    return { target: t, wantsZip: !!wantsZip };
  };
  let m = q.match(/^(.+?)の郵便番号(?:は|を|って|が)?/);
  if (m) return finish(m[1], true);
  m = q.match(/^(.+?)の住所(?:は|を|って|が)?/);
  if (m) return finish(m[1], false);
  m = q.match(/^(.+?)の(?:座標|緯度・経度|緯度経度)(?:は|を|って|が)?/);
  if (m) return finish(m[1], false);
  m = q.match(/^(.+?)の場所はどこ/);
  if (m) {
    const r = finish(m[1], false);
    return r && !isGenreOnlyTarget(r.target) ? r : null;
  }
  m = q.match(/^(.+?)(?:は|って)どこにある/);
  if (m) {
    const r = finish(m[1], false);
    return r && !isGenreOnlyTarget(r.target) ? r : null;
  }
  return null;
}

/**
 * /addr に {zipcode} または {query} を投げ、検証済みの形に整えて
 * 返す。サーバーが ok:false（not_found / upstream）を返した場合は
 * そのまま返し、呼び出し側が正直な文面で完結させる。通信失敗・
 * 形が違う場合は null（呼び出し側の判断で、従来経路に倒すか
 * 接続エラーを出す）。
 */
async function fetchAddress(body) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/addr`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      ADDR_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      kind: typeof data.kind === "string" ? data.kind : "",
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  const raw = Array.isArray(data.results) ? data.results : [];
  if (data.kind === "zipcode") {
    const results = [];
    for (const r of raw) {
      if (!r || typeof r.address !== "string" || !r.address) continue;
      results.push({
        prefecture: typeof r.prefecture === "string" ? r.prefecture : "",
        city: typeof r.city === "string" ? r.city : "",
        town: typeof r.town === "string" ? r.town : "",
        address: r.address,
      });
    }
    if (!results.length) return null;
    return {
      ok: true,
      kind: "zipcode",
      zipcode: typeof data.zipcode === "string" ? data.zipcode : "",
      results,
      source: typeof data.source === "string" ? data.source : "zipcloud",
    };
  }
  if (data.kind === "geocode") {
    const results = [];
    for (const r of raw) {
      if (!r || typeof r.lat !== "number" || typeof r.lng !== "number") {
        continue;
      }
      results.push({
        title: typeof r.title === "string" ? r.title : "",
        lat: r.lat,
        lng: r.lng,
      });
    }
    if (!results.length) return null;
    return {
      ok: true,
      kind: "geocode",
      results,
      source: typeof data.source === "string" ? data.source : "国土地理院",
    };
  }
  return null;
}

/* 郵便番号→住所の答え文。町名が無い番号は、ある範囲までだけを
   言い、補完しない。 */
function zipcodeAnswerText(data) {
  const list = data.results.map((r) => r.address);
  if (list.length === 1) {
    return `〒${data.zipcode} の住所は ${list[0]} です。`;
  }
  return `〒${data.zipcode} の住所は次のとおりです。\n` +
    list.map((a) => `・${a}`).join("\n");
}

/* 地名→座標の答え文。複数候補は同名の別地点と取り違えないよう
   title（住所表記）を必ず併記する。 */
function geocodeAnswerText(data) {
  const list = data.results;
  if (list.length === 1) {
    const r = list[0];
    return `「${r.title}」の位置は、緯度 ${r.lat} / 経度 ${r.lng} です。`;
  }
  return "候補が複数見つかりました。\n" + list.map((r) =>
    `・${r.title}（緯度 ${r.lat} / 経度 ${r.lng}）`).join("\n");
}

/* 地図リンク（キー不要の Google Maps 検索URL、座標クエリ）。 */
function addressMapsUrl(lat, lng) {
  return `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
}

/* 位置1件を表示用カードにする（タイトルが地図へのリンク）。 */
function addressCardOf(r) {
  return {
    kind: "address",
    title: r.title,
    lat: r.lat,
    lng: r.lng,
    link: addressMapsUrl(r.lat, r.lng),
  };
}

/* ================================================================== */
/* 外部サービス連携 追加5件レーン（2026-11-20 本人決定）              */
/*                                                                    */
/* 地震・警報 / 最寄り駅 / 祝日 / 為替・株価 / 音楽検索。いずれも    */
/* cube-server の実データ API（/quake /station /holiday /market     */
/* /music）で答え、モデル生成・検証ゲートは通さない。数値が要る    */
/* レーン（地震・為替・株価）は、発火したら必ずこのレーンで完結    */
/* させ、取れなかった場合も正直な文面を出す（モデルの推測に        */
/* フォールスルーしない）。結果は変わる情報なので知識メモ帳には    */
/* 保存しない。                                                       */
/*                                                                    */
/* Wikipedia 定義レーンは本人の判断（索引レーンと重複）で同日に     */
/* 撤去した。サーバー側の /wiki は残してあるが、ページからは        */
/* 呼ばない。                                                         */
/* ================================================================== */

const QUAKE_TIMEOUT_MS = 15000;
const STATION_TIMEOUT_MS = 20000;
const HOLIDAY_TIMEOUT_MS = 15000;
const MARKET_TIMEOUT_MS = 15000;
const MUSIC_TIMEOUT_MS = 15000;
const PAPERS_TIMEOUT_MS = 30000;
const TRANSLATE_TIMEOUT_MS = 30000;
const WOLFRAM_TIMEOUT_MS = 90000;

/* ISO 日時を JST の「M月D日 HH:MM」に整える（取得・発表時刻の表示用）。 */
function jstDateTimeLabel(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso || "");
  try {
    const parts = new Intl.DateTimeFormat("ja-JP", {
      timeZone: "Asia/Tokyo",
      month: "numeric", day: "numeric",
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).formatToParts(d);
    const pick = (t) => (parts.find((p) => p.type === t) || {}).value || "";
    return `${pick("month")}月${pick("day")}日 ${pick("hour")}:${pick("minute")}`;
  } catch {
    return String(iso || "");
  }
}

/* ---- 地震・警報レーン（cube-server の /quake） ---- */

/* 気象庁の府県予報区コード（警報JSONのURLに使う）。 */
const JMA_PREFECTURE_CODES = {
  "北海道": "010000", "青森県": "020000", "岩手県": "030000",
  "宮城県": "040000", "秋田県": "050000", "山形県": "060000",
  "福島県": "070000", "茨城県": "080000", "栃木県": "090000",
  "群馬県": "100000", "埼玉県": "110000", "千葉県": "120000",
  "東京都": "130000", "神奈川県": "140000", "新潟県": "150000",
  "富山県": "160000", "石川県": "170000", "福井県": "180000",
  "山梨県": "190000", "長野県": "200000", "岐阜県": "210000",
  "静岡県": "220000", "愛知県": "230000", "三重県": "240000",
  "滋賀県": "250000", "京都府": "260000", "大阪府": "270000",
  "兵庫県": "280000", "奈良県": "290000", "和歌山県": "300000",
  "鳥取県": "310000", "島根県": "320000", "岡山県": "330000",
  "広島県": "340000", "山口県": "350000", "徳島県": "360000",
  "香川県": "370000", "愛媛県": "380000", "高知県": "390000",
  "福岡県": "400000", "佐賀県": "410000", "長崎県": "420000",
  "熊本県": "430000", "大分県": "440000", "宮崎県": "450000",
  "鹿児島県": "460000", "沖縄県": "470000",
};

/* 文中の都道府県名（略称でも可）を取り出す。無ければ null。 */
function extractPrefecture(text) {
  const t = String(text || "");
  for (const name of Object.keys(JMA_PREFECTURE_CODES)) {
    if (t.includes(name)) return { name, code: JMA_PREFECTURE_CODES[name] };
  }
  for (const name of Object.keys(JMA_PREFECTURE_CODES)) {
    const short = name.replace(/[都府県]$/, "");
    if (short.length >= 2 && t.includes(short)) {
      return { name, code: JMA_PREFECTURE_CODES[name] };
    }
  }
  return null;
}

/**
 * 地震・警報の質問かどうか。直近の地震そのもの、または警報・
 * 注意報の有無を聞いているときだけ発火する。仕組み・原因・
 * 対処法などの知識質問と、ニュース紹介（話題レーン）は拾わない。
 */
function parseQuakeQuery(question) {
  const q = String(question || "");
  if (!q) return null;
  if (/ニュース|話題|トレンド/.test(q)) return null;
  if (/なぜ|なんで|原因|仕組み|メカニズム|とは|どうすれば|対処|避難|備え/.test(q)) {
    return null;
  }
  if (/警報|注意報/.test(q) &&
      /出てる|出ている|出ています|出てます|出た|ありますか|ある？|ある\?|発令|今|現在|今日/.test(q)) {
    return { kind: "warning", pref: extractPrefecture(q) };
  }
  if (/地震|津波/.test(q) &&
      /さっき|今の|いま|最近|今日|あった|ありました|揺れた|揺れ|大丈夫|震度|どこ|いつ|大き|心配|マグニチュード/.test(q)) {
    return { kind: "quake" };
  }
  return null;
}

/* /quake に投げ、検証済みの形に整えて返す（fetchAddress と同型）。 */
async function fetchQuake(body) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/quake`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      QUAKE_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  if (data.kind === "quake") {
    const items = [];
    for (const it of Array.isArray(data.items) ? data.items : []) {
      if (!it || typeof it.place !== "string" || !it.place ||
          typeof it.scaleLabel !== "string" || !it.scaleLabel) continue;
      items.push({
        time: typeof it.time === "string" ? it.time : "",
        scaleLabel: it.scaleLabel,
        place: it.place,
        magnitude: typeof it.magnitude === "number" ? it.magnitude : null,
        depth: typeof it.depth === "number" ? it.depth : null,
        tsunami: typeof it.tsunami === "string" ? it.tsunami : "",
      });
    }
    return {
      ok: true, kind: "quake", items,
      fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
    };
  }
  if (data.kind === "warning") {
    const warnings = [];
    for (const w of Array.isArray(data.warnings) ? data.warnings : []) {
      if (!w || typeof w.name !== "string" || !w.name) continue;
      warnings.push({ name: w.name });
    }
    return {
      ok: true, kind: "warning",
      pref: typeof data.pref === "string" ? data.pref : "",
      headline: typeof data.headline === "string" ? data.headline : "",
      reportDatetime:
        typeof data.reportDatetime === "string" ? data.reportDatetime : "",
      warnings,
      fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
    };
  }
  return null;
}

/* P2P の発生時刻 "2026/11/20 13:50:00" → "11月20日 13:50"。 */
function quakeTimeLabel(time) {
  const m = String(time || "")
    .match(/(\d{4})\/(\d{1,2})\/(\d{1,2})\s+(\d{1,2}):(\d{2})/);
  if (!m) return String(time || "");
  return `${Number(m[2])}月${Number(m[3])}日 ${m[4]}:${m[5]}`;
}

/* 地震情報の答え文。数字はすべて API の値の転記。 */
function quakeAnswerText(data) {
  if (!data.items.length) {
    return "最近の地震情報が取得できませんでした。";
  }
  return data.items.slice(0, 3).map((it) => {
    let s = `${quakeTimeLabel(it.time)}ごろ、${it.place}で最大震度${it.scaleLabel}の地震がありました`;
    const sub = [];
    if (it.magnitude !== null) sub.push(`マグニチュード${it.magnitude}`);
    if (it.depth !== null) sub.push(`深さ約${it.depth}km`);
    if (sub.length) s += `（${sub.join("、")}）`;
    s += "。";
    if (it.tsunami === "None") s += "津波の心配はありません。";
    else if (it.tsunami === "Watch") s += "津波注意報が出ています。";
    else if (it.tsunami === "Warning") s += "津波警報が出ています。";
    else if (it.tsunami === "NonEffective") {
      s += "若干の海面変動があるかもしれませんが、被害の心配はありません。";
    }
    return s;
  }).join("\n");
}

/* 警報・注意報の答え文。出ている種別は API の集計だけを使う。 */
function warningAnswerText(data) {
  const when = data.reportDatetime
    ? `（${jstDateTimeLabel(data.reportDatetime)}発表）` : "";
  if (!data.warnings.length) {
    return `${data.pref}では、現在発表されている警報・注意報はありません${when}。`;
  }
  const names = [...new Set(data.warnings.map((w) => w.name))];
  let s = `${data.pref}では現在、${names.join("・")}が出ている地域があります${when}。`;
  if (data.headline) s += `\n${data.headline}`;
  return s;
}

/* ---- 最寄り駅レーン（住所レーンの座標 → cube-server の /station） ---- */

/**
 * 「〜の最寄り駅は？」形の質問から対象を取り出す。行き方・
 * ルートの質問と、「近くの〜」のような対象の無い言い方は
 * 拾わない（場所レーンの担当）。
 */
function parseStationQuery(question) {
  const q = String(question || "");
  if (!q) return null;
  if (/行き方|どうやって行く|ルート|経路|乗り換え|作り方|とは/.test(q)) {
    return null;
  }
  const finish = (raw) => {
    const t = cleanAddressTarget(raw);
    if (!t || t.length > 40) return null;
    if (/[、。？！?!]/.test(t)) return null;
    if (/^(近く|近所|周辺|ここ|そこ|それ|これ|何|なに)/.test(t)) return null;
    return { target: t };
  };
  let m = q.match(/^(.+?)の最寄り?駅(?:は|を|って|が|に)?/);
  if (m) return finish(m[1]);
  m = q.match(/^(.+?)から(?:一番|いちばん)?近い駅/);
  if (m) return finish(m[1]);
  return null;
}

/* /station に座標を投げ、検証済みの形に整えて返す。 */
async function fetchStation(lat, lng) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/station`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng }),
      },
      STATION_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  const results = [];
  for (const r of Array.isArray(data.results) ? data.results : []) {
    if (!r || typeof r.name !== "string" || !r.name) continue;
    results.push({
      name: r.name,
      line: typeof r.line === "string" ? r.line : "",
      prefecture: typeof r.prefecture === "string" ? r.prefecture : "",
      distance: typeof r.distance === "string" ? r.distance : "",
    });
  }
  if (!results.length) return null;
  return { ok: true, results };
}

/* 最寄り駅の答え文（一番近い駅を本文、残りはカード）。 */
function stationAnswerText(target, data) {
  const r = data.results[0];
  let s = `「${target}」の最寄り駅は、${r.name}駅`;
  if (r.line) s += `（${r.line}）`;
  if (r.distance) s += `で、約${r.distance}のところにあります。`;
  else s += "です。";
  if (data.results.length > 1) {
    s += `ほかの近い駅はカードに表示しています。`;
  }
  return s;
}

/* 駅1件を表示用カードにする（タイトルが地図へのリンク）。 */
function stationCardOf(r) {
  return {
    kind: "station",
    name: r.name,
    line: r.line,
    distance: r.distance,
    prefecture: r.prefecture,
    link: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${r.name}駅`)}`,
  };
}

/* ---- 祝日レーン（cube-server の /holiday） ---- */

/**
 * 祝日の質問かどうか。今日・明日・明後日・次の祝日・日付指定・
 * 一覧の形だけを拾い、店の営業質問や意味の質問は拾わない。
 */
function parseHolidayQuery(question) {
  const q = String(question || "");
  if (!q || !/祝日/.test(q)) return null;
  if (/営業|とは|意味/.test(q)) return null;
  if ((/一覧|全部|今年の祝日/.test(q)) && !/次の|今日|明日/.test(q)) {
    return { kind: "list" };
  }
  const m = q.match(/(\d{1,2})月(\d{1,2})日/);
  if (m) {
    const month = Number(m[1]), day = Number(m[2]);
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
      return { kind: "date", month, day };
    }
  }
  if (/明後日/.test(q)) return { kind: "offset", days: 2 };
  if (/明日/.test(q)) return { kind: "offset", days: 1 };
  if (/昨日/.test(q)) return { kind: "offset", days: -1 };
  if (/次の|今度|直近/.test(q)) return { kind: "next" };
  if (/今日|本日/.test(q)) return { kind: "offset", days: 0 };
  if (/(?:は|が|って)いつ|いつ[？?ですか]/.test(q)) {
    return { kind: "next" };
  }
  return null;
}

/* /holiday から祝日表（"YYYY-MM-DD" → 名前）を取得する。 */
async function fetchHoliday() {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/holiday`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
      HOLIDAY_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  if (!data.holidays || typeof data.holidays !== "object") return null;
  const holidays = {};
  for (const [k, v] of Object.entries(data.holidays)) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(k) && typeof v === "string" && v) {
      holidays[k] = v;
    }
  }
  if (!Object.keys(holidays).length) return null;
  return { ok: true, holidays };
}

function jstDatePlus(days) {
  return new Date(Date.now() + days * 86400000);
}

/* 祝日の答え文。判定はすべて祝日表（API の値）に対して行う。 */
function holidayAnswerText(parsed, holidays) {
  if (parsed.kind === "offset") {
    const key = pointsJstDateKey(jstDatePlus(parsed.days));
    const [, mo, d] = key.split("-");
    const label = parsed.days === 0 ? "今日"
      : parsed.days === 1 ? "明日"
      : parsed.days === 2 ? "明後日" : "昨日";
    const dateLabel = `${Number(mo)}月${Number(d)}日`;
    const name = holidays[key];
    return name
      ? `${label}（${dateLabel}）は、${name}で祝日です。`
      : `${label}（${dateLabel}）は祝日ではありません。`;
  }
  if (parsed.kind === "date") {
    const y = Number(pointsJstDateKey().slice(0, 4));
    const key = `${y}-${String(parsed.month).padStart(2, "0")}-` +
      String(parsed.day).padStart(2, "0");
    const name = holidays[key];
    return name
      ? `${y}年${parsed.month}月${parsed.day}日は、${name}で祝日です。`
      : `${y}年${parsed.month}月${parsed.day}日は祝日ではありません。`;
  }
  if (parsed.kind === "next") {
    for (let i = 0; i <= 370; i++) {
      const key = pointsJstDateKey(jstDatePlus(i));
      if (holidays[key]) {
        const [, mo, d] = key.split("-");
        const when = i === 0 ? "今日"
          : i === 1 ? "明日" : `${Number(mo)}月${Number(d)}日`;
        return `次の祝日は${when}、${holidays[key]}です（${key.replace(/-/g, "/")}）。`;
      }
    }
    return "次の祝日は見つかりませんでした。";
  }
  const y = pointsJstDateKey().slice(0, 4);
  const keys = Object.keys(holidays)
    .filter((k) => k.startsWith(`${y}-`)).sort();
  if (!keys.length) return "今年の祝日データが取得できませんでした。";
  return `${y}年の祝日です。\n` + keys.map((k) => {
    const [, mo, d] = k.split("-");
    return `・${Number(mo)}月${Number(d)}日：${holidays[k]}`;
  }).join("\n");
}

/* ---- 為替・株価レーン（cube-server の /market） ---- */

/* 通貨の日本語名 → ISOコード（円への換算・ペア判定に使う）。 */
const FX_CURRENCY_NAMES = {
  "香港ドル": "HKD", "シンガポールドル": "SGD",
  "オーストラリアドル": "AUD", "ニュージーランドドル": "NZD",
  "カナダドル": "CAD", "スイスフラン": "CHF",
  "南アフリカランド": "ZAR", "インドルピー": "INR",
  "ブラジルレアル": "BRL", "メキシコペソ": "MXN",
  "トルコリラ": "TRY", "人民元": "CNY", "豪ドル": "AUD",
  "NZドル": "NZD", "ドル": "USD", "ユーロ": "EUR",
  "ポンド": "GBP", "ウォン": "KRW", "元": "CNY",
  "フラン": "CHF", "ランド": "ZAR", "バーツ": "THB",
  "ルピー": "INR", "ペソ": "MXN", "リラ": "TRY", "レアル": "BRL",
};

/* 「〜円」のペア名 → 円に対する通貨コード。 */
const FX_PAIR_NAMES = [
  ["ドル円", "USD"], ["ユーロ円", "EUR"], ["ポンド円", "GBP"],
  ["豪ドル円", "AUD"], ["オーストラリアドル円", "AUD"],
  ["カナダドル円", "CAD"], ["スイスフラン円", "CHF"],
  ["人民元円", "CNY"], ["ウォン円", "KRW"], ["NZドル円", "NZD"],
  ["ニュージーランドドル円", "NZD"], ["香港ドル円", "HKD"],
];

/* 会社名・銘柄名 → 証券コード（Yahoo Finance のシンボル）。 */
const STOCK_NAME_CODES = {
  "トヨタ自動車": "7203.T", "トヨタ": "7203.T",
  "ソニーグループ": "6758.T", "ソニー": "6758.T",
  "任天堂": "7974.T", "本田技研": "7267.T", "ホンダ": "7267.T",
  "日産自動車": "7201.T", "日産": "7201.T",
  "パナソニック": "6752.T", "キーエンス": "6861.T",
  "ソフトバンクグループ": "9984.T", "ソフトバンク": "9984.T",
  "三菱UFJ": "8306.T", "三菱ＵＦＪ": "8306.T",
  "ファーストリテイリング": "9983.T", "ユニクロ": "9983.T",
  "セブン&アイ": "3382.T", "セブン＆アイ": "3382.T",
  "イオン": "8267.T", "楽天グループ": "4755.T", "楽天": "4755.T",
  "メルカリ": "4385.T", "NTT": "9432.T", "KDDI": "9433.T",
  "JR東日本": "9020.T", "ANA": "9202.T", "JAL": "9201.T",
  "東京電力": "9501.T", "関西電力": "9503.T",
  "武田薬品": "4502.T", "日本郵船": "9101.T",
  "アップル": "AAPL", "Apple": "AAPL",
  "マイクロソフト": "MSFT", "Microsoft": "MSFT",
  "エヌビディア": "NVDA", "NVIDIA": "NVDA",
  "テスラ": "TSLA", "Tesla": "TSLA",
  "アマゾン": "AMZN", "Amazon": "AMZN",
  "グーグル": "GOOGL", "Google": "GOOGL", "アルファベット": "GOOGL",
  "メタ": "META", "Meta": "META",
  "ネットフリックス": "NFLX", "Netflix": "NFLX",
  "ディズニー": "DIS", "Disney": "DIS",
  "コカ・コーラ": "KO", "コカコーラ": "KO",
  "マクドナルド": "MCD", "スターバックス": "SBUX", "ナイキ": "NKE",
};

/**
 * 為替・株価の質問かどうか。ペア名・換算（「100ドルは何円？」）・
 * 株価（会社名辞書・証券コード）の形だけを拾う。仕組みや理由を
 * 聞く質問は拾わない。会社名が辞書に無く対象だけ分かる場合は
 * stock_unknown を返し、レーン側で正直に完結させる（モデルの
 * 推測価格を出させないため）。
 */
function parseMarketQuery(question) {
  const q = String(question || "");
  if (!q) return null;
  if (/とは|仕組み|買い方|売り方|始め方|なぜ|なんで|理由|要因|ニュース|話題/.test(q)) {
    return null;
  }
  let m = q.match(/([\d,]+(?:\.\d+)?)\s*(香港ドル|シンガポールドル|オーストラリアドル|ニュージーランドドル|カナダドル|スイスフラン|南アフリカランド|インドルピー|ブラジルレアル|メキシコペソ|トルコリラ|人民元|豪ドル|NZドル|ドル|ユーロ|ポンド|ウォン|元|フラン|ランド|バーツ|ルピー|ペソ|リラ|レアル)は(?:今)?(?:何円|いくら)/);
  if (m) {
    const code = FX_CURRENCY_NAMES[m[2]];
    if (code && code !== "JPY") {
      return {
        kind: "fxconvert",
        amount: Number(m[1].replace(/,/g, "")),
        from: code,
        fromName: m[2],
      };
    }
  }
  if (/いくら|レート|何円|今|教えて|ですか|は？|は\?/.test(q)) {
    for (const [name, code] of FX_PAIR_NAMES) {
      if (q.includes(name)) {
        return { kind: "fx", base: code, quote: "JPY", pairName: name };
      }
    }
    const names = Object.keys(FX_CURRENCY_NAMES)
      .sort((a, b) => b.length - a.length);
    for (const name of names) {
      if (q.includes(name)) {
        const code = FX_CURRENCY_NAMES[name];
        return { kind: "fx", base: code, quote: "JPY", pairName: `${name}円` };
      }
    }
  }
  if (/株価/.test(q)) {
    const names = Object.keys(STOCK_NAME_CODES)
      .sort((a, b) => b.length - a.length);
    for (const name of names) {
      if (q.includes(name)) {
        return { kind: "stock", symbol: STOCK_NAME_CODES[name], name };
      }
    }
    m = q.match(/(\d{4})(?:\.T)?の株価/);
    if (m) {
      return { kind: "stock", symbol: `${m[1]}.T`, name: `証券コード${m[1]}` };
    }
    m = q.match(/([A-Z]{2,5})の株価/);
    if (m) return { kind: "stock", symbol: m[1], name: m[1] };
    m = q.match(/^(.+?)の株価/);
    if (m) {
      const t = m[1].trim();
      if (t.length >= 2 && t.length <= 20 &&
          !/^(今日|今|全体|日本|アメリカ|米国|最近)/.test(t)) {
        return { kind: "stock_unknown", name: t };
      }
    }
  }
  return null;
}

/* /market に投げ、検証済みの形に整えて返す。 */
async function fetchMarket(body) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/market`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      MARKET_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  if (data.kind === "fx") {
    if (typeof data.rate !== "number" || !Number.isFinite(data.rate)) {
      return null;
    }
    return {
      ok: true, kind: "fx",
      base: typeof data.base === "string" ? data.base : "",
      quote: typeof data.quote === "string" ? data.quote : "",
      rate: data.rate,
      date: typeof data.date === "string" ? data.date : "",
      fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
    };
  }
  if (data.kind === "stock") {
    if (typeof data.price !== "number" || !Number.isFinite(data.price)) {
      return null;
    }
    return {
      ok: true, kind: "stock",
      symbol: typeof data.symbol === "string" ? data.symbol : "",
      name: typeof data.name === "string" ? data.name : "",
      price: data.price,
      prevClose: typeof data.prevClose === "number" ? data.prevClose : null,
      change: typeof data.change === "number" ? data.change : null,
      changePct: typeof data.changePct === "number" ? data.changePct : null,
      currency: typeof data.currency === "string" ? data.currency : "",
      marketTime: typeof data.marketTime === "string" ? data.marketTime : "",
      fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
    };
  }
  return null;
}

/* レートの表示（小数3桁まで、末尾の0は落とす）。 */
function rateText(rate) {
  if (typeof rate !== "number" || !Number.isFinite(rate)) return "";
  return String(Number(rate.toFixed(3)));
}

/* Frankfurter の基準日 "2026-11-19" → "2026年11月19日"。 */
function fxDateLabel(date) {
  const m = String(date || "").match(/(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return String(date || "");
  return `${Number(m[1])}年${Number(m[2])}月${Number(m[3])}日`;
}

/* 為替の答え文。1単位で小さすぎるレートは100単位で出す。 */
function marketFxText(data, pairName) {
  const when = data.date ? `（${fxDateLabel(data.date)}時点）` : "";
  if (data.rate < 1) {
    return `${pairName}は、100${data.base}＝${rateText(data.rate * 100)}円です${when}。`;
  }
  return `${pairName}は、1${data.base}＝${rateText(data.rate)}円です${when}。`;
}

/* 株価の答え文。価格・前日比はすべて API の値の転記。 */
function marketStockText(data, displayName) {
  const name = displayName || data.name || data.symbol;
  const price = data.price
    .toLocaleString("ja-JP", { maximumFractionDigits: 2 });
  const unit = data.currency === "JPY" ? "円" : ` ${data.currency}`;
  let s = `${name}（${data.symbol}）の株価は、${price}${unit}です。`;
  if (data.change !== null && data.changePct !== null) {
    const sign = data.change > 0 ? "+" : "";
    const chg = data.change
      .toLocaleString("ja-JP", { maximumFractionDigits: 2 });
    s += `前日比は${sign}${chg}（${sign}${data.changePct.toFixed(2)}%）です。`;
  }
  if (data.marketTime) s += `（${jstDateTimeLabel(data.marketTime)}時点）`;
  return s;
}

/* ---- 音楽検索レーン（cube-server の /music） ---- */

/**
 * 曲の検索・試聴の質問かどうか。アーティスト名＋「の曲を
 * 検索／教えて／聴きたい」等の形だけを拾う。歌詞の質問は
 * 拾わない（歌詞そのものは出さない）。
 */
function parseMusicQuery(question) {
  const q = String(question || "");
  if (!q) return null;
  if (/歌詞|作り方|作曲|とは|意味|行き方/.test(q)) return null;
  const finish = (raw) => {
    const t = String(raw || "")
      .trim()
      .replace(/^[「『]/, "")
      .replace(/[」』]$/, "")
      .replace(/(の|は|が|って)$/, "")
      .trim();
    if (!t || t.length > 40) return null;
    if (/^(それ|これ|何|なに)/.test(t)) return null;
    return { query: t };
  };
  let m = q.match(/^(.+?)の(?:曲|歌|楽曲)を(?:検索|探して|探す|教えて|知りたい|聴きたい|聞きたい|聞かせて)/);
  if (m) return finish(m[1]);
  m = q.match(/^(.+?)の(?:曲|歌|楽曲)(?:を|は|って)?(?:試聴|検索)/);
  if (m) return finish(m[1]);
  m = q.match(/^(.+?)の(?:曲|歌|楽曲)(?:は|って)\s*[？?]\s*$/);
  if (m) return finish(m[1]);
  m = q.match(/^(.+?)を試聴(?:したい|させて|できますか|できる)?/);
  if (m) return finish(m[1]);
  m = q.match(/^(.+?)の試聴(?:を|は|したい|させて)?/);
  if (m) return finish(m[1]);
  return null;
}

/* /music に投げ、検証済みの形に整えて返す。 */
async function fetchMusic(query) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/music`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      },
      MUSIC_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  const results = [];
  for (const r of Array.isArray(data.results) ? data.results : []) {
    if (!r || typeof r.track !== "string" || !r.track ||
        typeof r.artist !== "string" || !r.artist) continue;
    results.push({
      track: r.track,
      artist: r.artist,
      album: typeof r.album === "string" ? r.album : "",
      trackUrl: httpUrlOrEmpty(typeof r.trackUrl === "string" ? r.trackUrl : ""),
      previewUrl:
        httpUrlOrEmpty(typeof r.previewUrl === "string" ? r.previewUrl : ""),
      artwork: httpUrlOrEmpty(typeof r.artwork === "string" ? r.artwork : ""),
    });
  }
  if (!results.length) return null;
  return { ok: true, results };
}

/* 曲1件を表示用カードにする。 */
function musicCardOf(r) {
  return {
    kind: "music",
    track: r.track,
    artist: r.artist,
    album: r.album,
    link: r.trackUrl,
    preview: r.previewUrl,
    artwork: r.artwork,
  };
}

/* ---- 論文検索レーン（cube-server の /papers） ---- */
/* 発動条件の正本はサーバー側 papers_endpoint.js の matchPapersLane。
   ここはその移植で、学術意図が明示的な質問だけを拾う。誤爆ガード
   （自由研究・小論文・研究所などの施設名・場所の質問）もサーバー
   と同一。検索語の抽出もサーバー側で行われるが、表示文のために
   ページ側でも同じ抽出を行う。結果は変わる情報なので知識メモ帳
   には保存しない（他のデータレーンと同じ）。 */

/* 質問文からフィラーを除いて検索語を作る（サーバー側と同一）。 */
const PAPERS_FILLER_PATTERNS = [
  /に関する(論文|研究|学術論文)/g,
  /(の|に)(関する|ついての?)/g,
  /(論文|研究|学術論文)を?(探して|検索して|教えて|知りたい|見せて|紹介して|ありますか|ある\?|ある？)/g,
  /(論文|研究)(を|が|は|も|に)/g,
  /(について|とは|って何|の動向|の最先端|の成果|のテーマ|の分野)/g,
  /\b(recent\s+)?(papers?|research|studies|literature)\s+(on|about|regarding)\b/gi,
  /\b(research\s+)?papers?\b/gi,
  /\barxiv\b/gi,
  /(教えて|知りたい|探して|検索して|紹介して|見せて|ありますか|あります|ください)/g,
  /[？?！!。、\s]+/g,
];

function extractPapersTerms(text) {
  let t = String(text || "");
  for (const re of PAPERS_FILLER_PATTERNS) t = t.replace(re, " ");
  /* 汎用の学術語は意図の印であって検索語ではない。 */
  t = t.replace(/学術論文|論文|研究|査読付き|査読/g, " ");
  t = t.replace(/の?\s*(動向|成果|最先端)/g, " ");
  const tokens = t.split(/\s+/).map((tok) =>
    tok.replace(/^[をがはのにでへとも、。]+/, "")
      .replace(/[をがはのにでへとも、。]+$/, ""),
  ).filter(Boolean);
  return tokens.join(" ").slice(0, 120);
}

/**
 * 論文を探す質問かどうか。学術意図が明示的なときだけ発火し、
 * { query（元の質問）, terms（抽出した検索語） } を返す。
 */
function parsePapersQuery(question) {
  const raw = String(question || "").trim();
  if (!raw) return null;
  const lower = raw.toLowerCase();

  /* --- 誤爆ガード（サーバー側 matchPapersLane と同一） --- */
  const hasStrong = /論文|arxiv|学術|査読/i.test(raw);
  if (/自由研究/.test(raw) && !hasStrong) return null;
  if (/小論文/.test(raw) && !/学術|査読|arxiv/i.test(raw)) return null;
  if (/(研究所|研究室|研究会)/.test(raw) && !hasStrong) return null;
  if (/(はどこ|への行き方|までの道|の場所|の住所)/.test(raw) && !hasStrong) {
    return null;
  }

  /* --- 発火条件 --- */
  let fire = false;
  if (/論文/.test(raw) && !/小論文/.test(raw)) fire = true;
  if (/\barxiv\b/i.test(raw)) fire = true;
  if (/学術/.test(raw)) fire = true;
  if (/査読/.test(raw)) fire = true;
  if (/研究/.test(raw) &&
      /(に関する研究|の研究|研究(を|が|は|も|の)|研究されている|研究が進)/.test(raw)) {
    fire = true;
  }
  if (/\b(papers?\s+(on|about)|research\s+paper|studies\s+on|research\s+on|scientific\s+literature)\b/i.test(lower)) {
    fire = true;
  }
  if (!fire) return null;
  return { query: raw, terms: extractPapersTerms(raw) };
}

/* /papers に投げ、検証済みの形に整えて返す。 */
async function fetchPapers(query) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/papers`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      },
      PAPERS_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  const results = [];
  for (const r of Array.isArray(data.results) ? data.results : []) {
    if (!r || typeof r.title !== "string" || !r.title) continue;
    results.push({
      title: r.title,
      authors: (Array.isArray(r.authors) ? r.authors : [])
        .filter((a) => typeof a === "string" && a),
      year: typeof r.year === "number" || typeof r.year === "string"
        ? r.year : "",
      summary: typeof r.summary === "string" ? r.summary : "",
      url: httpUrlOrEmpty(typeof r.url === "string" ? r.url : ""),
      source: typeof r.source === "string" ? r.source : "",
    });
  }
  if (!results.length) return null;
  return {
    ok: true,
    results,
    source: typeof data.source === "string" ? data.source : "",
    attribution:
      typeof data.attribution === "string" ? data.attribution : "",
    fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
  };
}

/* 論文1件を表示用カードにする。 */
function paperCardOf(r) {
  return {
    kind: "paper",
    title: r.title,
    authors: r.authors.join("、"),
    year: r.year,
    summary: r.summary,
    link: r.url,
    source: r.source,
  };
}

/* ---- 翻訳レーン（cube-server の /translate、DeepL） ---- */
/* 発動条件の正本はサーバー側 deepl_endpoint.js の matchTranslateLane。
   ここはその忠実な移植で、翻訳の意図が明示的な質問だけを拾う。
   翻訳の話題・相談（翻訳アプリのおすすめ、翻訳の仕事など）は
   発火しない。原文の取り出しは引用符の中身が最優先で、無ければ
   命令部分を取り除いた残り。指示語だけ（「この文章を翻訳して」）
   の場合は原文なしのまま発火させ、サーバー側の no_source_text
   の正直な案内に委ねる。DeepL の無料枠は通算100万字の一回きり
   （リセットなし）なので、成功表示には残量（usage）を必ず添える。
   結果は変わる情報なので知識メモ帳には保存しない（他のデータ
   レーンと同じ）。 */

/* 言語名 -> DeepL の target_lang コード（サーバー側と同一）。 */
const TRANSLATE_LANG_CODES = {
  "英語": "EN-US", "英": "EN-US",
  "日本語": "JA",
  "中国語": "ZH",
  "韓国語": "KO",
  "フランス語": "FR",
  "ドイツ語": "DE",
  "スペイン語": "ES",
  "イタリア語": "IT",
  "ポルトガル語": "PT-PT",
  "ロシア語": "RU",
  "オランダ語": "NL",
};
const TRANSLATE_LANG_CODES_EN = {
  english: "EN-US", japanese: "JA", chinese: "ZH", korean: "KO",
  french: "FR", german: "DE", spanish: "ES", italian: "IT",
  portuguese: "PT-PT", russian: "RU", dutch: "NL",
};
const TRANSLATE_LANG_WORDS = Object.keys(TRANSLATE_LANG_CODES).join("|");

/* DeepL の言語コード -> 表示用の日本語名。原文の言語は DeepL の
   検出結果（"JA" や "EN" のような基底コード）で返るため、訳先の
   地域つきコードと両方を引けるようにしてある。 */
const TRANSLATE_LANG_NAMES = {
  "EN-US": "英語", "EN": "英語",
  "JA": "日本語",
  "ZH": "中国語",
  "KO": "韓国語",
  "FR": "フランス語",
  "DE": "ドイツ語",
  "ES": "スペイン語",
  "IT": "イタリア語",
  "PT-PT": "ポルトガル語", "PT": "ポルトガル語",
  "RU": "ロシア語",
  "NL": "オランダ語",
};

function translateLangName(code) {
  return TRANSLATE_LANG_NAMES[code] || "";
}

/* 原文が指示語だけ（「この文章を翻訳して」）なら訳す文が無い。 */
const TRANSLATE_REFER_ONLY =
  /^(この文章|この文|これ|それ|あれ|次の文|次の文章|以下|上記)$/;
const TRANSLATE_HAS_JA = /[\u3040-\u30ff\u4e00-\u9faf]/;

function cleanTranslateSource(s) {
  return String(s || "")
    .replace(/^[「『"“\s]+/, "")
    .replace(/[」』"”\s]+$/, "")
    .replace(/^[をがはのにでへとも、。\s]+/, "")
    .replace(/[、。！？!?\s]+$/, "")
    .trim();
}

/* 言語名が無いときの訳先：日本語の文なら英語、それ以外は日本語。 */
function defaultTranslateTarget(sourceText) {
  return TRANSLATE_HAS_JA.test(sourceText) ? "EN-US" : "JA";
}

/* マッチャーの共通の締め（サーバー側 finish の移植）。発火しない
   ときは null、発火するときは { sourceText, targetLang } を返す。
   原文が取れない場合は sourceText を空にして発火させる。 */
function finishTranslate(raw, sourceRaw, targetLang, aboutTopic, quotedText) {
  /* 引用符の原文があれば、分解した原文より優先する（サーバー側と同一）。 */
  const sourceText = cleanTranslateSource(quotedText || sourceRaw);
  if (aboutTopic && !quotedText) return null;
  if (!sourceText || TRANSLATE_REFER_ONLY.test(sourceText)) {
    return {
      sourceText: "",
      targetLang: targetLang || defaultTranslateTarget(raw),
    };
  }
  return {
    sourceText,
    targetLang: targetLang || defaultTranslateTarget(sourceText),
  };
}

/* サーバー側 matchTranslateLane の本体の移植。最初に一致した
   パターンが判定を決める（サーバー側の早期 return と同じ）。
   発火しないときは null、発火するときは
   { sourceText, targetLang } を返す。 */
function matchTranslateIntent(raw) {
  /* 引用符つきの原文は最も確実な信号。 */
  const quoted = raw.match(/[「『]([^」』]+)[」』]/) || raw.match(/"([^"]+)"/);
  const quotedText = quoted ? quoted[1].trim() : "";

  /* --- 誤爆ガード：翻訳の「話題」は発火しない（引用符つき原文と
     命令形が両方ある場合だけ例外的に発火する） --- */
  const aboutTopic =
    /(翻訳アプリ|翻訳ソフト|翻訳サイト|翻訳サービス|翻訳ツール|翻訳機|翻訳者|翻訳の仕事|翻訳会社|翻訳料金|翻訳精度|機械翻訳|翻訳とは|翻訳って)/.test(raw);
  let m;

  /* --- 日本語の命令形 --- */
  /* Xを<言語名>に翻訳して / Xを<言語名>に訳して */
  if ((m = raw.match(new RegExp(`^(.+?)を(${TRANSLATE_LANG_WORDS})に(?:翻訳|訳)(?:して|する|してほしい|してください)?`)))) {
    return finishTranslate(raw, m[1], TRANSLATE_LANG_CODES[m[2]], aboutTopic, quotedText);
  }
  /* Xを英訳して / Xを和訳して / Xを翻訳して */
  if ((m = raw.match(/^(.+?)を(英訳|和訳|翻訳)(?:して|する|してほしい|してください)?/))) {
    const target = m[2] === "英訳" ? "EN-US" : m[2] === "和訳" ? "JA" : "";
    return finishTranslate(raw, m[1], target, aboutTopic, quotedText);
  }
  /* Xを<言語名>にして（言語名が必須なので「赤にして」は発火しない） */
  if ((m = raw.match(new RegExp(`^(.+?)を(${TRANSLATE_LANG_WORDS})にして`)))) {
    return finishTranslate(raw, m[1], TRANSLATE_LANG_CODES[m[2]], aboutTopic, quotedText);
  }
  /* Xは<言語名>で何と言う / Xを<言語名>で言うと / Xって<言語名>で何 */
  if ((m = raw.match(new RegExp(`^(.+?)(?:は|を|って)(${TRANSLATE_LANG_WORDS})で(?:何と|なんて|どう)?(?:言|いう|いえば)`)))) {
    return finishTranslate(raw, m[1], TRANSLATE_LANG_CODES[m[2]], aboutTopic, quotedText);
  }
  /* 翻訳して：X / 英訳して: X（命令が先・コロン・原文） */
  if ((m = raw.match(/^(翻訳|英訳|和訳)(?:して)?[：:]\s*(.+)$/))) {
    const target = m[1] === "英訳" ? "EN-US" : m[1] === "和訳" ? "JA" : "";
    return finishTranslate(raw, m[2], target, aboutTopic, quotedText);
  }
  /* 引用符つき原文＋裸の命令形（「Hello」を翻訳して） */
  if (quotedText && /(翻訳して|英訳して|和訳して|訳して)/.test(raw)) {
    let target = "";
    const lm = raw.match(new RegExp(`(${TRANSLATE_LANG_WORDS})に(?:翻訳|訳)`));
    if (lm) target = TRANSLATE_LANG_CODES[lm[1]];
    else if (/英訳/.test(raw)) target = "EN-US";
    else if (/和訳/.test(raw)) target = "JA";
    return finishTranslate(raw, quotedText, target, aboutTopic, quotedText);
  }

  /* --- 英語の命令形（原文の大文字小文字は保つ） --- */
  if ((m = raw.match(/^translate\s+(.+?)\s+(?:in)?to\s+([a-zA-Z]+)\s*$/i))) {
    const target = TRANSLATE_LANG_CODES_EN[m[2].toLowerCase()] || "";
    if (target) return finishTranslate(raw, m[1], target, aboutTopic, quotedText);
  }
  if ((m = raw.match(/^how do you say\s+(.+?)\s+in\s+([a-zA-Z]+)\s*\??\s*$/i))) {
    const target = TRANSLATE_LANG_CODES_EN[m[2].toLowerCase()] || "";
    if (target) return finishTranslate(raw, m[1], target, aboutTopic, quotedText);
  }
  if ((m = raw.match(/^translate\s+(.+?)\s*$/i))) {
    return finishTranslate(raw, m[1], "", aboutTopic, quotedText);
  }

  return null;
}

/**
 * 翻訳の質問かどうか。翻訳の意図が明示的なときだけ発火し、
 * { query（元の質問）, sourceText（取り出した原文）,
 *   targetLang（DeepL の言語コード） } を返す。
 */
function parseTranslateQuery(question) {
  const raw = String(question || "").trim();
  if (!raw) return null;
  const hit = matchTranslateIntent(raw);
  if (!hit) return null;
  return { query: raw, sourceText: hit.sourceText, targetLang: hit.targetLang };
}

/* /translate に投げ、検証済みの形に整えて返す。usage（DeepL の
   残量）は表示のために必ず保持する。 */
async function fetchTranslate(parsed) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/translate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: parsed.query }),
      },
      TRANSLATE_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  if (typeof data.translatedText !== "string" || !data.translatedText) {
    return null;
  }
  const usage = data.usage && typeof data.usage === "object" ? {
    character_count: typeof data.usage.character_count === "number"
      ? data.usage.character_count : null,
    character_limit: typeof data.usage.character_limit === "number"
      ? data.usage.character_limit : null,
    character_remaining: typeof data.usage.character_remaining === "number"
      ? data.usage.character_remaining : null,
  } : null;
  return {
    ok: true,
    sourceText: typeof data.sourceText === "string" && data.sourceText
      ? data.sourceText : parsed.sourceText,
    translatedText: data.translatedText,
    sourceLang: typeof data.sourceLang === "string" ? data.sourceLang : "",
    targetLang: typeof data.targetLang === "string" && data.targetLang
      ? data.targetLang : parsed.targetLang,
    chars: typeof data.chars === "number" ? data.chars : 0,
    usage,
    localTotalChars: typeof data.localTotalChars === "number"
      ? data.localTotalChars : null,
    source: typeof data.source === "string" && data.source
      ? data.source : "DeepL",
    fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
  };
}

/* 翻訳結果を表示用カードにする。 */
function translateCardOf(d) {
  return {
    kind: "translate",
    sourceText: d.sourceText,
    translatedText: d.translatedText,
    sourceLang: d.sourceLang,
    targetLang: d.targetLang,
  };
}

/* ---- Wolfram 計算レーン（cube-server の /calc-wolfram） ---- */
/* 計算の主軸（2026-11-22 本人決定）。式・方程式・単位・割合・
   累乗はサーバー側で決定的に Wolfram クエリへ変換し、文章題は
   サーバー側で Ultra に変換だけさせてから Wolfram|Alpha に
   計算させる。失敗・未設定・解釈不能のときは従来の計算レーン
   （CalcLane / evalArithmetic）へフォールバックする。
   規約上の注意：結果はキャッシュしない（メモ帳に保存しない）、
   答えには Wolfram|Alpha の結果ページへのリンクを必ず出す、
   AppID はサーバー側だけに置く（ページには置かない）。
   発動条件の正本はサーバー側 matchCalcWolfram
   （wolfram_endpoint.js）で、以下はその移植。変換そのものは
   サーバーが行うため、ここでは発火の形だけを判定する。 */

const WOLFRAM_UNIT_ALT =
  "平方センチメートル|平方メートル|キロメートル|センチメートル|" +
  "ミリメートル|メートル|キログラム|グラム|デシリットル|" +
  "ミリリットル|リットル|ポンド|フィート|インチ|マイル|ヤード|" +
  "時間|分|秒";
const WOLFRAM_UNIT_RE = new RegExp(
  `[\\d.]+\\s*(?:${WOLFRAM_UNIT_ALT})\\s*(?:は|を)\\s*` +
  `(?:何|なん|いくつ)?\\s*(?:${WOLFRAM_UNIT_ALT})`);
const WOLFRAM_WORD_KEYWORDS =
  /(全部で|合計|合わせて|おつり|残り|差は|時速|速さ|平均|単価|割り勘|等分|分けると|分けた|配る|配ると|払い|払う|買うと|買った|値段|金額|いくら|いくつ|何個|何本|何枚|何人|何リットル|何メートル|何キロ|何グラム|何時間|何分|何冊|何台|何匹|何組|何箱|何円)/;

/* サーバー側 normCalc と同じ正規化（全角→半角）。 */
function wolframNormCalc(s) {
  return String(s || "")
    .replace(/[０-９]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0))
    .replace(/＋/g, "+")
    .replace(/[－−]/g, "-")
    .replace(/×/g, "*")
    .replace(/÷/g, "/")
    .replace(/＝/g, "=")
    .replace(/％/g, "%")
    .replace(/（/g, "(")
    .replace(/）/g, ")")
    .replace(/．/g, ".")
    .replace(/＾/g, "^")
    .replace(/　/g, " ");
}

/* サーバー側 buildWolframQuery が変換できる形かどうか
   （方程式・単位・割合・累乗・式のいずれか）。 */
function hasWolframDeterministicShape(t) {
  // 方程式: "=" を含み、文字と数字の両方がある断片
  const s = t.replace(/([a-zA-Z])の(\d+)乗/g, "$1^$2");
  const segments = s.match(/[0-9a-zA-Z.+\-*/^()= ]+/g) || [];
  for (const seg of segments) {
    const e = seg.trim();
    if (e.includes("=") && /[a-zA-Z]/.test(e) && /\d/.test(e)) return true;
  }
  // 単位: 温度
  if (/(摂氏|華氏)\s*[\d.]+\s*度?\s*は\s*(摂氏|華氏)\s*何度/.test(t)) {
    return true;
  }
  // 単位: A <単位> は何 <単位>
  if (WOLFRAM_UNIT_RE.test(t)) return true;
  // 割合・百分率
  if (/[\d.]+\s*は\s*[\d.]+\s*の\s*何\s*%/.test(t)) return true;
  if (/[\d.]+\s*から\s*[\d.]+\s*(?:へ|に)\s*の?\s*(?:増加率|減少率)/.test(t)) {
    return true;
  }
  if (/[\d.]+\s*(?:人|個|枚|本|冊|台)?\s*のうち\s*[\d.]+/.test(t)) return true;
  if (/[\d.]+\s*を\s*百分率/.test(t)) return true;
  if (/[\d.]+\s*円[^。]*?(?:消費税|税金|税)\s*[\d.]+\s*%/.test(t)) return true;
  if (/[\d.]+\s*%\s*の\s*(?:消費税|税金|税)[^。]*?[\d.]+\s*円/.test(t)) {
    return true;
  }
  if (/[\d.]+\s*円[^。]*?[\d.]+\s*%\s*(?:引き|割引)/.test(t)) return true;
  if (/[\d.]+\s*%\s*(?:引き|割引)[^。]*?[\d.]+\s*円/.test(t)) return true;
  if (/[\d.]+\s*の\s*[\d.]+\s*%/.test(t)) return true;
  // 累乗・平方根
  if (/[\d.]+\s*の\s*-?[\d.]+\s*乗/.test(t)) return true;
  if (/[\d.]+\s*の\s*平方根/.test(t)) return true;
  if (/(?:√|ルート)\s*[\d.]+/.test(t)) return true;
  // 式: 小数・括弧を含む最大の式ラン（日付の形は除く）
  if (!/\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}/.test(t)) {
    const runs = t.match(/[0-9.+\-*/()]+/g) || [];
    for (const run of runs) {
      const e = run.replace(/^[+*/.]+/, "").replace(/[+*/.]+$/, "");
      if (!e) continue;
      const nums = e.match(/\d+(?:\.\d+)?|\.\d+/g) || [];
      if (nums.length < 2) continue;
      if (/[+\-*/]/.test(e.replace(/^[-+]/, ""))) return true;
    }
  }
  return false;
}

/* サーバー側 isWordProblem と同じ文章題の形。 */
function isWolframWordProblem(raw) {
  const nums = String(raw || "").match(/\d+(?:\.\d+)?/g) || [];
  if (!nums.length) return false;
  if (!WOLFRAM_WORD_KEYWORDS.test(raw)) return false;
  return /(？|\?|ですか|ますか|何|いくつ|いくら)/.test(raw);
}

/**
 * 計算の質問かどうか（Wolfram レーンの発動判定）。
 * 発火する場合は { query（元の質問） } を返す。
 */
function parseWolframQuery(question) {
  const raw = String(question || "").trim();
  if (!raw) return null;
  if (/論文|arxiv|学術|査読/i.test(raw)) return null;
  if (/(翻訳|英訳|和訳)して/.test(raw)) return null;
  if (/計算(とは|の意味|って)|電卓/.test(raw)) return null;
  if (/^\d{3}-\d{4}/.test(raw) && /(どこ|住所|場所)/.test(raw)) return null;
  if (hasWolframDeterministicShape(wolframNormCalc(raw)) ||
      isWolframWordProblem(raw)) {
    return { query: raw };
  }
  return null;
}

/* /calc-wolfram に投げ、検証済みの形に整えて返す。
   出典リンク（resultUrl）が無い成功応答は、リンク必須の
   規約を守れないため失敗として扱う（従来レーンへ倒れる）。 */
async function fetchWolfram(parsed) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/calc-wolfram`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: parsed.query }),
      },
      WOLFRAM_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (!res || !res.ok || !data || typeof data.ok !== "boolean") return null;
  if (!data.ok) {
    return {
      ok: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  }
  if (typeof data.resultText !== "string" || !data.resultText) return null;
  const resultUrl = httpUrlOrEmpty(
    typeof data.resultUrl === "string" ? data.resultUrl : "");
  if (!resultUrl) return null;
  return {
    ok: true,
    resultText: data.resultText,
    interpretation: typeof data.interpretation === "string"
      ? data.interpretation : "",
    resultUrl,
    wolframQuery: typeof data.wolframQuery === "string"
      ? data.wolframQuery : "",
    via: typeof data.via === "string" ? data.via : "",
    source: typeof data.source === "string" && data.source
      ? data.source : "Wolfram|Alpha",
    fetchedAt: typeof data.fetchedAt === "string" ? data.fetchedAt : "",
  };
}

/* Wolfram の計算結果を表示用カードにする。 */
function wolframCardOf(d) {
  return {
    kind: "wolfram",
    resultText: d.resultText,
    interpretation: d.interpretation,
    resultUrl: d.resultUrl,
    wolframQuery: d.wolframQuery,
  };
}

/* ---- Google連携レーン（cube-server の /google・/gmail・/gcal） ---- */
/* 本人の決定（2026-11-20）：Gmail 送信もカレンダーの作成・変更も
   全自動（送信前の確認は挟まない）。条件は「送信ログ＋通知」。
   宛先は会話内に明示された単一のメールアドレスだけ。名前だけ・
   推測の宛先には絶対に送らず、揃わなければ聞き返す。
   認証は Supabase セッションの JWT を cube-server に渡し、
   サーバー側が保管する Google のリフレッシュトークンで実行する
   （ページに Google の秘密は置かない）。未ログイン・未連携の
   ときは、このレーンの中で正直に案内して完結させる。設計の
   正本は ~/workspace/cube_ext/GOOGLE_PLAN.md §5。 */

const GOOGLE_API_TIMEOUT_MS = 30000;
const GOOGLE_SENDLOG_SEEN_KEY = "cube.googleSendlogSeen.v1";

/* cube-server の Google 系エンドポイントへの POST（JWT つき）。
   応答 JSON をそのまま返す（401 の {ok:false} も含む）。通信
   失敗・JSON でない応答は null。 */
async function googlePost(path, token, body) {
  if (!token) return null;
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}${path}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body || {}),
      },
      GOOGLE_API_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  if (!res) return null;
  const data = await res.json().catch(() => null);
  return data && typeof data === "object" ? data : null;
}

async function fetchGoogleStatus(token) {
  const data = await googlePost("/google/status", token, {});
  if (!data || typeof data.ok !== "boolean") return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return {
    ok: true,
    configured: !!data.configured,
    linked: !!data.linked,
    googleEmail:
      typeof data.google_email === "string" ? data.google_email : "",
  };
}

async function startGoogleLink(token) {
  const data = await googlePost("/google/link/start", token, {});
  if (!data || !data.ok || typeof data.url !== "string" || !data.url) {
    return {
      ok: false,
      error: data && typeof data.error === "string" ? data.error : "",
    };
  }
  return { ok: true, url: data.url };
}

async function unlinkGoogle(token) {
  const data = await googlePost("/google/unlink", token, {});
  return !!data && data.ok === true;
}

/* ---- Gmail ---- */

const GMAIL_ADDRESS_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/;

/* 本文の末尾に残った命令表現（「、とメールを送って」等）を落とす。 */
function cleanGmailBodyTail(body) {
  return String(body || "")
    .replace(/[、,]\s*と\s*(メールを|メールで)?\s*(送ってください|送信してください|送って|送信して|送る|送信|送付)[。！!]*$/, "")
    .replace(/(メールを|メールで)(送ってください|送信してください|送って|送信して|送る|送信|送付)[。！!]*$/, "")
    .trim();
}

function extractGmailAddress(text) {
  const m = String(text || "").match(GMAIL_ADDRESS_RE);
  return m ? m[0] : "";
}

/**
 * メール送信の依頼かどうか。宛先・件名・本文を取り出し、揃わ
 * ないものは missing に入れる。送信の意図が無ければ null。
 * 宛先は文中の明示されたアドレスだけ（名前からは作らない）。
 */
function parseGmailSendQuery(question) {
  const q = String(question || "");
  if (!q || !/メール/.test(q)) return null;
  if (/とは|書き方|文例|例文|意味/.test(q)) return null;
  if (!/送って|送信|送る|送付|メールして|メールで知らせ|メールで送/.test(q)) {
    return null;
  }
  const to = extractGmailAddress(q);
  let subject = "";
  const sm =
    q.match(/件名[は:：]?\s*[「『]([^」』]+)[」』]/) ||
    q.match(/件名[は:：]\s*([^、。\n]+)/) ||
    q.match(/[「『]([^」』]+)[」』]という件名/);
  if (sm) subject = sm[1].trim();
  let body = "";
  const bm = q.match(/本文[は:：]\s*([\s\S]+)$/);
  if (bm) body = cleanGmailBodyTail(bm[1].trim());
  if (!body) {
    const qm = q.match(/[「『]([^」』]+)[」』]\s*と(?:いう内容で)?\s*(?:メール|送)/);
    if (qm && qm[1].trim() && qm[1].trim() !== subject) body = qm[1].trim();
  }
  if (!body) {
    // 宛先・件名・命令表現を取り除いた残りを本文とみなす。
    let rest = q;
    if (to) rest = rest.replace(to, " ");
    if (subject) rest = rest.replace(subject, " ");
    if (!to) {
      // アドレスが無い場合、先頭の「（名前）に」は宛先の指し方
      // なので本文には入れない。
      rest = rest.replace(/^[^\s、。]{1,20}[にへ][、,\s]*/, "");
    }
    rest = rest
      .replace(/件名[は:：]?/g, " ")
      .replace(/本文[は:：]?/g, " ")
      .replace(/[「『」』]/g, " ")
      .replace(/メールで知らせて|メールを送って|メールして|メールで|メールを|メール/g, " ")
      .replace(/送ってください|送信してください|送って|送信して|送る|送信|送付/g, " ")
      .replace(/という内容で|という件名で/g, " ")
      .replace(/^[にへ、。・\s]+|[にへ、。・\s]+$/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (rest) body = rest;
  }
  const missing = [];
  if (!to) missing.push("宛先のメールアドレス");
  if (!subject) missing.push("件名");
  if (!body) missing.push("本文");
  return { kind: "send", to, subject, body, missing };
}

/**
 * 聞き返しへの返信で下書きを埋める。マーカー（アドレス・
 * 件名：・本文：）は常に取り込む。マーカー無しの素の文は、
 * 足りない項目が1つだけのときにその項目として取り込む
 * （別の話題を下書きに吸い込まないための制限）。
 */
function mergeGmailDraft(draft, question) {
  const q = String(question || "");
  const next = {
    to: draft.to || "",
    subject: draft.subject || "",
    body: draft.body || "",
  };
  const em = extractGmailAddress(q);
  if (em) next.to = em;
  const sm =
    q.match(/件名[は:：]?\s*[「『]([^」』]+)[」』]/) ||
    q.match(/件名[は:：]\s*([^、。\n]+)/);
  if (sm) next.subject = sm[1].trim();
  const bm = q.match(/本文[は:：]\s*([\s\S]+)$/);
  if (bm) next.body = bm[1].trim();
  if (!em && !sm && !bm) {
    const missingCount = [next.to, next.subject, next.body]
      .filter((v) => !v).length;
    const text = q.trim();
    if (missingCount === 1 && text && !/[？?]/.test(text)) {
      if (!next.body) next.body = text;
      else if (!next.subject) next.subject = text;
    }
  }
  return next;
}

function gmailSendAskBackText(draft) {
  const missing = [];
  if (!draft.to) missing.push("宛先のメールアドレス");
  if (!draft.subject) missing.push("件名");
  if (!draft.body) missing.push("本文");
  return (
    `メールを送るには、${missing.join("と")}が必要です。` +
    "例えば「相手のアドレスに、件名：明日の会議、本文：10時に集合です、とメールを送って」のように教えてください。" +
    "宛先は名前だけでは送れません（間違った相手に送らないためです）。分かった分だけでも送ってくれれば、残りを聞き直します。"
  );
}

/**
 * メールの読み取りの依頼かどうか（新着の確認・一覧・最新を
 * 読む）。送信の依頼は送信レーンの担当なので拾わない。
 */
function parseGmailReadQuery(question) {
  const q = String(question || "");
  if (!q || !/メール/.test(q)) return null;
  if (/とは|書き方|文例|例文|意味/.test(q)) return null;
  if (parseGmailSendQuery(q)) return null;
  if (!/来てる|来ている|届い|新着|受信|未読|見せて|読んで|開いて|ある[？?]|チェック|確認|一覧/.test(q)) {
    return null;
  }
  let query = "";
  const em = extractGmailAddress(q);
  if (em) {
    query = `from:${em}`;
  } else {
    const fm = q.match(/([^\s、。・]+)からのメール/);
    if (fm && fm[1] !== "メール") query = `from:${fm[1]}`;
  }
  if (/未読/.test(q)) query = query ? `${query} is:unread` : "is:unread";
  const openFirst = /読んで|開いて|本文/.test(q);
  return { kind: "list", q: query, openFirst };
}

async function sendGmail(token, mail) {
  const data = await googlePost("/gmail/send", token, mail);
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return {
    ok: true,
    messageId: String(data.messageId || ""),
    log: data.log && typeof data.log === "object" ? data.log : null,
  };
}

async function fetchGmailList(token, opts) {
  const body = { maxResults: (opts && opts.maxResults) || 10 };
  if (opts && opts.q) body.q = opts.q;
  const data = await googlePost("/gmail/list", token, body);
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  const messages = (Array.isArray(data.messages) ? data.messages : [])
    .filter((m) => m && typeof m === "object")
    .map((m) => ({
      id: String(m.id || ""),
      from: String(m.from || ""),
      subject: String(m.subject || ""),
      date: String(m.date || ""),
      snippet: String(m.snippet || ""),
      unread: !!m.unread,
    }));
  return { ok: true, messages };
}

async function fetchGmailMessage(token, id) {
  const data = await googlePost("/gmail/get", token, { id });
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  const m = data.message || {};
  return {
    ok: true,
    message: {
      id: String(m.id || ""),
      from: String(m.from || ""),
      to: String(m.to || ""),
      subject: String(m.subject || ""),
      date: String(m.date || ""),
      bodyText: String(m.bodyText || ""),
    },
  };
}

async function fetchGmailSendLog(token, limit) {
  const data = await googlePost("/gmail/sendlog", token, {
    limit: limit || 20,
  });
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  const entries = (Array.isArray(data.entries) ? data.entries : [])
    .filter((e) => e && typeof e === "object")
    .map((e) => ({
      ts: String(e.ts || ""),
      to: String(e.to || ""),
      subject: String(e.subject || ""),
      messageId: String(e.messageId || ""),
    }));
  return { ok: true, entries };
}

function gmailSendLogKey(entry) {
  return `${entry.ts}|${entry.messageId}|${entry.to}|${entry.subject}`;
}

function gmailListText(messages) {
  if (!messages.length) return "条件に合うメールはありませんでした。";
  const lines = messages.map((m, i) => {
    const unread = m.unread ? "【未読】" : "";
    const subj = m.subject || "（件名なし）";
    const from = m.from || "（差出人不明）";
    return `${i + 1}. ${unread}${subj}（${from}）`;
  });
  return `メールが${messages.length}件あります。\n` + lines.join("\n");
}

/* ---- Googleカレンダー ---- */

const GCAL_WEEKDAYS = ["日", "月", "火", "水", "木", "金", "土"];

/* JST の日付キー（YYYY-MM-DD）。今日から days 日ずらす。 */
function gcalDateKeyPlus(days) {
  return pointsJstDateKey(jstDatePlus(days));
}

/* 日付キーの翌日（YYYY-MM-DD）。 */
function gcalNextDateKey(key) {
  const [y, m, d] = String(key).split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d) + 86400000)
    .toISOString()
    .slice(0, 10);
}

function gcalWeekdayOf(dateKey) {
  const [y, m, d] = String(dateKey).split("-").map(Number);
  return GCAL_WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
}

/**
 * 質問文から日付を取り出す。{key, label, explicit} か null。
 * 今日・明日・明後日・昨日・M月D日・M/D・曜日に対応する。
 */
function parseGcalDate(question) {
  const q = String(question || "");
  if (/明後日/.test(q)) return { key: gcalDateKeyPlus(2), label: "明後日", explicit: false };
  if (/明日/.test(q)) return { key: gcalDateKeyPlus(1), label: "明日", explicit: false };
  if (/今日|本日/.test(q)) return { key: gcalDateKeyPlus(0), label: "今日", explicit: false };
  if (/昨日/.test(q)) return { key: gcalDateKeyPlus(-1), label: "昨日", explicit: false };
  const md = q.match(/(\d{1,2})月(\d{1,2})日/);
  if (md) {
    const month = Number(md[1]);
    const day = Number(md[2]);
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
      const y = Number(gcalDateKeyPlus(0).slice(0, 4));
      const key = `${y}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      return { key, label: `${month}月${day}日`, explicit: true };
    }
  }
  const slash = q.match(/(?:^|[^\d])(\d{1,2})\/(\d{1,2})(?:[^\d]|$)/);
  if (slash) {
    const month = Number(slash[1]);
    const day = Number(slash[2]);
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
      const y = Number(gcalDateKeyPlus(0).slice(0, 4));
      const key = `${y}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      return { key, label: `${month}月${day}日`, explicit: true };
    }
  }
  const wd = q.match(/([日月火水木金土])曜日/);
  if (wd) {
    const target = GCAL_WEEKDAYS.indexOf(wd[1]);
    const todayKey = gcalDateKeyPlus(0);
    const todayWd = GCAL_WEEKDAYS.indexOf(gcalWeekdayOf(todayKey));
    const delta = (target - todayWd + 7) % 7;
    return {
      key: gcalDateKeyPlus(delta),
      label: `${wd[1]}曜日`,
      explicit: false,
    };
  }
  return null;
}

/* 文中の時刻をすべて "HH:MM" で取り出す（「3時間」のような
   所要時間は除く）。範囲指定は [開始, 終了] の順になる。 */
function gcalTimeTokens(question) {
  const q = String(question || "");
  const tokens = [];
  const re = /(午前|午後)?\s*(\d{1,2})時(?:(\d{1,2})分)?|(\d{1,2}):(\d{2})/g;
  let m;
  while ((m = re.exec(q)) !== null) {
    if (q[re.lastIndex] === "間") continue;
    if (m[4] !== undefined) {
      const h = Number(m[4]);
      const min = Number(m[5]);
      if (h > 23 || min > 59) continue;
      tokens.push(
        `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`);
    } else {
      let h = Number(m[2]);
      const min = m[3] ? Number(m[3]) : 0;
      if (m[1] === "午後" && h < 12) h += 12;
      if (m[1] === "午前" && h === 12) h = 0;
      if (h > 23 || min > 59) continue;
      tokens.push(
        `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`);
    }
  }
  return tokens;
}

/* 日付・時刻・命令表現を取り除いた残り（予定名の候補）。 */
function gcalStripScheduleWords(q) {
  let s = String(q || "");
  s = s.replace(/(\d{1,2}月\d{1,2}日|\d{1,2}\/\d{1,2}|明後日|明日|今日|本日|昨日|来週|今週|[日月火水木金土]曜日)/g, " ");
  s = s.replace(/(午前|午後)?\s*\d{1,2}時(\d{1,2}分)?/g, " ");
  s = s.replace(/\d{1,2}:\d{2}/g, " ");
  s = s.replace(/から|まで/g, " ");
  return s;
}

function extractGcalTitle(question) {
  const q = String(question || "");
  const quoted = q.match(/[『「]([^』」]+)[』」]/);
  if (quoted) return quoted[1].trim();
  let s = gcalStripScheduleWords(q);
  s = s.replace(/(の)?予定(を|に)?(入れて|追加して|登録して|作って|作成して|入れる|追加|登録|作成)?/g, " ");
  s = s.replace(/スケジュール/g, " ");
  s = s.replace(/お願いします|ください/g, " ");
  s = s.replace(/^[にへ、。・\s]+|[にへ、。・\s]+$/g, "");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

/* 削除・変更の対象を指す手がかり（予定名の一部）。 */
function extractGcalHint(question) {
  const q = String(question || "");
  const quoted = q.match(/[『「]([^』」]+)[』」]/);
  if (quoted) return quoted[1].trim();
  let s = gcalStripScheduleWords(q);
  s = s.replace(/(の)?予定(を|は|に)?/g, " ");
  s = s.replace(/スケジュール/g, " ");
  s = s.replace(/消して|削除|取り消し|キャンセル|変更して|変更|ずらして|移動して|延期して|延期/g, " ");
  s = s.replace(/\d+\s*番目|最初|さっき/g, " ");
  s = s.replace(/お願いします|ください/g, " ");
  s = s.replace(/^[をにへの、。・\s]+|[をにへの、。・\s]+$/g, "");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function parseGcalOrdinal(question) {
  const q = String(question || "");
  const m = q.match(/(\d+)\s*番目/);
  if (m) return Number(m[1]) - 1;
  if (/最初|さっき|1つ目|一つ目/.test(q)) return 0;
  if (/2つ目|二つ目/.test(q)) return 1;
  if (/3つ目|三つ目/.test(q)) return 2;
  return null;
}

/**
 * カレンダーの依頼かどうか。kind は list / create / update /
 * delete。作成で揃わないものは missing に入れる（聞き返す）。
 * 予定の話題でも操作でもない質問は null。
 */
function parseGcalQuery(question) {
  const q = String(question || "");
  if (!q || !/予定|スケジュール/.test(q)) return null;
  if (/とは|意味/.test(q)) return null;
  const date = parseGcalDate(q);
  const times = gcalTimeTokens(q);
  if (/消して|削除|取り消し|キャンセル/.test(q)) {
    return {
      kind: "delete",
      hint: extractGcalHint(q),
      ordinal: parseGcalOrdinal(q),
      date,
    };
  }
  if (/変更|ずらし|移動|延期/.test(q)) {
    const quoted = q.match(/[『「]([^』」]+)[』」]/);
    return {
      kind: "update",
      hint: extractGcalHint(q),
      ordinal: parseGcalOrdinal(q),
      date,
      times,
      title: quoted ? quoted[1].trim() : "",
    };
  }
  if (/入れて|追加|登録|作って|作成|予約/.test(q)) {
    const title = extractGcalTitle(q);
    const missing = [];
    if (!date) missing.push("日付");
    if (!times.length) missing.push("時刻");
    if (!title) missing.push("予定の名前");
    return { kind: "create", date, times, title, missing };
  }
  // 確認（一覧）：日付指定があればその日、無ければ今後1週間。
  if (date) {
    return {
      kind: "list",
      label: date.label,
      timeMin: `${date.key}T00:00:00+09:00`,
      timeMax: `${gcalNextDateKey(date.key)}T00:00:00+09:00`,
    };
  }
  return {
    kind: "list",
    label: "今後1週間",
    timeMin: `${gcalDateKeyPlus(0)}T00:00:00+09:00`,
    timeMax: `${gcalDateKeyPlus(7)}T00:00:00+09:00`,
  };
}

function sanitizeGcalEvents(list) {
  return (Array.isArray(list) ? list : [])
    .filter((e) => e && typeof e === "object" && e.id)
    .map((e) => ({
      id: String(e.id),
      summary: String(e.summary || ""),
      start: String(e.start || ""),
      end: String(e.end || ""),
      location: String(e.location || ""),
      description: String(e.description || ""),
      htmlLink: String(e.htmlLink || ""),
    }));
}

async function fetchGcalEvents(token, timeMin, timeMax) {
  const data = await googlePost("/gcal/list", token, { timeMin, timeMax });
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return { ok: true, events: sanitizeGcalEvents(data.events) };
}

async function createGcalEvent(token, ev) {
  const data = await googlePost("/gcal/create", token, ev);
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return { ok: true, event: sanitizeGcalEvents([data.event])[0] || null };
}

async function updateGcalEvent(token, eventId, patch) {
  const data = await googlePost("/gcal/update", token, {
    eventId,
    ...patch,
  });
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return { ok: true, event: sanitizeGcalEvents([data.event])[0] || null };
}

async function deleteGcalEvent(token, eventId) {
  const data = await googlePost("/gcal/delete", token, { eventId });
  if (!data) return null;
  if (!data.ok) return { ok: false, error: String(data.error || "") };
  return { ok: true };
}

/* "YYYY-MM-DDTHH:MM:00+09:00" に1時間を足した ISO 文字列。 */
function gcalPlusOneHour(iso) {
  const key = iso.slice(0, 10);
  const [h, m] = iso.slice(11, 16).split(":").map(Number);
  const total = h * 60 + m + 60;
  const hh = Math.floor(total / 60) % 24;
  const day = total >= 24 * 60 ? gcalNextDateKey(key) : key;
  return (
    `${day}T${String(hh).padStart(2, "0")}:` +
    `${String(total % 60).padStart(2, "0")}:00+09:00`
  );
}

/* epoch ms を JST の ISO 文字列（+09:00）に戻す。 */
function gcalIsoFromMs(ms) {
  const jst = new Date(ms + 9 * 3600000);
  const key = jst.toISOString().slice(0, 10);
  const hh = String(jst.getUTCHours()).padStart(2, "0");
  const mm = String(jst.getUTCMinutes()).padStart(2, "0");
  return `${key}T${hh}:${mm}:00+09:00`;
}

/* 予定の日時表示（例：11月21日（金） 10:00〜11:00）。 */
function gcalWhenText(ev) {
  const s = String(ev.start || "");
  const e = String(ev.end || "");
  const key = s.slice(0, 10);
  const [, mo, d] = key.split("-");
  const dateLabel = `${Number(mo)}月${Number(d)}日（${gcalWeekdayOf(key)}）`;
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return `${dateLabel} 終日`;
  const st = s.slice(11, 16);
  const et = e.slice(0, 10) === key ? e.slice(11, 16) : "";
  return et ? `${dateLabel} ${st}〜${et}` : `${dateLabel} ${st}〜`;
}

function gcalEventLine(ev) {
  const loc = ev.location ? `（場所：${ev.location}）` : "";
  return `・${gcalWhenText(ev)}：${ev.summary || "（無題）"}${loc}`;
}

function gcalListText(label, events) {
  if (!events.length) return `${label}の予定はありません。`;
  return `${label}の予定です。\n` + events.map(gcalEventLine).join("\n");
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
  let content = effectiveContentTerms(modelAnswer, question);
  if (content.length === 0) {
    // 答えの中心が1文字の語（「元素記号Feは鉄の元素です」の鉄）だと
    // 主張語が空になる。値の位置から拾う短い1語で代用する（下の
    // 単独語ルール＝タイトル錨＋同じ窓の共起がそのまま縛る）。
    content = gateContentTerms(modelAnswer, question);
  }
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
  // 日付の質問で、答えの主張が数字（年）だけの場合：「人類が初めて
  // 月面に着陸したのは1969年です」のような文は、主張語が全部質問の
  // 言い換えとして消えて数字だけが残るため、通常の照合では通らない。
  // その数字と質問の大事な語3つ以上が同じ窓にそろう記事だけを
  // 裏づけにする（語のそろいが話題の錨、数字の一致が出来事の裏づけ）。
  if (content.length === 1 && /^\d{4}$/.test(content[0]) && qKind === "date") {
    const qTerms = keyTerms(normalizeJa(question)).filter(
      (t) => !/^\d/.test(t) && t.length >= 2,
    );
    if (qTerms.length >= 3) {
      for (const r of results) {
        if (!isUsableEvidenceResult(r)) continue;
        for (const w of evidenceWindows(r)) {
          if (!windowSupportsAnswer(w, modelAnswer, content)) continue;
          // 年は「N年」の形で窓に現れていなければならない（「8」の
          // ような数字の断片や、無関係な箇所の同じ数字では通さない）。
          if (!w.includes(`${content[0]}年`)) continue;
          const hits = qTerms.filter((t) => w.includes(t)).length;
          if (hits >= 3) return r;
        }
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
  // 単独語は答えの値そのもの（または「Xの〜」の X）でなければならない。
  // 「元素記号Feは元素記号の記号です」のような循環文では、質問の
  // 枠語（記号）が主張語として残り、その語の記事で通ってしまう。
  if (!value) return null;
  if (term !== value && term !== value.split("の")[0]) return null;
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

/* ------------------------------------------------------------------ */
/* 文単位の出典検証ゲート（2026-11-16）                                */
/*                                                                    */
/* モデルの答えを文（主張）単位に割り、各文を手元の証拠（種類つき      */
/* 事実・検索結果）で裏づける。表示側は「裏づけの取れた文だけ」を      */
/* 使う——裏づけの無い主張文は、モデル優先表示でも出さない。          */
/* 設計は hybrid/VERIFICATION_LAYER_DESIGN.md を参照。                */
/* ------------------------------------------------------------------ */

/**
 * ゲート用の文分割：既存 splitSentences は8字未満の文を捨てるが、
 * 「水です。」のような短い主張も検証対象にしなければならないため、
 * 句点・感嘆符・疑問符・改行で割り、空でない断片を全部残す。
 */
function splitForGate(text) {
  const parts = String(text ?? "").split(/(?<=[。！？\n])/);
  const out = [];
  for (const p of parts) {
    const s = p.trim();
    if (s) out.push(s);
  }
  if (out.length === 0 && String(text ?? "").trim()) {
    out.push(String(text).trim());
  }
  return out;
}

/** 棄権の表明か（「分かりません」等で、主張語も数字も持たない文）。 */
function isAbstentionSentence(sentence, question) {
  if (
    !/分かりません|わかりません|分からない|確認できません|不明です|知りません|存じ上げません|分かりかねます/.test(
      sentence,
    )
  ) {
    return false;
  }
  // 棄権の言い回し自体を取り除いてから中身を測る（「分かりません」
  // は語の抽出で「かりません」に断片化し、主張語に見えてしまうため）。
  const rest = sentence.replace(
    /分かりません|わかりません|分からない|確認できません|不明です|知りません|存じ上げません|分かりかねます/g,
    "",
  );
  return (
    effectiveContentTerms(rest, question).length === 0 &&
    numbersIn(rest).length === 0
  );
}

/**
 * 定義質問用のアンカー：その記事が質問の対象についての記事か。
 * タイトルが対象を指すか、冒頭600字が対象を名指していること。
 * リダイレクト先の記事（モーリーファンタジー → イオンファンタジー）は
 * タイトルが対象と一致しないが冒頭で元の名前を名指すため、本文側の
 * 名指しでも拾う（factSourceRelatedToEntity と同じ基準）。
 */
function definitionAnchored(result, question) {
  const entity = subjectOf(question) ?? topicFromQuestion(question);
  if (!entity) return false;
  if (titleMatchesSubject(result.title, entity)) return true;
  const ne = normalizeJa(entity);
  if (ne.length >= 2) {
    if (normalizeJa(result.title).includes(ne)) return true;
    if (normalizeJa((result.text ?? "").slice(0, 600)).includes(ne)) {
      return true;
    }
  }
  // 最上級の質問では話題文（「太陽系で一番大きい惑星」）が記事に
  // そのまま出てこないため、基底の主語（「太陽系」）でも拾う。
  if (superlativeRank(question)) {
    const base = String(entity).split(/で(?:一番|最も|最大|最高)/)[0];
    const nb = normalizeJa(base ?? "");
    if (nb.length >= 2 && nb !== ne) {
      if (titleMatchesSubject(result.title, base)) return true;
      if (normalizeJa((result.text ?? "").slice(0, 600)).includes(nb)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * 経路C用の主張語：effectiveContentTerms の語から末尾の助詞を落とした
 * もの。かな主体の語（モーリーファンタジー等）は質問文の中では1つの
 * かなの連なりとして抽出されるため、答え側の「モーリーファンタジーは」
 * のような助詞つきの語が質問語と一致せず主張語に残り、そのままでは
 * 記事の窓（「モーリーファンタジー」をはじめとする…）と一致しない。
 * 窓照合の前に助詞を落として、語の芯で比べる。
 */
function gateContentTerms(sentence, question) {
  const out = [];
  for (const t of effectiveContentTerms(sentence, question)) {
    if (/^\d/.test(t)) {
      if (!out.includes(t)) out.push(t);
      continue;
    }
    const s = t.replace(/(は|が|を|に|の|へ|と|も|から|まで|より)+$/, "");
    if (s.length >= 2 && !out.includes(s)) out.push(s);
  }
  if (out.length === 0) {
    // 答えの中心が1文字の語（水・鉄など）だと、2文字以上の語だけを
    // 集める上の処理では主張語が空になる。は〜です／は〜をの値の
    // 位置から短い1語（「Xの〜」ならX）だけを拾う。拾った語は
    // 各経路の単独語ルール（対象との共起＋アンカー）で縛られる。
    const vm = /は「?(.{1,10}?)」?(?:です|である|を|が|に)/.exec(
      sentence ?? "",
    );
    const v = (vm?.[1] ?? "").trim();
    if (v && v.length <= 3 && !/^\d+$/.test(v)) {
      out.push(v);
    } else if (v) {
      const hm = /^(.{1,3}?)の/.exec(v);
      if (hm && !/^\d+$/.test(hm[1])) out.push(hm[1]);
    }
  }
  return out;
}

/**
 * 定義質問専用の裏づけ（経路C）：対象にアンカーされた記事の冒頭窓
 * （タイトル＋リード）に、文の主張語と数字が全部そろっているか。
 * 経路B（findCorroboratingResult）は定義質問を意図的に通さないため、
 * 定義文はこの窓限定の経路でだけ裏づける。タイトルに限定（（…））の
 * 付く記事（同名別エンティティ）は後回しにし、まず限定の無い記事で
 * 裏づけを探す。
 */
/** 順位・最上級の印（順位の食い違いを検出するため、順位で比べる）。 */
function superlativeRank(text) {
  const t = String(text ?? "");
  if (/二番目|2番目|第二/.test(t)) return 2;
  if (/三番目|3番目|第三/.test(t)) return 3;
  if (/一番|最も|最大|最高|1番|第一/.test(t)) return 1;
  return 0;
}

/**
 * 順位・最上級の質問の対象の頭名詞：「日本で二番目に高い山」なら
 * 「山」、「日本で一番高いビル」なら「ビル」。順位の印の後ろから
 * 形容詞を除いて取る。取れなければ null。
 */
function superlativeHeadNoun(question) {
  const topic = topicFromQuestion(question) ?? "";
  const m =
    /(二番目|2番目|三番目|3番目|一番|最も|最大|最小|最高|最低|最古|最新|最長|最短)/.exec(
      topic,
    );
  if (!m) return null;
  let rest = topic.slice(m.index + m[0].length);
  rest = rest.replace(/^(に|の|は|が)+/, "");
  rest = rest.replace(
    /^(高い|大きい|大きな|小さい|深い|長い|短い|古い|新しい|多い|少ない|強い|広い|狭い|重い|軽い|速い|早い)/,
    "",
  );
  const noun = rest.match(/^[一-鿿々ァ-ヶー]+/)?.[0] ?? "";
  return noun.length >= 1 ? noun : null;
}

/** 最上級の質問の属性語と、その属性での最大級・最小級の言い方。 */
function superlativeAttrWord(q) {
  const small = /(小さい|少ない|短い|低い|狭い|軽い|最小|最少|最短|最低)/.test(
    q,
  );
  const table = [
    [/人口/, "人口", "最大", "最小"],
    [/面積/, "面積", "最大", "最小"],
    [/長さ|全長|長い|短い/, "長さ", "最長", "最短"],
    [/高さ|標高|高い|低い/, "高さ", "最高", "最低"],
    [/大きさ|大きい|小さい/, "大きさ", "最大", "最小"],
    [/重さ|重量|重い|軽い/, "重さ", "最大", "最小"],
    [/速さ|速度|速い|遅い/, "速さ", "最速", "最遅"],
    [/深さ|深い|浅い/, "深さ", "最大", "最小"],
  ];
  for (const [re, word, big, sml] of table) {
    if (re.test(q)) return { word, rank: small ? sml : big };
  }
  return null;
}

/** 拡張検索の語並べから落とす、質問の言い回しだけの語。 */
const EXTRA_QUERY_STOP = new Set([
  "何年",
  "何人",
  "何",
  "ですか",
  "ますか",
  "一番",
  "最も",
  "教えて",
  "について",
  "ください",
  "どんな",
  "どこ",
  "いつ",
  "誰",
]);

/** 文章の最上級の向き：大きい側なら 1、小さい側なら -1、無ければ 0。 */
function superlativeDirection(text) {
  const t = String(text ?? "");
  const big =
    /(最大|最多|最長|最高|最速|最も多|一番多|大きい|大きな|多い|長い|高い|広い|重い|速い)/.test(
      t,
    );
  const small =
    /(最小|最少|最短|最低|最も小|一番小|小さい|少ない|短い|低い|狭い|軽い|遅い)/.test(
      t,
    );
  if (big && !small) return 1;
  if (small && !big) return -1;
  return 0;
}

const SUPERLATIVE_ATTRS = [
  [/人口/, ["人口"]],
  [/面積/, ["面積"]],
  [/長さ|全長|長い|短い/, ["長さ", "全長", "長い"]],
  [/高さ|標高|高い|低い/, ["高さ", "標高", "高い"]],
  [/大きさ|大きい|小さい/, ["大きさ", "大きい", "大きな", "面積"]],
  [/重さ|重量|重い|軽い/, ["重さ", "重量", "重い"]],
  [/深さ|深い|浅い/, ["深さ", "深い", "水深"]],
  [/速さ|速度|速い|遅い/, ["速さ", "速度", "速い"]],
];

/**
 * 最上級の質問への「勝者」型の答え（「世界で一番人口が多い国は
 * インドです」）の裏づけ。勝者の記事自身のリードが、同じ範囲・
 * 同じ属性・同じ順位・同じ向きの最上級を自称している場合に限る。
 * 一覧記事が「香川県が最も狭い」の形で勝者を名指している場合も、
 * その形（勝者＋が＋最も／一番）でだけ認める。勝者の記事が最上級を
 * 自称していない答えは、ここでは通さない。
 */
function superlativeWinnerSource(sentence, results, question) {
  const qRank = superlativeRank(question);
  if (!qRank) return null;
  // 答えが名指す勝者：「はXです／はXで、」の X を候補にする。
  const winners = [];
  for (const m of String(sentence ?? "").matchAll(
    /は「?([^、。！？]{1,24}?)」?(?=です|である|だ|で、|、)/g,
  )) {
    const w = (m[1] ?? "").replace(/^(約|およそ)/, "").trim();
    if (w && !/^\d/.test(w) && !/[はがをにへと]/.test(w) && !winners.includes(w)) {
      winners.push(w);
    }
  }
  if (winners.length === 0) return null;
  const attr = SUPERLATIVE_ATTRS.find(([re]) => re.test(question));
  if (!attr) return null;
  const scope = /(世界|日本|太陽系|全国)/.exec(question)?.[1] ?? null;
  const scopeAlt =
    scope === "日本" && /(県|都道府県)/.test(question) ? "都道府県" : null;
  const qDir = superlativeDirection(question);
  const claimNums = numbersIn(sentence);
  for (const r of results ?? []) {
    if (!isUsableEvidenceResult(r)) continue;
    const lead = (r.text ?? "").slice(0, 1600);
    if (claimNums.length > 0) {
      const leadNums = numbersIn(
        normalizeUnits(normalizeJa(`${r.title}\n${lead}`)),
      );
      if (
        !claimNums.every((n) => leadNums.some((e) => numbersMatchLoose(n, e)))
      ) {
        continue;
      }
    }
    for (const winner of winners) {
      const titleEq =
        titleMatchesSubject(r.title, winner) ||
        normalizeJa(
          (r.title ?? "").replace(/[（(][^（）()]*[）)]\s*$/, ""),
        ) === normalizeJa(winner);
      for (const ls of splitSentences(lead)) {
        const rankOk =
          superlativeRank(ls) === qRank ||
          /第1位|世界一|日本一|首位/.test(ls);
        if (!rankOk) continue;
        if (!attr[1].some((w) => ls.includes(w))) continue;
        if (
          scope &&
          !ls.includes(scope) &&
          !(scopeAlt && ls.includes(scopeAlt)) &&
          !(r.title ?? "").includes(scope) &&
          !(scopeAlt && (r.title ?? "").includes(scopeAlt))
        ) {
          continue;
        }
        const d = superlativeDirection(ls);
        if (qDir !== 0 && d !== 0 && d !== qDir) continue;
        const asserts =
          titleEq ||
          ls.includes(`${winner}が最も`) ||
          ls.includes(`${winner}が一番`) ||
          ls.includes(`${winner}は最も`) ||
          ls.includes(`${winner}は日本一`) ||
          ls.includes(`${winner}は世界一`) ||
          ls.includes(`のは${winner}`) ||
          ls.includes(`は${winner}である`);
        if (!asserts) continue;
        if (/ではなく|でなく|誤り|間違い/.test(ls)) continue;
        return r;
      }
    }
  }
  return null;
}

/**
 * 答えの文が名指す「勝者」（最上級の答えの値）を1つ取り出す。
 * 「XはYです」型は Y、「Xは、ZにあるWです」型（「日本で一番深い
 * 湖は、滋賀県にある琵琶湖です」）は末尾の W を取る。取れなければ
 * null（その場合は勝者の一致は主張語の照合に任せる）。
 */
function superlativeClaimWinner(sentence) {
  const s = String(sentence ?? "");
  for (const m of s.matchAll(
    /は「?([^、。！？]{1,24}?)」?(?=です|である|だ|で、|、)/g,
  )) {
    const w = (m[1] ?? "").replace(/^(約|およそ)/, "").trim();
    if (w && !/^\d/.test(w) && !/[はがをにへと]/.test(w)) return w;
  }
  const vm = /は「?([^。「」]{1,30}?)」?(?:です|である)[。]?$/.exec(s);
  let v = (vm?.[1] ?? "").replace(/^[、,]/, "").trim();
  if (v) {
    const tail = v.split(/にある|に位置する/).pop().trim();
    if (tail && !/^\d/.test(tail) && !/[はがをにへと]/.test(tail)) {
      return tail;
    }
  }
  return null;
}

/**
 * 証拠側の1文が、質問の最上級と「同じ主張」をしているか：
 * 順位が同じで、属性（深さ・面積・高さ…）が同じ系統で、向き
 * （大きい側／小さい側）が矛盾せず、答えの名指す勝者がその文の
 * 中に現れていること。順位と頭名詞だけでは「日本最大の湖である
 * 琵琶湖」（面積の主張）が「日本で一番深い湖」（深さの質問）の
 * 裏づけになってしまう（第2回実測 a15 の誤確認）。属性の系統が
 * 質問から読み取れない場合は、その項目は判定しない。
 */
function superlativeSentenceAgrees(ls, question, claimSentence) {
  const qRank = superlativeRank(question);
  if (qRank) {
    const rankOk =
      superlativeRank(ls) === qRank ||
      (qRank === 1 && /第1位|世界一|日本一|首位/.test(ls));
    if (!rankOk) return false;
  }
  const attr = SUPERLATIVE_ATTRS.find(([re]) => re.test(question));
  if (attr && !attr[1].some((w) => ls.includes(w))) return false;
  const qDir = superlativeDirection(question);
  const d = superlativeDirection(ls);
  if (qDir !== 0 && d !== 0 && d !== qDir) return false;
  const winner = superlativeClaimWinner(claimSentence);
  if (winner && !normalizeJa(ls).includes(normalizeJa(winner))) {
    return false;
  }
  return true;
}

function definitionSentenceSource(sentence, results, question) {
  let content = gateContentTerms(sentence, question);
  if (content.length === 0) {
    // 「H2Oは水を表します」のように、答えの値が1文字の語だと主張語が
    // 空になる。値の位置（は〜です／は〜を）から短い1語だけを拾い、
    // 対象名と同じ窓にある場合だけ使う（下の1語ルールがそのまま効く）。
    const vm = /は「?(.{1,10}?)」?(?:です|である|を|が|に)/.exec(sentence);
    const v = (vm?.[1] ?? "").trim();
    if (v && v.length <= 3 && !/^\d+$/.test(v)) {
      content = [v];
    } else if (v) {
      // 「元素記号Feは鉄の元素です」のように値が「Xの〜」の形で、
      // X自体が1〜3文字の場合はXを唯一の主張語として扱う。
      const hm = /^(.{1,3}?)の/.exec(v);
      if (hm && !/^\d+$/.test(hm[1])) content = [hm[1]];
    }
  }
  // 主張語が1つも無い文は裏づけない（英数字の綴りから取れた数字
  // だけで「裏づけられた」ことになる穴を塞ぐ）。
  if (content.length === 0) return null;
  // 順位・最上級の質問では、順位そのものが主張の一部：裏づけ窓にも
  // 同じ順位の印と、対象の頭名詞（山・湖・ビル等）が無ければ通さない
  // （「二番目に高い山は富士山」と言っても、一番の記事では裏づけない）。
  // さらに属性（深さ・面積など）と勝者の一致も必須にする：順位と
  // 頭名詞だけだと、面積で日本一の琵琶湖が「一番深い湖」の裏づけに
  // なってしまう（第2回実測 a15 の誤確認の経路）。
  const qRank = superlativeRank(question);
  const headNoun = qRank ? superlativeHeadNoun(question) : null;
  const usable = (results ?? []).filter(
    (r) => isUsableEvidenceResult(r) && definitionAnchored(r, question),
  );
  const passes = [
    usable.filter((r) => !/[（(][^（）()]+[）)]/.test(r.title ?? "")),
    usable,
  ];
  // 窓はリードの1文単位（タイトルを添える）。リード全体を1つの窓に
  // すると、語が別々の文に散らばっているだけでも裏づけになって
  // しまう（日本の記事に富士山と3,776が別々の文にある等）ため。
  // 主張語が1つだけの文は、その語と対象名が同じ1文にそろっている
  // ことも求める（別対象の記事に偶然出てくる語で通さない）。
  const entity = subjectOf(question) ?? topicFromQuestion(question);
  let ne = entity ? normalizeJa(entity) : "";
  // 最上級の質問では、共起を求める対象名も基底の主語にする（話題文
  // そのままは記事の文に出てこないため）。
  if (qRank && entity) {
    const base = String(entity).split(/で(?:一番|最も|最大|最高)/)[0];
    const nb = normalizeJa(base ?? "");
    if (nb.length >= 2) ne = nb;
  }
  for (const pass of passes) {
    for (const r of pass) {
      const leadSentences = splitSentences((r.text ?? "").slice(0, 1600));
      for (const ls of leadSentences) {
        if (qRank) {
          if (superlativeRank(ls) !== qRank) continue;
          if (headNoun && !ls.includes(headNoun)) continue;
          if (!superlativeSentenceAgrees(ls, question, sentence)) continue;
        }
        const w = normalizeUnits(normalizeJa(`${r.title}\n${ls}`));
        if (content.length === 1 && !/^\d/.test(content[0]) && ne) {
          if (!w.includes(ne)) continue;
          // 1語だけの主張では、対象名がタイトルではなく文そのものに
          // 現れていることも求める。タイトルにだけ対象名がある記事の
          // 無関係な文（例：ドラマ記事のあらすじに出てくる「水」）が、
          // 窓へのタイトル添えで共起したことになって通るのを防ぐ。
          if (!normalizeJa(ls).includes(ne)) continue;
        }
        if (windowSupportsAnswer(w, sentence, content)) {
          // 主語のすり替え点検：候補文の主題が、この裏づけ文の主題と
          // 別の対象を指している場合は、その文の主張を候補文の主題の
          // 話として裏づけない（a6型）。点検は、裏づけ文の主題が記事
          // タイトル（記事自身の主題）と一致する場合に限る（「相撲は
          // 日本の国技です」のような答えの値を主語にする文まで巻き
          // 込まないため）。最上級の質問では勝者の点検
          // （superlativeWinnerGrounded）が担うため、ここでは除く。
          const lsSubject = sentenceHeadSubject(ls);
          if (
            !qRank &&
            titleNamesSubject(r.title, lsSubject) &&
            headSubjectsConflict(sentenceHeadSubject(sentence), lsSubject)
          ) {
            continue;
          }
          return r;
        }
      }
    }
  }
  return null;
}

/**
 * その文が事実の主張を含むか。主張語・数字がある場合はもちろん、
 * 「Feは鉄です」のように値が1文字で主張語の抽出（2文字以上の語が
 * 対象）から漏れる文や、「鉄の肖像です」のように主語の無い値だけの
 * 文も、断定形（〜です／〜である等）で終わっていれば主張として拾う。
 * 拾い漏らすと検証を素通りして表示されてしまうため、安全側に倒す。
 * 会話の定型（相づち・挨拶）だけは主張から除く。
 */
function isClaimSentence(sentence, question) {
  // 定型の挨拶・相づちで始まり、値の言明も数字も無い文は主張ではない
  // （語の抽出では挨拶も「主張語」を持ってしまうため、先に落とす）。
  if (
    /^(そう|はい|ええ|うん|なるほど|ありがとう|ごめん|すみません|こんにちは|こんばんは|おはよう|よろしく|お疲れさま|おつかれさま|またね|じゃあ|では)/.test(
      sentence,
    ) &&
    numbersIn(sentence).length === 0 &&
    !/は「?[^。「」]{1,30}」?(?:です|である)[。！]?$/.test(sentence)
  ) {
    return false;
  }
  if (effectiveContentTerms(sentence, question).length > 0) return true;
  if (numbersIn(sentence).length > 0) return true;
  if (!/(です|である|ます|でした|だった)[。！]?$/.test(sentence)) {
    return false;
  }
  if (
    /^(そう|はい|ええ|うん|なるほど|ありがとう|ごめん|すみません|こんにちは|こんばんは|おはよう|よろしく|お疲れさま|おつかれさま|またね|じゃあ|では)/.test(
      sentence,
    )
  ) {
    return false;
  }
  const m = sentence.match(/は「?([^。「」]{1,30}?)」?(?:です|である)[。]?$/);
  const v = m?.[1]?.trim();
  if (
    v &&
    (EVIDENCE_STOP_TERMS.has(v) || /^(そう|これ|それ|はい|いいえ)$/.test(v))
  ) {
    return false;
  }
  return true;
}

/**
 * 崩壊出力か：同じ4文字の並びが何度も繰り返される生成（モデルの
 * ループ）は、事実の主張として扱わず丸ごと表示しない。普通の答えで
 * 同じ4文字が5回以上繰り返されることはまず無い。
 */
function isDegenerateAnswer(text) {
  const s = String(text ?? "");
  if (s.length < 24) return false;
  const grams = new Map();
  for (let i = 0; i + 4 <= s.length; i += 1) {
    const g = s.slice(i, i + 4);
    grams.set(g, (grams.get(g) ?? 0) + 1);
  }
  for (const c of grams.values()) {
    if (c >= 5) return true;
  }
  return false;
}

/**
 * 『…』の質問で、括弧の外側にある文脈語（対象を修飾する語）。
 * 「夏目漱石の小説『東京の空』」なら夏目漱石——その作品が実在する
 * 前提の材料であり、裏づけ記事がこの文脈を満たさなければ、その記事は
 * 同名の別物（同名のアルバム等）だと疑う。種類・役割・質問形式の語は
 * 文脈に数えない（『羅生門』の作者は…の「作者」は文脈ではない）。
 */
const CONTEXT_STOP_TERMS = new Set([
  "作者", "著者", "監督", "小説", "映画", "アニメ", "漫画", "本社", "本店",
  "高さ", "標高", "海抜", "全長", "長さ", "重さ", "面積", "人口", "人数",
  "発売日", "所在地", "場所", "名前", "意味", "首都", "国歌", "名物", "特産",
  "運営会社", "運営元", "社長", "会長", "知事", "総理大臣", "大統領", "国王",
  "首相", "創業者", "代表", "駅", "路線",
]);

function questionContextTerms(question) {
  if (!subjectOf(question)) return [];
  const outside = String(question ?? "").replace(/『[^』]*』/g, "");
  return keyTerms(normalizeJa(outside)).filter((t) => {
    if (/^\d/.test(t)) return false;
    if (t.length < 2) return false;
    if (EVIDENCE_STOP_TERMS.has(t)) return false;
    if (CONTEXT_STOP_TERMS.has(t)) return false;
    if (/教えて|ください|ですか|ますか|でしょう|知りたい|聞きたい|ほしい/.test(t)) {
      return false;
    }
    return true;
  });
}

/** そのテキストが質問の文脈語を（1つでも）満たしているか。 */
function contextSatisfiedByText(text, question) {
  const terms = questionContextTerms(question);
  if (terms.length === 0) return true;
  const nt = normalizeJa(text ?? "");
  return terms.some((t) => nt.includes(normalizeJa(t)));
}

/**
 * 1文の裏づけ。{source, via} か null を返す。
 * 経路A：種類つき事実との一致（出典＝事実の記事）。
 * 経路B：窓照合（findCorroboratingResult の厳格さそのまま）。
 * 経路C：定義質問専用の冒頭窓照合。
 * 経路B/Cの前には2つの関門を置く：
 * - 種類の適合：質問の種類と文の主張する種類が食い違えば裏づけない
 *   （ビルを聞かれて山の答えが窓照合を通る、を防ぐ。本社と場所は
 *   どちらも所在地の答えなので相互に許す）。
 * - 文脈の適合：『…』の質問では、文脈語を満たす記事だけで裏づける
 *   （同名別物の記事で裏づけられないようにする）。
 */
/**
 * 場所の事実が複数の都道府県を値に持つのに、答えが一部しか言って
 * いないか（「富士山は静岡県にあります」vs 事実「静岡県と山梨県」）。
 * 部分的な答えを「事実と一致した」ことにしないための点検。
 */
function placeAnswerMissesFactValue(modelAnswer, fact, question) {
  if (!fact || classifyQuestion(question) !== "place") return false;
  const factPrefs = new Set(
    (fact.value ?? "").match(/[一-鿿々]{1,5}[都道府県]/g) ?? [],
  );
  if (factPrefs.size < 2) return false;
  const ansPrefs = new Set(
    String(modelAnswer ?? "").match(/[一-鿿々]{1,5}[都道府県]/g) ?? [],
  );
  return [...factPrefs].some((p) => !ansPrefs.has(p));
}

/**
 * 場所の文の窓照合の完全性：裏づけ窓が「A県とB県」のように複数の
 * 都道府県を並べているのに、文が一部しか言っていない場合は、部分的な
 * 答えを裏づけられたことにしない。
 */
function placeWindowComplete(sentence, question, source) {
  if (!source || !source.text) return true;
  const content = effectiveContentTerms(sentence, question);
  const inSentence = new Set(
    sentence.match(/[一-鿿々]{1,5}[都道府県]/g) ?? [],
  );
  for (const w of evidenceWindows(source)) {
    if (!windowSupportsAnswer(w, sentence, content)) continue;
    const prefs = new Set(w.match(/[一-鿿々]{1,5}[都道府県]/g) ?? []);
    const extra = [...prefs].filter((p) => !inSentence.has(p));
    if (extra.length > 0 && /(と|、|および|またが)/.test(w)) return false;
  }
  return true;
}

/**
 * 順位・最上級の質問に対する窓の追加条件：裏づけ記事の中に、同じ順位
 * の印と対象の頭名詞を持ち、文の数字と主張語（少なくとも1つ）を
 * 含む文が1つあること。加えて、その文が質問と同じ属性・向きで、
 * 答えの名指す勝者を名指していること（superlativeSentenceAgrees）：
 * 順位と頭名詞だけでは、別の属性で1位の対象が裏づけになってしまう。
 */
function rankWindowSupports(result, sentence, question, qRank) {
  const headNoun = superlativeHeadNoun(question);
  const content = gateContentTerms(sentence, question);
  const nums = numbersIn(sentence);
  const texts = [result?.text ?? "", result?.snippet ?? ""];
  for (const t of texts) {
    if (!t) continue;
    for (const ws of splitForGate(t)) {
      if (superlativeRank(ws) !== qRank) continue;
      if (headNoun && !ws.includes(headNoun)) continue;
      const w = normalizeUnits(normalizeJa(ws)).replace(/,/g, "");
      if (!nums.every((n) => w.includes(String(n).replace(/,/g, "")))) {
        continue;
      }
      if (
        content.length > 0 &&
        !content.some((term) => w.includes(normalizeJa(term)))
      ) {
        continue;
      }
      if (!superlativeSentenceAgrees(ws, question, sentence)) continue;
      return true;
    }
  }
  return false;
}

/**
 * 文の頭の主題（主語）を取り出す。記事文の流用で主語がすり替わって
 * いないかを機械的に比べるためのもので、修飾節（「〜に位置する
 * 田沢湖は」の位置する節）や法人格の前置（株式会社）は落とし、
 * 主題の名詞句だけを返す。読み取れなければ null（その場合は
 * 主語の点検をしない＝従来どおりの照合に任せる）。
 */
function sentenceHeadSubject(text) {
  const first = String(text ?? "").split(/[。！？\n]/)[0] ?? "";
  const s = normalizeJa(first).replace(/^『/, "").trim();
  const m = /^(.{2,40}?)(?:とは|は)/.exec(s);
  if (!m) return null;
  let subj = (m[1].split(/[、,]/).pop() ?? "").trim();
  const tail =
    /(?:する|ある|いる|なる)([一-鿿々ァ-ヶーA-Za-z0-9・]{2,})$/.exec(subj);
  if (tail) subj = tail[1];
  subj = subj
    .replace(/^(株式会社|有限会社|合同会社|一般社団法人|一般財団法人)/, "")
    .replace(/[（(][^（）()]*[）)]$/, "")
    .replace(/『|』/g, "")
    .trim();
  return subj.length >= 2 ? subj : null;
}

/**
 * 2つの主題が「別の対象」を指すか。どちらかが読み取れない場合は
 * 食い違いとは言わない。包含（末尾一致）は同じ対象の表記ゆれ・
 * 修飾の有無として許す。
 */
function headSubjectsConflict(a, b) {
  if (!a || !b) return false;
  if (a === b) return false;
  if (a.endsWith(b) || b.endsWith(a)) return false;
  return true;
}

/** その主題が記事タイトル（記事自身の主題）を指しているか。 */
function titleNamesSubject(title, subj) {
  if (!title || !subj) return false;
  if (titleMatchesSubject(title, subj)) return true;
  const nt = normalizeJa(title);
  const ns = normalizeJa(subj);
  if (!nt || !ns) return false;
  const base = nt.replace(/[（(].*?[）)]/g, "").trim();
  return base === ns || nt.startsWith(ns) || base.startsWith(ns);
}

/**
 * 定義の事実（記事の冒頭文そのもの）との主語すり替え：候補文の
 * 主題が、記事文の主題と別の対象を指している場合は、その文は
 * 記事の主張を別対象の話に化けさせているので、事実では裏づけない
 * （モーリーファンタジーの質問にイオンファンタジーの冒頭文を
 * 主語だけ差し替えて通した 2026-11-19 実測 a6 の経路）。点検は、
 * 事実文の主題が記事タイトル（記事自身の主題）と一致する場合に
 * 限る（記事の主題でない文との食い違いは、従来の照合に任せる）。
 */
function definitionSubjectSwapped(sentence, fact) {
  if (!fact || fact.kind !== "definition") return false;
  const factSubject = sentenceHeadSubject(fact.value);
  if (!titleNamesSubject(fact.sourceTitle, factSubject)) return false;
  return headSubjectsConflict(sentenceHeadSubject(sentence), factSubject);
}

/* 最上級の属性ごとの、証拠文側の言い方（最長・最高などの最X形を含む）。 */
const SUPERLATIVE_ATTR_FORMS = [
  [/人口/, ["人口", "最多", "最少"]],
  [/面積/, ["面積", "最大", "最小"]],
  [/長さ|全長|長い|短い/, ["長さ", "全長", "長い", "最長", "最短"]],
  [/高さ|標高|高い|低い/, ["高さ", "標高", "高い", "最高", "最低"]],
  [/大きさ|大きい|小さい/, ["大きさ", "大きい", "大きな", "面積", "最大", "最小"]],
  [/重さ|重量|重い|軽い/, ["重さ", "重量", "重い", "最大", "最小"]],
  [/速さ|速度|速い|遅い/, ["速さ", "速度", "速い", "最速", "最遅"]],
  [/深さ|深い|浅い/, ["深さ", "深い", "水深", "最大"]],
];

/**
 * 証拠側の1文が、勝者を「その順位の持ち主」として主張しているか。
 * 勝者の名前が文に出ているだけ（候補の列挙・別対象の文への出現）
 * では主張とは言わない：勝者が主題・主語として最上級を担う形、
 * 「のは勝者」「は勝者である」の名指し形、または勝者自身の記事
 * （タイトル一致）の中の順位文であることを求める。属性の系統と
 * 向き（大きい側／小さい側）も質問と矛盾しないこと。最X形
 * （最長・最高など）も順位の印として数える（superlativeRank は
 * 最X形を見ないため、ここでは印の集合を広げてある）。
 */
function sentenceAssertsWinner(ls, winner, question, titleEq) {
  const w = normalizeJa(winner);
  const s = normalizeJa(ls);
  if (!w) return false;
  if (/ではなく|でなく|誤り|間違い/.test(ls)) return false;
  const attr = SUPERLATIVE_ATTR_FORMS.find(([re]) => re.test(question));
  if (attr && !attr[1].some((x) => ls.includes(x))) return false;
  const qRank = superlativeRank(question);
  const rankOk =
    superlativeRank(ls) === qRank ||
    (qRank === 1 &&
      /第1位|世界一|日本一|首位|最[長短高低大小速重広狭深浅新古多]/.test(ls));
  if (!rankOk) return false;
  const qDir = superlativeDirection(question);
  const d = superlativeDirection(ls);
  if (qDir !== 0 && d !== 0 && d !== qDir) return false;
  if (titleEq) return true;
  if (!s.includes(w)) return false;
  const we = escapeRegExp(w);
  if (s.includes(`${w}が最も`) || s.includes(`${w}が一番`)) return true;
  if (
    new RegExp(`${we}が[^、。]{0,10}(?:最|第1位|世界一|日本一)`).test(s)
  ) {
    return true;
  }
  if (s.includes(`のは${w}`)) return true;
  if (new RegExp(`は${we}(?:である|です|だ)(?!という)`).test(s)) {
    return true;
  }
  const idx = s.indexOf(`${w}は`);
  if (idx >= 0) {
    const clause = s.slice(idx, idx + 60).split(/[、。]/)[0] ?? "";
    if (
      /最|一番|第1位|世界一|日本一/.test(clause) &&
      !/と言われ|という説|という主張|という意見|こともある/.test(clause)
    ) {
      return true;
    }
  }
  return false;
}

/**
 * 最上級の質問への答えが名指す勝者が、証拠の中で「勝者として
 * 主張」されているか。名指しが読み取れない文は対象外（従来の
 * 照合に任せる）。2026-11-19 実測 c12：根拠ページが「ナイル川は
 * 世界最長」と明記しているのに「アマゾン川です」が通った——
 * 勝者の名前が候補として文に出ているだけで裏づけになっていた。
 */
function superlativeWinnerGrounded(sentence, question, ctx) {
  const winner = superlativeClaimWinner(sentence);
  if (!winner || normalizeJa(winner).length < 2) return true;
  const sources = [];
  for (const r of ctx?.results ?? []) {
    if (r) sources.push(r);
  }
  if (ctx?.article) {
    sources.push({
      title: ctx.article.title ?? "",
      text: ctx.article.lead ?? "",
      snippet: "",
    });
  }
  if (ctx?.fact) {
    sources.push({
      title: ctx.fact.sourceTitle ?? "",
      text: ctx.fact.evidence ?? "",
      snippet: "",
    });
  }
  for (const r of sources) {
    const titleEq =
      titleMatchesSubject(r.title ?? "", winner) ||
      normalizeJa((r.title ?? "").replace(/[（(][^（）()]*[）)]\s*$/, "")) ===
        normalizeJa(winner);
    const texts = [r.text ?? "", r.snippet ?? ""];
    for (const t of texts) {
      if (!t) continue;
      for (const ls of splitSentences(t)) {
        if (sentenceAssertsWinner(ls, winner, question, titleEq)) {
          return true;
        }
      }
    }
  }
  return false;
}

function verifySentence(sentence, question, ctx) {
  const fact = ctx?.fact ?? null;
  if (
    fact &&
    modelAgreesWithFact(sentence, fact) &&
    !placeAnswerMissesFactValue(sentence, fact, question) &&
    !definitionSubjectSwapped(sentence, fact)
  ) {
    return {
      source: { title: fact.sourceTitle, url: fact.sourceUrl },
      via: "fact",
    };
  }
  const qKind = classifyQuestion(question);
  const aKind = answerClaimKind(sentence);
  if (qKind && aKind && qKind !== aKind) {
    const locationPair =
      (qKind === "hq" && aKind === "place") ||
      (qKind === "place" && aKind === "hq");
    if (!locationPair) return null;
  }
  // 質問が特定の英数字トークン（Fe、H2O等）を名指している場合、
  // そのトークンに触れていない文は質問への答えとして裏づけない
  // （Feを聞かれて元素記号一般の定義が照合を通る、を防ぐ）。
  const latinTokens = (
    String(question).match(/[A-Za-z][A-Za-z0-9]{1,}/g) ?? []
  ).map((t) => t.toLowerCase());
  if (latinTokens.length > 0) {
    const ls = String(sentence).toLowerCase();
    if (!latinTokens.every((t) => ls.includes(t))) return null;
  }
  // 最上級の質問で勝者を名指す文は、証拠がその勝者を「勝者として
  // 主張」している場合に限る。名前が候補として挙がっているだけ・
  // 別の勝者を主張する記事に出てくるだけでは裏づけない（c12型）。
  if (
    superlativeRank(question) &&
    !superlativeWinnerGrounded(sentence, question, ctx)
  ) {
    return null;
  }
  let results = ctx?.results ?? [];
  if (questionContextTerms(question).length > 0) {
    results = results.filter((r) =>
      contextSatisfiedByText(
        `${r?.title ?? ""}\n${(r?.text ?? "").slice(0, 600)}`,
        question,
      ),
    );
  }
  const src = findCorroboratingResult(sentence, results, question);
  if (src) {
    if (qKind === "place" && !placeWindowComplete(sentence, question, src)) {
      return null;
    }
    // 順位・最上級の質問では、裏づけ窓にも順位の印と頭名詞が要る
    // （値と数字がそろうだけの窓は順位を裏づけない）。満たさなければ
    // この経路は不成立として、定義経路の順位つき窓を探し直す。
    const qRank = superlativeRank(question);
    if (qRank && !rankWindowSupports(src, sentence, question, qRank)) {
      // fall through
    } else {
      return { source: src, via: "window" };
    }
  }
  // 最上級の質問への勝者型の答えは、勝者の記事自身が最上級を
  // 自称しているかで裏づける（通常の窓照合では、順位の文と数字の
  // 文が分かれているだけで正しい答えまで落ちてしまうため）。
  const ssrc = superlativeWinnerSource(sentence, results, question);
  if (ssrc) return { source: ssrc, via: "superlative" };
  if (qKind === "definition") {
    const dsrc = definitionSentenceSource(sentence, results, question);
    if (dsrc) return { source: dsrc, via: "definition" };
  }
  return null;
}

/**
 * モデルの答え全体の検証ゲート。
 * 返り値：{ status, text, source, perSentence }
 * - status: "all" | "partial" | "none" | "abstain" | "nonclaim"
 * - text: 表示してよい文だけをつないだもの（all は元の答えそのまま）
 * - source: 最初に裏づけをくれた記事（出典表示用）
 */
function verifyModelAnswer(modelAnswer, question, ctx) {
  // 同語反復の答え（「日本で一番高いビルは日本で一番高いビルです」）
  // は中身の無い文であり、どの文も裏づけない。断片化した語の偶然の
  // 一致で照合を通ることがあるため、入り口で落とす。
  if (isTautologicalAnswer(modelAnswer)) {
    return {
      status: "none",
      text: "",
      source: null,
      perSentence: splitForGate(modelAnswer).map((s) => ({
        sentence: s,
        kind: "claim",
        verified: false,
        via: "tautology",
      })),
    };
  }
  // 崩壊出力（同じ並びの繰り返し）は、どの文も裏づけない。
  if (isDegenerateAnswer(modelAnswer)) {
    return {
      status: "none",
      text: "",
      source: null,
      perSentence: splitForGate(modelAnswer).map((s) => ({
        sentence: s,
        kind: "claim",
        verified: false,
        via: "degenerate",
      })),
    };
  }
  const sentences = splitForGate(modelAnswer);
  const perSentence = [];
  let claimCount = 0;
  let verifiedCount = 0;
  let firstSource = null;
  let hasAbstention = false;
  const kept = [];
  for (const s of sentences) {
    if (isAbstentionSentence(s, question)) {
      hasAbstention = true;
      perSentence.push({ sentence: s, kind: "abstention", verified: true });
      kept.push(s);
      continue;
    }
    const isClaim = isClaimSentence(s, question);
    if (!isClaim) {
      perSentence.push({ sentence: s, kind: "filler", verified: true });
      kept.push(s);
      continue;
    }
    claimCount += 1;
    const v = verifySentence(s, question, ctx);
    if (v) {
      verifiedCount += 1;
      if (!firstSource) firstSource = v.source;
      perSentence.push({
        sentence: s,
        kind: "claim",
        verified: true,
        via: v.via,
      });
      kept.push(s);
    } else {
      perSentence.push({ sentence: s, kind: "claim", verified: false });
    }
  }
  if (claimCount === 0) {
    return {
      status: hasAbstention ? "abstain" : "nonclaim",
      text: String(modelAnswer ?? ""),
      source: null,
      perSentence,
    };
  }
  if (verifiedCount === 0) {
    return { status: "none", text: "", source: null, perSentence };
  }
  if (verifiedCount === claimCount) {
    return {
      status: "all",
      text: String(modelAnswer ?? ""),
      source: firstSource,
      perSentence,
    };
  }
  return {
    status: "partial",
    text: kept.join(""),
    source: firstSource,
    perSentence,
  };
}

/**
 * 「争いのある事実」か：定義の質問で、事実の記事タイトルに限定
 * （（…））が付くか、検索結果に曖昧さ回避ページが含まれる場合。
 * 同名で意味が複数ある語の定義は、1記事の冒頭だけでは質問の意味だと
 * 断定できない（H2O＝水 vs H2O (Webサーバ) の事故の一般化）。
 */
function factIsContested(fact, results) {
  if (!fact || fact.kind !== "definition") return false;
  if (/[（(][^（）()]+[）)]/.test(fact.sourceTitle ?? "")) return true;
  return (results ?? []).some((r) => /曖昧さ回避/.test(r?.title ?? ""));
}

/**
 * 質問の対象が「見つからない」状態か：対象が特定でき、対象記事が
 * 取れず、検索は実行できたのにタイトルに対象を持つ結果が1件も無い。
 * 実在しないもの（架空の建物・作品・製品）への捏造表示を止める判定。
 * 検索の接続失敗時は false（見つからないのではなく確かめられなかった）。
 */
function entityNotFound(question, article, results, searchError) {
  if (searchError) return false;
  if (article) return false;
  const entity = subjectOf(question) ?? topicFromQuestion(question);
  if (!entity || normalizeJa(entity).length < 2) return false;
  return !(results ?? []).some((r) =>
    titleMatchesSubject(r?.title ?? "", entity),
  );
}

/**
 * 数値の事実の値が、根拠文の中で「年」としてしか現れないか。
 * 例：エベレストの標高の根拠文「1954年にインド測量局が…8848 m」から
 * 抽出が「1,954メートル」を作ってしまう事故——年を測った値を事実と
 * して出さないための点検。値の数字の出現がすべて「年」の直前なら
 * 年とみなす（どこか1か所でも単位等を伴って現れれば事実として通す）。
 */
function factValueIsYearOnly(fact) {
  const nums = numbersIn(fact?.value ?? "");
  if (nums.length === 0) return false;
  const ev = toHalfWidth(fact.evidence ?? "").replace(/,/g, "");
  const n = String(Math.trunc(Number(nums[0])));
  if (!n || !ev.includes(n)) return false;
  let idx = 0;
  let found = 0;
  while (true) {
    const i = ev.indexOf(n, idx);
    if (i < 0) break;
    found += 1;
    if (ev[i + n.length] !== "年") return false;
    idx = i + n.length;
  }
  return found > 0;
}

/**
 * 種類つき事実を表示・保存に使う前の点検（事実も検証を通す）：
 * - 出典記事が質問の対象と関係していること（アンカー）。
 * - 『…』の質問では、記事の冒頭か根拠文が質問の文脈語を満たすこと
 *   （同名別物の記事の事実を出さない）。
 * - 数値の事実は、値が根拠文で「年」としてしか現れないものは落とす。
 * どれかに落ちれば事実なしとして進む。
 */
function sanitizeFact(fact, article, question) {
  if (!fact) return null;
  const entity = subjectOf(question) ?? topicFromQuestion(question);
  if (entity && !factSourceRelatedToEntity(fact, article, entity)) {
    return null;
  }
  if (
    !contextSatisfiedByText(
      `${article?.lead ?? ""}\n${fact.evidence ?? ""}`,
      question,
    )
  ) {
    return null;
  }
  if (fact.kind === "measure" && factValueIsYearOnly(fact)) return null;
  return fact;
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
  if (/元素記号/.test(a)) return "attribute";
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
  if (/国歌|首都|元素記号/.test(q)) return "attribute";
  // 紙幣の肖像：「新しい千円札の肖像は誰ですか」は人物の質問として
  // 扱い、紙幣の記事から肖像の人物を取りに行く（extractPortrait）。
  if (/肖像/.test(q) && /(誰|どなた|何|なに|どの|どれ)/.test(q)) {
    return "person";
  }
  if (/いつ|何年|設立|創立|創業|結成|完成|竣工|開業|できた/.test(q)) {
    return "date";
  }
  if (
    /高さ|標高|海抜|全長|長さ|重さ|面積|人口|何メートル|何キロ|いくつ|何人|何円|何歳|何本|何個|何枚/.test(
      q,
    )
  ) {
    return "measure";
  }
  if (/(どこ|所在地|場所|ある？|ありますか)/.test(q)) return "place";
  // 「なんですか／なん？」は「何ですか／なに？」の口語形。ここに無いと
  // 「モーリーファンタジーはなんですか」が未分類のまま事実を取りに
  // 行けなかった（とは形は「とは」自体で当たるため無事だった）。
  if (
    /とは|何ですか|なんですか|何が|何を|何に|何から|なに？|何？|なん？|どんなもの|意味|教えて|教え|知りたい|聞きたい/.test(
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
    /(.+?)に本社を(?:置く|置き|置いて|置いた)/,
    /(.+?)に本店を(?:置く|置き|置いて|置いた)/,
    /本社所在地は、?(.{3,50}?)(?:である|です|にあたる)[。]?/,
    /本社は、?(.{3,40}?)(?:に所在|に位置|にあります)/,
    /本社が?、?(.{3,40}?)に所在/,
  ];
  for (const re of patterns) {
    const hit = findSentence(article.lead, (s) => re.test(s), 14);
    if (!hit) continue;
    const m = re.exec(hit);
    if (!m?.[1]) continue;
    // 捕捉部は文頭（会社名）から始まるため、そのまま placeFromSpan に
    // 渡すと会社名の途中から切れてしまう。都道府県名（漢字2〜4字＋
    // 都・府・県）の最初の出現位置から切り直してから場所として
    // 整える（「新都心」の都のような紛れを避けるため、直前が文の
    // 区切りや助詞である出現を優先する）。
    let cut = m[1];
    let idx = -1;
    let firstIdx = -1;
    for (const mm of cut.matchAll(/(?:[一-鿿]{2,4}(?:都|府|県)|北海道)/g)) {
      if (firstIdx < 0) firstIdx = mm.index;
      const prev = mm.index > 0 ? cut[mm.index - 1] : "";
      if (mm.index === 0 || /[はが、）)\s　]/.test(prev)) {
        idx = mm.index;
        break;
      }
    }
    if (idx < 0) idx = firstIdx;
    if (idx > 0) cut = cut.slice(idx);
    const place = placeFromSpan(cut);
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
  // 発祥地：「讃岐うどんは、香川県（旧讃岐国）のうどんである」の形。
  // 料理自身の記事が本場を自称している場合だけ、その土地を答えにする。
  if (/発祥/.test(question)) {
    const hit = findSentence(
      article.lead,
      (s) =>
        /の(?:郷土料理|名物|名産品?|うどん|料理|食べ物)/.test(s) ||
        /発祥/.test(s),
      8,
    );
    if (hit) {
      const m =
        /は、?(.{2,20}?)(?:（[^）]*）)?の(?:郷土料理|名物|名産品?|うどん|料理|食べ物)/.exec(
          hit,
        ) ?? /発祥地は、?(.{2,20}?)(?:です|である|にあ)/.exec(hit);
      const place = m?.[1] ? placeFromSpan(m[1]) : null;
      if (place) {
        return makeFact(
          article,
          "place",
          place,
          `${label}の発祥地は${place}です。`,
          hit,
        );
      }
    }
  }
  // 複数の都道府県に跨る対象（富士山は山梨県と静岡県）：所在地が
  // 1つに定まらないため、「XとYに跨る」の形そのものを事実にする。
  {
    const hit = findSentence(article.lead, (s) => /に跨[がりる]/.test(s), 8);
    if (hit) {
      const m =
        /([一-鿿]{2,10}?(?:都|道|府|県)(?:（[^）]*）)?と[一-鿿]{2,10}?(?:都|道|府|県)(?:（[^）]*）)?)に跨/.exec(
          hit,
        );
      if (m) {
        const value = m[1].replace(/（[^）]*）/g, "");
        return makeFact(
          article,
          "place",
          value,
          `${label}は${value}に跨っています。`,
          hit,
        );
      }
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
  // 出来事の語幹フォールバック：質問が名指す出来事（施行・初出場・
  // 着陸など）を含む文から、その出来事の節の日付だけを取る。節単位で
  // 見るのは、「1946年…公布、1947年…施行」のように複数の出来事と
  // 日付が並ぶ文で、聞かれた出来事の日付を取り違えないため。節で
  // 決まらない場合は、文中の年が1種類だけのときに限って文全体から取る。
  const stemMatch =
    /(?:が|に|を|は)([一-鿿々]{2,7}?)(?:された|した|する|だった)(?:のは|のが|いつ|何年|何月|何日)/.exec(
      question,
    ) ??
    /(?:が|に|を|は)([一-鿿々]{2,7}?)(?:された|した)(?:こと|年|日)/.exec(
      question,
    );
  const stem = stemMatch?.[1];
  if (stem && !/設立|創立|創業|結成|完成|竣工|開業/.test(stem)) {
    const hitStem = findSentence(
      article.lead,
      (s) => s.includes(stem) && dateRe.test(s),
      60,
    );
    if (hitStem) {
      let value = null;
      for (const clause of hitStem.split(/[、；;]/)) {
        if (!clause.includes(stem)) continue;
        const m = dateRe.exec(clause.replace(/（[^）]*）/g, ""));
        if (m) {
          value = m[0];
          break;
        }
      }
      if (!value) {
        const bare = hitStem.replace(/（[^）]*）/g, "");
        const years = [
          ...new Set([...bare.matchAll(/(\d{3,4})年/g)].map((x) => x[1])),
        ];
        if (years.length === 1) {
          const m = dateRe.exec(bare);
          if (m) value = m[0];
        }
      }
      if (value) {
        return makeFact(
          article,
          "date",
          value,
          `${label}の${stem}は${value}です。`,
          hitStem,
        );
      }
    }
  }
  return null;
}

const MEASURE_SPECS = [
  {
    // 速さは「何キロメートル」と聞かれるため、重さの判定（何キロ）より
    // 先に見る。値は「秒速340m」の形でリードに現れることが多い。
    test: /速さ|速度/,
    fields: ["速さ", "速度"],
    name: "速さ",
    defaultUnit: "メートル",
    leadKeyword: /速さ|速度|秒速/,
    leadPatterns: [
      /秒速\s*(\d+(?:\.\d+)?)\s*(?:メートル|m(?![a-zA-Z]))/,
      /毎秒\s*(\d+(?:\.\d+)?)\s*(?:メートル|m(?![a-zA-Z]))/,
    ],
    valuePrefix: "秒速",
  },
  {
    // 距離も「何キロメートル」と聞かれるため重さより先に見る。
    test: /距離/,
    fields: ["距離"],
    name: "距離",
    defaultUnit: "キロメートル",
    leadKeyword: /距離/,
    leadPatterns: [
      /平均距離は\s*([\d,]+(?:\.\d+)?[億万]?[\d,千]*)\s*(?:km|キロメートル)/,
      /距離は\s*([\d,]+(?:\.\d+)?[億万]?[\d,千]*)\s*(?:km|キロメートル)/,
    ],
  },
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
  {
    // 骨の数：「成人になるにつれ…206個になる」の形でリードに現れる。
    // 生まれた時の数（270個）と取り違えないよう、成人の言及がある
    // パターンを優先する。
    test: /骨.*(?:何本|何個)|(?:何本|何個).*骨/,
    fields: [],
    name: "骨の数",
    defaultUnit: "個",
    leadKeyword: /骨/,
    leadPatterns: [
      /(?:成人|大人)[^。]{0,40}?(\d+)\s*個になる/,
      /(\d+)\s*個の骨からなる/,
    ],
  },
];

const MEASURE_RE =
  /([\d,]+(?:\.\d+)?)\s*(メートル|キロメートル|センチメートル|ミリメートル|平方キロメートル|キログラム|グラム|トン|km2|km²|km|cm|mm|kg|m|t|g|人|名|円|歳)?/;

// 億・万を含む表記（「14億人」「38万4,403km」）：素の正規表現だと
// 「14」だけを拾って「14人」という誤った事実を作ってしまうため、
// 億・万の塊ごと表面の表記を値として保持する。
const MEASURE_RE_JP =
  /([\d,]+(?:\.\d+)?億(?:[\d,千]+万)?|[\d,]+(?:\.\d+)?万(?:[\d,千]+)?(?:\.\d+)?)\s*(メートル|キロメートル|センチメートル|ミリメートル|平方キロメートル|キログラム|グラム|トン|km2|km²|km|cm|mm|kg|m|t|g|人|名|円|歳)?/;

function measureValue(raw, defaultUnit) {
  const s = raw.replace(/,/g, ",");
  const mj = MEASURE_RE_JP.exec(s);
  const m = mj ?? MEASURE_RE.exec(s);
  if (!m || !m[1]) return null;
  const num = mj ? m[1] : withCommas(m[1]);
  const unitRaw = m[2] ?? defaultUnit;
  const unit = UNIT_NORM[unitRaw] ?? unitRaw;
  return `${num}${unit}`;
}

function extractMeasure(question, article, label) {
  // チームの人数：「サッカーの1チームの人数は何人ですか」は人口の
  // 質問ではない。記事の中で「1チームN人／N人制」の形が1種類だけ
  // 現れる場合に限って取り出す（15人制と13人制が併記される記事では
  // どちらの人数か決められないため取り出さない）。
  if (/チーム/.test(question) && /(何人|人数)/.test(question)) {
    const values = new Set();
    for (const m of (article.lead ?? "").matchAll(
      /1チーム(?:は|が|の)?[^。]{0,10}?(\d+)\s*人/g,
    )) {
      values.add(m[1]);
    }
    for (const m of (article.lead ?? "").matchAll(/(\d+)\s*人制/g)) {
      values.add(m[1]);
    }
    if (values.size === 1) {
      const n = [...values][0];
      const hit =
        findSentence(article.lead, (s) => /1チーム\d+\s*人|\d+\s*人制/.test(s), 20) ??
        article.lead.slice(0, 200);
      return makeFact(
        article,
        "measure",
        `${n}人`,
        `${label}の1チームの人数は${n}人です。`,
        hit,
      );
    }
  }
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
  // 専用パターン（秒速N・平均距離はN・成人はN個）：パターンの優先順に
  // リード全体から探す。一般のキーワード探索より先に試すのは、
  // 「0℃では毎秒331.5メートル」のような条件つきの値より先に、
  // 標準の値（「一般に音速を秒速340m」）を拾うため。
  if (spec.leadPatterns) {
    for (const re of spec.leadPatterns) {
      const hitP = findSentence(article.lead, (s) => re.test(s), 120);
      if (!hitP) continue;
      const m = re.exec(hitP);
      if (m?.[1]) {
        const value = `${spec.valuePrefix ?? ""}${m[1]}${spec.defaultUnit}`;
        return makeFact(
          article,
          "measure",
          value,
          `${label}の${spec.name}は${value}です。`,
          hitP,
        );
      }
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

/**
 * 紙幣の肖像：紙幣の記事は「2024年（令和6年）から発行されている
 * 北里柴三郎の肖像のF号券である」の形で、発行年と肖像の人物を並べる。
 * 質問が年を名指していればその年の肖像だけを、名指していなければ
 * 記事中の肖像が1人だけのときに限って取り出す（旧券と新券の肖像が
 * 併記される記事で、どちらか分からない肖像は出さない）。
 */
function extractPortrait(question, article, label) {
  const re =
    /(\d{4})年[^。]{0,32}?から発行されている([^、。\s]{2,12})の肖像/g;
  const found = [];
  for (const m of (article.lead ?? "").matchAll(re)) {
    found.push({ year: m[1], name: (m[2] ?? "").trim() });
  }
  if (found.length === 0) return null;
  const qYear = /(\d{4})年/.exec(question)?.[1];
  let pick = null;
  if (qYear) {
    pick = found.find((f) => f.year === qYear) ?? null;
    if (!pick) return null;
  } else {
    const names = [...new Set(found.map((f) => f.name))];
    if (names.length !== 1) return null;
    pick = found[0];
  }
  if (!pick.name || pick.name.length < 2) return null;
  const hit =
    findSentence(
      article.lead,
      (s) => s.includes(pick.name) && s.includes("肖像"),
      20,
    ) ?? article.lead.slice(0, 200);
  return makeFact(
    article,
    "person",
    pick.name,
    `${label}の肖像は${pick.name}です。`,
    hit,
  );
}

function extractPerson(question, article, label) {
  if (/肖像/.test(question)) {
    return extractPortrait(question, article, label);
  }
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
  { test: /元素記号/, field: "元素記号", name: "元素記号", quote: false },
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
  if (spec.field === "元素記号") {
    // 元素の記事は冒頭で「元素記号はFe」と自称する。その形だけを拾う。
    const re = /元素記号は\s*([A-Z][a-z]?\d?)\b/;
    const hit = findSentence(article.lead, (s) => re.test(s), 6);
    if (hit) {
      const m = re.exec(hit);
      const value = (m?.[1] ?? "").trim();
      if (value) {
        return makeFact(
          article,
          "attribute",
          value,
          `${label}の元素記号は${value}です。`,
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

  /* Search round 2 (2026-11-17): LRU cache helpers. The engine's
   * caches used to be wiped wholesale once they passed a size cap
   * (this._blocks.clear() at 257 entries, _idRecs/_leads.clear() at
   * 8193, _termCache.clear() at 513), and the lane transport wiped its
   * whole 96MB range cache on overflow. Measurement notes: the
   * _blocks cap (256 shared ~64KB blocks across titles/redir/tri
   * directories) and the transport wipe were the refetch multipliers
   * that actually fired; the record caches' clear-all was a cliff
   * that rarely fired because primes bypass it, so their LRU caps
   * are sized as safety valves (see below), not working-set limits.
   * IO-only change: cached values are identical, ranking untouched. */
  function cacheGet(map, key) {
    const v = map.get(key);
    if (v !== undefined) { map.delete(key); map.set(key, v); }
    return v;
  }
  function cacheSet(map, key, val, cap) {
    if (map.has(key)) map.delete(key);
    map.set(key, val);
    while (map.size > cap) map.delete(map.keys().next().value);
  }
  /* Cap sizes (search round 2, revised after measurement): the
   * prime path caches every gid inside a fetched ids span, and later
   * primes/searches rely on those span-covered records being present
   * (restricting the cache to requested gids only measured WORSE:
   * ids.bin requests x7, bytes x3.9). So the caps are sized to hold
   * the realistic session working set — up to the full 1.52M-record
   * ids table — and act purely as LRU safety valves against the old
   * clear-all cliff, not as working-set limits. */
  const BLOCKS_CACHE_BYTES = 64 * 1024 * 1024; // ~64KB blocks -> ~1000
  const IDREC_CACHE_CAP = 1600000;             // tiny fixed-size recs
  const LEAD_CACHE_CAP = 98304;                // ~1.2KB lead texts
  const TERM_CACHE_CAP = 1024;
  const TRI_ENTRY_CACHE_CAP = 32768;
  const POST_CACHE_IDS_CAP = 3000000;          // parsed posting ids

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
      this._blocksBytes = 0;      // search round 2: byte-capped LRU
      this._leads = new Map();    // gid -> {title, lead}
      this._idRecs = new Map();   // gid -> rec|null
      this._termCache = new Map(); // src|term -> Map(gid -> score)
      this._triEntries = new Map(); // (t|b)+trigram -> {off,len,df}|null
      this._postLists = new Map();  // file|off -> {p, n} parsed postings
      this._postIds = 0;            // total ids held in _postLists
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
      const hit = cacheGet(this._blocks, key);
      if (hit) return hit;
      const buf = await this._range(file, off, off + size - 1);
      if (this._blocks.has(key)) {
        this._blocksBytes -= this._blocks.get(key).length;
        this._blocks.delete(key);
      }
      this._blocks.set(key, buf);
      this._blocksBytes += buf.length;
      while (this._blocksBytes > BLOCKS_CACHE_BYTES &&
             this._blocks.size > 1) {
        const oldest = this._blocks.keys().next().value;
        if (oldest === key) break;
        this._blocksBytes -= this._blocks.get(oldest).length;
        this._blocks.delete(oldest);
      }
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
      if (this._idRecs.has(gid)) return cacheGet(this._idRecs, gid);
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
      cacheSet(this._idRecs, gid, rec, IDREC_CACHE_CAP);
      return rec;
    }
    async leadOf(rec) {
      const gid = rec.id;
      if (gid !== undefined && this._leads.has(gid)) {
        return cacheGet(this._leads, gid);
      }
      const buf = await this._range("leads.bin", rec.leadOff,
                                    rec.leadOff + rec.leadLen - 1);
      const text = td.decode(buf);
      const nl = text.indexOf("\n");
      const out = { title: nl >= 0 ? text.slice(0, nl) : text,
                    lead: nl >= 0 ? text.slice(nl + 1) : "" };
      if (gid !== undefined) {
        cacheSet(this._leads, gid, out, LEAD_CACHE_CAP);
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
        // Search round 2: span fetches 4 -> 16 concurrent ( ditto ).
        await mapLimit(spans, 16, async ([lo, hi]) => {
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
            cacheSet(this._idRecs, g, rec, IDREC_CACHE_CAP);
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
      // Search round 2: span fetches 6 -> 24 concurrent (the lane
      // transport semaphore caps total in-flight at 24).
      await mapLimit(spans, 24, async (recs) => {
        const lo = recs[0].leadOff;
        let hi = 0;
        for (const r of recs) hi = Math.max(hi, r.leadOff + r.leadLen);
        const buf = await this._range("leads.bin", lo, hi - 1);
        for (const r of recs) {
          const text = td.decode(buf.subarray(
            r.leadOff - lo, r.leadOff - lo + r.leadLen));
          const nl = text.indexOf("\n");
          cacheSet(this._leads, r.id, {
            title: nl >= 0 ? text.slice(0, nl) : text,
            lead: nl >= 0 ? text.slice(nl + 1) : "" }, LEAD_CACHE_CAP);
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
      // Search round 2: memoize the directory lookup itself. The same
      // trigrams recur across the many terms/searches of one question;
      // without this, each recurrence re-scans (and, on block-cache
      // eviction, re-fetches) a ~64KB directory block.
      const ck = (titleOnly ? "t|" : "b|") + tg;
      if (this._triEntries.has(ck)) return cacheGet(this._triEntries, ck);
      const ent = await this._trigramEntryFetch(tg, titleOnly);
      cacheSet(this._triEntries, ck, ent, TRI_ENTRY_CACHE_CAP);
      return ent;
    }
    async _trigramEntryFetch(tg, titleOnly) {
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
      // Search round 2: cache the parsed id list per postings range.
      // termScores() intersects the same term once per source, and the
      // evidence lanes re-run the same terms across searches; without
      // this, every recurrence re-parses (and, after transport-cache
      // eviction, re-fetches) up to 1MB of varints. Values are the
      // exact arrays the uncached path would return.
      const file = titleOnly ? "title_postings.bin" : "postings.bin";
      const key = file + "|" + entry.off;
      const hit = this._postLists.get(key);
      if (hit) {
        this._postLists.delete(key);
        this._postLists.set(key, hit);
        return hit.p;
      }
      const rec = { p: null, n: -1 };
      rec.p = (async () => {
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
      })();
      this._postLists.set(key, rec);
      rec.p.then(
        (ids) => {
          rec.n = ids.length;
          this._postIds += ids.length;
          while (this._postIds > POST_CACHE_IDS_CAP) {
            let evicted = false;
            for (const [k, e] of this._postLists) {
              if (e !== rec && e.n >= 0) {
                this._postLists.delete(k);
                this._postIds -= e.n;
                evicted = true;
                break;
              }
            }
            if (!evicted) break;
          }
        },
        () => {
          if (this._postLists.get(key) === rec) this._postLists.delete(key);
        },
      );
      return rec.p;
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
      if (this._termCache.has(ck)) return cacheGet(this._termCache, ck);
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
      cacheSet(this._termCache, ck, out, TERM_CACHE_CAP);
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
 * - bounded concurrency (default 24 in-flight requests; search round 2,
 *   originally 12),
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
    // Search round 2: default concurrency 12 -> 24. Per-question
    // request counts are dominated by first-touch fetches of distinct
    // index ranges (measured on the 50-question bench: 64,960 of
    // 69,606 requests are leads.bin spans), which caching cannot
    // remove; the parallel bulk (record primes, re-rank reads) is
    // semaphore-bound, so doubling in-flight requests is the lever
    // that cuts device wall time. The per-question budget (60k
    // requests / 150s in ask()) still bounds total load. NOTE: the
    // gain is a device-side model (offline shards are local disk and
    // cannot measure HF RTT); reverting is this one constant.
    const maxConc = opts.concurrency ?? 24;
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
        // A budget the host opened explicitly for one question is
        // sticky (see autoBegin). Implicit budgets are not.
        sticky: qopts.sticky !== false,
      };
    }
    function budgetActive() {
      return !!budget && !budget.aborted && Date.now() <= budget.deadline;
    }
    function autoBegin() {
      // An explicit per-question budget is sticky: once it has been
      // exceeded or has expired, later adapter calls in the same
      // question keep failing fast (checkBudget throws) instead of
      // silently minting a fresh budget — otherwise the question's
      // cap never bounds the question as a whole (same repair as the
      // fullstack lane v31). Calls made with no explicit question
      // budget (standalone seam use) keep the historical behavior of
      // refreshing an implicit, non-sticky budget.
      if (!budget) beginQuestion({ sticky: false });
      else if (!budget.sticky && !budgetActive()) {
        beginQuestion({ sticky: false });
      }
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
        // Search round 2: LRU recency refresh on hit.
        cache.delete(key);
        cache.set(key, hit);
        return hit.p;
      }
      const entry = { p: null, bytes: 0 };
      entry.p = make();
      cache.set(key, entry);
      entry.p.then(
        (buf) => {
          entry.bytes = buf.length;
          cacheBytes += buf.length;
          // Search round 2: evict oldest resolved entries down to 95%
          // of the cap instead of wiping the whole cache. The old
          // wipe-on-overflow meant one heavy question (postings alone
          // can exceed the 96MB cap) destroyed every cached range and
          // the rest of the question re-fetched blocks, bodies and
          // leads it had already paid for.
          if (cacheBytes > cacheCapBytes) {
            const target = Math.floor(cacheCapBytes * 0.95);
            for (const [k, e] of cache) {
              if (cacheBytes <= target) break;
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

  /* The host's query language is the Wikipedia API's: it builds
   * operator queries (intitle:…) and quoted phrases ("…") meant for
   * the web API. The index engine has no such operators — fed
   * verbatim, an intitle: query degrades into a keyword search over
   * the operator text and returns unrelated short titles (e.g.
   * intitle:モーリーファンタジーと -> ファンタジー, ファン, リーファ).
   * Translate at the lane boundary: strip a leading intitle: and one
   * pair of surrounding quotes. The dispatcher's web fallback keeps
   * the original query untouched. */
  function engineQuery(q) {
    let s = String(q).trim();
    s = s.replace(/^intitle:/i, "");
    if (s.length >= 2 && s.startsWith('"') && s.endsWith('"')) {
      s = s.slice(1, -1);
    }
    return s.trim();
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
        const hits = await e.search(engineQuery(q), limit);
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
        const hits = await e.search(engineQuery(q), limit, { withBody: true });
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
        // The web version only calls a page a disambiguation when it
        // has NO infobox fields; the index stores plain text, so no
        // infobox ever survives here and that condition cannot be
        // applied as-is. Closest proxy: an article that opens with a
        // substantive prose sentence naming the title is a real
        // article, even when list-shaped lines follow in the first
        // lines — e.g. イオンファンタジー, whose chronology lines
        // ("1988年（昭和63年） - …") otherwise trip `listy` and got
        // the company article misjudged as a disambiguation page.
        const baseTitle = t.replace(/[（(][^（）()]*[）)]\s*$/, "").trim();
        const firstLine = firstLines[0] ?? "";
        const opensWithProseLead =
          firstLine.length >= 40 &&
          /。$/.test(firstLine) &&
          (firstLine.includes(t) ||
            (baseTitle.length >= 2 && firstLine.includes(baseTitle)));
        return {
          title: t,
          url: deps.wikiUrl(t),
          lead: body.slice(0, 5000),
          fields: {},
          disambiguation:
            templateMark ||
            (candidates.length > 0 && listy && !opensWithProseLead),
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
      return /国歌/.test(q) ? "国歌" : /元素記号/.test(q) ? "元素記号" : "首都";
    case "person": {
      const role = /(知事|総理大臣|首相|社長|会長|創業者|創立者|肖像)/.exec(q);
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
  // 収集はラウンド単位で並列に行う。各ラウンドは「そのラウンドが
  // 従来どおり push していた順序のままの記事配列」を返し、push は
  // ラウンドの定義順に行う。集める対象・件数・push 順序・重複排除は
  // 直列時代と同一で、変わるのはラウンド同士が重なって走ることだけ
  // （各取得の失敗時の倒れ方——索引→ウェブ・失敗時は null/[]——も
  // 従来どおり）。ラウンド内の予期せぬ例外はそのラウンドだけ空に
  // 倒し、集まった分だけ返す方針を保つ。
  const rounds = [
    // 『…』の対象そのものの記事＋曖昧さ回避の派生記事。
    async () => {
      if (!subject) return [];
      const page = await wikiFetchExtract(subject);
      if (!page) return [];
      // 派生記事を先に：曖昧さ回避ページの1行項目と同点でも本記事が勝つ。
      // 候補の取得は並列に行い、並びは候補の順序どおりに保つ。
      const candPages = await Promise.all(
        disambiguationCandidates(page, subject).map((cand) =>
          wikiFetchExtract(cand),
        ),
      );
      return [...candPages, page];
    },
    // 題名そのものの記事は、検索順位に頼らず必ず先頭候補に入れる：
    // 検索の上位に来なかったという理由だけで、本命の記事が証拠から
    // 漏れて正しい答えまで「未確認」になるのを防ぐ。
    async () => {
      if (!topic || (subject && normalizeJa(topic) === normalizeJa(subject)))
        return [];
      const page = await wikiFetchExtract(topic);
      if (!page) return [];
      const candPages = await Promise.all(
        disambiguationCandidates(page, topic).map((cand) =>
          wikiFetchExtract(cand),
        ),
      );
      return [...candPages, page];
    },
    // 題名でのトピック検索。
    async () => {
      if (!topic || normalizeJa(topic) === normalizeJa(q)) return [];
      const topicTitles = await wikiSearchTitles(topic, 4);
      return Promise.all(topicTitles.map(wikiFetchExtract));
    },
    // タイトル内検索（intitle:）も併せる：題名がタイトルに含まれる記事を
    // 優先的に拾い、本文にたまたま題名が出るだけの記事の混入を減らす。
    async () => {
      if (!topic || /\s/.test(topic)) return [];
      const intitleTitles = await wikiSearchTitles(`intitle:${topic}`, 4);
      return Promise.all(intitleTitles.map(wikiFetchExtract));
    },
    // 種類の語を足した検索と、引用符つきの完全一致検索も併せる。
    // 題名だけでは同名の別対象（化学のイオン等）が上位に来て本命が
    // 埋もれることがある。集めた候補は後のスコアと厳しい照合で
    // ふるいにかけるため、網を広げても裏づけの精度は下がらない。
    async () => {
      if (!topic) return [];
      const phrase = kindSearchPhrase(q);
      if (!phrase || topic.includes(phrase)) return [];
      const kindTitles = await wikiSearchTitles(`${topic} ${phrase}`, 4);
      return Promise.all(kindTitles.map(wikiFetchExtract));
    },
    async () => {
      if (!topic || /\s/.test(topic) || topic.length < 2) return [];
      const quotedTitles = await wikiSearchTitles(`"${topic}"`, 3);
      return Promise.all(quotedTitles.map(wikiFetchExtract));
    },
    // 質問文そのままでの検索。
    async () => {
      const titles = await wikiSearchTitles(q, 4);
      return Promise.all(titles.map(wikiFetchExtract));
    },
    // 『…』の対象での検索。
    async () => {
      if (!subject) return [];
      const extra = await wikiSearchTitles(subject, 3);
      return Promise.all(extra.map(wikiFetchExtract));
    },
    // 拡張検索（後ろに足すだけ）：最上級の質問は、質問文のままだと
    // 勝者の記事が検索に現れないことが多い（「世界で一番人口が多い
    // 国」ではインドが出ない）。「属性＋範囲＋最大級」の形に言い換えた
    // 検索（「人口 世界最大」「面積 日本最小」）と、質問の大事な語だけ
    // を並べた検索（「人類 月面 着陸」）を併せ、新しく見つかった記事
    // だけを後ろに足す。既存の候補と並びは変えず、照合の厳しさも
    // 変えない（裏づけの判定は従来どおり記事の中身で行う）。
    // 拡張クエリ同士も独立なので並列に取り、並びはクエリ順に保つ。
    async () => {
      const extraQueries = [];
      if (superlativeRank(q) > 0) {
        const attr = superlativeAttrWord(q);
        const scope = /(世界|日本|太陽系|全国)/.exec(q)?.[1];
        if (attr && scope) {
          extraQueries.push(`${attr.word} ${scope}${attr.rank}`);
          const head = superlativeHeadNoun(q);
          if (head) extraQueries.push(`${head} ${attr.word} ${scope}一`);
        }
      }
      const kwTerms = keyTerms(normalizeJa(q))
        .filter((t) => !/^\d+$/.test(t) && !EXTRA_QUERY_STOP.has(t))
        .slice(0, 5);
      if (kwTerms.length >= 2) extraQueries.push(kwTerms.join(" "));
      const perQuery = await Promise.all(
        extraQueries.map(async (eq) => {
          if (normalizeJa(eq) === normalizeJa(q)) return [];
          const titles = await wikiSearchTitles(eq, 4).catch(() => []);
          return Promise.all(titles.map(wikiFetchExtract));
        }),
      );
      return perQuery.flat();
    },
  ];
  try {
    const roundResults = await Promise.all(
      rounds.map((run) => run().catch(() => [])),
    );
    for (const pages of roundResults) {
      for (const r of pages) push(r);
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
    person: /社長|会長|代表|創業者|総理大臣|首相|知事|現職|肖像/,
    operator: /運営/,
    specialty: /名物|郷土料理|特産/,
    attribute: /国歌|首都|元素記号/,
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
  // 紙幣の肖像の質問：題名は額面の紙幣そのもの（「千円札」→千円紙幣）。
  // 文全体を題名にすると紙幣の記事に届かないため、額面だけを抜く。
  const den = q.match(/([一二三四五六七八九十千万円]{2,}札)/);
  if (den && /肖像/.test(q)) return den[1].replace(/札$/, "紙幣");
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
  // 「地球から月までの距離」型：題名は到着点の側（月）。文全体では
  // どちらの天体の記事にも届かない。
  const distCut = t.match(/^.{1,12}?から(.{1,8}?)までの距離/);
  if (distCut?.[1]) {
    t = distCut[1];
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
    /(の|で)(名物|特産品?|名産品?|郷土料理|国歌|首都|元素記号|化学式|有名な食べ物)(は|が|って)?.*$/,
    "",
  );
  t = t.replace(
    // 「人口とは何ですか」「人口の意味は？」のような定義質問では、
    // 数値の名詞の後ろの とは／の意味 を数値カットで食わない
    // （題名そのものが消えてしまうため）。
    /(の)?(高さ|標高|海抜|全長|長さ|重さ|面積|人口|速さ|値段|価格|年齢)(?!(?:とは|の意味|の定義))(は|が|って)?.*$/,
    "",
  );
  // 疑問詞のカット：「何／なに」の口語形「なん」も同じ位置に来る
  // （「モーリーファンタジーとはなんですか」）。これが無いと「なん」が
  // 残って末尾の「とは」落としも効かず、題名が「…とはなん」になって
  // 記事もリダイレクトも外れてしまう。
  t = t.replace(/(とは|って)(何|なに|なん|どんな).*$/, "");
  t = t.replace(/(は|が)(誰|だれ|いつ|どこ|何|なに|なん|いくつ|どんな|どう).*$/, "");
  t = t.replace(/(ですか|ますか|でしょうか|だろうか)$/, "");
  // 末尾に残った「とは」（「モーリーファンタジーとは？」型）：後段の
  // 助詞落としでは「は」だけが落ちて「…と」が残り、題名のタイトル
  // 完全一致もリダイレクト解決も外れてしまうため、ここで丸ごと落とす。
  const defTail = t.replace(/とは$/, "");
  if (defTail.length >= 2) t = defTail;
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
  // 1文字の漢字の題名（鉄・月・音など）は、それ自体が記事名なので
  // 題名として認める（かな1文字のような断片は従来どおり認めない）。
  return t.length >= 2 ? t : t.length === 1 && /[一-鿿]/.test(t) ? t : null;
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
      // 複合タイトル候補：「音の速さ」→音速、「光の速さ」→光速のよう
      // に、題名の頭の名詞＋質問の属性語がそのまま記事名になる場合。
      // 実在し、種類つき事実が実際に抜ける記事だけを採用する。
      {
        const head = topic.split("の")[0] ?? "";
        if (head.length >= 1 && head.length <= 4) {
          // 質問の属性語→記事名の接尾辞（速さ→速、高さ→高、長さ→長）。
          const suffixes = [
            ["速さ", "速"],
            ["速度", "速度"],
            ["高さ", "高"],
            ["長さ", "長"],
            ["重さ", "重"],
            ["深さ", "深"],
            ["広さ", "広"],
            ["面積", "面積"],
          ];
          for (const [qw, suf] of suffixes) {
            if (!q.includes(qw)) continue;
            const cand = `${head}${suf}`;
            if (cand === topic) continue;
            const src = await wikiArticleSource(cand);
            if (src && !src.disambiguation && extractFact(q, src)) {
              return src;
            }
          }
        }
      }
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
/* 証拠取得のサーバー側レーン（cube-server の /evidence）               */
/*                                                                    */
/* 事実質問の証拠集め（対象記事＋照合用検索）は、ページ側の索引        */
/* レーンが HF への Range 取得を1問あたり約1,000〜15,000回叩くため、  */
/* ブラウザ文脈では半分以上の質問が完走しなかった（2026-11-18/19 の  */
/* 実測）。そこで取得そのものを cube-server 側に寄せる：サーバーは    */
/* ページと同じ取得コード（resolveArticleSource / searchWikipedia）  */
/* を iMac 上で実行し、証拠一式を1回の往復で返す。検証ゲートの判定    */
/* は従来どおりページ側に残り、サーバーは証拠テキストを返すだけ。    */
/* サーバーに届かない・失敗した場合は、従来のページ側レーンに         */
/* そのまま倒れる（フェイルソフト）。                               */
/* ================================================================== */

/**
 * サーバー側レーンに証拠一式を問い合わせる。成功したら
 * {article, results, error}（ページ側レーンと同じ形）、失敗したら
 * null を返す。サーバーが「収集は完走したが証拠が無かった」と
 * 答えた場合は空の束を返し、無駄なページ側レーンは走らせない。
 * サーバー側で収集自体が失敗した（証拠ゼロ＋エラーあり）場合は
 * null に倒し、ページ側レーンで取り直す。
 */
async function fetchServerEvidence(question) {
  let res;
  try {
    res = await fetchWithTimeout(
      `${CUBE_SERVER_BASE_URL}/evidence`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: question }),
      },
      EVIDENCE_SERVER_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
  const data = res ? await res.json().catch(() => null) : null;
  if (
    !res ||
    !res.ok ||
    !data ||
    data.lane !== "evidence" ||
    typeof data.ms !== "number"
  ) {
    return null;
  }
  let article = null;
  const a = data.article;
  if (
    a &&
    typeof a.title === "string" &&
    typeof a.url === "string" &&
    typeof a.lead === "string" &&
    a.fields &&
    typeof a.fields === "object"
  ) {
    article = { title: a.title, url: a.url, lead: a.lead, fields: a.fields };
  }
  const results = [];
  if (Array.isArray(data.results)) {
    for (const r of data.results) {
      if (!r || typeof r.title !== "string" || typeof r.text !== "string") {
        continue;
      }
      results.push({
        title: r.title,
        url: typeof r.url === "string" ? r.url : "",
        snippet: typeof r.snippet === "string" ? r.snippet : "",
        text: r.text,
      });
    }
  }
  const error = typeof data.error === "string" ? data.error : "";
  if (!article && results.length === 0 && error) return null;
  WikiNet.via = "server";
  WikiNet.lastError = error;
  return { article, results, error };
}

/**
 * 証拠の入口：まずサーバー側レーンを試し、失敗・タイムアウトの
 * ときだけ従来のページ側レーン（fetchArticle ＋ searchWikipedia）
 * を開始する。ask() が使う2つの約束（articlePromise /
 * searchPromise）を返す。ページ側レーンはサーバーの失敗が確定
 * するまで開始しない（二重に取得を走らせないため）。
 */
function startEvidencePromises(question) {
  const bundlePromise = fetchServerEvidence(question);
  let pageArticle = null;
  let pageSearch = null;
  const ensurePageLane = () => {
    if (!pageArticle) {
      pageArticle = fetchArticle(question).catch(() => null);
      pageSearch = searchWikipedia(question).catch(() => []);
    }
  };
  const articlePromise = (async () => {
    const bundle = await bundlePromise;
    if (bundle) return bundle.article;
    ensurePageLane();
    return pageArticle;
  })();
  const searchPromise = (async () => {
    const bundle = await bundlePromise;
    if (bundle) return bundle.results;
    ensurePageLane();
    return pageSearch;
  })();
  return { articlePromise, searchPromise };
}

/* ================================================================== */
/* 知識メモ帳（この端末のブラウザ内に保存）                             */
/* ================================================================== */

const NB_KEY = "jisaku-ai-codepen-notebook-v1";
const BRIEFING_DATE_KEY = "cube-briefing-date";
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

/* 証拠取得の予算（第2回実測の教訓で再設計）：
   1問の証拠集め（対象記事＋検索）は、索引レーンへの Range 取得が
   約1,000〜15,000リクエストに及ぶ（実測：富士山の所在で約3,000、
   「日本で一番深い湖」で約15,000）。以前は照合全体に20秒の締切を
   質問開始時点から掛けていたため、生成の速いモデルでは証拠が
   そろう前に締切が来て、集めた証拠ごと捨てられていた（実在する
   記事でも「記事が見つかりませんでした」になる事故）。生成の遅い
   Max だけが、生成の裏で証拠が間に合い、設計どおりに動いた。
   そこで予算を2つに分ける：
   - レーン予算（EVIDENCE_LANE_BUDGET_MS / EVIDENCE_LANE_MAX_REQUESTS）：
     索引レーンの暴走防止だけに使う大きな上限。
   - 待ち予算（EVIDENCE_WAIT_MS）：生成が終わってから証拠の到着を
     待つ上限。生成時間とは独立に数えるため、モデルの速さで
     証拠のそろい方が変わらない。証拠が早くそろえば即座に進む。
   証拠集めの主経路はサーバー側レーン（cube-server の /evidence、
   startEvidencePromises 参照）：ページと同じ取得コードを iMac 上で
   実行し、証拠一式を1回の往復で受け取る。サーバー実測（2026-11-19、
   索引は iMac ローカルのミラーから取得、PC 測定と同時実行の負荷下）
   では通常の質問が0.4〜1.4秒、索引リクエスト約1.5万回の重量級
   （日本で一番深い湖）でも7.4秒で完走したため、待ち予算は
   旧来の90秒から60秒へ詰める。サーバーが落ちている・届かない
   ときは従来のページ側レーンに倒れ、その場合の暴走防止は引き続き
   レーン予算が担う（待ち予算より長く残すのは、生成と並行して
   証拠集めが先行している分を打ち切らないため）。 */
const EVIDENCE_LANE_BUDGET_MS = 150000;
const EVIDENCE_LANE_MAX_REQUESTS = 60000;
const EVIDENCE_WAIT_MS = 60000;
/* /evidence へのページ側タイムアウト。サーバー側の処理（レーン
   予算150秒が上限）より短く切り、届かなければページ側レーンへ
   切り替えるための上限。サーバー実測の最長7.4秒（重量級の質問の
   コールド実行）に対して約6倍の余裕を持たせた値。 */
const EVIDENCE_SERVER_TIMEOUT_MS = 45000;

const BADGES = {
  notebook: { label: "知識メモ帳の答え", className: "badge-notebook" },
  "notebook-stale": {
    label: "知識メモ帳の答え（今回ネットで確認できなかった）",
    className: "badge-unverified",
  },
  calc: { label: "検算の答え", className: "badge-calc" },
  memory: { label: "記憶の答え", className: "badge-memory" },
  self: {
    label: "Cube自身のこと（アプリ固定の情報）",
    className: "badge-self",
  },
  weather: {
    label: "天気情報（気象庁・Open-Meteo）",
    className: "badge-weather",
  },
  "weather-om": { label: "天気情報（Open-Meteo）", className: "badge-weather" },
  briefing: { label: "朝のブリーフィング（Cube Ultra）", className: "badge-briefing" },
  places: {
    label: "場所の検索結果（Yahoo! JAPAN・OpenStreetMap）",
    className: "badge-places",
  },
  topics: {
    label: "今の話題（ニュース・トレンドの紹介）",
    className: "badge-topics",
  },
  shop: {
    label: "買い物候補（Yahoo!ショッピング）",
    className: "badge-shop",
  },
  pc: {
    label: "PC操作の結果（仮想PCで実行）",
    className: "badge-pc",
  },
  youtube: {
    label: "動画の要約（YouTubeの字幕から）",
    className: "badge-youtube",
  },
  address: {
    label: "住所・位置の検索結果（郵便番号・国土地理院）",
    className: "badge-address",
  },
  wiki: {
    label: "Wikipediaの記事から（出典つき）",
    className: "badge-wiki",
  },
  quake: {
    label: "地震・警報の情報（P2P地震情報・気象庁）",
    className: "badge-quake",
  },
  station: {
    label: "最寄り駅の検索結果（HeartRails）",
    className: "badge-station",
  },
  holiday: {
    label: "祝日の情報（holidays-jp）",
    className: "badge-holiday",
  },
  market: {
    label: "為替・株価の情報（Frankfurter・Yahoo Finance）",
    className: "badge-market",
  },
  music: {
    label: "音楽の検索結果（iTunes）",
    className: "badge-music",
  },
  papers: {
    label: "論文の検索結果（arXiv / Semantic Scholar）",
    className: "badge-papers",
  },
  translate: {
    label: "翻訳の結果（DeepL）",
    className: "badge-translate",
  },
  wolfram: {
    label: "計算の答え（Wolfram|Alpha）",
    className: "badge-wolfram",
  },
  gmail: {
    label: "Gmail（Google連携で実行）",
    className: "badge-gmail",
  },
  gcal: {
    label: "Googleカレンダー（Google連携で実行）",
    className: "badge-gcal",
  },
  model: {
    label: "Cubeの答え（ネットで確認済み）",
    className: "badge-model",
  },
  image: {
    label: "画像を見た答え（Cube Ultra）",
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
  serverLoadSeq: 0, // サーバーモデル（Twin / Pro / Max）接続の世代。接続中にモデルが切り替わったとき、古い接続の後処理（状態表示の上書き）を無効化するために使う
  baseUrl: DEFAULT_BASE_URL,
  weatherPlace: DEFAULT_WEATHER_PLACE,
  confirmMarkAll: false,

  init() {
    Notebook.load();
    this.initMemory();
    this.initAccount();
    this.renderMeter();
    // 選択中のモデルを復元する（初期値は v1）。ベースURLはモデルごとの
    // 上書きとして読み込む（v1 は従来キー、v2 / Twin / Pro / Max は専用キー）。
    const savedId = store.get(MODEL_ID_KEY);
    this.modelId = MODELS[savedId] ? savedId : "v1";
    const def = modelDef(this.modelId);
    // Ultra とサーバーモデル（Twin / Pro / Max）と雑談専門（会話）
    // にはベースURLが無い（どれも iMac サーバーで動かす）。
    this.baseUrl = def.engine === "ultra" || def.engine === "server" || def.engine === "chat"
      ? ""
      : store.get(def.urlKey) || def.defaultBaseUrl;
    el("base-url").value = this.baseUrl;
    el("current-url").textContent = this.baseUrl;
    for (const btn of document.querySelectorAll(".model-switch-btn")) {
      btn.addEventListener("click", () => void this.switchModel(btn.dataset.model));
    }
    // サーバー接続ボタン：iMac で動かすモデル（Twin / Pro / Max /
    // Ultra）が未接続・接続エラーのときに状態バーへ出し、押された
    // ときだけ接続する。エンジン選択（ONNX / GGUF）はサーバー固定化
    // で廃止した（ボタン自体を index.html から外してある）。
    const serverConnectBtn = el("server-connect-btn");
    if (serverConnectBtn) {
      serverConnectBtn.addEventListener("click", () => void this.connectServerModel());
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
    // 画像添付（Ultra 専用）：ボタンでファイル選択を開き、選ばれたら
    // 縮小して保持する。取り消しボタンで添付を外す。
    const attachBtn = el("attach-btn");
    const imageFile = el("image-file");
    if (attachBtn && imageFile) {
      attachBtn.addEventListener("click", () => imageFile.click());
      imageFile.addEventListener("change", () => {
        const f = imageFile.files && imageFile.files[0];
        imageFile.value = "";
        if (f) void this.attachImageFile(f);
      });
      const attachRemove = el("attach-remove");
      if (attachRemove) {
        attachRemove.addEventListener("click", () => this.clearAttachedImage());
      }
    }
    this.syncAttach();
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
    // 朝のブリーフィング（条件に合うときだけ、裏で静かに試す）。
    void this.maybeShowBriefing();
    // PC操作パネル（cube_pc.js）。モジュールが無い環境では何もしない。
    // 成否の文面はサーバー側の判定をそのまま出す（ここでは作らない）。
    if (typeof CubePc !== "undefined") {
      try {
        const pcPanel = CubePc.createPanel({
          container: el("cube-pc-panel"),
          baseUrl: CUBE_SERVER_BASE_URL,
        });
        pcPanel.mount();
        const pcOpen = el("cube-pc-open");
        if (pcOpen) {
          pcOpen.addEventListener("click", () => {
            const s = el("cube-pc-panel");
            if (s) s.hidden = !s.hidden;
          });
        }
      } catch {
        /* パネルの失敗は本体を妨げない */
      }
    }
    // 電話デモパネル（cube_call_demo.js）。モジュールが無い環境では
    // 何もしない。シナリオ取得に失敗してもパネル内に出るだけで、
    // チャット本体には影響しない。音声読み上げフックは任意で、
    // CubeVoice に一発読み上げ用の口が無いため渡さない（省略可の設計）。
    if (typeof CubeCallDemo !== "undefined") {
      try {
        CubeCallDemo.createPanel(el("call-demo-panel"), {
          baseUrl: CUBE_SERVER_BASE_URL,
          voice: null,
        });
      } catch {
        /* パネルの失敗は本体を妨げない */
      }
    }
  },

  showTab(name) {
    for (const btn of document.querySelectorAll(".tab-btn")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.tab === name));
    }
    el("view-chat").hidden = name !== "chat";
    el("view-notebook").hidden = name !== "notebook";
    const fcView = el("view-factcheck");
    if (fcView) fcView.hidden = name !== "factcheck";
    const labView = el("view-lab");
    if (labView) labView.hidden = name !== "lab";
    const doctorView = el("view-doctor");
    if (doctorView) doctorView.hidden = name !== "doctor";
    const memView = el("view-memory");
    if (memView) memView.hidden = name !== "memory";
    el("view-settings").hidden = name !== "settings";
    if (name === "notebook") this.renderNotebook();
    if (name === "memory") this.renderMemory();
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
    this.baseUrl = def.engine === "ultra" || def.engine === "server" || def.engine === "chat"
      ? ""
      : store.get(def.urlKey) || def.defaultBaseUrl;
    el("base-url").value = this.baseUrl;
    el("current-url").textContent = this.baseUrl;
    this.renderModelSwitch();
    // 画像の添付は Ultra 専用：モデルを離れたら添付を外す
    // （他のモデルへ画像が黙って送られないように）。
    if (id !== "ultra" && pendingChatImage) pendingChatImage = null;
    this.syncAttach();
    this.syncSend();
    // iMac で動かすモデル（Twin / Pro / Max / Ultra / 会話）は、
    // 選ばれた時点でそのまま接続する（待機表示で止めない）。端末内
    // モデルは従来どおり loadModel で読み込む。
    if (def.engine === "ultra" || def.engine === "server" || def.engine === "chat") {
      await this.connectServerModel();
    } else {
      await this.loadModel(this.baseUrl);
    }
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
      text.textContent = `${label} 読み込み中… ${formatBytes(st.loadedBytes)}${st.totalBytes > 0 ? ` / ${formatBytes(st.totalBytes)}` : ""}（${st.label}）`;
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
    // iMac で動かすモデル（Twin / Pro / Max / Ultra）の選択中は、
    // ベースURLの上書き欄（端末内モデル専用）を無効化し、現在の
    // ベースURL欄にはその旨を表示する。未接続（standby）・接続
    // エラー（error）のときは、接続／再試行のボタンも出す。
    const def = modelDef(this.modelId);
    const noBaseUrl = def.engine === "ultra" || def.engine === "server" || def.engine === "chat";
    const baseInput = el("base-url");
    if (baseInput) baseInput.disabled = noBaseUrl;
    const loadBtn = el("load-model-btn");
    if (loadBtn) loadBtn.disabled = noBaseUrl;
    if (noBaseUrl) {
      const cur = el("current-url");
      if (cur) {
        cur.textContent = def.engine === "ultra"
          ? "（Cube Ultraにはありません。iMacサーバーで動かします）"
          : "（このモデルはiMacサーバーで動かすため、ベースURLはありません）";
      }
    }
    this.setServerConnect(noBaseUrl && (st.kind === "standby" || st.kind === "error"));
  },

  /* サーバー接続ブロック（未接続・接続エラー時の「接続／再試行」
     ボタン）の表示／非表示。表示条件の判定は setModelStatus 側で
     行い、ここは表示だけを受け持つ。 */
  setServerConnect(show) {
    const box = el("server-connect");
    if (!box) return;
    box.hidden = !show;
  },

  /* エンジン選択ブロックの表示／非表示。show のときは、前回読み込みに
     成功したエンジンが保存されていれば、その案内を添える。Ultra の
     エンジンはどちらも自動では開始せず、このブロックのボタンが
     押されたときだけ開始する。
     ※サーバー固定化で選択ブロック自体を index.html から外したため、
     現在は常に何もしない（要素が無い）。旧エンジンの読み込みメソッド
     （loadUltraOnnx / loadUltraGguf）互換のために残してある。 */
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

  /* iMac で動かすモデルへの接続：モデルが選ばれたとき（switchModel）
     と、未接続・接続エラー時に出る接続ボタンから使う。Ultra は既存の
     loadUltraImac（llama-server への到達確認）、サーバーモデル
     （Twin / Pro / Max）は ServerModel.connect（health の確認と、
     サーバー側が読み込み中なら ready までのポーリング）。どちらも
     端末へのダウンロードは無い。失敗したら素直な文面をエラー表示に
     し、接続ボタンで再試行できるようにする（setModelStatus が
     standby / error のときにボタンを出す）。 */
  async connectServerModel() {
    const def = modelDef(this.modelId);
    if (def.engine === "ultra") return this.loadUltraImac();
    if (def.engine === "chat") return this.connectChatModel();
    if (def.engine !== "server") return;
    if (ServerModel.readyFor(def.id)) {
      this.setModelStatus({
        kind: "ready",
        paramCount: ServerModel.paramCountFor(def.id),
      });
      return;
    }
    const seq = ++this.serverLoadSeq;
    // 他のエンジン（Ultra 系）があれば破棄する（アクティブな
    // エンジンは常に1つだけ）。
    disposeUltra();
    ChatModel.dispose();
    this.setUltraGgufNotice(null);
    this.setUltraEngineChoice(false);
    this.modelBusy = true;
    this.renderModelSwitch();
    this.setModelStatus({
      kind: "loading",
      loadedBytes: 0,
      totalBytes: 0,
      label: `${def.label}（iMacサーバーに接続しています）`,
    });
    try {
      await ServerModel.connect(def.id, () => {
        if (seq !== this.serverLoadSeq) return;
        this.setModelStatus({
          kind: "loading",
          loadedBytes: 0,
          totalBytes: 0,
          label: `${def.label}（iMac側で読み込み中です）`,
        });
      });
      if (seq !== this.serverLoadSeq) return;
      this.setModelStatus({
        kind: "ready",
        paramCount: ServerModel.paramCountFor(def.id),
      });
    } catch (e) {
      if (seq !== this.serverLoadSeq) return;
      const message = e instanceof Error ? e.message : String(e);
      this.setModelStatus({ kind: "error", message });
    } finally {
      if (seq === this.serverLoadSeq) {
        this.modelBusy = false;
        this.renderModelSwitch();
      }
    }
  },

  /* 雑談専門（Cube 会話）の接続：サーバー側はオンデマンド起動で、
     初回はモデルの読み込みに数十秒かかる（/chat/start が起動完了
     まで待って返す）。開始時に他のエンジンの接続状態は破棄する
     （アクティブなエンジンは常に1つだけ）。 */
  async connectChatModel() {
    const def = modelDef(this.modelId);
    if (ChatModel.readyFor(def.id)) {
      this.setModelStatus({ kind: "ready", paramCount: ChatModel.paramCountFor() });
      return;
    }
    const seq = ++this.serverLoadSeq;
    disposeUltra();
    ServerModel.dispose();
    this.setUltraGgufNotice(null);
    this.setUltraEngineChoice(false);
    this.modelBusy = true;
    this.renderModelSwitch();
    this.setModelStatus({
      kind: "loading",
      loadedBytes: 0,
      totalBytes: 0,
      label: `${def.label}（iMacサーバーに接続しています）`,
    });
    try {
      await ChatModel.connect(() => {
        if (seq !== this.serverLoadSeq) return;
        this.setModelStatus({
          kind: "loading",
          loadedBytes: 0,
          totalBytes: 0,
          label: `${def.label}（iMac側でモデルを起動中です。初回は少し時間がかかります）`,
        });
      });
      if (seq !== this.serverLoadSeq) return;
      this.setModelStatus({ kind: "ready", paramCount: ChatModel.paramCountFor() });
    } catch (e) {
      if (seq !== this.serverLoadSeq) return;
      const message = e instanceof Error ? e.message : String(e);
      this.setModelStatus({ kind: "error", message });
    } finally {
      if (seq === this.serverLoadSeq) {
        this.modelBusy = false;
        this.renderModelSwitch();
      }
    }
  },

  async loadModel(url) {
    const def = modelDef(this.modelId);
    if (def.engine === "ultra") {
      // Cube Ultra（サーバー固定）：UI から開始できるのは iMac
      // エンジンだけ。ここ（ページ表示時・選択の復元時）は自動では
      // 接続せず、未接続の待機表示にする。接続はモデルが選ばれた
      // とき（switchModel）と接続ボタンから、connectServerModel
      // 経由で行う。旧エンジン（ONNX / GGUF）の試行の印・フェーズ
      // 記録が残っていても、その文面はもう出さない（両エンジンとも
      // UI から到達できない。印はモデルを選び直したときに
      // switchModel が消す）。
      this.ultraLoadSeq++;
      this.setUltraGgufNotice(null);
      this.setUltraEngineChoice(false);
      ServerModel.dispose();
      ChatModel.dispose();
      if (UltraImacModel.ready) {
        this.setModelStatus({ kind: "ready", paramCount: ULTRA_PARAM_COUNT });
        return;
      }
      this.setModelStatus({
        kind: "standby",
        message: "Cube Ultra は未接続です。iMacサーバーに接続してください",
      });
      return;
    }
    if (def.engine === "server") {
      // サーバーモデル（Twin / Pro / Max）：端末には読み込まない。
      // ここ（ページ表示時・選択の復元時）は自動では接続せず、
      // 未接続の待機表示にする。接続はモデルが選ばれたとき
      // （switchModel）と接続ボタンから、connectServerModel 経由で
      // 行う。
      this.ultraLoadSeq++;
      this.setUltraGgufNotice(null);
      this.setUltraEngineChoice(false);
      disposeUltra();
      ChatModel.dispose();
      if (ServerModel.readyFor(def.id)) {
        this.setModelStatus({
          kind: "ready",
          paramCount: ServerModel.paramCountFor(def.id),
        });
        return;
      }
      this.setModelStatus({
        kind: "standby",
        message: `${def.label} は未接続です。iMacサーバーに接続してください`,
      });
      return;
    }
    if (def.engine === "chat") {
      // 雑談専門（Cube 会話）：端末には読み込まない。ここ（ページ
      // 表示時・選択の復元時）は自動では接続せず、未接続の待機表示
      // にする。接続はモデルが選ばれたとき（switchModel）と接続
      // ボタンから、connectServerModel 経由で行う。
      this.ultraLoadSeq++;
      this.setUltraGgufNotice(null);
      this.setUltraEngineChoice(false);
      disposeUltra();
      ServerModel.dispose();
      if (ChatModel.readyFor(def.id)) {
        this.setModelStatus({ kind: "ready", paramCount: ChatModel.paramCountFor() });
        return;
      }
      this.setModelStatus({
        kind: "standby",
        message: `${def.label} は未接続です。iMacサーバーに接続してください`,
      });
      return;
    }
    // 端末内モデル（Nano / Flash）が選択された：Ultra のエンジンと
    // サーバーモデルの接続状態があれば破棄し、GGUF の案内も隠す
    // （アクティブなエンジンは常に1つだけ）。進行中の Ultra 読み込み
    // の後処理も無効化する。
    this.ultraLoadSeq++;
    this.setUltraGgufNotice(null);
    this.setUltraEngineChoice(false);
    disposeUltra();
    ServerModel.dispose();
    ChatModel.dispose();
    // ベースURLの上書きは選択中モデルごとのキーに保存する
    // （Nano は従来の jisaku-ai-model-base-url、Flash は専用キー）。
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
      // Flash の重みがまだアップロードされていない場合の案内を
      // 添える（黙って Nano で回答することは絶対にしない）。
      if (def.id === "v2" && message.includes("manifest.json")) {
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

  /* iMacサーバーエンジンの開始：サーバー固定化後の Ultra の唯一の
     接続経路で、モデルが選ばれたとき（switchModel）と接続ボタンから
     connectServerModel 経由で始める。端末へのダウンロードは無く、
     iMac への到達確認だけ。開始時に他のエンジン（通常版・軽量
     エンジン）とサーバーモデルの接続状態は破棄する（アクティブな
     エンジンは常に1つだけ）。成功したら接続済みの印を保存する。
     届かなかったら素直な文面をエラー表示にし、接続ボタンで再試行
     できるようにする。 */
  async loadUltraImac() {
    if (this.modelId !== "ultra" || UltraImacModel.ready) return;
    const seq = ++this.ultraLoadSeq;
    UltraModel.dispose();
    UltraGgufModel.dispose();
    ServerModel.dispose();
    ChatModel.dispose();
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
      // 再試行は接続ボタンから（エラー表示にすると setModelStatus が
      // サーバー接続ブロックを出す）。
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
    el("send-btn").disabled =
      !(el("question-input").value.trim() || pendingChatImage) || this.phase !== "idle";
  },

  scrollChat() {
    const sc = el("chat-scroll");
    sc.scrollTo({ top: sc.scrollHeight, behavior: "smooth" });
  },

  pushUser(text, imageDataUrl) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (imageDataUrl) {
      const img = document.createElement("img");
      img.className = "bubble-img";
      img.src = imageDataUrl;
      img.alt = "添付した画像";
      bubble.appendChild(img);
    }
    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);
    li.appendChild(bubble);
    el("msg-list").appendChild(li);
    this.historyCaptureUser(text);
    this.scrollChat();
  },

  /* 画像添付（Ultra 専用）：選ばれたファイルを縮小して保持し、
     プレビューを出す。読み込めない画像はプレビュー欄に理由を出す。 */
  async attachImageFile(file) {
    if (this.modelId !== "ultra") return;
    try {
      pendingChatImage = await fileToResizedJpegDataUrl(file);
    } catch {
      pendingChatImage = null;
      this.syncAttach("その画像は読み込めませんでした。別の画像で試してください。");
      this.syncSend();
      return;
    }
    this.syncAttach();
    this.syncSend();
  },

  clearAttachedImage() {
    pendingChatImage = null;
    this.syncAttach();
    this.syncSend();
  },

  /* 添付ボタンとプレビューの表示を、選択中モデルと添付状態に合わせる。
     画像を見られるのは Ultra だけなので、他のモデルではボタンを
     無効化し、理由を title に出す。 */
  syncAttach(errorNote) {
    const btn = el("attach-btn");
    if (btn) {
      const isUltra = this.modelId === "ultra";
      btn.disabled = !isUltra;
      btn.title = isUltra
        ? "画像を添付する（Cube Ultra が見ます）"
        : "画像は Cube Ultra で見られます";
    }
    const box = el("attach-preview");
    if (!box) return;
    const note = el("attach-note");
    const thumb = el("attach-thumb");
    if (errorNote) {
      box.hidden = false;
      if (thumb) {
        thumb.hidden = true;
        thumb.removeAttribute("src");
      }
      if (note) note.textContent = errorNote;
      return;
    }
    if (pendingChatImage) {
      box.hidden = false;
      if (thumb) {
        thumb.hidden = false;
        thumb.src = pendingChatImage;
      }
      if (note) note.textContent = "画像を添付しました（Cube Ultra が見ます）";
    } else {
      box.hidden = true;
    }
  },

  /* 画像つき質問の実行と表示（Ultra 専用レーンで完結させる）。
     画像の内容はネットの裏づけ検証の対象にならないため、通常の
     質問経路（レーン振り分け・検証ゲート・知識メモ帳）には入れない。
     どの失敗でも通常経路には倒さず、理由を注記に出す（YouTube
     レーンと同じ作法）。 */
  async askWithImage(question, imageDataUrl) {
    this.setPhase("thinking");
    let text = "";
    let errorNote = "";
    try {
      text = await generateUltraImage(
        question,
        imageDataUrl,
        this.historyMessagesForUltra(),
      );
    } catch (e) {
      errorNote = e instanceof Error ? e.message : String(e);
    }
    this.setPhase("idle");
    if (text) {
      this.pushAssistant({
        text,
        badge: "image",
        note: "添付された画像を Cube Ultra が見て答えています。画像はサーバーに保存されません。",
      });
    } else {
      this.pushAssistant({
        text: "画像を見て答えられませんでした。",
        badge: "unknown",
        note: errorNote || "生成に失敗しました。少し待ってからもう一度試してください。",
      });
    }
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
    // 場所・話題・買い物レーンの構造化カード（API の値の転記だけ）。
    if (Array.isArray(msg.cards) && msg.cards.length) {
      try {
        li.appendChild(this.buildLaneCards(msg.cards));
      } catch {
        /* カード描画の失敗は会話を妨げない */
      }
    }
    if (msg.evidence) {
      const ev = document.createElement("p");
      ev.className = "evidence";
      ev.textContent = `根拠: ${msg.evidence}`;
      li.appendChild(ev);
    }
    if (msg.modelSaid) {
      // 検証で採用されなかったモデルの下書きは、透明性のために
      // 残すが、回答ブロックの本文と並べて出すと「答えの一部」に
      // 見えて利用者が混乱する（実例：Web検索の答えの下に、全面
      // 却下された無関係な下書きがそのまま出た）。折りたたみに
      // 畳み、開いた人だけが見られるようにする。
      const det = document.createElement("details");
      det.className = "model-said-fold";
      const sum = document.createElement("summary");
      sum.textContent = "検証で採用されなかったCubeの下書き";
      det.appendChild(sum);
      const ms = document.createElement("p");
      ms.className = "model-said";
      ms.textContent = msg.modelSaid;
      det.appendChild(ms);
      li.appendChild(det);
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
    // 嘘チェック可視化：検証内訳パネルをこのメッセージに付ける
    // （履歴から復元したメッセージでは保存済みの内訳を使う）。
    const fcData = msg.factcheck || this._fcBreakdown;
    if (fcData && typeof CubeFactCheck !== "undefined") {
      this._fcBreakdown = null;
      msg.factcheck = fcData;
      try {
        const panel = CubeFactCheck.renderPanel(fcData);
        if (panel) li.appendChild(panel);
      } catch {
        /* パネルの失敗は会話を妨げない */
      }
    }
    el("msg-list").appendChild(li);
    this.historyCaptureAssistant(msg);
    this.scrollChat();
  },

  /* ---- YouTube要約レーンの実行と表示 ----
     字幕の取得と要約の生成が終わるまで検索中表示にし、結果は
     通常の回答として出す。どの失敗でも通常経路には倒さず、
     理由ごとの正直な文面をこのレーンで完結させる。 */
  async runYoutubeLane(question, videoId) {
    this.setPhase("searching");
    let r = null;
    try {
      r = await youtubeSummary(question, this.modelId);
    } catch {
      r = null;
    }
    this.setPhase("idle");
    const watchUrl = `https://www.youtube.com/watch?v=${videoId}`;
    if (!r || r.kind === "unreachable") {
      this.pushAssistant({
        text: "字幕の取得サーバーに接続できませんでした。",
        badge: "youtube",
        sourceTitle: "YouTube",
        sourceUrl: watchUrl,
        note: "時間をおいてもう一度試してください。",
      });
      return;
    }
    const title = (r.data && r.data.title) || "";
    const channel = (r.data && r.data.channel) || "";
    const sourceTitle = title || "YouTube";
    if (r.kind === "summary") {
      const noteParts = [
        `動画「${title || videoId}」${channel ? `（${channel}）` : ""}の${ytLangLabel(r.data)}をもとに、選択中のモデルが要約しました。`,
      ];
      if (r.truncated) {
        noteParts.push(`字幕が長いため、冒頭約${r.cap}字の範囲での要約です。`);
      }
      this.pushAssistant({
        text: r.text,
        badge: "youtube",
        sourceTitle,
        sourceUrl: watchUrl,
        note: noteParts.join(""),
      });
      return;
    }
    if (r.kind === "no_captions") {
      this.pushAssistant({
        text: "この動画は字幕が無いため、要約できませんでした。",
        badge: "youtube",
        sourceTitle,
        sourceUrl: watchUrl,
        note: "字幕（日本語・英語など）が付いている動画なら要約できます。",
      });
      return;
    }
    if (r.kind === "meter_empty") {
      this.pushAssistant({
        text: "サーバーモデルの時間メーターが0のため、要約を作れませんでした。",
        badge: "youtube",
        sourceTitle,
        sourceUrl: watchUrl,
        note: "Cube Nano / Cube Flash（この端末の中で動くモデル）に切り替えると、メーターなしで要約できます。ログイン中なら、ポイントでメーター時間を追加することもできます。",
      });
      return;
    }
    this.pushAssistant({
      text: r.kind === "failed"
        ? "字幕を取得できませんでした。"
        : "字幕は取得できましたが、要約を作れませんでした。",
      badge: "youtube",
      sourceTitle,
      sourceUrl: watchUrl,
      note: "時間をおくか、モデルを切り替えてもう一度試してください。",
    });
  },

  /* ---- 住所レーンの実行と表示 ----
     郵便番号形は形式が決定的なので、どの結果でもこのレーンで
     完結させる（見つからない場合も、推測の住所は出さない）。
     キーワード形はサーバーに届かなかった場合だけ従来経路に
     倒れるため、完結したら true、倒れる場合は false を返す。 */
  async runAddressZipLane(zipDigits) {
    this.setPhase("searching");
    const data = await fetchAddress({ zipcode: zipDigits });
    this.setPhase("idle");
    const zipLabel = `${zipDigits.slice(0, 3)}-${zipDigits.slice(3)}`;
    if (!data) {
      this.pushAssistant({
        text: "住所の取得サーバーに接続できませんでした。",
        badge: "address",
        note: "時間をおいてもう一度試してください。",
      });
      return;
    }
    if (!data.ok) {
      if (data.error === "not_found") {
        this.pushAssistant({
          text: `〒${zipLabel} の住所は見つかりませんでした。`,
          badge: "address",
          note: "郵便番号が正しいか確認してください。分からない住所を推測で出すことはしません。",
        });
      } else {
        this.pushAssistant({
          text: "住所を取得できませんでした。",
          badge: "address",
          note: "時間をおいてもう一度試してください。",
        });
      }
      return;
    }
    this.pushAssistant({
      text: zipcodeAnswerText(data),
      badge: "address",
      note: "出典：zipcloud（郵便番号検索）。番号によっては市区町村までの範囲しか載っていません。",
    });
  },

  async runAddressQueryLane(parsed) {
    this.setPhase("searching");
    const data = await fetchAddress({ query: parsed.target });
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      if (data.error === "not_found") {
        this.pushAssistant({
          text: parsed.wantsZip
            ? `「${parsed.target}」の郵便番号も位置も見つかりませんでした。`
            : `「${parsed.target}」の位置は見つかりませんでした。`,
          badge: "address",
          note: "地名や施設名を変えて（例：都道府県名から書く）もう一度聞いてください。分からない場所を推測で出すことはしません。",
        });
      } else {
        this.pushAssistant({
          text: "位置を取得できませんでした。",
          badge: "address",
          note: "時間をおいてもう一度試してください。",
        });
      }
      return true;
    }
    const lead = parsed.wantsZip
      ? `「${parsed.target}」の郵便番号は、名前からは検索できませんでした。代わりに位置をお知らせします。\n`
      : "";
    this.pushAssistant({
      text: lead + geocodeAnswerText(data),
      badge: "address",
      cards: data.results.map(addressCardOf),
      note: "出典：国土地理院（住所・地名の検索）。地図はカードのリンクから開けます。",
    });
    return true;
  },

  /* ---- 地震・警報レーンの実行と表示 ----
     発火したらこのレーンで完結させる（true）。サーバーに届か
     なかった場合だけ false を返し、従来の手順に倒れる。警報で
     都道府県が質問に無い場合は、天気の既定地から補う。 */
  async runQuakeLane(parsed) {
    this.setPhase("searching");
    let data;
    if (parsed.kind === "warning") {
      const pref = parsed.pref || extractPrefecture(this.weatherPlace);
      if (!pref) {
        this.setPhase("idle");
        return false;
      }
      data = await fetchQuake({ prefCode: pref.code, prefName: pref.name });
    } else {
      data = await fetchQuake({});
    }
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      this.pushAssistant({
        text: parsed.kind === "warning"
          ? "警報・注意報を取得できませんでした。"
          : "地震情報を取得できませんでした。",
        badge: "quake",
        note: "時間をおいてもう一度試してください。分からない情報を推測で出すことはしません。",
      });
      return true;
    }
    this.pushAssistant({
      text: data.kind === "warning"
        ? warningAnswerText(data) : quakeAnswerText(data),
      badge: "quake",
      note: data.kind === "warning"
        ? "出典：気象庁（警報・注意報）。詳しい地域は気象庁の警報・注意報ページで確認してください。"
        : "出典：P2P地震情報（気象庁の発表をもとにした情報）。",
    });
    return true;
  },

  /* ---- 最寄り駅レーンの実行と表示 ----
     対象を住所レーンの geocode で座標にし、/station で一番近い
     駅を引く。対象が見つからない場合は正直に完結させ、サーバー
     に届かない場合だけ false で従来の手順に倒れる。 */
  async runStationLane(parsed) {
    this.setPhase("searching");
    const geo = await fetchAddress({ query: parsed.target });
    if (!geo) {
      this.setPhase("idle");
      return false;
    }
    if (!geo.ok) {
      this.setPhase("idle");
      if (geo.error === "not_found") {
        this.pushAssistant({
          text: `「${parsed.target}」の場所が見つからなかったため、最寄り駅を調べられませんでした。`,
          badge: "station",
          note: "地名や施設名を変えて（例：都道府県名から書く）もう一度聞いてください。分からない場所を推測で出すことはしません。",
        });
        return true;
      }
      return false;
    }
    const top = geo.results[0];
    const data = await fetchStation(top.lat, top.lng);
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      this.pushAssistant({
        text: `「${parsed.target}」の最寄り駅を取得できませんでした。`,
        badge: "station",
        note: "時間をおいてもう一度試してください。",
      });
      return true;
    }
    this.pushAssistant({
      text: stationAnswerText(parsed.target, data),
      badge: "station",
      cards: data.results.map(stationCardOf),
      note: "出典：HeartRails Express（駅情報）。距離は目安です。",
    });
    return true;
  },

  /* ---- 祝日レーンの実行と表示 ---- */
  async runHolidayLane(parsed) {
    this.setPhase("searching");
    const data = await fetchHoliday();
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      this.pushAssistant({
        text: "祝日の情報を取得できませんでした。",
        badge: "holiday",
        note: "時間をおいてもう一度試してください。",
      });
      return true;
    }
    this.pushAssistant({
      text: holidayAnswerText(parsed, data.holidays),
      badge: "holiday",
      note: "出典：holidays-jp（内閣府の「国民の祝日」データ）。",
    });
    return true;
  },

  /* ---- 為替・株価レーンの実行と表示 ----
     数値が要る質問なので、発火したら必ずこのレーンで完結
     させる。取れなかった場合も正直な文面を出し、モデルの
     推測価格には倒さない。 */
  async runMarketLane(parsed) {
    if (parsed.kind === "stock_unknown") {
      this.pushAssistant({
        text: `「${parsed.name}」の証券コードが対応表に無いため、株価を調べられませんでした。`,
        badge: "market",
        note: "証券コード（4桁の数字）が分かれば、「7203の株価は？」のように聞いてください。分からない株価を推測で出すことはしません。",
      });
      return;
    }
    this.setPhase("searching");
    let data = null;
    if (parsed.kind === "fx") {
      data = await fetchMarket({ fx: `${parsed.base}${parsed.quote}` });
    } else if (parsed.kind === "fxconvert") {
      data = await fetchMarket({ fx: `${parsed.from}JPY` });
    } else {
      data = await fetchMarket({ symbol: parsed.symbol });
    }
    this.setPhase("idle");
    if (!data || !data.ok) {
      this.pushAssistant({
        text: "為替・株価の情報を取得できませんでした。",
        badge: "market",
        note: "時間をおいてもう一度試してください。分からない価格を推測で出すことはしません。",
      });
      return;
    }
    if (parsed.kind === "fxconvert") {
      const result = parsed.amount * data.rate;
      const rounded = result >= 100
        ? Math.round(result) : Math.round(result * 100) / 100;
      this.pushAssistant({
        text: `${parsed.amount.toLocaleString("ja-JP")}${parsed.fromName}は、約${rounded.toLocaleString("ja-JP")}円です（1${parsed.from}＝${rateText(data.rate)}円、${fxDateLabel(data.date)}時点のレートで計算しました）。`,
        badge: "market",
        note: "出典：Frankfurter（欧州中央銀行の参考レート）。実際の両替レートとは異なります。",
      });
      return;
    }
    if (data.kind === "fx") {
      this.pushAssistant({
        text: marketFxText(data, parsed.pairName),
        badge: "market",
        note: "出典：Frankfurter（欧州中央銀行の参考レート）。",
      });
      return;
    }
    this.pushAssistant({
      text: marketStockText(data, parsed.name),
      badge: "market",
      note: "出典：Yahoo Finance。株価は遅れて表示される場合があります。",
    });
  },

  /* ---- 音楽検索レーンの実行と表示 ---- */
  async runMusicLane(parsed) {
    this.setPhase("searching");
    const data = await fetchMusic(parsed.query);
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      if (data.error === "not_found") {
        this.pushAssistant({
          text: `「${parsed.query}」の曲は見つかりませんでした。`,
          badge: "music",
          note: "アーティスト名や曲名を変えてもう一度聞いてください。",
        });
      } else {
        this.pushAssistant({
          text: "音楽情報を取得できませんでした。",
          badge: "music",
          note: "時間をおいてもう一度試してください。",
        });
      }
      return true;
    }
    this.pushAssistant({
      text: `「${parsed.query}」の曲を${data.results.length}件見つけました。`,
      badge: "music",
      cards: data.results.map(musicCardOf),
      note: "出典：iTunes。カードの「試聴する」から一部を聴けます。",
    });
    return true;
  },

  /* ---- 論文検索レーンの実行と表示 ---- */
  async runPapersLane(parsed) {
    this.setPhase("searching");
    const data = await fetchPapers(parsed.query);
    this.setPhase("idle");
    if (!data) return false;
    const topic = parsed.terms || parsed.query;
    if (!data.ok) {
      if (data.error === "not_found") {
        this.pushAssistant({
          text: `「${topic}」に関する論文は見つかりませんでした。`,
          badge: "papers",
          note: "キーワードを変えると見つかることがあります（英語のキーワードだと見つかりやすいです）。",
        });
      } else {
        this.pushAssistant({
          text: "論文情報を取得できませんでした。",
          badge: "papers",
          note: "時間をおいてもう一度試してください。",
        });
      }
      return true;
    }
    let note = `出典：${data.source || "arXiv / Semantic Scholar"}。`;
    if (data.attribution) note += `${data.attribution} `;
    if (data.fetchedAt) {
      note += `（${jstDateTimeLabel(data.fetchedAt)}取得）`;
    }
    this.pushAssistant({
      text: `「${topic}」に関する論文を${data.results.length}件見つけました。`,
      badge: "papers",
      cards: data.results.map(paperCardOf),
      note,
    });
    return true;
  },

  /* ---- 翻訳レーンの実行と表示 ---- */
  async runTranslateLane(parsed) {
    this.setPhase("searching");
    const data = await fetchTranslate(parsed);
    this.setPhase("idle");
    if (!data) return false;
    if (!data.ok) {
      /* サーバー側で翻訳の意図でないと判定されたら従来の手順に
         倒す（ページ側とサーバー側で判定がずれた場合の逃げ道。
         それ以外の失敗はこのレーンの中で正直に案内して完結）。 */
      if (data.error === "not_translation") return false;
      if (data.error === "no_source_text") {
        this.pushAssistant({
          text: "翻訳したい文が質問の中に見つかりませんでした。",
          badge: "translate",
          note: "「『ありがとう』を英語に翻訳して」のように、訳したい文を一緒に入れて聞いてください。",
        });
      } else if (data.error === "quota_exceeded") {
        this.pushAssistant({
          text: "翻訳の無料分を使い切りました。",
          badge: "translate",
          note: "DeepL の無料枠は通算100万字の一回きりで、リセットされません。使い切ったため、この翻訳機能はここまでです。",
        });
      } else if (data.error === "not_configured") {
        this.pushAssistant({
          text: "翻訳機能はまだ使える状態になっていません。",
          badge: "translate",
          note: "サーバー側の設定が済んでいないため翻訳できませんでした。時間をおいてもう一度試してください。",
        });
      } else if (data.error === "source_too_long") {
        this.pushAssistant({
          text: "文が長すぎて翻訳できませんでした。",
          badge: "translate",
          note: "原文は1,200字までです。短く分けて試してください。",
        });
      } else {
        this.pushAssistant({
          text: "翻訳を取得できませんでした。",
          badge: "translate",
          note: "時間をおいてもう一度試してください。",
        });
      }
      return true;
    }
    const targetName = translateLangName(data.targetLang) || data.targetLang;
    let note = `出典：${data.source || "DeepL"}。`;
    if (data.usage && typeof data.usage.character_remaining === "number") {
      note += `無料分の残りは約${data.usage.character_remaining.toLocaleString("ja-JP")}字です。`;
    }
    if (data.fetchedAt) {
      note += `（${jstDateTimeLabel(data.fetchedAt)}取得）`;
    }
    this.pushAssistant({
      text: `『${data.sourceText}』を${targetName}に翻訳しました。`,
      badge: "translate",
      cards: [translateCardOf(data)],
      note,
    });
    return true;
  },

  /* ---- Wolfram 計算レーンの実行と表示 ----
     計算の主軸。未設定・失敗・解釈不能（not_calc / no_query /
     upstream 等）はすべて false を返し、従来の計算レーン
     （0.75 の CalcLane と手順2の evalArithmetic）へ素通り
     させる。結果は規約上キャッシュできないため、メモ帳には
     保存しない。 */
  async runWolframLane(parsed) {
    this.setPhase("searching");
    const data = await fetchWolfram(parsed);
    this.setPhase("idle");
    if (!data || !data.ok) return false;
    let note = `出典：${data.source || "Wolfram|Alpha"}。`;
    if (data.fetchedAt) {
      note += `（${jstDateTimeLabel(data.fetchedAt)}取得）`;
    }
    this.pushAssistant({
      text: `「${parsed.query}」を計算しました。`,
      badge: "wolfram",
      cards: [wolframCardOf(data)],
      note,
    });
    return true;
  },

  /* ---- Google連携レーンの実行と表示 ----
     本人の決定どおり全自動で実行する。送信・作成・変更・削除の
     前に確認は挟まない。失敗・未連携・未ログインは、この中で
     正直に案内して完結させる（モデルの推測には倒さない）。 */

  googleToken() {
    const mgr = this.account;
    if (!mgr || !mgr.signedIn) return "";
    return (mgr.session && mgr.session.access_token) || "";
  },

  /* 連携状態をサーバーに確認する（60秒はキャッシュを使う）。 */
  async refreshGoogleStatus(force) {
    const token = this.googleToken();
    if (!token) {
      this.googleState = null;
      return null;
    }
    if (
      !force &&
      this.googleState &&
      Date.now() - this.googleStateAt < 60000
    ) {
      return this.googleState;
    }
    const st = await fetchGoogleStatus(token);
    if (st && st.ok) {
      this.googleState = st;
      this.googleStateAt = Date.now();
    }
    return this.googleState;
  },

  /* Google 系レーンの前段：進めない状態なら案内を出して
     false を返す。 */
  async ensureGoogleReady(badge) {
    if (!this.account || !this.account.signedIn) {
      this.pushAssistant({
        text: "Google連携を使うには、まずCubeにログインしてください。サイドバーの「Googleでログイン」からログインできます。",
        badge,
      });
      return false;
    }
    const st = await this.refreshGoogleStatus(false);
    if (!st) {
      this.pushAssistant({
        text: "Google連携の状態を確認できませんでした。",
        badge,
        note: "サーバーに届かなかった可能性があります。時間をおいてもう一度試してください。",
      });
      return false;
    }
    if (!st.configured) {
      this.pushAssistant({
        text: "Google連携はまだ準備中です。",
        badge,
      });
      return false;
    }
    if (!st.linked) {
      this.pushAssistant({
        text: "まだGoogleと連携していません。サイドバーのアカウント欄にある「Googleと連携する」ボタンを押して、Googleの画面で許可してください。",
        badge,
        note: "連携すると、CubeからGmailの読み書きと、Googleカレンダーの予定の読み書きができるようになります。",
      });
      return false;
    }
    return true;
  },

  /* Google 系エンドポイントの失敗を、理由に応じて正直に出す。 */
  googleFailMessage(data, badge) {
    if (!data) {
      this.pushAssistant({
        text: "サーバーに届きませんでした。",
        badge,
        note: "時間をおいてもう一度試してください。",
      });
      return;
    }
    if (data.error === "not_linked" || data.error === "relink_required") {
      if (this.googleState) this.googleState.linked = false;
      this.renderGoogleLink();
      this.pushAssistant({
        text: "Google連携が切れています。サイドバーの「Googleと連携する」ボタンから、もう一度連携してください。",
        badge,
        note: "連携の許可は一定期間で切れることがあります。",
      });
      return;
    }
    if (data.error === "not_configured") {
      this.pushAssistant({
        text: "Google連携はまだ準備中です。",
        badge,
      });
      return;
    }
    this.pushAssistant({
      text: "うまくいきませんでした。",
      badge,
      note: "時間をおいてもう一度試してください。分からない情報を推測で出すことはしません。",
    });
  },

  async runGmailSendLane(parsed) {
    if (!(await this.ensureGoogleReady("gmail"))) return;
    if (parsed.missing.length) {
      this._gmailDraft = {
        to: parsed.to,
        subject: parsed.subject,
        body: parsed.body,
      };
      this.pushAssistant({
        text: gmailSendAskBackText(this._gmailDraft),
        badge: "gmail",
      });
      return;
    }
    await this.executeGmailSend({
      to: parsed.to,
      subject: parsed.subject,
      body: parsed.body,
    });
  },

  /* 聞き返しへの返信で下書きが埋まれば送信まで進める。
     埋まらなければ下書きを捨てて通常処理に倒す（true=処理済み）。 */
  async handleGmailDraftReply(question) {
    const draft = this._gmailDraft;
    if (!draft) return false;
    const merged = mergeGmailDraft(draft, question);
    const before = [draft.to, draft.subject, draft.body]
      .filter(Boolean).length;
    const after = [merged.to, merged.subject, merged.body]
      .filter(Boolean).length;
    if (after === 3) {
      this._gmailDraft = null;
      if (!(await this.ensureGoogleReady("gmail"))) return true;
      await this.executeGmailSend(merged);
      return true;
    }
    if (after > before) {
      this._gmailDraft = merged;
      this.pushAssistant({
        text: gmailSendAskBackText(merged),
        badge: "gmail",
      });
      return true;
    }
    this._gmailDraft = null;
    return false;
  },

  async executeGmailSend(mail) {
    this.setPhase("searching");
    const res = await sendGmail(this.googleToken(), mail);
    this.setPhase("idle");
    if (!res) {
      this.googleFailMessage(null, "gmail");
      return;
    }
    if (!res.ok) {
      this.googleFailMessage(res, "gmail");
      return;
    }
    this._gmailDraft = null;
    this.pushAssistant({
      text: `${mail.to} へメールを送信しました。\n件名：${mail.subject}`,
      badge: "gmail",
      note: "送信は送信ログに記録され、サイドバーの通知欄にも表示されます。",
    });
    if (res.log) {
      this.showSendLogNotices([res.log]);
    } else {
      void this.checkSendLogNotices();
    }
  },

  async runGmailReadLane(parsed) {
    if (!(await this.ensureGoogleReady("gmail"))) return;
    this.setPhase("searching");
    const data = await fetchGmailList(this.googleToken(), {
      maxResults: 10,
      q: parsed.q,
    });
    if (!data) {
      this.setPhase("idle");
      this.googleFailMessage(null, "gmail");
      return;
    }
    if (!data.ok) {
      this.setPhase("idle");
      this.googleFailMessage(data, "gmail");
      return;
    }
    if (parsed.openFirst && data.messages.length) {
      const full = await fetchGmailMessage(
        this.googleToken(), data.messages[0].id);
      this.setPhase("idle");
      if (full && full.ok) {
        const m = full.message;
        const body =
          m.bodyText.length > 1200
            ? `${m.bodyText.slice(0, 1200)}\n…（以下省略）`
            : m.bodyText;
        this.pushAssistant({
          text:
            `最新のメールです。\n件名：${m.subject || "（件名なし）"}\n` +
            `差出人：${m.from || "（差出人不明）"}\n\n${body}`,
          badge: "gmail",
        });
        return;
      }
    }
    this.setPhase("idle");
    this.pushAssistant({
      text: gmailListText(data.messages),
      badge: "gmail",
      note: parsed.q
        ? "条件に合う新しい順のメールです。"
        : "新しい順のメールです。",
    });
  },

  async runGcalLane(parsed) {
    if (!(await this.ensureGoogleReady("gcal"))) return;
    const token = this.googleToken();
    if (parsed.kind === "list") {
      this.setPhase("searching");
      const data = await fetchGcalEvents(
        token, parsed.timeMin, parsed.timeMax);
      this.setPhase("idle");
      if (!data) return this.googleFailMessage(null, "gcal");
      if (!data.ok) return this.googleFailMessage(data, "gcal");
      this._gcalEvents = data.events;
      this.pushAssistant({
        text: gcalListText(parsed.label, data.events),
        badge: "gcal",
      });
      return;
    }
    if (parsed.kind === "create") {
      if (parsed.missing.length) {
        this.pushAssistant({
          text:
            `予定を入れるには、${parsed.missing.join("と")}が必要です。` +
            "例えば「明日の10時に『歯医者』の予定を入れて」のように教えてください。",
          badge: "gcal",
        });
        return;
      }
      let dateKey = parsed.date.key;
      // 日付指定が過去なら来年の同じ日とみなす（今年の過去日に
      // 予定を作る聞き間違いを避ける）。
      if (parsed.date.explicit && dateKey < gcalDateKeyPlus(0)) {
        dateKey = `${Number(dateKey.slice(0, 4)) + 1}${dateKey.slice(4)}`;
      }
      const start = `${dateKey}T${parsed.times[0]}:00+09:00`;
      const end = parsed.times[1]
        ? `${dateKey}T${parsed.times[1]}:00+09:00`
        : gcalPlusOneHour(start);
      this.setPhase("searching");
      const res = await createGcalEvent(token, {
        summary: parsed.title,
        start,
        end,
      });
      this.setPhase("idle");
      if (!res) return this.googleFailMessage(null, "gcal");
      if (!res.ok) return this.googleFailMessage(res, "gcal");
      this.pushAssistant({
        text: `予定を入れました。\n${res.event ? gcalEventLine(res.event) : `・${parsed.title}`}`,
        badge: "gcal",
      });
      return;
    }
    // 変更・削除：対象の予定を特定してから実行する。
    const target = await this.resolveGcalTarget(parsed, token);
    if (!target) return;
    if (parsed.kind === "delete") {
      this.setPhase("searching");
      const res = await deleteGcalEvent(token, target.id);
      this.setPhase("idle");
      if (!res) return this.googleFailMessage(null, "gcal");
      if (!res.ok) return this.googleFailMessage(res, "gcal");
      this._gcalEvents = (this._gcalEvents || [])
        .filter((e) => e.id !== target.id);
      this.pushAssistant({
        text: `予定を削除しました。\n${gcalEventLine(target)}`,
        badge: "gcal",
      });
      return;
    }
    // update
    const patch = {};
    if (parsed.title) patch.summary = parsed.title;
    if (parsed.date || parsed.times.length) {
      const baseDate = parsed.date
        ? parsed.date.key
        : target.start.slice(0, 10);
      const baseStart = parsed.times[0] || target.start.slice(11, 16);
      if (!baseStart) {
        this.pushAssistant({
          text: "その予定は終日の予定なので、時刻の変更はできませんでした。日付だけの変更は「○月○日に変更して」と言ってください。",
          badge: "gcal",
        });
        return;
      }
      patch.start = `${baseDate}T${baseStart}:00+09:00`;
      if (parsed.times[1]) {
        patch.end = `${baseDate}T${parsed.times[1]}:00+09:00`;
      } else if (
        /^\d{4}-\d{2}-\d{2}T/.test(target.start) &&
        /^\d{4}-\d{2}-\d{2}T/.test(target.end)
      ) {
        // 元の所要時間を保つ。
        const durMs =
          Date.parse(target.end) - Date.parse(target.start);
        patch.end = gcalIsoFromMs(Date.parse(patch.start) + durMs);
      } else {
        patch.end = gcalPlusOneHour(patch.start);
      }
    }
    if (!Object.keys(patch).length) {
      this.pushAssistant({
        text: "変更の内容が分かりませんでした。「○○の予定を明日の15時に変更して」のように、新しい日時や名前を含めて言ってください。",
        badge: "gcal",
      });
      return;
    }
    this.setPhase("searching");
    const res = await updateGcalEvent(token, target.id, patch);
    this.setPhase("idle");
    if (!res) return this.googleFailMessage(null, "gcal");
    if (!res.ok) return this.googleFailMessage(res, "gcal");
    this.pushAssistant({
      text: `予定を変更しました。\n${res.event ? gcalEventLine(res.event) : gcalEventLine(target)}`,
      badge: "gcal",
    });
  },

  /* 変更・削除の対象を、直近表示した予定→今後2週間の予定の
     順で特定する。特定できない・複数ある場合は正直に出して
     null を返す（推測で消したり変えたりはしない）。 */
  async resolveGcalTarget(parsed, token) {
    const pick = (events) => {
      if (
        parsed.ordinal != null &&
        parsed.ordinal >= 0 &&
        events[parsed.ordinal]
      ) {
        return { event: events[parsed.ordinal] };
      }
      if (parsed.hint) {
        const hits = events.filter(
          (e) =>
            e.summary &&
            (e.summary.includes(parsed.hint) ||
              parsed.hint.includes(e.summary)),
        );
        if (hits.length === 1) return { event: hits[0] };
        if (hits.length > 1) return { candidates: hits };
      }
      if (parsed.ordinal == null && !parsed.hint && events.length === 1) {
        return { event: events[0] };
      }
      return {};
    };
    let pool = Array.isArray(this._gcalEvents) ? this._gcalEvents : [];
    let r = pick(pool);
    if (!r.event && !r.candidates) {
      this.setPhase("searching");
      const data = await fetchGcalEvents(
        token,
        `${gcalDateKeyPlus(0)}T00:00:00+09:00`,
        `${gcalDateKeyPlus(14)}T00:00:00+09:00`,
      );
      this.setPhase("idle");
      if (data && data.ok) {
        pool = data.events;
        this._gcalEvents = pool;
        r = pick(pool);
      }
    }
    if (r.event) return r.event;
    if (r.candidates) {
      this.pushAssistant({
        text:
          "対象の予定が複数見つかりました。どれですか？\n" +
          r.candidates.map(gcalEventLine).join("\n"),
        badge: "gcal",
        note: "「2番目の予定を消して」のように番号で言うか、予定の名前を含めてもう一度言ってください。",
      });
      return null;
    }
    this.pushAssistant({
      text: "対象の予定が見つかりませんでした。",
      badge: "gcal",
      note: "「今週の予定は？」などで確認してから、予定の名前を含めてもう一度言ってください。推測で消したり変更したりはしません。",
    });
    return null;
  },

  /* ---- Google連携（アカウント欄の UI） ---- */

  /* 連携 UI の DOM をアカウント欄に差し込む（index.html は
     触らず、script.js から作る）。 */
  initGoogleLinkUi() {
    const area = el("account-area");
    if (!area || el("google-link-area")) return;
    const box = document.createElement("div");
    box.id = "google-link-area";
    box.className = "google-link-area";
    box.hidden = true;
    const status = document.createElement("p");
    status.id = "google-link-status";
    status.className = "google-link-status";
    const linkBtn = document.createElement("button");
    linkBtn.type = "button";
    linkBtn.id = "google-link-btn";
    linkBtn.className = "google-link-btn";
    linkBtn.textContent = "Googleと連携する";
    linkBtn.addEventListener("click", () => void this.googleLinkStart());
    const unlinkBtn = document.createElement("button");
    unlinkBtn.type = "button";
    unlinkBtn.id = "google-unlink-btn";
    unlinkBtn.className = "google-unlink-btn";
    unlinkBtn.textContent = "連携を解除";
    unlinkBtn.hidden = true;
    unlinkBtn.addEventListener("click", () => void this.googleUnlink());
    box.appendChild(status);
    box.appendChild(linkBtn);
    box.appendChild(unlinkBtn);
    area.appendChild(box);
    const notices = document.createElement("div");
    notices.id = "google-notices";
    notices.className = "google-notices";
    notices.hidden = true;
    area.appendChild(notices);
    // Google 同意からの戻り（?google=linked / ?google=error）。
    try {
      const params = new URLSearchParams(location.search);
      const g = params.get("google");
      if (g === "linked") {
        this.googleReturnNotice = "Googleと連携しました。";
      } else if (g === "error") {
        this.googleReturnNotice =
          "Google連携が完了しませんでした。もう一度「Googleと連携する」から試してください。";
      }
      if (g) {
        params.delete("google");
        params.delete("reason");
        const qs = params.toString();
        history.replaceState(
          null, "",
          location.pathname + (qs ? `?${qs}` : "") + location.hash);
      }
    } catch {
      /* URL が読めない環境では何もしない */
    }
  },

  renderGoogleLink() {
    const box = el("google-link-area");
    if (!box) return;
    const mgr = this.account;
    if (!mgr || !mgr.signedIn) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const status = el("google-link-status");
    const linkBtn = el("google-link-btn");
    const unlinkBtn = el("google-unlink-btn");
    const st = this.googleState;
    if (!st) {
      if (status && !this.googleReturnNotice) {
        status.textContent = "Google連携：確認中…";
      }
      if (linkBtn) linkBtn.hidden = true;
      if (unlinkBtn) unlinkBtn.hidden = true;
      return;
    }
    if (!st.configured) {
      if (status) status.textContent = "Google連携：準備中";
      if (linkBtn) linkBtn.hidden = true;
      if (unlinkBtn) unlinkBtn.hidden = true;
      return;
    }
    if (st.linked) {
      if (status) {
        status.textContent =
          this.googleReturnNotice ||
          `Google連携済み：${st.googleEmail || ""}`;
      }
      if (linkBtn) linkBtn.hidden = true;
      if (unlinkBtn) unlinkBtn.hidden = false;
    } else {
      if (status) {
        status.textContent =
          this.googleReturnNotice || "Google連携：未連携";
      }
      if (linkBtn) linkBtn.hidden = false;
      if (unlinkBtn) unlinkBtn.hidden = true;
    }
  },

  async googleLinkStart() {
    const token = this.googleToken();
    if (!token) return;
    const status = el("google-link-status");
    if (status) status.textContent = "Googleの許可画面を開いています…";
    const r = await startGoogleLink(token);
    if (r && r.ok && r.url) {
      location.href = r.url;
      return;
    }
    if (status) {
      status.textContent =
        "連携を開始できませんでした。時間をおいてもう一度試してください。";
    }
  },

  async googleUnlink() {
    const token = this.googleToken();
    if (!token) return;
    await unlinkGoogle(token);
    this.googleState = {
      ok: true,
      configured: true,
      linked: false,
      googleEmail: "",
    };
    this.googleStateAt = Date.now();
    this.googleReturnNotice = "";
    this.renderGoogleLink();
    const notices = el("google-notices");
    if (notices) {
      notices.innerHTML = "";
      notices.hidden = true;
    }
  },

  /* 送信ログの未表示分を通知欄に出す（本人の条件：送信ログ＋
     通知）。表示した分は既読として localStorage に残す。 */
  async checkSendLogNotices() {
    const st = this.googleState;
    if (!st || !st.linked) return;
    const data = await fetchGmailSendLog(this.googleToken(), 20);
    if (!data || !data.ok) return;
    let seen = [];
    try {
      seen = JSON.parse(
        localStorage.getItem(GOOGLE_SENDLOG_SEEN_KEY) || "[]");
    } catch {
      seen = [];
    }
    const seenSet = new Set(Array.isArray(seen) ? seen : []);
    const fresh = data.entries
      .filter((e) => !seenSet.has(gmailSendLogKey(e)));
    if (fresh.length) this.showSendLogNotices(fresh);
    try {
      localStorage.setItem(
        GOOGLE_SENDLOG_SEEN_KEY,
        JSON.stringify(data.entries.map(gmailSendLogKey).slice(-100)));
    } catch {
      /* 保存できなくても通知の表示は続ける */
    }
  },

  showSendLogNotices(entries) {
    const box = el("google-notices");
    if (!box || !entries.length) return;
    box.hidden = false;
    for (const e of entries) {
      const item = document.createElement("div");
      item.className = "google-notice";
      const text = document.createElement("span");
      text.textContent =
        `メール送信：${e.to} へ「${e.subject}」を送信しました`;
      const close = document.createElement("button");
      close.type = "button";
      close.className = "google-notice-close";
      close.textContent = "×";
      close.setAttribute("aria-label", "通知を閉じる");
      close.addEventListener("click", () => {
        item.remove();
        if (!box.children.length) box.hidden = true;
      });
      item.appendChild(text);
      item.appendChild(close);
      box.appendChild(item);
    }
    try {
      const seen = JSON.parse(
        localStorage.getItem(GOOGLE_SENDLOG_SEEN_KEY) || "[]");
      const set = new Set(Array.isArray(seen) ? seen : []);
      for (const e of entries) set.add(gmailSendLogKey(e));
      localStorage.setItem(
        GOOGLE_SENDLOG_SEEN_KEY,
        JSON.stringify([...set].slice(-100)));
    } catch {
      /* 既読の保存に失敗しても表示は済んでいる */
    }
  },

  /* ---- PC操作レーンの実行と表示 ----
     進行中は履歴に残らない仮メッセージでステップを見せ、完了後に
     最終結果を通常の回答として出す（履歴に残るのは結果だけ）。
     進行中はライブ画面（仮想PCのスクリーンショット）も流し続け、
     画面タップや「自分で操作する」で AI を止めてユーザーが直接
     操作できる（乗っ取り）。操作モードの正本はサーバーの
     control イベントで、手入力は mode:"user" の間だけ通る。 */
  async runPcLane(question) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-ai";
    const b = document.createElement("span");
    b.className = `badge mono ${BADGES.pc.className}`;
    b.textContent = BADGES.pc.label;
    li.appendChild(b);
    const p = document.createElement("p");
    p.className = "answer";
    p.textContent = "仮想PCを操作しています。少し時間がかかります…";
    li.appendChild(p);
    const n = document.createElement("p");
    n.className = "note";
    n.textContent = "開始しています…";
    li.appendChild(n);

    /* ---- ライブ画面と乗っ取りコントロール ---- */
    const liveImg = document.createElement("img");
    liveImg.className = "pc-live-screen";
    liveImg.alt = "仮想PCの画面（ライブ）";
    liveImg.hidden = true;
    li.appendChild(liveImg);
    const live = { taskId: null, mode: "ai", busy: false };
    const canInput = typeof CubePc !== "undefined" &&
      typeof CubePc.postInput === "function" &&
      typeof CubePc.toScreenCoords === "function";
    const controls = document.createElement("div");
    controls.className = "pc-live-controls";
    controls.hidden = true;
    const takeoverBtn = document.createElement("button");
    takeoverBtn.type = "button";
    takeoverBtn.className = "pc-live-btn";
    takeoverBtn.textContent = "操作する";
    const releaseBtn = document.createElement("button");
    releaseBtn.type = "button";
    releaseBtn.className = "pc-live-btn";
    releaseBtn.textContent = "AIに戻す";
    releaseBtn.hidden = true;
    const liveState = document.createElement("span");
    liveState.className = "pc-live-state";
    controls.appendChild(takeoverBtn);
    controls.appendChild(releaseBtn);
    controls.appendChild(liveState);
    li.appendChild(controls);
    const keyRow = document.createElement("div");
    keyRow.className = "pc-live-keys";
    keyRow.hidden = true;
    const KEY_BUTTONS = [
      ["Enter", "Return"], ["Backspace", "BackSpace"], ["Tab", "Tab"],
      ["Esc", "Escape"], ["↑", "Up"], ["↓", "Down"], ["←", "Left"],
      ["→", "Right"], ["Space", "space"],
    ];
    const typeRow = document.createElement("div");
    typeRow.className = "pc-live-keys";
    typeRow.hidden = true;
    const typeInput = document.createElement("input");
    typeInput.type = "text";
    typeInput.className = "pc-live-type-input";
    typeInput.placeholder = "入力する文字";
    const typeBtn = document.createElement("button");
    typeBtn.type = "button";
    typeBtn.className = "pc-live-btn";
    typeBtn.textContent = "入力する";
    typeRow.appendChild(typeInput);
    typeRow.appendChild(typeBtn);
    li.appendChild(keyRow);
    li.appendChild(typeRow);

    /* ---- 実行前確認（削除・送信のゲート） ----
       実行側が削除・送信の直前で止まると confirm イベントが届く。
       要約と「実行する／やめる」ボタンを出し、応答は input 経由
       （乗っ取りと同じ経路）で返す。確定は confirm_resolved
       イベントが正本。応答しなければ実行側は実行せずに終わる。 */
    const confirmBox = document.createElement("div");
    confirmBox.className = "pc-confirm";
    confirmBox.hidden = true;
    const confirmText = document.createElement("p");
    confirmText.className = "pc-confirm-text";
    confirmBox.appendChild(confirmText);
    const confirmBtns = document.createElement("div");
    confirmBtns.className = "pc-confirm-btns";
    const confirmOkBtn = document.createElement("button");
    confirmOkBtn.type = "button";
    confirmOkBtn.className = "pc-live-btn pc-confirm-ok";
    confirmOkBtn.textContent = "実行する";
    const confirmCancelBtn = document.createElement("button");
    confirmCancelBtn.type = "button";
    confirmCancelBtn.className = "pc-live-btn";
    confirmCancelBtn.textContent = "やめる";
    confirmBtns.appendChild(confirmOkBtn);
    confirmBtns.appendChild(confirmCancelBtn);
    confirmBox.appendChild(confirmBtns);
    li.appendChild(confirmBox);
    const confirmState = { confirmId: null, busy: false };
    const answerConfirm = async (decision) => {
      if (!live.taskId || !confirmState.confirmId || confirmState.busy) return;
      if (typeof CubePc === "undefined" ||
          typeof CubePc.postInput !== "function") return;
      confirmState.busy = true;
      confirmOkBtn.disabled = true;
      confirmCancelBtn.disabled = true;
      try {
        const r = await CubePc.postInput(
          CUBE_SERVER_BASE_URL, live.taskId,
          { type: "confirm", confirmId: confirmState.confirmId, decision });
        if (!r.ok) {
          confirmState.busy = false;
          confirmOkBtn.disabled = false;
          confirmCancelBtn.disabled = false;
          confirmText.textContent =
            (r.data && r.data.detail) || "確認の応答を受け付けられませんでした。";
        }
        // 成功時は confirm_resolved イベントでボックスを閉じる
      } catch {
        confirmState.busy = false;
        confirmOkBtn.disabled = false;
        confirmCancelBtn.disabled = false;
        confirmText.textContent = "サーバーに接続できませんでした。";
      }
    };
    confirmOkBtn.addEventListener("click", () => { void answerConfirm("approve"); });
    confirmCancelBtn.addEventListener("click", () => { void answerConfirm("deny"); });

    const updateLiveUi = () => {
      takeoverBtn.hidden = live.mode === "user";
      releaseBtn.hidden = live.mode !== "user";
      keyRow.hidden = live.mode !== "user";
      typeRow.hidden = live.mode !== "user";
      liveState.textContent = live.mode === "user"
        ? "あなたが操作中です（AIは止まっています）。画面をタップするとクリックできます。"
        : live.busy
          ? "AIを止めています…"
          : "AIが操作中です。自分で操作したいときは「操作する」を押してください。";
    };
    const sendInput = async (cmd) => {
      if (!live.taskId || !canInput) return;
      try {
        const r = await CubePc.postInput(
          CUBE_SERVER_BASE_URL, live.taskId, cmd);
        if (!r.ok) {
          liveState.textContent =
            (r.data && r.data.detail) || "入力を受け付けられませんでした。";
        }
      } catch {
        liveState.textContent = "サーバーに接続できませんでした。";
      }
    };
    takeoverBtn.addEventListener("click", () => {
      if (live.mode === "user" || live.busy) return;
      live.busy = true;
      updateLiveUi();
      void sendInput({ type: "takeover" });
      const t = setTimeout(() => {
        if (live.mode !== "user") { live.busy = false; updateLiveUi(); }
      }, 10000);
      if (t && typeof t.unref === "function") t.unref();
    });
    releaseBtn.addEventListener("click", () => {
      void sendInput({ type: "release" });
    });
    for (const [label, key] of KEY_BUTTONS) {
      const kb = document.createElement("button");
      kb.type = "button";
      kb.className = "pc-live-btn";
      kb.textContent = label;
      kb.addEventListener("click", () => { void sendInput({ type: "key", key }); });
      keyRow.appendChild(kb);
    }
    typeBtn.addEventListener("click", () => {
      const text = typeInput.value;
      if (text) {
        void sendInput({ type: "type", text });
        typeInput.value = "";
      }
    });
    liveImg.addEventListener("click", (ev) => {
      if (!live.taskId || !canInput) return;
      if (!ev || typeof ev.clientX !== "number") return;
      const rect = typeof liveImg.getBoundingClientRect === "function"
        ? liveImg.getBoundingClientRect() : null;
      const pt = CubePc.toScreenCoords(rect, ev.clientX, ev.clientY);
      if (!pt) return;
      if (live.mode === "user") {
        void sendInput({ type: "click", x: pt.x, y: pt.y });
      }
      // AI操作中のタップは何もしない（乗っ取りは「操作する」
      // ボタンからだけ。誤タップで AI を止めないため）
    });
    el("msg-list").appendChild(li);
    this.scrollChat();

    const outcome = await executePcTask(question, (ev) => {
      if (ev.type === "step") {
        const labels = (ev.actions || [])
          .map((a) => (typeof CubePc !== "undefined" && CubePc.actionLabel)
            ? CubePc.actionLabel(a.raw || "") : (a.raw || ""))
          .filter(Boolean);
        n.textContent = labels.length
          ? `ステップ${ev.step}：${labels[labels.length - 1]}`
          : `ステップ${ev.step} を実行中…`;
        this.scrollChat();
      } else if (ev.type === "started") {
        live.taskId = ev.taskId;
        if (canInput) {
          controls.hidden = false;
          updateLiveUi();
        }
      } else if (ev.type === "screenshot") {
        liveImg.hidden = false;
        liveImg.src = `${CUBE_SERVER_BASE_URL}${ev.url}` +
          (ev.step === "live" ? `?t=${Date.now()}` : "");
        this.scrollChat();
      } else if (ev.type === "control") {
        if (ev.mode === "user" || ev.mode === "ai") {
          live.mode = ev.mode;
          live.busy = false;
          updateLiveUi();
        }
      } else if (ev.type === "verification" && ev.phase === "done_reject") {
        n.textContent = "確認：まだ完了していないため差し戻しました";
        this.scrollChat();
      } else if (ev.type === "confirm") {
        confirmState.confirmId = ev.confirmId;
        confirmState.busy = false;
        confirmOkBtn.disabled = false;
        confirmCancelBtn.disabled = false;
        confirmText.textContent =
          (ev.summary || "この操作を実行しますか？") +
          (ev.detail ? `（${ev.detail}）` : "");
        confirmBox.hidden = false;
        n.textContent = "確認待ちです。内容を見て「実行する」か「やめる」を選んでください。";
        this.scrollChat();
      } else if (ev.type === "confirm_resolved") {
        confirmState.confirmId = null;
        confirmState.busy = false;
        confirmBox.hidden = true;
        n.textContent = ev.decision === "approve"
          ? "確認しました。実行します…"
          : ev.decision === "deny"
            ? "キャンセルしました（実行しません）"
            : ev.decision === "timeout"
              ? "確認の返事がなかったため、実行しませんでした"
              : "確認が終了しました";
        this.scrollChat();
      }
    });

    li.remove();
    if (outcome.kind === "finished" && outcome.result) {
      this.pushAssistant({
        text: outcome.result.message || "終わりました。",
        badge: "pc",
        note: pcResultNote(outcome.result),
      });
    } else if (outcome.kind === "unsupported" || outcome.kind === "busy") {
      this.pushAssistant({
        text: outcome.detail,
        badge: "pc",
        note: "今できる操作は「おためし」タブのPC操作パネルに例があります。",
      });
    } else {
      this.pushAssistant({
        text: "PC操作サーバーに接続できませんでした。",
        badge: "pc",
        note: "仮想PCやモデルの準備ができていない可能性があります。時間をおいてもう一度試してください。",
      });
    }
  },

  /* ---- 場所・話題・買い物レーンのカード描画 ----
     数字や固有名詞はカードのデータ（API の値）からだけ作る。
     文章の生成はしない。 */

  buildLaneCards(cards) {
    const wrap = document.createElement("div");
    wrap.className = "lane-cards";
    for (const card of cards) {
      if (!card || typeof card !== "object") continue;
      if (card.kind === "place") wrap.appendChild(this.buildPlaceCard(card));
      else if (card.kind === "address") {
        wrap.appendChild(this.buildAddressCard(card));
      }
      else if (card.kind === "topic") {
        wrap.appendChild(this.buildTopicCard(card));
      } else if (card.kind === "station") {
        wrap.appendChild(this.buildStationCard(card));
      } else if (card.kind === "music") {
        wrap.appendChild(this.buildMusicCard(card));
      } else if (card.kind === "paper") {
        wrap.appendChild(this.buildPaperCard(card));
      } else if (card.kind === "translate") {
        wrap.appendChild(this.buildTranslateCard(card));
      } else if (card.kind === "wolfram") {
        wrap.appendChild(this.buildWolframCard(card));
      } else if (card.kind === "shop") {
        wrap.appendChild(this.buildShopCard(card));
      } else if (card.kind === "suggest") {
        wrap.appendChild(this.buildSuggestChip(card));
      }
    }
    return wrap;
  },

  buildPlaceCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = card.name;
      title.appendChild(a);
    } else {
      title.textContent = card.name;
    }
    div.appendChild(title);
    const meta = [];
    if (card.address) meta.push(card.address);
    const dist = formatDistanceLabel(card.distance_m);
    if (dist) meta.push(dist);
    if (card.nearest_station) meta.push(`最寄り：${card.nearest_station}`);
    if (Array.isArray(card.genre) && card.genre.length) {
      meta.push(card.genre.join("・"));
    }
    if (card.hours) meta.push(`営業時間：${card.hours}`);
    if (card.tel) meta.push(`電話：${card.tel}`);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join("　");
      div.appendChild(m);
    }
    const srcLabels = (Array.isArray(card.sources) ? card.sources : [])
      .map((s) => (s === "yahoo" ? "Yahoo! JAPAN"
        : s === "osm" ? "OpenStreetMap" : s))
      .filter(Boolean);
    if (srcLabels.length) {
      const sp = document.createElement("p");
      sp.className = "lane-card-meta";
      sp.textContent = `出典：${srcLabels.join(" / ")}`;
      div.appendChild(sp);
    }
    return div;
  },

  /* 住所レーンの位置カード。タイトルが地図へのリンクになり、
     座標は API の値をそのまま転記する。 */
  buildAddressCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = `${card.title}（地図で開く）`;
      title.appendChild(a);
    } else {
      title.textContent = card.title;
    }
    div.appendChild(title);
    const m = document.createElement("p");
    m.className = "lane-card-meta";
    m.textContent = `緯度 ${card.lat} / 経度 ${card.lng}`;
    div.appendChild(m);
    return div;
  },

  /* 最寄り駅カード。駅名・路線・距離は API の値の転記だけ。 */
  buildStationCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = `${card.name}駅（地図で開く）`;
      title.appendChild(a);
    } else {
      title.textContent = `${card.name}駅`;
    }
    div.appendChild(title);
    const meta = [];
    if (card.line) meta.push(card.line);
    if (card.distance) meta.push(`約${card.distance}`);
    if (card.prefecture) meta.push(card.prefecture);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join("　");
      div.appendChild(m);
    }
    return div;
  },

  /* 音楽カード。曲名が Apple Music へのリンク、試聴リンクつき。 */
  buildMusicCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = card.track;
      title.appendChild(a);
    } else {
      title.textContent = card.track;
    }
    div.appendChild(title);
    const meta = [];
    if (card.artist) meta.push(card.artist);
    if (card.album) meta.push(card.album);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join(" / ");
      div.appendChild(m);
    }
    if (card.preview) {
      const p = document.createElement("p");
      p.className = "lane-card-meta";
      const a = document.createElement("a");
      a.href = card.preview;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = "試聴する";
      p.appendChild(a);
      div.appendChild(p);
    }
    if (card.artwork) {
      const img = document.createElement("img");
      img.className = "lane-card-img";
      img.src = card.artwork;
      img.alt = "";
      img.loading = "lazy";
      div.appendChild(img);
    }
    return div;
  },

  /* 論文カード。タイトルが論文ページへのリンクになり、著者・
     年・出典・抄録の抜粋は API の値の転記だけ（文章は作らない）。 */
  buildPaperCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = card.title;
      title.appendChild(a);
    } else {
      title.textContent = card.title;
    }
    div.appendChild(title);
    const meta = [];
    if (card.authors) meta.push(card.authors);
    if (card.year) meta.push(String(card.year));
    if (card.source) meta.push(card.source);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join(" / ");
      div.appendChild(m);
    }
    if (card.summary) {
      const s = document.createElement("p");
      s.className = "lane-card-meta";
      s.textContent = card.summary;
      div.appendChild(s);
    }
    return div;
  },

  /* 翻訳カード。訳文が主で、原文と変換方向（言語名）は API の
     値とコード対応表からだけ作る（文章は作らない）。 */
  buildTranslateCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    title.textContent = card.translatedText;
    div.appendChild(title);
    const meta = [];
    if (card.sourceText) meta.push(`原文：${card.sourceText}`);
    const srcName = translateLangName(card.sourceLang) || card.sourceLang;
    const tgtName = translateLangName(card.targetLang) || card.targetLang;
    if (srcName && tgtName) meta.push(`${srcName} → ${tgtName}`);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join(" / ");
      div.appendChild(m);
    }
    return div;
  },

  /* Wolfram 計算カード。答えが主で、Wolfram が解釈した式と、
     結果ページへのリンク（出典表示として必須）を添える。
     リンクの URL は API の応答に含まれるものだけを使う。 */
  buildWolframCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    title.textContent = card.resultText;
    div.appendChild(title);
    const meta = [];
    if (card.interpretation) meta.push(`解釈：${card.interpretation}`);
    else if (card.wolframQuery) meta.push(`式：${card.wolframQuery}`);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join(" / ");
      div.appendChild(m);
    }
    if (card.resultUrl) {
      const p = document.createElement("p");
      p.className = "lane-card-meta";
      const a = document.createElement("a");
      a.href = card.resultUrl;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = "Wolfram|Alpha で詳しく見る";
      p.appendChild(a);
      div.appendChild(p);
    }
    return div;
  },

  buildTopicCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.link) {
      const a = document.createElement("a");
      a.href = card.link;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = card.title;
      title.appendChild(a);
    } else {
      title.textContent = card.title;
    }
    div.appendChild(title);
    const meta = [];
    if (card.sourceLabel) meta.push(card.sourceLabel);
    const when = formatTimestampLabel(card.published);
    if (when) meta.push(when);
    if (meta.length) {
      const m = document.createElement("p");
      m.className = "lane-card-meta";
      m.textContent = meta.join("　");
      div.appendChild(m);
    }
    if (card.snippet && card.snippet !== card.title) {
      const s = document.createElement("p");
      s.className = "lane-card-snippet";
      s.textContent = card.snippet;
      div.appendChild(s);
    }
    return div;
  },

  buildShopCard(card) {
    const div = document.createElement("div");
    div.className = "lane-card";
    const title = document.createElement("p");
    title.className = "lane-card-title";
    if (card.url) {
      const a = document.createElement("a");
      a.href = card.url;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = card.name;
      title.appendChild(a);
    } else {
      title.textContent = card.name;
    }
    div.appendChild(title);
    if (card.image) {
      const img = document.createElement("img");
      img.className = "lane-card-img";
      img.src = card.image;
      img.alt = card.name;
      img.loading = "lazy";
      div.appendChild(img);
    }
    if (typeof card.price === "number") {
      const price = document.createElement("p");
      price.className = "lane-card-price";
      price.textContent =
        `${card.price.toLocaleString("ja-JP")}円` +
        (card.tax_included ? "（税込）" : "（税別）");
      div.appendChild(price);
    }
    const meta = [];
    meta.push(card.shipping_label || "送料は商品ページでご確認ください");
    if (card.in_stock === false) meta.push("在庫なし");
    if (typeof card.review_rate === "number" && card.review_count > 0) {
      meta.push(`評価 ${card.review_rate}（${card.review_count}件）`);
    }
    if (card.shop_name) meta.push(card.shop_name);
    if (card.rank_reason) meta.push(card.rank_reason);
    const m = document.createElement("p");
    m.className = "lane-card-meta";
    m.textContent = meta.join("　");
    div.appendChild(m);
    return div;
  },

  /* 条件を緩める提案チップ（ゼロ件のとき）。押すとその質問で
     もう一度聞く。 */
  buildSuggestChip(card) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lane-suggest";
    btn.textContent = card.label || "";
    btn.addEventListener("click", () => {
      void this.ask(card.query);
    });
    return btn;
  },

  /* 買い物レーンの検索が表示まで至ったら、モール横断検索と同じ
     mall_search 種別で +1pt（1日3pt上限はサーバー側 RPC が種別
     単位で数えるため、モールサイト分と合算になる）。冪等キーは
     cube_shop: 前置で発生元を区別する。未ログイン・失敗時は
     何もしない（ポイントは付加機能）。 */
  async earnShopPoints(keyword, conditions) {
    try {
      const backend = this.pointsBackend();
      const mgr = this.account;
      if (!backend || !mgr || !mgr.user) return;
      const key =
        `cube_shop:${pointsJstDateKey()}:${shopConditionsHash(keyword, conditions)}`;
      let before = null;
      try {
        before = await backend.fetchBalance();
      } catch { /* 残高が取れなくても付与の試行は続ける */ }
      const after = await backend.earn(
        "mall_search", key, { source: "cube_shop" });
      if (typeof after === "number") {
        if (before !== null && after > before) {
          this.showPointsNote("+1 pt（買い物検索）");
        }
        this.pointsBalance = after;
        this.renderPoints();
      }
    } catch {
      /* ポイントの失敗は会話を妨げない */
    }
  },

  /* 朝のブリーフィングを会話の一番上に差す（pushAssistant と同じ
     形だが、末尾ではなく先頭に入れる）。 */
  pushBriefing(text, briefingTopics) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-ai";
    const b = document.createElement("span");
    b.className = `badge mono ${BADGES.briefing.className}`;
    b.textContent = BADGES.briefing.label;
    li.appendChild(b);
    const p = document.createElement("p");
    p.className = "answer";
    p.textContent = text;
    li.appendChild(p);
    // 今の話題（/topics）：見出しと出典をそのまま並べる「紹介」で、
    // Ultra の文章とは混ぜない。stale のときは少し前の情報と明示する。
    if (
      briefingTopics &&
      Array.isArray(briefingTopics.cards) &&
      briefingTopics.cards.length
    ) {
      const sec = document.createElement("div");
      sec.className = "lane-cards";
      const head = document.createElement("p");
      head.className = "lane-card-title";
      head.textContent = briefingTopics.stale
        ? "今の話題（少し前の時点の情報）"
        : "今の話題";
      sec.appendChild(head);
      for (const card of briefingTopics.cards) {
        try {
          sec.appendChild(this.buildTopicCard(card));
        } catch {
          /* 1件の描画失敗でブリーフィング全体は落とさない */
        }
      }
      li.appendChild(sec);
    }
    const list = el("msg-list");
    list.insertBefore(li, list.children[0] || null);
  },

  /* 朝のブリーフィング：05:00–11:00 の間に、その日初めて Cube を
     開いたときだけ、iMac の /briefing から材料を取り、Ultra に
     書かせて会話の先頭に出す。init から非同期に呼び、どの段階で
     失敗しても何も出さない（通常の Cube の邪魔をしない）。
     Ultra が書けなければ材料のデータだけの文面で出す。ユーザーが
     既に質問していたら出さない（印も付けないので、次の機会に
     改めて試す）。表示できたら当日の日付を記録し、同日は再表示
     しない。 */
  async maybeShowBriefing() {
    try {
      const now = new Date();
      const hour = now.getHours();
      if (hour < BRIEFING_START_HOUR || hour >= BRIEFING_END_HOUR) return;
      const today = localDateKey(now);
      if (store.get(BRIEFING_DATE_KEY) === today) return;
      let data = null;
      try {
        const res = await fetchWithTimeout(
          `${CUBE_SERVER_BASE_URL}/briefing`,
          { method: "GET" },
          BRIEFING_FETCH_TIMEOUT_MS,
        );
        if (res && res.ok) data = await res.json().catch(() => null);
      } catch {
        data = null;
      }
      // サーバー側の日付（JST）と端末の日付が違うデータは使わない。
      if (!data || data.date !== today) return;
      const material = buildBriefingMaterial(data);
      if (!material) return;
      let text = "";
      try {
        await UltraImacModel.load(() => {});
        // 節ごとに書かせる（1回で全部書かせると最初の話題だけで
        // 終わる）。書けた節は Ultra の文、書けなかった節は材料の
        // 文でつなぎ、1節でも書けたらブリーフィングとして出す。
        const parts = [];
        let anyWritten = false;
        for (const section of material.sections) {
          let written = "";
          try {
            const out = await generateUltra(
              `文：${section.passage}\n質問：${BRIEFING_SECTION_QUESTIONS[section.key] || "この内容を簡潔に伝えてください。です・ます調で書いてください。"}`,
            );
            const t = (out || "").trim();
            if (t && !isUltraAbstention(t)) {
              written = trimBriefingTail(normalizeUltraWrittenStyle(t));
            }
          } catch {
            written = "";
          }
          if (written) anyWritten = true;
          parts.push(written || section.prose);
        }
        if (anyWritten) {
          text = `おはようございます。${parts.join("")}`.replace(
            /^おはようございます。おはようございます。/,
            "おはようございます。",
          );
        }
      } catch {
        text = "";
      }
      if (!text) text = material.plain;
      if (!text) return;
      // 既に会話が始まっていたら、邪魔をしない。
      const chatted = [...el("msg-list").children].some(
        (c) => c && c.className === "msg-user",
      );
      if (chatted) return;
      // 今の話題（/topics）：見出しと出典をそのまま並べる紹介で、
      // Ultra には言い換えさせない。取れなければ話題なしで出す
      // （ブリーフィング自体は落とさない）。表示が確定してから
      // 取りに行くので、会話が始まっていたら取得自体をしない。
      let briefingTopics = null;
      try {
        const t = await fetchTopics(5, 5000);
        if (t && t.topics.length) {
          briefingTopics = {
            stale: t.stale,
            cards: t.topics.slice(0, 5).map(topicCardOf),
          };
        }
      } catch {
        briefingTopics = null;
      }
      this.pushBriefing(text, briefingTopics);
      store.set(BRIEFING_DATE_KEY, today);
    } catch {
      /* ブリーフィングの失敗は黙って流す */
    }
  },

  /* ---- Cube 記憶（cube_memory.js） ---- */

  /* 記憶モジュールの実体（initMemory で設定。無い環境では null）。 */
  memory: null,

  /* 記憶の初期化。保存先はこの端末の localStorage を基本形とする
     （公開版では訪問者のブラウザ内にだけ残る。DESIGN.md §3）。
     MEMORY_SERVER_ENABLED が true のときだけ、iMac の /memory との
     ハイブリッド保存にする（エンドポイント配置後に有効化すること）。
     モジュールが読み込まれていない環境（Node テスト等）では何も
     しない——記憶が無い状態では従来の挙動と一切変わらない。 */
  initMemory() {
    this.memory = null;
    try {
      if (typeof CubeMemory === "undefined") return;
      const store = MEMORY_SERVER_ENABLED
        ? CubeMemory.makeHybridStore(
            CubeMemory.makeServerStore(CUBE_SERVER_BASE_URL),
            CubeMemory.makeLocalStore(),
          )
        : CubeMemory.makeLocalStore();
      const mem = CubeMemory.create({ store });
      this.memory = mem;
      window.CubeMem = mem;
      mem
        .load()
        .then(() => this.updateMemoryTabCount())
        .catch(() => { /* 記憶が読めなくても通常利用は妨げない */ });
      const form = el("mem-add-form");
      if (form) {
        form.addEventListener("submit", (e) => {
          e.preventDefault();
          void this.addMemoryFromForm();
        });
      }
    } catch {
      this.memory = null;
    }
  },

  /* 質問の入口での記憶処理：明示コマンド（覚えて／忘れて）の解釈と
     発言からの候補抽出を行い、Ultra 向け注入サフィックスを今回の
     質問用に差し替える。明示コマンドだった場合は通知だけを出して
     true を返す（そのターンはそこで完了する）。記憶が無い・失敗
     した場合は黙って false を返し、通常の回答フローに影響しない。 */
  async memoryObserve(question) {
    cubeMemorySystemSuffix = "";
    const mem = this.memory;
    if (!mem || !mem.loaded) return false;
    let result = null;
    try {
      result = await mem.observeUserMessage(question);
    } catch {
      return false;
    }
    try {
      cubeMemorySystemSuffix = mem.systemSuffix(question) || "";
    } catch {
      cubeMemorySystemSuffix = "";
    }
    if (result && (result.added.length || result.forgotten.length)) {
      this.updateMemoryTabCount();
    }
    if (result && result.notices && result.notices.length) {
      for (const notice of result.notices) this.pushMemoryNotice(notice);
      return true;
    }
    return false;
  },

  /* 記憶の通知（覚えました／忘れました等）を、通常の返答とは別に
     「記憶の答え」バッジつきの小さな注記として会話へ出す。 */
  pushMemoryNotice(text) {
    el("empty-state").hidden = true;
    const li = document.createElement("li");
    li.className = "msg-ai msg-memnote";
    const b = document.createElement("span");
    b.className = `badge mono ${BADGES.memory.className}`;
    b.textContent = BADGES.memory.label;
    li.appendChild(b);
    const p = document.createElement("p");
    p.className = "mem-note";
    p.textContent = text;
    li.appendChild(p);
    el("msg-list").appendChild(li);
    this.historyCaptureNotice(text);
    this.scrollChat();
  },

  /* 自作モデル（Nano/Flash/Twin/Pro/Max）向けの質問装飾：QA 形式に
     文脈枠が無いため、関連が強い記憶だけを最大2件・短く質問の末尾に
     添える（モジュール側で強い主題一致のあるものだけに絞られる）。
     記憶が無い・一致が弱ければ質問はそのまま返る。Twin/Pro/Max は
     質問文字列がそのままサーバーへ送られるため、この形で届く。 */
  memoryDecorate(question) {
    const mem = this.memory;
    if (!mem || !mem.loaded) return question;
    try {
      return mem.decorateQuestion(question, { maxEntries: 2 }) || question;
    } catch {
      return question;
    }
  },

  /* 直接参照レーン：ユーザー自身についての質問（私の〜は？／覚えて
     いる？）に、モデルを介さず記憶から即答する。全モデル共通の
     経路。Cube 自身の固定情報（手順0.5）とは対象が違う（あちらは
     Cube のこと、こちらはユーザーのこと）ので衝突しない。記憶に
     該当が無ければ null を返し、通常フローに委ねる。 */
  memoryDirectAnswer(question) {
    const mem = this.memory;
    if (!mem || !mem.loaded) return null;
    const q = String(question || "");
    const selfRef = /(私|僕|あたし|自分|俺|おれ)/.test(q);
    try {
      // モジュール側の直接参照（自己言及＋主題一致）。
      const hit = mem.findDirectAnswer(q);
      if (hit) return `「${hit.text}」と覚えています。`;
      if (!selfRef && !/(覚えて|おぼえて)/.test(q)) return null;
      if (selfRef && /好きな(もの|こと|食べ物|食べもの|スポーツ|遊び|曲|本|映画)?/.test(q)) {
        const prefs = mem.list({ kind: "preference" });
        if (prefs.length) {
          return `あなたの好きなものとして、${prefs.map((e) => `「${e.text}」`).join("、")}を覚えています。`;
        }
      }
      if (selfRef && /名前/.test(q)) {
        const named = mem
          .list({ kind: "user" })
          .find((e) => (e.topic && e.topic.includes("名前")) || /名前/.test(e.text));
        if (named) return `「${named.text}」と覚えています。`;
      }
      if (selfRef && /(住んで|住所)/.test(q)) {
        const home = mem
          .list({ kind: "user" })
          .find((e) => e.topic === "住んでいる場所");
        if (home) return `「${home.text}」と覚えています。`;
      }
      if (selfRef && /通って/.test(q)) {
        const place = mem
          .list({ kind: "user" })
          .find((e) => e.topic === "通っている場所");
        if (place) return `「${place.text}」と覚えています。`;
      }
      // 「私のこと覚えてる？」への一般形：現行の記憶を要約して返す。
      if (/(覚えて|おぼえて)/.test(q)) {
        const actives = mem.list({}).filter((e) => e.kind !== "daily");
        if (actives.length) {
          const shown = actives.slice(0, 3).map((e) => `「${e.text}」`).join("、");
          return `あなたのこととして、${shown}${actives.length > 3 ? " など" : ""}を覚えています。`;
        }
      }
    } catch {
      return null;
    }
    return null;
  },

  updateMemoryTabCount() {
    const span = el("mem-tab-count");
    if (!span) return;
    const mem = this.memory;
    const n = mem && mem.loaded ? mem.list({}).length : 0;
    span.textContent = n > 0 ? ` (${n})` : "";
  },

  /* 記憶ビューの「手動で覚えさせる」フォーム。 */
  async addMemoryFromForm() {
    const note = el("mem-add-note");
    const input = el("mem-add-text");
    const kindSel = el("mem-add-kind");
    const mem = this.memory;
    if (!mem) {
      if (note) note.textContent = "記憶モジュールが読み込まれていません。";
      return;
    }
    const text = input ? input.value.trim() : "";
    if (!text) return;
    const r = await mem.addExplicit(text, kindSel ? kindSel.value : undefined);
    if (note) {
      note.textContent = r.added
        ? `覚えました：${r.entry.text}`
        : r.reason === "looks-secret"
          ? "秘密情報らしい内容は覚えられません。"
          : r.reason === "duplicate"
            ? "それは既に覚えています。"
            : "覚えられませんでした。";
    }
    if (r.added && input) input.value = "";
    this.updateMemoryTabCount();
    this.renderMemory();
  },

  /* ---- 記憶タブ ---- */
  renderMemory() {
    const listEl = el("mem-list");
    if (!listEl) return;
    const mem = this.memory;
    const actives = mem && mem.loaded ? mem.list({}) : [];
    const counts = el("mem-counts");
    if (counts) {
      counts.textContent = actives.length > 0 ? `覚えていること ${actives.length}件` : "";
      counts.hidden = actives.length === 0;
    }
    const empty = el("mem-empty");
    if (empty) empty.hidden = actives.length > 0;
    this.updateMemoryTabCount();
    listEl.textContent = "";
    for (const e of actives) listEl.appendChild(this.memoryItemEl(e, false));
    const hist = mem && mem.loaded
      ? [...mem.list({ status: "superseded" }), ...mem.list({ status: "forgotten" })]
      : [];
    const wrap = el("mem-history-wrap");
    const histList = el("mem-history");
    if (wrap && histList) {
      wrap.hidden = hist.length === 0;
      histList.textContent = "";
      for (const e of hist) histList.appendChild(this.memoryItemEl(e, true));
    }
  },

  memoryItemEl(e, isHistory) {
    const li = document.createElement("li");
    li.className = "nb-item";
    const flex = document.createElement("div");
    flex.className = "nb-flex";
    const main = document.createElement("div");
    main.className = "nb-main";
    const tags = document.createElement("p");
    tags.className = "nb-tags";
    const kt = document.createElement("span");
    kt.className = "tag mono";
    kt.textContent = MEM_KIND_LABELS[e.kind] || e.kind;
    tags.appendChild(kt);
    if (e.topic) {
      const tt = document.createElement("span");
      tt.className = "tag mono";
      tt.textContent = e.topic;
      tags.appendChild(tt);
    }
    if (isHistory) {
      const st = document.createElement("span");
      st.className = "tag mono tag-untrained";
      st.textContent = e.status === "superseded" ? "上書き済み" : "忘れたもの";
      tags.appendChild(st);
    }
    const src = document.createElement("span");
    src.className = "tag mono";
    src.textContent = `出所：${MEM_SOURCE_LABELS[e.source && e.source.type] || "会話から"}`;
    tags.appendChild(src);
    main.appendChild(tags);
    const textP = document.createElement("p");
    textP.className = "nb-answer";
    textP.textContent = e.text;
    main.appendChild(textP);
    const d = document.createElement("p");
    d.className = "nb-date mono";
    d.textContent = e.date || "";
    main.appendChild(d);
    flex.appendChild(main);
    const actions = document.createElement("div");
    actions.className = "nb-actions";
    if (!isHistory) {
      const eb = document.createElement("button");
      eb.type = "button";
      eb.className = "small-btn";
      eb.textContent = "編集";
      eb.setAttribute("aria-label", `「${e.text}」を編集`);
      eb.addEventListener("click", () => {
        if (!this.memory) return;
        textP.hidden = true;
        eb.disabled = true;
        const box = document.createElement("div");
        box.className = "mem-edit-box";
        const input = document.createElement("input");
        input.type = "text";
        input.value = e.text;
        input.className = "mem-edit-input";
        box.appendChild(input);
        const save = document.createElement("button");
        save.type = "button";
        save.className = "small-btn";
        save.textContent = "保存";
        save.addEventListener("click", () => {
          void (async () => {
            const updated = await this.memory.updateText(e.id, input.value);
            if (!updated) {
              input.setAttribute("aria-invalid", "true");
              return;
            }
            this.renderMemory();
          })();
        });
        box.appendChild(save);
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "small-btn";
        cancel.textContent = "キャンセル";
        cancel.addEventListener("click", () => this.renderMemory());
        box.appendChild(cancel);
        main.appendChild(box);
      });
      actions.appendChild(eb);
      const fb = document.createElement("button");
      fb.type = "button";
      fb.className = "small-btn";
      fb.textContent = "忘れる";
      fb.setAttribute("aria-label", `「${e.text}」を忘れる`);
      fb.addEventListener("click", () => {
        void (async () => {
          if (!this.memory) return;
          await this.memory.forget(e.text);
          this.renderMemory();
        })();
      });
      actions.appendChild(fb);
    }
    const db = document.createElement("button");
    db.type = "button";
    db.className = "small-btn";
    db.textContent = "削除";
    db.setAttribute("aria-label", `「${e.text}」を削除`);
    db.addEventListener("click", () => {
      void (async () => {
        if (!this.memory) return;
        await this.memory.remove(e.id);
        this.renderMemory();
      })();
    });
    actions.appendChild(db);
    flex.appendChild(actions);
    li.appendChild(flex);
    return li;
  },

  saveNotebook(entry) {
    // 同語反復の文は知識として固定しない（最後の関門）。
    if (isTautologicalAnswer(entry.answer)) return;
    Notebook.add(entry);
    el("nb-tab-count").textContent = ` (${Notebook.items.length})`;
  },

  /* ---- Cube アカウントとチャット履歴（cube_account.js） ---- */

  /* アカウントマネージャ（initAccount で設定。config 未設定・
     モジュール未読込の環境では null のままで、アカウント機能と
     履歴は一切動かない＝従来挙動と完全同一）。 */
  account: null,

  /* Cubeポイントの残高（未ログイン・未取得の間は null）。 */
  pointsBalance: null,

  /* ポイントで追加したメーター分（%、端末に保持）。 */
  pointsBonusPct: 0,

  /* ポイント処理（取得・付与・消費）の多重実行ガード。 */
  pointsBusy: false,

  /* ログインボーナスを試した相手 "userId:JST日付"。
     同じ相手・同じ日にはページ内で再試行しない（RPC 側も冪等）。 */
  pointsDailyFor: "",

  /* トップアップの未確定キー（pct → 冪等キー）。結果不明の失敗の
     あと再試行しても二重に引かれないよう、確定するまで使い回す。 */
  pointsTopupKeys: {},

  /* 履歴ストア（ローカル保存＋ログイン時のクラウド同期）。
     無効な環境では null。 */
  historyStore: null,

  /* Google連携の状態（/google/status の結果）と取得時刻。 */
  googleState: null,
  googleStateAt: 0,

  /* Google 同意から戻った直後の案内文（連携欄に一度出す）。 */
  googleReturnNotice: "",

  /* メール送信の下書き（聞き返しの間だけ保持する）。 */
  _gmailDraft: null,

  /* 直近に表示したカレンダーの予定（変更・削除の対象特定用）。 */
  _gcalEvents: [],

  /* いま開いている会話 {id,title,created_at,updated_at,turns}。
     まだ何も話していない間は null で、最初の発言で作られる。 */
  historyCurrent: null,

  /* 履歴から開いた（再開した）会話かどうか。再開した会話では、
     続きの質問に文脈を引き継ぐ（Ultra は履歴メッセージ列、
     自作モデルは直近のやり取りの抜粋）。 */
  historyResumed: false,

  /* 履歴の復元中はキャプチャを止めるための印。 */
  historyRestoring: false,

  historyRenderSeq: 0,

  /* アカウントと履歴の初期化。CUBE_ACCOUNT_CONFIG が空の間は
     何もしない（アカウント領域・履歴 UI は hidden のまま）。 */
  initAccount() {
    this.account = null;
    this.historyStore = null;
    try {
      if (typeof CubeAccount === "undefined") return;
      if (!CubeAccount.isConfigured(CUBE_ACCOUNT_CONFIG)) return;
      const mgr = CubeAccount.createManager({
        config: CUBE_ACCOUNT_CONFIG,
        clientFactory:
          (typeof window !== "undefined" &&
            window.__CUBE_ACCOUNT_CLIENT_FACTORY__) || undefined,
      });
      if (!mgr.enabled) return;
      this.account = mgr;
      this.historyStore = CubeAccount.makeCloudHistoryStore(
        CubeAccount.makeLocalHistoryStore(),
        mgr,
      );
      const loginBtn = el("account-login-btn");
      if (loginBtn) {
        loginBtn.addEventListener("click", () => void this.accountSignIn());
      }
      const logoutBtn = el("account-logout-btn");
      if (logoutBtn) {
        logoutBtn.addEventListener("click", () => void this.accountSignOut());
      }
      for (const item of POINTS_TOPUP_MENU) {
        const btn = el(`points-topup-${item.pct}`);
        if (btn) {
          btn.addEventListener("click",
            () => void this.pointsTopup(item.pct, item.price));
        }
      }
      this.pointsLoadBonus();
      this.initGoogleLinkUi();
      const newChat = el("new-chat-btn");
      if (newChat) {
        newChat.addEventListener("click", (e) => {
          e.preventDefault();
          this.newConversation();
        });
      }
      mgr.onChange(() => void this.handleAccountChange());
      this.renderAccount();
      void mgr
        .init()
        .then(() => this.handleAccountChange())
        .catch(() => this.renderAccount());
      void this.renderHistoryList();
    } catch {
      this.account = null;
      this.historyStore = null;
    }
  },

  /* サイドバーのアカウント領域を、認証状態に合わせて描く。 */
  renderAccount() {
    const area = el("account-area");
    if (!area) return;
    const mgr = this.account;
    if (!mgr || !mgr.enabled) {
      area.hidden = true;
      return;
    }
    area.hidden = false;
    const user = mgr.user;
    const loginBtn = el("account-login-btn");
    const userBox = el("account-user");
    const note = el("account-note");
    if (user) {
      if (loginBtn) loginBtn.hidden = true;
      if (userBox) userBox.hidden = false;
      const nameEl = el("account-name");
      if (nameEl) nameEl.textContent = user.name || user.email || "ログイン中";
      const av = el("account-avatar");
      if (av) {
        if (user.avatarUrl) {
          av.src = user.avatarUrl;
          av.hidden = false;
        } else {
          av.hidden = true;
        }
      }
      if (note) note.textContent = "ログイン中：会話履歴はアカウントに同期されます";
    } else {
      if (loginBtn) loginBtn.hidden = false;
      if (userBox) userBox.hidden = true;
      if (note) note.textContent = "ログインすると会話履歴を他の端末と同期できます";
    }
    this.renderPoints();
    this.renderGoogleLink();
  },

  /* 認証状態が変わったら：アカウント表示を更新し、履歴を
     クラウドと突き合わせてリストを取り直す。ポイントも
     ログイン状態に合わせて取り直す（失敗は静かに握る）。 */
  async handleAccountChange() {
    this.renderAccount();
    void this.refreshPoints();
    void this.refreshGoogleStatus(true).then(() => {
      this.renderGoogleLink();
      void this.checkSendLogNotices();
    });
    await this.renderHistoryList();
  },

  async accountSignIn() {
    if (!this.account) return;
    try {
      await this.account.signInWithGoogle();
    } catch {
      /* 遷移できなかった場合はそのまま（未ログインで使い続ける） */
    }
  },

  async accountSignOut() {
    if (!this.account) return;
    try {
      await this.account.signOut();
    } catch {
      /* ログアウトの失敗は表示側だけでも未ログインに倒す */
    }
    this.renderAccount();
    await this.renderHistoryList();
  },

  /* --- Cubeポイント（付加機能。失敗はすべて静かに握る） --- */

  /* ログイン中だけポイント用バックエンドを返す。 */
  pointsBackend() {
    const mgr = this.account;
    if (!mgr || !mgr.enabled || !mgr.user) return null;
    try {
      return typeof mgr.pointsBackend === "function"
        ? mgr.pointsBackend() : null;
    } catch {
      return null;
    }
  },

  pointsLoadBonus() {
    try {
      const v = Number(localStorage.getItem(POINTS_BONUS_KEY));
      this.pointsBonusPct =
        Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
    } catch {
      this.pointsBonusPct = 0;
    }
  },

  pointsSaveBonus() {
    try {
      localStorage.setItem(POINTS_BONUS_KEY, String(this.pointsBonusPct));
    } catch { /* 保存できなくても表示は続ける */ }
  },

  /* ポイント表示を認証状態と残高に合わせて描く。未ログインでは
     領域ごと出さない。 */
  renderPoints() {
    const area = el("points-area");
    if (!area) return;
    const mgr = this.account;
    if (!mgr || !mgr.enabled || !mgr.user) {
      area.hidden = true;
      return;
    }
    area.hidden = false;
    const bal = el("points-balance");
    if (bal) {
      bal.textContent =
        this.pointsBalance === null ? "…" : String(this.pointsBalance);
    }
    const bonus = el("points-bonus");
    if (bonus) {
      bonus.hidden = !(this.pointsBonusPct > 0);
      const bp = el("points-bonus-pct");
      if (bp) bp.textContent = String(this.pointsBonusPct);
    }
    for (const item of POINTS_TOPUP_MENU) {
      const btn = el(`points-topup-${item.pct}`);
      if (btn) btn.disabled = this.pointsBusy;
    }
    /* ポイント残高・追加分の変化はメーター表示にも反映する。 */
    this.renderMeter();
  },

  showPointsNote(text) {
    const note = el("points-note");
    if (note) note.textContent = text || "";
  },

  /* 時間制メーターの表示（cube_meter.js）。モジュールが無い環境
     では出さない。無料枠は端末単位なので、ログイン状態とは
     無関係に常時表示する。消費のたびに onChange 経由でも
     呼ばれる。 */
  renderMeter() {
    const area = el("meter-area");
    if (!area) return;
    const meter = getCubeMeter();
    if (!meter || typeof CubeMeter === "undefined") {
      area.hidden = true;
      return;
    }
    const snap = meter.snapshot();
    area.hidden = false;
    const rem = el("meter-remaining");
    if (rem) rem.textContent = CubeMeter.formatDuration(snap.remainingSec);
    const plan = el("meter-plan");
    if (plan) {
      plan.textContent = snap.congestion
        ? "（混雑時の上限が適用中です）"
        : !Number.isFinite(snap.baseTotalSec)
          ? "（無制限プラン）"
          : "";
    }
    const fill = el("meter-bar-fill");
    if (fill && fill.style) {
      let pct = 100;
      if (Number.isFinite(snap.baseTotalSec)) {
        const total = snap.baseTotalSec + snap.bonusTotalSec;
        pct = total > 0
          ? Math.max(0, Math.min(100, (snap.remainingSec / total) * 100))
          : 0;
      }
      fill.style.width = `${pct}%`;
    }
    const bonus = el("meter-bonus");
    if (bonus) {
      bonus.hidden = !(snap.bonusRemainingSec > 0);
      const br = el("meter-bonus-remaining");
      if (br) br.textContent = CubeMeter.formatDuration(snap.bonusRemainingSec);
    }
    const note = el("meter-note");
    if (note) {
      note.textContent = snap.remainingSec <= 0
        ? "今日の分は使い切りました。Cube Nano / Cube Flash はメーターなしで使えます。"
        : "";
    }
  },

  /* 残高を取り直し、今日のログインボーナス（1日1回）を付けに
     行く。RPC は冪等なので多重実行しても二重付与にはならない。
     付与前後の残高が増えたときだけ「+1 pt」を出す。 */
  async refreshPoints() {
    const backend = this.pointsBackend();
    const mgr = this.account;
    if (!backend || !mgr || !mgr.user) {
      this.pointsBalance = null;
      this.renderPoints();
      return;
    }
    this.renderPoints();
    if (this.pointsBusy) return;
    this.pointsBusy = true;
    try {
      let balance = null;
      try {
        balance = await backend.fetchBalance();
      } catch { /* 残高が取れなくても付与の試行は続ける */ }
      const dailyFor = `${mgr.user.id}:${pointsJstDateKey()}`;
      if (this.pointsDailyFor !== dailyFor) {
        this.pointsDailyFor = dailyFor;
        try {
          const after = await backend.earn(
            "daily_login", pointsDailyLoginKey(), { source: "cube_web" });
          if (typeof after === "number") {
            if (balance !== null && after > balance) {
              this.showPointsNote("+1 pt（ログインボーナス）");
            }
            balance = after;
          }
        } catch { /* 付与できなくても残高表示は続ける */ }
      }
      if (typeof balance === "number") this.pointsBalance = balance;
    } catch { /* ポイントの失敗は会話を妨げない */ }
    this.pointsBusy = false;
    this.renderPoints();
  },

  /* ポイントでメーター追加。消費は RPC が成功したときだけ起き、
     成功したときだけ追加分を反映する。残高不足なら何も変えず
     「ポイントが足りません」と出す。通信失敗など結果不明のときは
     冪等キーを保持し、再試行で二重に引かれないようにする。 */
  async pointsTopup(pct, price) {
    const backend = this.pointsBackend();
    if (!backend || this.pointsBusy) return false;
    const item = POINTS_TOPUP_MENU
      .find((m) => m.pct === pct && m.price === price);
    if (!item) return false;
    this.pointsBusy = true;
    this.renderPoints();
    try {
      let key = this.pointsTopupKeys[pct];
      if (!key) {
        key = pointsNewTopupKey();
        this.pointsTopupKeys[pct] = key;
      }
      const after = await backend.spend(
        "meter_topup", item.price, key,
        { pct: item.pct, source: "cube_web" });
      delete this.pointsTopupKeys[pct];
      if (typeof after === "number") this.pointsBalance = after;
      this.pointsBonusPct += item.pct;
      this.pointsSaveBonus();
      this.showPointsNote(`メーターに +${item.pct}% 追加しました`);
      this.pointsBusy = false;
      this.renderPoints();
      return true;
    } catch (e) {
      this.pointsBusy = false;
      const msg = (e && e.message) || "";
      if (msg.includes("insufficient_balance")) {
        /* 確定した残高不足：キーは使い切り、残高だけ取り直す。 */
        delete this.pointsTopupKeys[pct];
        this.showPointsNote("ポイントが足りません");
        try {
          const b = await backend.fetchBalance();
          if (typeof b === "number") this.pointsBalance = b;
        } catch { /* 表示更新の失敗は静かに握る */ }
      } else {
        /* 結果不明：キーを保持して二重消費を防ぐ。 */
        this.showPointsNote("追加できませんでした。通信を確認してもう一度どうぞ");
      }
      this.renderPoints();
      return false;
    }
  },

  /* --- 履歴へのキャプチャ（描画フックから呼ばれる） --- */

  historyEnsureCurrent(firstText) {
    if (!this.historyCurrent) {
      const now = new Date().toISOString();
      this.historyCurrent = {
        id: CubeAccount.newConversationId(),
        title: CubeAccount.titleFromQuestion(firstText),
        created_at: now,
        updated_at: now,
        turns: [],
      };
      this.historyResumed = false;
    }
    return this.historyCurrent;
  },

  historyCaptureUser(text) {
    if (!this.historyStore || this.historyRestoring) return;
    try {
      const conv = this.historyEnsureCurrent(text);
      conv.turns.push({ role: "user", text: String(text) });
      conv.updated_at = new Date().toISOString();
      this.persistCurrentConversation();
    } catch {
      /* 履歴の失敗は会話を妨げない */
    }
  },

  historyCaptureAssistant(msg) {
    if (!this.historyStore || this.historyRestoring) return;
    try {
      if (!this.historyCurrent) return;
      const turn = { role: "ai", text: String((msg && msg.text) || "") };
      for (const k of
        ["badge", "evidence", "modelSaid", "sourceUrl", "sourceTitle", "note"]) {
        if (msg && typeof msg[k] === "string" && msg[k]) turn[k] = msg[k];
      }
      // 嘘チェック可視化：検証内訳（プレーンなJSON）も一緒に残す。
      if (msg && msg.factcheck) turn.factcheck = msg.factcheck;
      // 場所・話題・買い物レーンのカード（プレーンなJSON）も一緒に
      // 残し、履歴から復元したときも同じ形で出し直す。
      if (msg && Array.isArray(msg.cards) && msg.cards.length) {
        try {
          turn.cards = JSON.parse(JSON.stringify(msg.cards));
        } catch {
          /* カードが残せなくても会話履歴は守る */
        }
      }
      this.historyCurrent.turns.push(turn);
      this.historyCurrent.updated_at = new Date().toISOString();
      this.persistCurrentConversation();
    } catch {
      /* 履歴の失敗は会話を妨げない */
    }
  },

  /* 記憶の通知（覚えました等）も、表示されたとおりに残す。 */
  historyCaptureNotice(text) {
    if (!this.historyStore || this.historyRestoring) return;
    try {
      if (!this.historyCurrent) return;
      this.historyCurrent.turns.push({
        role: "ai", text: String(text), badge: "memory", memnote: true,
      });
      this.historyCurrent.updated_at = new Date().toISOString();
      this.persistCurrentConversation();
    } catch {
      /* 履歴の失敗は会話を妨げない */
    }
  },

  persistCurrentConversation() {
    const conv = this.historyCurrent;
    if (!conv || !this.historyStore) return;
    let snapshot;
    try {
      snapshot = JSON.parse(JSON.stringify(conv));
    } catch {
      return;
    }
    void this.historyStore
      .saveConversation(snapshot)
      .then(() => this.renderHistoryList())
      .catch(() => {
        /* 保存の失敗は会話を妨げない */
      });
  },

  /* --- 続きの文脈（履歴から再開した会話だけ） --- */

  /* 現在の質問（最後の user ターン）を除いた、完了したやり取り
     （user/ai の組）だけを返す。再開した会話でなければ null。 */
  historyPriorTurns() {
    if (!this.historyStore || !this.historyResumed || !this.historyCurrent) {
      return null;
    }
    const turns = this.historyCurrent.turns || [];
    const prior =
      turns.length && turns[turns.length - 1].role === "user"
        ? turns.slice(0, -1)
        : turns.slice();
    const usable = prior.filter(
      (t) => t.role === "user" || (t.role === "ai" && !t.memnote),
    );
    return usable.length ? usable : null;
  },

  /* Ultra 用：直近のターンを ChatML のメッセージ列にする。
     llama-server の文脈長（-c 2048）に収めるため、直近6ターン・
     1ターン240字・合計900字で古い順に切り捨てる（モデルによる
     要約はしない：追加生成のコストと、未検証文を文脈に固定する
     リスクを避ける。設計書 ACCOUNT_DESIGN.md 参照）。 */
  historyMessagesForUltra() {
    const prior = this.historyPriorTurns();
    if (!prior) return null;
    const picked = [];
    let used = 0;
    for (let i = prior.length - 1; i >= 0 && picked.length < 6; i--) {
      let text = String(prior[i].text || "");
      if (text.length > 240) text = text.slice(0, 240) + "…";
      if (picked.length > 0 && used + text.length > 900) break;
      picked.unshift({
        role: prior[i].role === "user" ? "user" : "assistant",
        content: text,
      });
      used += text.length;
    }
    return picked.length ? picked : null;
  },

  /* 雑談専門（Cube 会話）用：現在の会話の直近のやり取りを
     user / assistant のメッセージ列にする。雑談は文脈が命なので、
     再開した会話でなくても、進行中の会話（historyCurrent）から
     取る。現在の質問（最後の user ターン）は除き、メモ注入の
     印つき AI ターンは混ぜない。上限は直近8ターン・1ターン
     300字・合計1600字（サーバー側の上限とも整合）。履歴が無い
     環境（未ログイン等）では null（単発の会話として送る）。 */
  historyMessagesForChat() {
    if (!this.historyCurrent) return null;
    const turns = this.historyCurrent.turns || [];
    const prior =
      turns.length && turns[turns.length - 1].role === "user"
        ? turns.slice(0, -1)
        : turns.slice();
    const usable = prior.filter(
      (t) => t.role === "user" || (t.role === "ai" && !t.memnote),
    );
    if (!usable.length) return null;
    const picked = [];
    let used = 0;
    for (let i = usable.length - 1; i >= 0 && picked.length < 8; i--) {
      let text = String(usable[i].text || "");
      if (text.length > 300) text = text.slice(0, 300) + "…";
      if (picked.length > 0 && used + text.length > 1600) break;
      picked.unshift({
        role: usable[i].role === "user" ? "user" : "assistant",
        content: text,
      });
      used += text.length;
    }
    return picked.length ? picked : null;
  },

  /* 自作モデル用：直近のやり取りの抜粋を質問の末尾に添える
     （記憶の装飾と同じ流儀で、QA 形式の中に収める）。再開した
     会話でない・前のやり取りが無い場合は質問をそのまま返す
     （バイト単位で同一＝通常質問への回帰なし）。 */
  historyDecorate(question) {
    const prior = this.historyPriorTurns();
    if (!prior) return question;
    const pairs = [];
    for (let i = prior.length - 1; i >= 1 && pairs.length < 2; i--) {
      if (prior[i].role === "ai" && prior[i - 1].role === "user") {
        pairs.unshift([prior[i - 1], prior[i]]);
        i--;
      }
    }
    if (!pairs.length) return question;
    const clip = (s, n) => {
      const t = String(s || "");
      return t.length > n ? t.slice(0, n) + "…" : t;
    };
    const parts = pairs.map(
      ([u, a]) => `質問「${clip(u.text, 50)}」に「${clip(a.text, 80)}」と答えた`,
    );
    const suffix = `（これまでのやり取り：${parts.join("。")}）`;
    if (suffix.length > 260) return question;
    return question + suffix;
  },

  /* --- 会話の切り替え・新規・削除 --- */

  clearMessages() {
    const list = el("msg-list");
    if (!list) return;
    while (list.children.length) list.removeChild(list.children[0]);
    list.textContent = "";
  },

  /* 新規チャット：いまの会話は履歴に残したまま、空の会話にする。
     生成中は切り替えない（回答の行き先がずれるため）。 */
  newConversation() {
    if (!this.historyStore || this.phase !== "idle") return;
    this.historyCurrent = null;
    this.historyResumed = false;
    this.clearMessages();
    const empty = el("empty-state");
    if (empty) empty.hidden = false;
    this.showTab("chat");
    void this.renderHistoryList();
  },

  /* 履歴から会話を開く：ターンをそのまま再描画し、その会話の
     続きとして話せる状態にする（historyResumed を立てる）。
     生成中は切り替えない。 */
  async openConversation(id) {
    if (!this.historyStore || this.phase !== "idle") return;
    let conv = null;
    try {
      conv = await this.historyStore.loadConversation(id);
    } catch {
      conv = null;
    }
    if (!conv) return;
    this.historyRestoring = true;
    try {
      this.clearMessages();
      for (const turn of conv.turns || []) {
        if (turn.role === "user") this.pushUser(turn.text);
        else if (turn.memnote) this.pushMemoryNotice(turn.text);
        else this.pushAssistant(turn);
      }
    } finally {
      this.historyRestoring = false;
    }
    this.historyCurrent = conv;
    this.historyResumed = true;
    const empty = el("empty-state");
    if (empty) empty.hidden = (conv.turns || []).length > 0;
    this.showTab("chat");
    this.scrollChat();
    void this.renderHistoryList();
  },

  async deleteConversation(id) {
    if (!this.historyStore) return;
    try {
      await this.historyStore.deleteConversation(id);
    } catch {
      /* 削除の失敗は表示側の更新だけ行う */
    }
    if (this.historyCurrent && this.historyCurrent.id === id) {
      this.historyCurrent = null;
      this.historyResumed = false;
      this.clearMessages();
      const empty = el("empty-state");
      if (empty) empty.hidden = false;
    }
    void this.renderHistoryList();
  },

  /* サイドバーの履歴リストを描く（新しい順・表示中の会話を強調）。 */
  async renderHistoryList() {
    const group = el("history-group");
    const listEl = el("history-list");
    if (!group || !listEl) return;
    if (!this.historyStore) {
      group.hidden = true;
      return;
    }
    group.hidden = false;
    const seq = ++this.historyRenderSeq;
    let convs = [];
    try {
      convs = await this.historyStore.listConversations();
    } catch {
      convs = [];
    }
    if (seq !== this.historyRenderSeq) return; // 古い描画は捨てる
    while (listEl.children.length) listEl.removeChild(listEl.children[0]);
    const emptyNote = el("history-empty");
    if (emptyNote) emptyNote.hidden = convs.length > 0;
    for (const c of convs) {
      const li = document.createElement("li");
      li.className =
        "history-item" +
        (this.historyCurrent && this.historyCurrent.id === c.id
          ? " active"
          : "");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-open";
      const title = document.createElement("span");
      title.className = "history-title";
      title.textContent = c.title || "無題の会話";
      const date = document.createElement("span");
      date.className = "history-date mono";
      date.textContent = formatHistoryDate(c.updated_at);
      btn.append(title, date);
      btn.addEventListener("click", () => void this.openConversation(c.id));
      const del = document.createElement("button");
      del.type = "button";
      del.className = "history-del";
      del.textContent = "×";
      del.setAttribute("aria-label", "この会話を削除");
      del.addEventListener("click", () => void this.deleteConversation(c.id));
      li.append(btn, del);
      listEl.appendChild(li);
    }
  },

  /* 質問への回答の流れ：
     0) 天気 → 気象庁の予報＋Open-Meteoの現在の状況（実データ）
     0.5) Cube自身の質問 → 固定情報
     0.6) 会話レーン → あいさつ・お礼・別れ・定番の雑談への即答
       （Cube 会話が選択されているときは使わない：専門モデルが返す）
     0.75) 計算レーン → 正確な計算結果（過程つき）
     1) 知識メモ帳 → 即答＋裏で再確認
     2) 計算 → 検算 2.5) メーター関門
     2.6) 雑談分岐 → Cube 会話選択時、事実でない会話は専門モデルへ
     3) モデル生成 → 対象記事の事実で照合 */
  async ask(rawQuestion) {
    const question = (rawQuestion || "").trim();
    if (this.phase !== "idle") return;
    // 画像添付つきの送信（Ultra 専用）：画像の内容はネットの裏づけ
    // 検証の対象にならないため、通常の質問経路には入れず、画像
    // レーン（askWithImage）で完結させる。文章が空なら既定の
    // 聞き方で送る。
    if (pendingChatImage && this.modelId === "ultra") {
      const image = pendingChatImage;
      const q = question || "この画像について教えてください。";
      this.pushUser(q, image);
      pendingChatImage = null;
      this.syncAttach();
      el("question-input").value = "";
      this.syncSend();
      await this.askWithImage(q, image);
      return;
    }
    if (!question) return;
    this.pushUser(question);
    el("question-input").value = "";
    this.syncSend();
    // 記憶：明示コマンド（覚えて／忘れて）の解釈と、発言からの候補
    // 抽出。明示コマンドだったら通知だけを出してこのターンを終える。
    // あわせて Ultra 向けの注入サフィックスを今回の質問用に差し替える
    // （記憶が無い環境では常に空＝従来動作）。
    if (await this.memoryObserve(question)) return;
    // 「回答可能」は、読み込み済みのモデルIDと選択中のIDが一致する
    // ときだけ真。切り替え先の読み込みに失敗した状態では、前のモデル
    // のエンジンで（新しいモデルの名前で）回答することはない。
    const modelReady = modelReadyFor(this.modelId);
    const normalized = normalizeQuestion(question);
    // 今回の検索の失敗理由はここから集める（前の質問の理由を持ち越さない）。
    WikiNet.lastError = "";
    // 嘘チェック可視化：前ターンの検証内訳を持ち越さない。
    this._fcBreakdown = null;

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

    // 0.6) 会話レーン：あいさつ・お礼・別れ・定番の雑談は、事実の
    // 主張を含まない会話なので、モデル生成・ネット検索・検証ゲート
    // をすべて素通りして、その場で返事を返す。以前は通常の質問と
    // 同じ経路に入り、「こんにちは」にすら裏づけ確認の末の
    // 「確認できませんでした」が返ることがあった。判定は発話全体が
    // 定番パターンに一致する場合だけに限られ、事実の質問や複合文は
    // conversationAnswer が拾わない（従来の経路に倒れる）。
    // 会話の返事は知識メモ帳にも保存しない。
    // 雑談専門（Cube 会話）が選択されているときはこのレーンを使わ
    // ない：あいさつ・相づちへの返事も専門モデル自身が会話として
    // 返すのが配線の目的（2026-11-20 本人指示）。定型文で先回り
    // すると専門モデルに会話が届かなくなるため、判定だけして
    // 採用はしない（convAnswer の計算自体は従来どおり）。
    const convAnswer = conversationAnswer(question);
    if (convAnswer && this.modelId !== "chat") {
      this.pushAssistant({
        text: convAnswer,
        badge: "none",
        note: "あいさつや雑談への返事です。事実の質問ではないため、ネット検索や裏づけ確認はせず、その場で返事をしています。",
      });
      return;
    }

    // 郵便番号＋場所の文脈の質問（「963-8001はどこ？」）は、計算
    // の形（963-8001＝引き算）に化けるため、計算レーン（0.75 と
    // 手順2）の両方で対象外にし、2.25 の住所レーンに委ねる。
    // 場所の文脈が無い純粋な計算は従来どおり計算で答える
    // （2026-11-20 実機バグ報告への対応）。
    const zipPlaceQ = isZipcodePlaceQuestion(question);

    // 0.7) Wolfram 計算レーン：計算の主軸。式・方程式・単位・
    // 割合・累乗・文章題の形に一致したら、サーバー経由で
    // Wolfram|Alpha に計算させる。未設定・失敗・解釈不能の
    // ときは false が返り、従来の計算レーン（0.75）に倒れる。
    // 結果は規約上キャッシュしないためメモ帳には残さない。
    if (!zipPlaceQ) {
      const wolframParsed = parseWolframQuery(question);
      if (wolframParsed) {
        const handled = await this.runWolframLane(wolframParsed);
        if (handled) return;
      }
    }

    // 0.75) 計算レーン：文章題を含む計算のパターンに一致したら、
    // モデル・メモ帳を素通りして、正確な計算結果（過程つき）を
    // そのまま答える。結果は従来の計算経路と同じ形でメモ帳に残す。
    const lane = zipPlaceQ ? null : CalcLane.solve(question);
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

    // 1.5) 記憶の直接参照：ユーザー自身についての質問（私の〜は？／
    // 覚えている？）は、モデルを介さず覚えている内容から即答する。
    // 全モデル共通の経路で、モデルの読み込み状態にも依存しない。
    // 記憶は世界の事実の証拠ではないので、索引レーンの検証とは
    // 混ぜず、「記憶の答え」バッジで出所を明示する。
    const memAnswer = this.memoryDirectAnswer(question);
    if (memAnswer) {
      this.pushAssistant({
        text: memAnswer,
        badge: "memory",
        note: "Cubeが覚えている内容から答えています。",
      });
      return;
    }

    // 2) 計算は実際に計算して即答する。表示する答えはローカルの
    // 決定的な計算（evalArithmetic）の結果だけで作り、モデルの生成文
    // は混ぜない。モデルによる照合（検算）は表示の前提にせず、表示の
    // 後で裏に回す（メモ帳レーンの再確認と同じ型）：Cube の答えの数字
    // が計算結果と食い違う時だけ、従来と同じ modelSaid＋注記を後追い
    // の返答として出す。裏の生成が失敗しても、表示済みの答え（生成
    // には依存していない）には影響させない。
    const calc = zipPlaceQ ? null : evalArithmetic(question);
    if (calc !== null) {
      this.setPhase("idle");
      this.pushAssistant({
        text: `答えは ${calc} です。`,
        badge: "calc",
      });
      // メモ帳への吸収（種類=計算、事実=計算結果）は表示の後——保存が
      // 表示を遅らせることはない。
      this.saveNotebook({
        question,
        answer: `答えは ${calc} です。`,
        source_title: "",
        source_url: "",
        question_type: "calc",
        fact: String(calc),
      });
      if (modelReady) {
        void (async () => {
          try {
            // 検算経路のモデル生成にも、選択中モデルを使う（生成先は
            // generateForModel がモデルごとに振り分ける）。
            const modelAns = await generateForModel(this.modelId, question);
            const modelNums = numbersIn(modelAns);
            if (
              modelAns &&
              modelNums.length > 0 &&
              !modelNums.some((n) => Number(n) === calc)
            ) {
              this.pushAssistant({
                text: `答えは ${calc} です。`,
                badge: "calc",
                modelSaid: modelAns,
                note: "Cubeの答えと食い違ったため、検算の答えを表示しています。",
              });
            }
          } catch {
            /* 裏の検算の失敗は内部だけに留める（表示済みの答えは不変） */
          }
        })();
      }
      return;
    }

    // 2.25) 場所・話題・買い物レーン：cube-server の実データ API
    // （/places・/topics・/shop）で答える。即答レーン（天気・Cube
    // 自身・会話・計算・メモ帳・記憶）の後、モデル生成の前に置く。
    // どのレーンもモデル生成・検証ゲートを通さず、API の値を
    // そのまま構造化して出す。サーバーに届かない・失敗した場合
    // は何も出さず、従来の手順（2.5 以降）にそのまま倒れる。
    // 結果は変わる情報なので知識メモ帳には保存しない。

    // YouTube要約：動画URLが明示された依頼なので、URLを含む質問は
    // 必ずこのレーンで完結させる（字幕なし・失敗でも通常経路に
    // 倒さない）。PC操作レーンより前に置くのは、URL の貼り付けを
    // 「ページを開く」操作と取り違えないため。
    const ytVideoId = extractYoutubeVideoId(question);
    if (ytVideoId) {
      await this.runYoutubeLane(question, ytVideoId);
      return;
    }

    // 住所（郵便番号）：番号の形式が決定的で誤爆がほぼ無いため、
    // YouTube レーンと同じ段で確定させる。当たれば必ずこの
    // レーンで完結させる（見つからない場合も推測はしない）。
    const zipDigits = extractZipcode(question);
    if (zipDigits) {
      await this.runAddressZipLane(zipDigits);
      return;
    }

    // ログイン：チャットからは実行しない（パスワードをチャットに
    // 書かせない）。PC操作パネルの専用フォーム（金庫方式）へ
    // 誘導して、このレーンで完結させる。
    if (detectLoginRequest(question)) {
      this.pushAssistant({
        text: "ログインはチャットからは実行しません。パスワードを" +
          "チャットに書くのもやめてください。「おためし」タブの" +
          "PC操作パネルにあるログインフォームに、サイトのURL・ID・" +
          "パスワードを入力して実行してください。パスワードはAIに" +
          "見せず、実行側が直接入力する金庫方式です。",
        badge: "pc",
        note: "認証情報はモデルの文脈にもログにも残りません。",
      });
      return;
    }

    // Google連携（Gmail・カレンダー）：本人のアカウントでの
    // 実操作なので、PC操作レーンより前に確定させる（「メールを
    // 送って」が仮想PCのフォーム送信に化けないため。ログイン
    // レーンと同じ理由）。未ログイン・未連携のときも、この
    // レーンの中で正直に案内して完結させる。
    if (this._gmailDraft) {
      const handledDraft = await this.handleGmailDraftReply(question);
      if (handledDraft) return;
    }
    const gmailSendParsed = parseGmailSendQuery(question);
    if (gmailSendParsed) {
      await this.runGmailSendLane(gmailSendParsed);
      return;
    }
    const gmailReadParsed = parseGmailReadQuery(question);
    if (gmailReadParsed) {
      await this.runGmailReadLane(gmailReadParsed);
      return;
    }
    const gcalParsed = parseGcalQuery(question);
    if (gcalParsed) {
      await this.runGcalLane(gcalParsed);
      return;
    }

    // PC操作：明確な操作の依頼表現のときだけ（detectPcQuery が
    // 事実・方法の質問を弾く）。実行は仮想PCの専門モデルが行い、
    // 結果（検証の成否を含む）は橋渡しが決めた文面をそのまま出す。
    if (detectPcQuery(question)) {
      await this.runPcLane(question);
      return;
    }

    // 地震・警報：直近の実データで答える。安全に関わる数値情報
    // なので、発火したらこのレーンで完結させ、取れない場合も
    // 正直な文面を出す。サーバーに届かない場合だけ従来の手順に
    // 倒れる。
    const quakeParsed = parseQuakeQuery(question);
    if (quakeParsed) {
      const handled = await this.runQuakeLane(quakeParsed);
      if (handled) return;
    }

    // 話題：定型の聞き方に一致したときだけ。
    if (detectTopicsQuery(question)) {
      this.setPhase("searching");
      const topicsData = await fetchTopics(10);
      this.setPhase("idle");
      if (topicsData) {
        if (topicsData.topics.length) {
          this.pushAssistant({
            text: topicsData.stale
              ? "今の話題です（少し前の時点の情報です）。各メディアの見出しを紹介します。"
              : "今の話題を紹介します。各メディアの見出しです。",
            badge: "topics",
            cards: topicsData.topics.slice(0, 5).map(topicCardOf),
            note: "話題の紹介です。事実の確認をした答えではないため、詳しくはリンク先の記事をご覧ください。出典：Googleトレンド・Yahoo!ニュース・NHKニュース・はてなブックマーク。",
          });
        } else {
          this.pushAssistant({
            text: "今の話題を取得できませんでした。",
            badge: "topics",
            note: "時間をおいてもう一度試してください。",
          });
        }
        return;
      }
      // 取得失敗時は何も出さず従来経路へ（このまま手順を進める）。
    }

    // 住所（キーワード）：住所・郵便番号・座標そのものを聞く
    // 質問は、場所レーンの直前で確定させる（近くの店探しは
    // 場所レーンの担当）。サーバーに届かなかった場合だけ
    // 従来の手順に倒れる。
    const addrParsed = parseAddressQuery(question);
    if (addrParsed) {
      const handled = await this.runAddressQueryLane(addrParsed);
      if (handled) return;
    }

    // 最寄り駅：対象の座標（住所レーンの geocode）から一番近い
    // 駅を引く。場所レーンの直前に置き、「〜の最寄り駅」を
    // 場所探しと取り違えない。サーバーに届かない場合だけ倒れる。
    const stationParsed = parseStationQuery(question);
    if (stationParsed) {
      const handled = await this.runStationLane(stationParsed);
      if (handled) return;
    }

    // 場所：ジャンル語＋場所の手がかりがあるときだけ。
    if (detectPlacesQuery(question)) {
      this.setPhase("searching");
      const placesData = await fetchPlaces(question);
      this.setPhase("idle");
      if (placesData && placesData.ok) {
        if (placesData.places.length) {
          const where =
            placesData.centerLabel || placesData.placeLabel || "";
          const kw = placesData.keyword || "場所";
          this.pushAssistant({
            text: where
              ? `${where}周辺で「${kw}」を${placesData.places.length}件見つけました。`
              : `「${kw}」を${placesData.places.length}件見つけました。`,
            badge: "places",
            cards: placesData.places.map(placeCardOf),
            note: `出典：${placesData.attribution.join(" / ") || "Web Services by Yahoo! JAPAN / © OpenStreetMap contributors"}。営業時間は載っていない場所があります。`,
          });
        } else {
          this.pushAssistant({
            text: "条件に合う場所が見つかりませんでした。",
            badge: "places",
            note: "地名や種類を変えて（例：郡山駅近くのコンビニ）もう一度聞いてください。",
          });
        }
        return;
      }
      // 失敗（届かない・両源とも失敗）時は何も出さず従来経路へ。
    }

    // 買い物：購入相談のシグナルが明確なときだけ。
    const shopQuery = parseShopQuery(question, this._lastShop || null);
    if (shopQuery) {
      if (shopQuery.askBack) {
        this.pushAssistant({
          text: "何をお探しですか？商品名や種類を教えてください（例：ワイヤレスイヤホン、5,000円以下の時計）。",
          badge: "none",
        });
        return;
      }
      this.setPhase("searching");
      const shopData = await fetchShop(
        shopQuery.keyword, shopQuery.conditions);
      this.setPhase("idle");
      if (shopData) {
        // 続きの条件変更（「もっと安いの」等）のために覚えておく。
        this._lastShop = {
          keyword: shopQuery.keyword,
          conditions: { ...shopQuery.conditions },
        };
        if (shopData.candidates.length) {
          const noteParts = [];
          const fetchedLabel = formatTimestampLabel(shopData.fetchedAt);
          noteParts.push(fetchedLabel
            ? `価格と在庫は${fetchedLabel}時点のYahoo!ショッピングの情報です。購入はリンク先のモールでどうぞ。`
            : "価格と在庫はYahoo!ショッピングの情報です。購入はリンク先のモールでどうぞ。");
          if (shopData.guardExcluded > 0) {
            noteParts.push(`1〜99円の商品${shopData.guardExcluded}件は候補から外しました。`);
          }
          noteParts.push(
            `出典：${shopData.attribution.join(" / ") || "Web Services by Yahoo! JAPAN"}`);
          this.pushAssistant({
            text: shopIntroText(shopQuery.keyword, shopQuery.conditions),
            badge: "shop",
            cards: shopData.candidates,
            note: noteParts.join(""),
          });
        } else {
          const hasConditions =
            typeof shopQuery.conditions.price_min === "number" ||
            typeof shopQuery.conditions.price_max === "number" ||
            !!shopQuery.conditions.free_shipping;
          this.pushAssistant({
            text: "条件に合う商品が見つかりませんでした。",
            badge: "shop",
            cards: hasConditions
              ? [{
                  kind: "suggest",
                  label: `条件を外して「${shopQuery.keyword}」で探す`,
                  query: `${shopQuery.keyword}が欲しい`,
                }]
              : [],
            note: "条件を緩める（価格帯を外す・キーワードを短くする）と見つかることがあります。無い商品名は作らず、正直にお伝えしています。",
          });
        }
        // 表示まで至った検索だけポイント付与（失敗・聞き返しは対象外）。
        void this.earnShopPoints(shopQuery.keyword, shopQuery.conditions);
        return;
      }
      // 失敗時は何も出さず従来経路へ（従来の棄権表示に落ちる）。
    }

    // 祝日：holidays-jp の実データで答える（今日・明日・次の
    // 祝日・日付指定・一覧）。サーバーに届かない場合だけ倒れる。
    const holidayParsed = parseHolidayQuery(question);
    if (holidayParsed) {
      const handled = await this.runHolidayLane(holidayParsed);
      if (handled) return;
    }

    // 為替・株価：数値が要る質問なので、発火したら必ずこの
    // レーンで完結させる（取れなかった場合も正直な文面を出し、
    // モデルの推測価格には倒さない）。
    const marketParsed = parseMarketQuery(question);
    if (marketParsed) {
      await this.runMarketLane(marketParsed);
      return;
    }

    // 音楽検索：iTunes の実データで答える。サーバーに届かない
    // 場合だけ従来の手順に倒れる。
    const musicParsed = parseMusicQuery(question);
    if (musicParsed) {
      const handled = await this.runMusicLane(musicParsed);
      if (handled) return;
    }

    // 論文検索：arXiv / Semantic Scholar の実データで答える。
    // 学術意図が明示的な質問だけが発火する（発動条件の正本は
    // サーバー側の matchPapersLane）。サーバーに届かない場合
    // だけ従来の手順に倒れる。
    const papersParsed = parsePapersQuery(question);
    if (papersParsed) {
      const handled = await this.runPapersLane(papersParsed);
      if (handled) return;
    }

    // 翻訳：DeepL の実データで答える。翻訳の意図が明示的な
    // 質問だけが発火する（発動条件の正本はサーバー側の
    // matchTranslateLane）。サーバーに届かない場合と、
    // サーバー側で翻訳でないと判定された場合だけ従来の
    // 手順に倒れる。
    const translateParsed = parseTranslateQuery(question);
    if (translateParsed) {
      const handled = await this.runTranslateLane(translateParsed);
      if (handled) return;
    }

    // 2.5) 時間制メーターの関門：サーバー側モデル（Twin / Pro /
    // Max / Ultra）が選択され、接続済みで、残量が0なら、生成も
    // 検索照合も始めずに穏やかに止める。端末内モデルへの案内と、
    // ポイントでの追加（ログイン中のみ可能）を出す。天気・計算・
    // メモ帳・記憶の即答レーンはメーターと無関係に使えるため、
    // 関門はそれらの後ろ（＝モデル生成の直前）に置く。
    if (modelReady && meterAppliesTo(this.modelId)) {
      const meter = getCubeMeter();
      if (meter && !meter.canUse()) {
        this.pushAssistant({
          text: "サーバーモデルの時間メーターが0になりました。",
          badge: "unknown",
          note: "Cube Nano / Cube Flash はこの端末の中で動くため、メーターなしでそのまま使えます。モデルを切り替えてお試しください。ログイン中なら、ポイントでメーター時間を追加することもできます（サイドバーの「ポイントでメーター追加」）。メーターの無料分は日付が変わると戻ります。",
        });
        return;
      }
    }

    // 2.6) 雑談専門（Cube 会話）の分岐：事実の主張を含まない会話は
    // 専門モデルが会話として返す。事実の質問（classifyQuestion が
    // 種類を返すもの）は専門モデルに渡さず、このまま手順3の証拠＋
    // 検証ゲート経路に倒す（専門モデルは事実に答えない設計で、
    // 裏づけの無い生成文を出す経路も作らない）。天気・計算・住所・
    // 買い物などの専用レーンはこの分岐より前で処理済み。
    // 会話の返事は知識メモ帳にも保存しない（会話レーンと同じ）。
    if (this.modelId === "chat" && modelReady && !classifyQuestion(question)) {
      this.setPhase("thinking");
      try {
        const chatText = await generateForModel("chat", question, {
          historyMessages: this.historyMessagesForChat(),
        });
        this.setPhase("idle");
        if (chatText) {
          this.pushAssistant({
            text: chatText,
            badge: "none",
            note: "Cube 会話（雑談専門モデル）の返事です。会話のためのモデルなので、事実の質問には答えません。事実を知りたいときは、他のモデルに切り替えるか、そのまま質問してください（裏づけ確認の経路で調べます）。",
          });
        } else {
          this.pushAssistant({
            text: "うまく返事を作れませんでした。もう一度送ってみてください。",
            badge: "unknown",
            note: "Cube 会話（雑談専門モデル）が空の返事を返しました。",
          });
        }
      } catch (e) {
        this.setPhase("idle");
        if (e && e.code === "METER_EMPTY") {
          this.pushAssistant({
            text: "サーバーモデルの時間メーターが0になりました。",
            badge: "unknown",
            note: "Cube Nano / Cube Flash はこの端末の中で動くため、メーターなしでそのまま使えます。モデルを切り替えてお試しください。ログイン中なら、ポイントでメーター時間を追加することもできます（サイドバーの「ポイントでメーター追加」）。メーターの無料分は日付が変わると戻ります。",
          });
        } else {
          const message = e instanceof Error ? e.message : String(e);
          this.pushAssistant({
            text: "雑談専門モデルでエラーが起きました。",
            badge: "unknown",
            note: `Cube 会話（iMacサーバー）への接続で問題が起きました: ${message}`,
          });
        }
      }
      return;
    }

    // 3) モデル生成＋ウェブ検索での照合。確認のリクエストは生成の前に
    // 開始する：生成はメインスレッドの重い処理だが、ネットワークは
    // その裏で進むため、Wikipediaの往復がモデルの思考の後ろに
    // 並ばずに済む。
    // 証拠の到着は生成と並行に追跡する（後段の待ち合わせで、到着
    // 済みの分だけでも使えるようにするため）。レーン予算は暴走防止
    // の大きな上限だけにし、実際の待ちは生成後に EVIDENCE_WAIT_MS
    // で数える（定数のコメント参照）。
    IndexLane.beginQuestion({
      maxMs: EVIDENCE_LANE_BUDGET_MS,
      maxRequests: EVIDENCE_LANE_MAX_REQUESTS,
    });
    // 証拠はまずサーバー側レーン（/evidence）に集めさせ、失敗した
    // ときだけ従来のページ側レーンに倒れる（startEvidencePromises）。
    const { articlePromise, searchPromise } = startEvidencePromises(question);
    let articleSettled = false;
    let searchSettled = false;
    let articleValue = null;
    let searchValue = [];
    articlePromise.then((a) => {
      articleSettled = true;
      articleValue = a;
    });
    searchPromise.then((s) => {
      searchSettled = true;
      searchValue = s;
    });
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
        // 通常の質問応答の生成：生成先は generateForModel がモデル
        // ごとに振り分ける（端末内／iMacサーバー／Ultra）。生成上限は
        // 各エンジン側が持つ（Nano・Pro・Max: 48 / Flash・Twin: 96 /
        // Ultra: 128）。
        // 記憶の注入：Ultra はシステムプロンプト側（入口で差し替え
        // 済み）。自作モデルは QA 形式に文脈枠が無いため、関連が強い
        // 記憶だけを質問の末尾に短く添える（無関係なら無変更）。
        // 続きの文脈：履歴から再開した会話のときだけ、直近のやり
        // 取りを引き継ぐ。Ultra は履歴をメッセージ列として渡し、
        // 自作モデルは抜粋を質問の末尾に添える。再開していない
        // 会話ではどちらも付けない（従来とバイト単位で同一）。
        // 検算（手順2）・読解・ブリーフィングの生成には付けない。
        let historyMessages = null;
        let genQuestion;
        if (this.modelId === "ultra") {
          genQuestion = question;
          historyMessages = this.historyMessagesForUltra();
        } else {
          genQuestion = this.historyDecorate(this.memoryDecorate(question));
        }
        modelAnswer = await generateForModel(this.modelId, genQuestion, {
          historyMessages,
        });
        if (!modelAnswer) modelIssue = "empty";
      } catch (e) {
        modelAnswer = "";
        modelIssue = "error";
        modelError = e instanceof Error ? e.message : String(e);
      }
    }

    this.setPhase("searching");
    let results = [];
    let article = null;
    let fact = null;
    // 証拠の到着を、生成が終わったこの時点から EVIDENCE_WAIT_MS まで
    // 待つ（質問開始からの締切ではない：モデルの速さに依存しない）。
    // 両方そろえば即座に進む。締切までに片方しか届かなかったら、
    // 届いた分だけを使う（以前は締切で証拠を丸ごと捨てていた）。
    // どちらが欠けたかは evidenceTimedOut に残し、「記事が見つから
    // なかった」という断定は、証拠集めが完走した場合にだけ使う。
    let evidenceTimedOut = false;
    {
      const waitDeadline = Date.now() + EVIDENCE_WAIT_MS;
      while (!(articleSettled && searchSettled)) {
        const remain = waitDeadline - Date.now();
        if (remain <= 0) break;
        await new Promise((r) => setTimeout(r, Math.min(200, remain)));
      }
      evidenceTimedOut = !(articleSettled && searchSettled);
      if (articleSettled) article = articleValue;
      if (searchSettled) results = searchValue;
      if (article) {
        try {
          fact = extractFact(question, article);
        } catch {
          fact = null;
        }
      }
    }
    // 検索への接続そのものが失敗していたら、その理由を覚えておく
    // （下の「確認できませんでした」で、理由なしにごまかさないため）。
    const searchError = WikiNet.lastError;
    // 『…』の質問では、その作品自身の証拠だけを数える。
    results = scopeToSubject(question, results);

    // 事実の点検（常時）：種類つき事実も検証を通してから使う。
    // 出典記事が質問の対象と無関係な事実、『…』の文脈を満たさない
    // 同名別物の事実、値を「年」と取り違えた数値の事実は、表示も
    // 保存もしない。以前はアンカー確認がモデルと食い違うときに
    // 限られていたため、モデルが無言・不在のときに無関係な記事の
    // 事実がそのまま出る余地があった。
    fact = sanitizeFact(fact, article, question);

    // 文単位の検証ゲート：モデルの答えを文ごとに裏づける。以後の
    // 表示分岐では、この結果を最上位の判断材料にする（モデル優先
    // 表示は維持するが、裏づけの無い主張文は出さない）。
    const gate = modelAnswer
      ? verifyModelAnswer(modelAnswer, question, { fact, article, results })
      : null;
    // 嘘チェック可視化：ゲートの判定をそのまま内訳化する（表示専用で、
    // 判定には一切関与しない）。モジュールが無い環境では何もしない。
    if (gate && typeof CubeFactCheck !== "undefined") {
      try {
        this._fcBreakdown = CubeFactCheck.buildBreakdown({
          question,
          gate,
          fact,
          results,
        });
      } catch {
        this._fcBreakdown = null;
      }
    }

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
      // 食い違い×争いのある事実×モデルの全検証がそろった場合は、
      // モデルの答えを正とする（同名別エンティティの事実でモデルの
      // 正答を上書きしない。例：H2O＝水 vs H2O (Webサーバ)）。
      // 争いのある事実と食い違い、モデルの答えも全検証できない場合は
      // どちらも出さない——誤った事実での上書きも、裏づけの無い
      // 答えの表示も、どちらもしない。
      const contested = factIsContested(fact, results);
      // 一致の判定は、場所の事実が複数の都道府県を持つ場合の部分一致
      // （一部だけ言って「一致」になる）を除いたものを使う。
      const agrees =
        modelAgreesWithFact(modelAnswer, fact) &&
        !placeAnswerMissesFactValue(modelAnswer, fact, question);
      if (modelAnswer && !agrees && contested) {
        this.setPhase("idle");
        if (gate && gate.status === "all") {
          this.pushAssistant({
            text: modelAnswer,
            badge: "model",
            sourceTitle: gate.source?.title,
            sourceUrl: gate.source?.url,
            note: "Cubeの答えを主表示しています。ネットの事実とは食い違いましたが、Cubeの答えは別の記事で裏づけが取れました。知識メモ帳に保存しました。",
          });
          this.saveNotebook({
            question,
            answer: modelAnswer,
            source_title: gate.source?.title ?? "",
            source_url: gate.source?.url ?? "",
            question_type: classifyQuestion(question) ?? "",
            fact: modelAnswer,
          });
        } else {
          this.pushAssistant({
            text: "確認できませんでした。",
            badge: "unknown",
            note: "Cubeの答えとネットの事実が食い違い、どちらが質問の対象か裏づけが取れなかったため、どちらも表示しませんでした。",
          });
        }
        return;
      }
      // 「ネット検索の答え」として提示する文。原則は記事から組み立てた
      // 事実文だが、Ultra が選択され読み込み済みのときだけ、対象記事を
      // Ultra に読ませて Ultra の言葉で書かせる（統合の設計）。
      // 確認済み経路（モデルの答えが事実と一致）では使わない。書けな
      // ければ事実文のまま。生成の間はフェーズを「検索中」のままに
      // して、二重の送信を防ぐ。
      let netText = fact.answer;
      if (
        this.modelId === "ultra" &&
        modelReady &&
        !(modelAnswer && modelAgreesWithFact(modelAnswer, fact))
      ) {
        const written = await ultraWriteFactAnswer(question, article, fact);
        // 書き直し文も検証ゲートを通す：裏づけの取れた書き直しだけを
        // 採用し、通らなければ事実文のまま出す（検索レーンの答えも
        // 同じ検証を通す原則。単なる事実との一致判定では、圧縮した
        // 忠実な書き直しまで落としてしまうため、ゲートで判定する）。
        if (
          written &&
          verifyModelAnswer(written, question, { fact, article, results })
            .status === "all"
        ) {
          netText = written;
        }
      }
      this.setPhase("idle");
      // モデル優先：モデルが答えを出したなら、それが主表示になる。
      // ウェブの事実は裏づけの確認——表記のゆれ（空白・カンマ・単位・
      // 小数の丸め）を許して一致すれば「ネットで確認済み」のラベルを
      // 付け、食い違えば従来の透明性ルール（ネットの事実を正として
      // 表示し、モデルの元の答えも消さずに併記する）。
      if (modelAnswer && agrees) {
        // 一致していても、文単位ゲートを通す：事実と一致する文に
        // 裏づけの無い主張文がくっついている場合は、その文を出さない。
        // 全文が通れば全文を、一部だけなら取れた文だけを表示する。
        // どの文も通らなければ、下の事実表示に落とす。
        if (gate && gate.status === "all") {
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
        if (gate && gate.status === "partial") {
          this.pushAssistant({
            text: gate.text,
            badge: "model",
            sourceTitle: fact.sourceTitle,
            sourceUrl: fact.sourceUrl,
            evidence: fact.evidence,
            note: "Cubeの答えのうち、ネットで確認が取れた部分だけを表示しています。事実ベースの答えを知識メモ帳に保存しました。",
          });
          saveFact();
          return;
        }
      }
      if (modelAnswer) {
        this.pushAssistant({
          text: netText,
          badge: "net",
          sourceTitle: fact.sourceTitle,
          sourceUrl: fact.sourceUrl,
          evidence: fact.evidence,
          modelSaid: modelAnswer,
          note: agrees
            ? "Cubeの答えはネットの事実と一致しましたが、文ごとの裏づけがそろわなかったため、ネットの事実を表示しています。事実ベースの答えを知識メモ帳に保存しました。"
            : "Cubeの答えとネットの事実が食い違ったため、ネットの事実を正として表示しています。事実ベースの答えを知識メモ帳に保存しました。",
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
        text: netText,
        badge: "net",
        sourceTitle: fact.sourceTitle,
        sourceUrl: fact.sourceUrl,
        evidence: fact.evidence,
        note: `${noModelNote}事実ベースの答えを知識メモ帳に保存しました。`,
      });
      saveFact();
      return;
    }

    // 5.5) 一般Web検索フォールバック：種類つき事実が取れず、モデルの
    // 答えも表示できない（裏づけが1つも無い／モデル自身が棄権した）
    // 場合に限り、Ultra が選択され読み込み済みのときだけ、サーバー
    // 側の一般Web検索（/gsearch）の本文を Ultra に読ませて答える。
    // 検索結果を読むのは Ultra のみ。書けた答えも検証ゲートを通った
    // ものだけを出し、通らなければ従来の棄権表示に落とす。生成の間は
    // フェーズを「検索中」のままにして、二重の送信を防ぐ。
    if (
      this.modelId === "ultra" &&
      modelReady &&
      (!gate || gate.status === "none" || gate.status === "abstain")
    ) {
      const web = await webSearchAnswer(question).catch(() => null);
      if (web) {
        // 嘘チェック可視化：表示する答え（Web経由）の検証内訳に差し替える。
        if (typeof CubeFactCheck !== "undefined") {
          try {
            this._fcBreakdown = CubeFactCheck.buildBreakdown({
              question,
              gate: web.gate,
              fact: null,
              results: web.pages,
            });
          } catch {
            this._fcBreakdown = null;
          }
        }
        this.setPhase("idle");
        this.pushAssistant({
          text: web.text,
          badge: "net",
          sourceTitle: web.sourceTitle,
          sourceUrl: web.sourceUrl,
          evidence: web.evidence,
          modelSaid:
            gate && gate.status === "none" && modelAnswer
              ? modelAnswer
              : undefined,
          note: "Wikipediaの索引では裏づけが取れなかったため、一般のWeb検索で見つけた記事をUltraが読んで答えています。答えは文ごとに記事と照合し、裏づけが取れたものだけを表示しています。",
        });
        return;
      }
    }

    // 5) 種類つき事実が取れなかった場合の表示は、文単位ゲートの結果で
    // 決める。出典は、実際に裏づけが成立した記事そのものだけにする
    // （裏づけてもいない記事を出典に表示することはしない）。
    this.setPhase("idle");
    if (modelAnswer && gate) {
      if (gate.status === "all") {
        // 全主張文が裏づけられた場合だけ、モデルの答えを全文表示し、
        // 知識メモ帳にも保存する（確認済みの保存はこの経路だけ）。
        this.pushAssistant({
          text: modelAnswer,
          badge: "model",
          sourceTitle: gate.source?.title,
          sourceUrl: gate.source?.url,
          note: "Cubeの答えを、ネット検索の結果と文単位で照合しました。知識メモ帳に保存しました。",
        });
        this.saveNotebook({
          question,
          answer: modelAnswer,
          source_title: gate.source?.title ?? "",
          source_url: gate.source?.url ?? "",
          question_type: classifyQuestion(question) ?? "",
          fact: modelAnswer,
        });
        return;
      }
      if (gate.status === "partial") {
        // 裏づけの取れた文だけを表示する。部分的な答えは保存しない。
        this.pushAssistant({
          text: gate.text,
          badge: "model",
          sourceTitle: gate.source?.title,
          sourceUrl: gate.source?.url,
          note: "Cubeの答えのうち、ネットの記事で確認が取れた部分だけを表示しています。確認が取れなかった部分は表示しませんでした。",
        });
        return;
      }
      if (gate.status === "abstain" || gate.status === "nonclaim") {
        // 棄権の表明と、事実の主張を含まない会話的な答えは、嘘に
        // なりようがないため従来どおり表示する（保存はしない）。
        this.pushAssistant({
          text: modelAnswer,
          badge: "unverified",
          note:
            gate.status === "abstain"
              ? "Cubeは、この質問には確認できないと答えています。"
              : "事実の主張を含まない答えのため、そのまま表示しています。",
        });
        return;
      }
      // gate.status === "none"：裏づけの取れた主張文が1つも無い。
      // モデルの答えは表示しない。対象そのものが見つからない場合は
      // その旨を言う（実在しないものにあるように答えないため）。
      // ただし証拠集めが時間切れで完走していない場合は「見つから
      // なかった」とは断定しない（確かめ切れなかっただけであり、
      // 実在する記事でも同じ見た目になるため）。
      if (
        entityNotFound(
          question,
          article,
          results,
          searchError || (evidenceTimedOut ? "evidence-timeout" : ""),
        )
      ) {
        const entity = subjectOf(question) ?? topicFromQuestion(question);
        this.pushAssistant({
          text: "確認できませんでした。",
          badge: "unknown",
          note: `「${entity}」についての記事が見つかりませんでした。存在が確認できないものについて、あるように答えることはしません。`,
        });
        return;
      }
      this.pushAssistant({
        text: "確認できませんでした。",
        badge: "unknown",
        note: evidenceTimedOut
          ? "記事の取得が時間内に終わらなかったため、裏づけの確認ができませんでした。Cubeの答えも表示しませんでした。"
          : searchError
            ? `検索に接続できなかったため、記事の事実では確認できませんでした（理由：${searchError}）。Cubeの答えも裏づけが取れなかったため、表示しませんでした。設定の「検索テスト」で接続を確かめられます。`
            : "Cubeの答えは、記事の事実で裏づけが取れなかったため表示しませんでした。",
      });
      return;
    }
    {
      // モデルが何も出さなかった場合：その理由も注記に言う。検索に
      // 接続できなかった場合は、その理由を必ず添える。
      this.pushAssistant({
        text: "確認できませんでした。",
        badge: "unknown",
        note: evidenceTimedOut
          ? "記事の取得が時間内に終わらなかったため、確認できませんでした。"
          : searchError
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
    IndexLane.beginQuestion();
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
    assembleUltraPassage, isUltraAbstention, ultraWriteFactAnswer, ULTRA_PASSAGE_CAP,
    fetchWebSearchPages, webSearchAnswer, GSEARCH_TIMEOUT_MS,
    fetchServerEvidence, startEvidencePromises, EVIDENCE_SERVER_TIMEOUT_MS,
    EVIDENCE_WAIT_MS,
    normalizeUltraWrittenStyle,
    UltraGgufModel, loadUltraGguf,
    UltraImacModel, loadUltraImac, ULTRA_IMAC_BASE_URL, ULTRA_IMAC_UNREACHABLE,
    buildBriefingMaterial, trimBriefingTail, localDateKey, BRIEFING_DATE_KEY,
    pointsJstDateKey, pointsDailyLoginKey, pointsNewTopupKey,
    POINTS_TOPUP_MENU, POINTS_BONUS_KEY,
    getCubeMeter, meterAppliesTo, meterEmptyError,
    ServerModel, ChatModel, CHAT_PARAM_COUNT, generateForModel, CUBE_SERVER_BASE_URL, CUBE_SERVER_UNREACHABLE,
    readUltraStage, writeUltraStage,
    ULTRA_REPO, ULTRA_PARAM_COUNT, ULTRA_MAX_NEW_TOKENS,
    ULTRA_GGUF_URL, WLLAMA_ESM_URL, WLLAMA_WASM_FROM_CDN_URL,
    evalArithmetic, CalcLane, extractFact, classifyQuestion,
    resolveArticleSource, searchWikipedia, topicFromQuestion, kindSearchPhrase,
    canonicalTensorName, expectedTensors, modelAgreesWithFact,
    isCorroborated, findCorroboratingResult,
    answerContentTerms, numbersMatchLoose,
    cubeSelfAnswer, isTautologicalAnswer, conversationAnswer,
    detectPlacesQuery, fetchPlaces, placeCardOf,
    detectTopicsQuery, fetchTopics, topicCardOf, topicSourceLabel,
    parseShopQuery, extractShopConditions, extractShopKeyword,
    fetchShop, shopCardOf, shopIntroText, shopConditionsHash,
    detectPcQuery, executePcTask, pcResultNote, detectLoginRequest,
    extractYoutubeVideoId, fetchYoutubeCaptions, youtubeSummary,
    youtubeSummaryCap, ytLangLabel, YT_TIMEOUT_MS,
    extractZipcode, parseAddressQuery, fetchAddress, zipcodeAnswerText,
    geocodeAnswerText, addressCardOf, addressMapsUrl,
    isZipcodePlaceQuestion,
    parseQuakeQuery, fetchQuake, quakeAnswerText, warningAnswerText,
    extractPrefecture, JMA_PREFECTURE_CODES,
    parseStationQuery, fetchStation, stationAnswerText, stationCardOf,
    parseHolidayQuery, fetchHoliday, holidayAnswerText,
    parseMarketQuery, fetchMarket, marketFxText, marketStockText,
    parseMusicQuery, fetchMusic, musicCardOf,
    parsePapersQuery, fetchPapers, paperCardOf,
    parseTranslateQuery, fetchTranslate, translateCardOf, translateLangName,
    parseWolframQuery, fetchWolfram, wolframCardOf, WOLFRAM_TIMEOUT_MS,
    fetchGoogleStatus, startGoogleLink, unlinkGoogle,
    parseGmailSendQuery, mergeGmailDraft, gmailSendAskBackText,
    parseGmailReadQuery, sendGmail, fetchGmailList, fetchGmailMessage,
    fetchGmailSendLog, gmailSendLogKey, gmailListText,
    parseGcalQuery, parseGcalDate, gcalTimeTokens, gcalNextDateKey,
    fetchGcalEvents, createGcalEvent, updateGcalEvent, deleteGcalEvent,
    sanitizeGcalEvents, gcalWhenText, gcalEventLine, gcalListText,
    gcalPlusOneHour, GOOGLE_SENDLOG_SEEN_KEY,
    formatTimestampLabel, formatDistanceLabel,
    factSourceRelatedToEntity,
    verifyModelAnswer, splitForGate, verifySentence,
    factIsContested, entityNotFound, scopeToSubject, subjectOf,
    sanitizeFact, questionContextTerms, contextSatisfiedByText,
    isClaimSentence, isDegenerateAnswer, placeAnswerMissesFactValue,
    isUsableEvidenceResult, resultAnchoredToQuestion, resultSupportsAnswer,
    modelAnswerFitsQuestion, answerClaimKind,
    parseWeatherQuery, fetchWeather, fetchJmaForecast, resolveJmaArea,
    jmaWeatherText, normalizeQuestion, IndexLane,
  };
}
