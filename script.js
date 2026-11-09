"use strict";
/*
 * Cube（GitHub Pages版） script.js
 *
 * 全部ブラウザの中で動く版：
 * - 自作LLM（約3,177万パラメータ）の推論エンジン（int8量子化・KVキャッシュ付き。
 *   Python版とNodeで10問全一致を検証済みのエンジンと同一計算）
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
  const out = new Float32Array(t.q.length);
  for (let i = 0; i < t.q.length; i++) out[i] = t.q[i] * t.scale;
  return out;
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
    const q = this.emb.q, s = this.emb.scale;
    for (let v = 0; v < V; v++) {
      const base = v * C; let acc = 0;
      for (let j = 0; j < C; j++) acc += q[base + j] * h[j];
      logits[v] = s * acc;
    }
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

function buildEngine(manifest, bpeJson, fileBuffers) {
  const tensors = {};
  for (const t of manifest.tensors) {
    const buf =
      fileBuffers[t.file] ??
      fileBuffers[manifest.files.indexOf(t.file)];
    if (!buf) throw new Error(`テンソル ${t.name} のファイルがありません`);
    const bytes = new Int8Array(buf.buffer, buf.byteOffset + t.offset, t.length);
    tensors[canonicalTensorName(t.name)] = new QTensor(bytes, t.scale, t.shape);
  }
  // 全テンソル検証：生成に必要な重みが全部そろい、大きさも合う時だけ通す。
  const problems = [];
  for (const [name, len] of expectedTensors(manifest.config)) {
    const t = tensors[name];
    if (!t) problems.push(`${name} がありません`);
    else if (t.q.length !== len) {
      problems.push(`${name} の大きさが合いません（${t.q.length} ≠ ${len}）`);
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

/* モデル本体の薄いラッパ：状態表示つきで読み込み、質問文から生成する */
const LocalModel = {
  engine: null,
  paramCount: 0,
  get ready() { return this.engine !== null; },
  async load(baseUrl, onProgress) {
    const base = baseUrl.replace(/\/?$/, "/");
    // manifest.json は常に取り直す（小さい）。その本文の指紋が重みの
    // 世代を決めるので、上げ直された重みを古いキャッシュで開くことはない。
    const manifestRes = await fetch(base + "manifest.json");
    if (!manifestRes.ok) {
      throw new Error(`manifest.json を取得できませんでした (HTTP ${manifestRes.status})`);
    }
    const manifestText = await manifestRes.text();
    const manifest = JSON.parse(manifestText);

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

    // シャード：キャッシュにあるものはそのまま使い、無いものだけ
    // バイト単位で進捗を数えながら読み、次回のために保存する。
    const totals = manifest.files.map(() => 0);
    let loadedTotal = 0;
    const buffers = [];
    for (let i = 0; i < manifest.files.length; i++) {
      const name = manifest.files[i];
      if (cache) {
        const hit = await modelCacheGet(cache, cacheKey(name));
        if (hit instanceof ArrayBuffer) {
          const buf = new Uint8Array(hit);
          buffers.push(buf);
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
        buffers.push(buf);
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
      buffers.push(buf);
      loadedTotal += got;
      if (cache) {
        void modelCachePut(cache, cacheKey(name), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      }
    }
    this.engine = buildEngine(manifest, bpeJson, buffers);
    this.paramCount = manifest.total_params ||
      manifest.tensors.reduce((a, t) => a + t.length, 0);
    // 試し生成：「読み込み完了」はファイルが届いたことではなく、
    // 実際に生成できること。ここで落ちる失敗は読み込みエラーとして出す。
    try {
      this.generate("テスト", 1);
    } catch (e) {
      this.engine = null;
      throw new Error(
        `試し生成に失敗しました: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
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
  return { place: place.length >= 2 ? place : null, day };
}

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

/** 1記事の平文エキストラクトをタイトル完全指定で取る。失敗時は null。 */
async function wikiFetchExtract(title) {
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

/** Wikipedia検索のタイトル一覧（順位順）。失敗時は空。 */
async function wikiSearchTitles(q, limit) {
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

/** タイトル完全指定で1記事分の事実源データ。失敗時は null。 */
async function wikiArticleSource(title) {
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
async function wikiFetchExtractHeads(titles) {
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
/* 天気クライアント（Open-Meteo：無料・キー不要、ブラウザから直接）    */
/*                                                                    */
/* 聞かれた時点の実データを読む：質問の場所（無ければ設定の既定の      */
/* 場所）をジオコーディングし、その地点の予報を取る。何も保存しない   */
/* ——天気は変わるので、昨日の答えを今日の答えとして出してはいけない。 */
/* ================================================================== */

/**
 * Open-Meteoの無料ジオコーディングで場所を探す。日本語の入力は
 * 都道府県と市区町村がくっついている（「福島県郡山市」）ことが多く、
 * そのままでは一致しないため、少しずつ短くした形も試す
 * （そのまま／都道府県なし／市区町村の接尾なし／両方なし）。
 * 日本の結果があれば優先する。失敗時は null。
 */
async function geocodePlace(place) {
  const noPref = place.replace(/^.+?(?:都|道|府|県)/, "");
  const noCity = place.replace(/(?:市|区|町|村)$/, "");
  const bare = noPref.replace(/(?:市|区|町|村)$/, "");
  const variants = [...new Set([place, noPref, noCity, bare])].filter(
    (v) => v.length >= 2,
  );
  for (const v of variants) {
    try {
      const res = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(v)}&count=5&language=ja&format=json`,
      );
      if (!res.ok) continue;
      const data = await res.json();
      const list = (data.results ?? []).filter(
        (r) => typeof r.latitude === "number" && typeof r.longitude === "number",
      );
      if (list.length === 0) continue;
      const pick = list.find((r) => r.country_code === "JP") ?? list[0];
      if (!pick) continue;
      return {
        name: pick.name ?? v,
        admin1: pick.admin1 ?? "",
        latitude: pick.latitude,
        longitude: pick.longitude,
      };
    } catch {
      /* 次の形を試す */
    }
  }
  return null;
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
 * 天気を取る。Muse版の weather アクションと同じ形で返す：
 * { ok, error, place_label, current_temp, current_text, days, source_url }。
 * 失敗は ok:false ＋理由で返し、天気をでっち上げない。
 */
async function fetchWeather(place, defaultPlace, day) {
  const empty = {
    ok: false,
    error: "",
    place_label: "",
    current_temp: null,
    current_text: "",
    days: [],
    source_url: "https://open-meteo.com/",
  };
  const wanted = (place || "").trim() || (defaultPlace || "").trim();
  const geo = await geocodePlace(wanted);
  if (!geo) return { ...empty, error: "place_not_found" };
  const placeLabel = geo.admin1 ? `${geo.admin1} ${geo.name}` : geo.name;
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
    const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
    if (!res.ok) return { ...empty, error: "fetch_failed" };
    const data = await res.json();
    const daily = data.daily;
    const times = daily?.time ?? [];
    if (times.length === 0) return { ...empty, error: "fetch_failed" };
    const indices =
      day === "week"
        ? times.map((_, i) => i)
        : [day === "today" ? 0 : day === "tomorrow" ? 1 : 2].filter(
            (i) => i < times.length,
          );
    if (indices.length === 0) return { ...empty, error: "fetch_failed" };
    const days = indices.map((i) => ({
      date_label: formatForecastDate(times[i] ?? "", i),
      text: weatherCodeText(daily?.weather_code?.[i] ?? -1),
      temp_max: daily?.temperature_2m_max?.[i] ?? null,
      temp_min: daily?.temperature_2m_min?.[i] ?? null,
      precip_prob: daily?.precipitation_probability_max?.[i] ?? null,
    }));
    return {
      ok: true,
      error: "",
      place_label: placeLabel,
      current_temp:
        typeof data.current?.temperature_2m === "number"
          ? data.current.temperature_2m
          : null,
      current_text:
        typeof data.current?.weather_code === "number"
          ? weatherCodeText(data.current.weather_code)
          : "",
      days,
      source_url: "https://open-meteo.com/",
    };
  } catch {
    return { ...empty, error: "fetch_failed" };
  }
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
  weather: { label: "天気情報（Open-Meteo）", className: "badge-weather" },
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
  return `${(n / 1_000_000).toFixed(2)}M`;
}

const el = (id) => document.getElementById(id);

const App = {
  phase: "idle", // idle | thinking | searching | saving
  modelStatus: { kind: "not_loaded" },
  baseUrl: DEFAULT_BASE_URL,
  weatherPlace: DEFAULT_WEATHER_PLACE,
  confirmMarkAll: false,

  init() {
    Notebook.load();
    this.baseUrl = store.get(URL_KEY) || DEFAULT_BASE_URL;
    el("base-url").value = this.baseUrl;
    el("current-url").textContent = this.baseUrl;
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

  setModelStatus(st) {
    this.modelStatus = st;
    const box = el("model-status");
    const dot = box.querySelector(".dot");
    const text = el("model-status-text");
    box.className = "model-status mono " +
      (st.kind === "ready" ? "ready" : st.kind === "error" ? "error" : st.kind === "loading" ? "loading" : "");
    dot.className = "dot";
    if (st.kind === "not_loaded") text.textContent = "モデル未配置";
    if (st.kind === "loading") {
      text.textContent = `読み込み中… ${formatBytes(st.loadedBytes)}${st.totalBytes > 0 ? ` / ${formatBytes(st.totalBytes)}` : ""}（${st.label}）`;
    }
    if (st.kind === "ready") {
      text.textContent = `読み込み完了（パラメータ数 ${formatParams(st.paramCount)}）`;
    }
    if (st.kind === "error") text.textContent = `モデル未配置（${st.message}）`;
    el("model-note").hidden = st.kind === "ready" || st.kind === "loading";
    el("nb-tab-count").textContent = Notebook.items.length > 0 ? ` (${Notebook.items.length})` : "";
    const s = el("settings-model-status");
    if (s) {
      s.textContent = "状態: " + (
        st.kind === "ready"
          ? `読み込み完了（パラメータ数 ${formatParams(st.paramCount)}）`
          : st.kind === "loading"
            ? "読み込み中…"
            : st.kind === "error"
              ? `エラー: ${st.message}`
              : "モデル未配置"
      );
    }
  },

  async loadModel(url) {
    store.set(URL_KEY, url);
    this.setModelStatus({ kind: "loading", loadedBytes: 0, totalBytes: 0, label: "manifest.json" });
    try {
      await LocalModel.load(url, (loaded, total, label) => {
        this.setModelStatus({ kind: "loading", loadedBytes: loaded, totalBytes: total, label });
      });
      this.setModelStatus({ kind: "ready", paramCount: LocalModel.paramCount });
    } catch (e) {
      this.setModelStatus({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      });
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
     0) 天気 → Open-Meteoの実データ 0.5) Cube自身の質問 → 固定情報
     1) 知識メモ帳 → 即答＋裏で再確認
     2) 計算 → 検算 3) モデル生成 → 対象記事の事実で照合 */
  async ask(rawQuestion) {
    const question = (rawQuestion || "").trim();
    if (!question || this.phase !== "idle") return;
    this.pushUser(question);
    el("question-input").value = "";
    this.syncSend();
    const modelReady = LocalModel.ready;
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
        const lines = [`${w.place_label}の天気`];
        if (wq.day === "today" && w.current_temp !== null) {
          lines.push(
            `現在: ${w.current_text ? `${w.current_text}、` : ""}気温 ${w.current_temp}°C`,
          );
        }
        for (const d of w.days) {
          const parts = [d.text];
          if (d.temp_max !== null && d.temp_min !== null) {
            parts.push(`最高 ${d.temp_max}°C / 最低 ${d.temp_min}°C`);
          } else if (d.temp_max !== null) {
            parts.push(`最高 ${d.temp_max}°C`);
          }
          if (d.precip_prob !== null) {
            parts.push(`降水確率 ${d.precip_prob}%`);
          }
          lines.push(`${d.date_label}: ${parts.join("、")}`);
        }
        this.pushAssistant({
          text: lines.join("\n"),
          badge: "weather",
          sourceTitle: "Open-Meteo",
          sourceUrl: w.source_url,
          note: "天気はその時点の予報です（変わる情報なので知識メモ帳には保存しません）。",
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
          const modelAns = LocalModel.generate(question);
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
        modelAnswer = LocalModel.generate(question);
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
    BPE, GPT, QTensor, buildEngine, LocalModel,
    evalArithmetic, extractFact, classifyQuestion,
    resolveArticleSource, searchWikipedia, topicFromQuestion, kindSearchPhrase,
    canonicalTensorName, expectedTensors, modelAgreesWithFact,
    isCorroborated, findCorroboratingResult,
    answerContentTerms, numbersMatchLoose,
    cubeSelfAnswer, isTautologicalAnswer,
    isUsableEvidenceResult, resultAnchoredToQuestion, resultSupportsAnswer,
    modelAnswerFitsQuestion, answerClaimKind,
    parseWeatherQuery, fetchWeather, normalizeQuestion,
  };
}
