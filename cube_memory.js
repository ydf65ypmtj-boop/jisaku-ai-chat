"use strict";
/*
 * Cube 記憶モジュール（cube_memory.js）
 *
 * Muse のような「ファイルベースの記憶」を Cube（公開版・静的版）に
 * 与えるための独立モジュール。ページ本体（index.html / style.css /
 * script.js）には一切依存せず、組み込みは INTEGRATION.md の手順で
 * 別工程が行う。
 *
 * 設計の要点：
 * - 記憶は3つの「ファイル」に分かれる（Muse の MEMORY.md /
 *   日次ログ / 人物メモに対応）：
 *     main   … ユーザー情報・好み・決めたこと（本体メモリ）
 *     people … 人物メモ
 *     daily  … 日次ログ（決めたこと・覚えたことの時系列の記録）
 *   保存実体はファイルごとに1本の JSONL（1行1エントリ）。
 *   ローカル保存では localStorage のキー1つ＝ファイル1本に対応し、
 *   サーバー保存（iMac の cube-server）では実ファイルに対応する。
 *   どちらの保存先でも読み書きの形は同じ。
 * - 保存できる出所は3つだけに限定する：
 *     explicit     … ユーザーが「覚えて」と明示したもの
 *     verified     … 検証ゲートを通った事実（出典つき必須）
 *     conversation … ユーザー自身の発言からアプリ側ルールで抽出した
 *                    durable な候補（好み・決定・約束・人物情報のみ）
 *   モデルが生成した未検証の事実主張は、どの経路でも保存しない
 *   （嘘を記憶に固定しないため）。秘密情報らしい文字列も保存しない。
 * - 同じ主題の新しいエントリは古いものを「上書き済み」にし、
 *   古い方もファイルに残す（監査可能性）。完全な重複は入れない。
 * - 読み出しは関連選択＋文字数予算つき。モデルが小さいので、
 *   注入する文脈は短い箇条書きに整形する。記憶は証拠（記事本文）
 *   には決して混ぜない（出典の混同防止）。
 *
 * 使い方（ブラウザ）：
 *   <script src="cube_memory.js"></script>
 *   const mem = CubeMemory.create({ store: CubeMemory.makeHybridStore(...) });
 *   await mem.load();
 * Node（テスト）：
 *   const CubeMemory = require("./cube_memory.js");
 */
(function (global) {

  /* ---------------------------------------------------------------- */
  /* 定数                                                              */
  /* ---------------------------------------------------------------- */

  const KINDS = ["user", "preference", "decision", "person", "daily"];
  const FILE_OF_KIND = {
    user: "main", preference: "main", decision: "main",
    person: "people", daily: "daily",
  };
  const FILES = ["main", "people", "daily"];
  const KIND_LABEL = {
    user: "ユーザー", preference: "好み", decision: "決めたこと",
    person: "人物", daily: "記録",
  };
  const STATUSES = ["active", "superseded", "forgotten"];

  const DEFAULT_MAX_CHARS = 700;   // 注入予算（数百トークン目安）
  const DEFAULT_MAX_ENTRIES = 8;
  const MAX_TEXT_LEN = 200;        // 1エントリの本文上限（文字）
  const MAX_TOPIC_LEN = 40;

  /* ローカル保存のキー（既存の知識メモ帳 NB_KEY と同じ命名家族）。 */
  const LOCAL_KEYS = {
    main: "jisaku-ai-memory-main-v1",
    people: "jisaku-ai-memory-people-v1",
    daily: "jisaku-ai-memory-daily-v1",
  };

  /* 秘密情報らしい文字列は記憶に保存しない。 */
  const SECRET_PATTERNS = [
    /パスワード/, /暗証番号/, /認証コード/, /ワンタイム/,
    /password/i, /passcode/i, /\bPIN\b/,
    /API\s*キー/i, /api[_-]?key/i, /secret/i, /access[_-]?token/i,
    /クレジットカード/, /カード番号/,
    /\b\d{12,}\b/, // 12桁以上の連続数字（カード番号・口座番号など）
  ];

  /* ---------------------------------------------------------------- */
  /* 小物ユーティリティ                                                */
  /* ---------------------------------------------------------------- */

  /* 照合用正規化：既存の normalizeQuestion と同じ流儀
     （空白・句読点を除き、小文字化）。 */
  function normalize(s) {
    return String(s || "")
      .replace(/[\s　]/g, "")
      .replace(/[、。・,.!?！？:：;；「」『』（）()［］""]/g, "")
      .toLowerCase();
  }

  /* JST の日付キー（YYYY-MM-DD）。 */
  function jstDateKey(d) {
    const t = (d ? d.getTime() : Date.now()) + 9 * 3600 * 1000;
    return new Date(t).toISOString().slice(0, 10);
  }

  function nowIso() { return new Date().toISOString(); }

  let idCounter = 0;
  function newId() {
    idCounter += 1;
    return "m_" + Date.now().toString(36) + "_" + idCounter.toString(36) +
      Math.floor(Math.random() * 1296).toString(36);
  }

  /* 関連スコア用：日本語の2グラムと英数字ワードを取り出す。 */
  function contentChunks(s) {
    const out = new Set();
    const t = String(s || "");
    const runs = t.match(/[ぁ-んァ-ヶ一-鿿ー]{2,}/g) || [];
    for (const run of runs) {
      for (let i = 0; i + 2 <= run.length; i++) out.add(run.slice(i, i + 2));
    }
    const words = t.toLowerCase().match(/[a-z0-9]{2,}/g) || [];
    for (const w of words) out.add(w);
    return out;
  }

  function looksSecret(text) {
    const t = String(text || "");
    if (SECRET_PATTERNS.some((re) => re.test(t))) return true;
    // 空白なし・英数字混在・24文字以上の塊はキー類の可能性がある。
    if (!/\s/.test(t) && t.length >= 24 && /[a-zA-Z]/.test(t) && /\d/.test(t) &&
        !/^https?:\/\//i.test(t)) return true;
    return false;
  }

  /* ---------------------------------------------------------------- */
  /* JSONL の読み書き                                                  */
  /* ---------------------------------------------------------------- */

  function parseJsonl(text) {
    const entries = [];
    for (const line of String(text || "").split("\n")) {
      const s = line.trim();
      if (!s) continue;
      try {
        const e = JSON.parse(s);
        if (e && typeof e === "object" && typeof e.text === "string") {
          entries.push(sanitizeEntry(e));
        }
      } catch { /* 壊れた行は読み飛ばす（ファイル全体は守る） */ }
    }
    return entries;
  }

  function serializeJsonl(entries) {
    return entries.map((e) => JSON.stringify(e)).join("\n") +
      (entries.length ? "\n" : "");
  }

  /* 外部から来たエントリを、既知の形に整えて取り込む。 */
  function sanitizeEntry(e) {
    const kind = KINDS.includes(e.kind) ? e.kind : "user";
    const status = STATUSES.includes(e.status) ? e.status : "active";
    const src = (e.source && typeof e.source === "object") ? e.source : {};
    return {
      id: typeof e.id === "string" && e.id ? e.id : newId(),
      kind,
      topic: String(e.topic || "").slice(0, MAX_TOPIC_LEN),
      text: String(e.text || "").slice(0, MAX_TEXT_LEN),
      source: {
        type: ["explicit", "verified", "conversation"].includes(src.type)
          ? src.type : "conversation",
        detail: String(src.detail || ""),
        source_title: String(src.source_title || ""),
        source_url: String(src.source_url || ""),
      },
      date: typeof e.date === "string" ? e.date : jstDateKey(),
      status,
      supersedes: typeof e.supersedes === "string" ? e.supersedes : null,
      created_at: typeof e.created_at === "string" ? e.created_at : nowIso(),
      updated_at: typeof e.updated_at === "string" ? e.updated_at : nowIso(),
    };
  }

  /* ---------------------------------------------------------------- */
  /* 保存フィルタ（嘘と秘密を記憶に固定しないための関門）              */
  /* ---------------------------------------------------------------- */

  /*
   * 候補を保存してよいか判定する。保存できるのは：
   * - source.type === "explicit"（ユーザーの明示）… 全種別可
   * - source.type === "verified"（検証済み）… 出典（title か url）が必須
   * - source.type === "conversation"（会話抽出）… 抽出器経由
   *   （viaExtractor === true）の、好み・決定・人物・ユーザー情報だけ。
   *   世界の事実の主張はこの経路では保存できない。
   * それ以外（モデルの生成文そのもの等）は常に拒否する。
   */
  function validateCandidate(c) {
    if (!c || typeof c !== "object") return { ok: false, reason: "not-an-object" };
    const text = String(c.text || "").trim();
    if (!text) return { ok: false, reason: "empty-text" };
    if (text.length > MAX_TEXT_LEN) return { ok: false, reason: "text-too-long" };
    if (!KINDS.includes(c.kind)) return { ok: false, reason: "bad-kind" };
    if (looksSecret(text)) return { ok: false, reason: "looks-secret" };
    const src = c.source || {};
    if (src.type === "explicit") return { ok: true };
    if (src.type === "verified") {
      if (src.source_title || src.source_url) return { ok: true };
      return { ok: false, reason: "verified-needs-source" };
    }
    if (src.type === "conversation") {
      if (!c.viaExtractor) return { ok: false, reason: "conversation-needs-extractor" };
      if (["preference", "decision", "person", "user"].includes(c.kind)) {
        return { ok: true };
      }
      return { ok: false, reason: "conversation-kind-not-allowed" };
    }
    return { ok: false, reason: "untrusted-source" };
  }

  /* ---------------------------------------------------------------- */
  /* 会話からの候補抽出（アプリ側ルール）                              */
  /*                                                                   */
  /* ユーザー自身の発言だけを対象にする保守的なパターン群。            */
  /* 質問文（〜？／教えて／とは）は対象外。1人称の durable な          */
  /* 発言（好み・住まい・活動・決定・人物）だけを候補にする。          */
  /* ---------------------------------------------------------------- */

  function isQuestionLike(text) {
    const t = String(text || "").trim();
    return /[？?]$/.test(t) || /(教えて|とは|ですか|ますか|でしょうか|なに|何|どこ|いつ|だれ|誰|なぜ|どう)/.test(t);
  }

  function extractCandidates(userText) {
    const t = String(userText || "").trim();
    const out = [];
    if (!t || t.length > 300 || isQuestionLike(t)) return out;
    const push = (kind, topic, text) => {
      text = String(text || "").trim();
      if (!text || looksSecret(text)) return;
      out.push({
        kind, topic: String(topic || "").slice(0, MAX_TOPIC_LEN), text,
        source: { type: "conversation", detail: "user-utterance" },
        viaExtractor: true,
      });
    };
    let m;
    // 好み：〜が好き／大好き／苦手／嫌い
    if ((m = t.match(/(.{1,24}?)が(好き|大好き|苦手|嫌い)/))) {
      push("preference", m[1], `${m[1]}が${m[2]}`);
    }
    // 住まい：〜に住んでいる
    if ((m = t.match(/(.{2,20}?)に住んで(いる|います)/))) {
      push("user", "住んでいる場所", `${m[1]}に住んでいる`);
    }
    // 活動：〜をやっている／習っている（部活・習い事など）
    if ((m = t.match(/(.{1,20}?)を(やっている|やっています|習っている|習っています)/))) {
      push("user", m[1], `${m[1]}を${m[2].replace(/います$/, "いる")}`);
    }
    // 決定・約束：〜することにした／〜するつもり
    if ((m = t.match(/(.{1,40}?)することにした/))) {
      push("decision", "", `${m[1]}することにした`);
    } else if ((m = t.match(/(.{1,40}?)するつもり/))) {
      push("decision", "", `${m[1]}するつもり`);
    }
    // 人物：〜は私の／僕の＋続柄・関係
    if ((m = t.match(/(.{1,12}?)は(私|僕|あたし|自分)の(母|父|兄|姉|弟|妹|友達|友だち|先生|祖母|祖父)/))) {
      push("person", m[1], `${m[1]}は${m[2]}の${m[3]}`);
    }
    // 通っている学校など：〜に通っている
    if ((m = t.match(/(.{2,24}?)に通って(いる|います)/))) {
      push("user", "通っている場所", `${m[1]}に通っている`);
    }
    return out;
  }

  /* 明示コマンドの解釈：「〜を覚えて」「覚えて：〜」「〜を忘れて」 */
  function parseExplicitCommand(userText) {
    const t = String(userText || "").trim();
    let m;
    if ((m = t.match(/^覚えて[:：]\s*(.+)$/))) {
      return { action: "remember", text: m[1].trim() };
    }
    if ((m = t.match(/^(.+?)を覚えて(?:おいて|ください|て)?[。!！]?$/))) {
      return { action: "remember", text: m[1].trim() };
    }
    if ((m = t.match(/^(.+?)を忘れて(?:ください)?[。!！]?$/))) {
      return { action: "forget", text: m[1].trim() };
    }
    if ((m = t.match(/^(.+?)の記憶を(?:消して|削除して)(?:ください)?[。!！]?$/))) {
      return { action: "forget", text: m[1].trim() };
    }
    return null;
  }

  /* 明示の「覚えて」の種別推定（軽いヒント付け）。 */
  function inferKindForExplicit(text) {
    const t = String(text || "");
    if (/(好き|大好き|苦手|嫌い)/.test(t)) return "preference";
    if (/(することにした|するつもり|約束)/.test(t)) return "decision";
    if (/(は私の|は僕の|母|父|友達|友だち|先生)/.test(t)) return "person";
    return "user";
  }

  /* ---------------------------------------------------------------- */
  /* 保存先アダプタ                                                    */
  /*                                                                   */
  /* どのアダプタも { readAll(): Promise<{file: jsonlText}>,           */
  /* writeAll(files): Promise<void> } を実装する。                     */
  /* ---------------------------------------------------------------- */

  /* メモリ内（テスト用・フォールバック用）。 */
  function makeMemoryStore(initial) {
    const data = Object.assign({ main: "", people: "", daily: "" }, initial || {});
    return {
      kind: "memory",
      async readAll() { return Object.assign({}, data); },
      async writeAll(files) {
        for (const f of FILES) if (typeof files[f] === "string") data[f] = files[f];
      },
      _data: data,
    };
  }

  /* localStorage 保存：ファイル1本＝キー1つに JSONL 文字列を入れる。
     既存の知識メモ帳と同じ保存の仕組み・場所（この端末のブラウザ内）。 */
  function makeLocalStore(storage) {
    const ls = storage ||
      (typeof window !== "undefined" ? window.localStorage : null);
    return {
      kind: "local",
      async readAll() {
        const out = {};
        for (const f of FILES) {
          try { out[f] = ls ? (ls.getItem(LOCAL_KEYS[f]) || "") : ""; }
          catch { out[f] = ""; }
        }
        return out;
      },
      async writeAll(files) {
        if (!ls) return;
        for (const f of FILES) {
          if (typeof files[f] !== "string") continue;
          try { ls.setItem(LOCAL_KEYS[f], files[f]); } catch { /* 保存できない環境では諦める */ }
        }
      },
    };
  }

  /* iMac の cube-server 保存：GET/PUT /memory（memory_endpoint.js）。
     Tailscale 経由で本人の端末同士だけが届くため、端末を跨いで
     同じ記憶が使える。公開版（ZeroGPU）では使わない（DESIGN.md）。 */
  function makeServerStore(baseUrl, fetchFn) {
    const base = String(baseUrl || "").replace(/\/$/, "");
    const doFetch = fetchFn || (typeof fetch !== "undefined" ? fetch : null);
    return {
      kind: "server",
      baseUrl: base,
      async readAll() {
        if (!doFetch) throw new Error("fetch unavailable");
        const res = await doFetch(`${base}/memory`, { credentials: "omit" });
        if (!res.ok) throw new Error(`memory GET failed: ${res.status}`);
        const j = await res.json();
        const out = {};
        for (const f of FILES) out[f] = (j && j.files && j.files[f]) || "";
        return out;
      },
      async writeAll(files) {
        if (!doFetch) throw new Error("fetch unavailable");
        const res = await doFetch(`${base}/memory`, {
          method: "PUT",
          credentials: "omit",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ files }),
        });
        if (!res.ok) throw new Error(`memory PUT failed: ${res.status}`);
      },
    };
  }

  /* ハイブリッド：サーバーが届けばサーバーを正とし、ローカルへ写す。
     届かなければローカルで動き続け、書込みはローカルへ先に入れて
     サーバーへはベストエフォートで送る（次回ロード時に id で統合）。 */
  function makeHybridStore(serverStore, localStore) {
    let serverOk = false;
    const mergeJsonl = (a, b) => {
      const byId = new Map();
      for (const e of parseJsonl(a)) byId.set(e.id, e);
      for (const e of parseJsonl(b)) if (!byId.has(e.id)) byId.set(e.id, e);
      return serializeJsonl([...byId.values()]);
    };
    return {
      kind: "hybrid",
      get serverOk() { return serverOk; },
      async readAll() {
        let local = { main: "", people: "", daily: "" };
        try { local = await localStore.readAll(); } catch { /* ローカルも無い場合は空 */ }
        try {
          const remote = await serverStore.readAll();
          serverOk = true;
          const merged = {};
          for (const f of FILES) merged[f] = mergeJsonl(remote[f], local[f]);
          try { await localStore.writeAll(merged); } catch { /* 写しは任意 */ }
          // ローカルにしか無いエントリがあればサーバーへも戻す。
          try { await serverStore.writeAll(merged); } catch { /* 次回に持ち越し */ }
          return merged;
        } catch {
          serverOk = false;
          return local;
        }
      },
      async writeAll(files) {
        try { await localStore.writeAll(files); } catch { /* 保存できない環境では諦める */ }
        try { await serverStore.writeAll(files); serverOk = true; }
        catch { serverOk = false; }
      },
    };
  }

  /* ---------------------------------------------------------------- */
  /* 記憶本体                                                          */
  /* ---------------------------------------------------------------- */

  function create(options) {
    const opts = options || {};
    const store = opts.store || makeLocalStore(opts.storage);
    const state = {
      loaded: false,
      entries: { main: [], people: [], daily: [] }, // ファイルごとの全エントリ
    };

    const fileOf = (entry) => FILE_OF_KIND[entry.kind] || "main";
    const allEntries = () => FILES.flatMap((f) => state.entries[f]);
    const activeEntries = () => allEntries().filter((e) => e.status === "active");

    async function persist(filesToWrite) {
      const files = {};
      for (const f of filesToWrite) files[f] = serializeJsonl(state.entries[f]);
      await store.writeAll(files);
    }

    async function load() {
      const files = await store.readAll();
      for (const f of FILES) state.entries[f] = parseJsonl(files[f]);
      state.loaded = true;
      return api;
    }

    /* --- 追加（フィルタを通ったものだけが入る） --- */
    async function add(candidate) {
      const v = validateCandidate(candidate);
      if (!v.ok) return { added: false, reason: v.reason };
      const entry = sanitizeEntry({
        id: newId(),
        kind: candidate.kind,
        topic: candidate.topic || "",
        text: String(candidate.text).trim(),
        source: candidate.source,
        date: candidate.date || jstDateKey(),
        status: "active",
        created_at: nowIso(), updated_at: nowIso(),
      });
      const file = fileOf(entry);
      const nText = normalize(entry.text);
      const nTopic = normalize(entry.topic);
      const siblings = state.entries[file].filter((e) => e.status === "active");
      // 完全な重複は入れない。
      if (siblings.some((e) => normalize(e.text) === nText)) {
        return { added: false, reason: "duplicate" };
      }
      // 同じ主題の現行エントリは上書き済みにして引き継ぐ。
      let supersededId = null;
      if (nTopic) {
        for (const e of siblings) {
          if (normalize(e.topic) === nTopic && e.kind === entry.kind) {
            e.status = "superseded";
            e.updated_at = nowIso();
            supersededId = e.id;
          }
        }
      }
      entry.supersedes = supersededId;
      state.entries[file].push(entry);
      const touched = [file];
      // 決定と明示は日次ログにも1行残す（Muse の日次ノート方式）。
      if (entry.kind === "decision" || entry.source.type === "explicit") {
        state.entries.daily.push(sanitizeEntry({
          id: newId(), kind: "daily", topic: entry.topic,
          text: entry.text, source: entry.source, date: entry.date,
          status: "active", created_at: nowIso(), updated_at: nowIso(),
        }));
        touched.push("daily");
      }
      await persist(touched);
      return { added: true, entry, supersededId };
    }

    /* ユーザーの明示（「覚えて」）からの追加。 */
    async function addExplicit(text, kindHint, topic) {
      const kind = kindHint || inferKindForExplicit(text);
      return add({
        kind, topic: topic || "", text,
        source: { type: "explicit", detail: "user-command" },
      });
    }

    /* 検証ゲートを通った事実の追加（出典が必須）。 */
    async function addVerified(text, o) {
      const oo = o || {};
      return add({
        kind: oo.kind || "user", topic: oo.topic || "", text,
        source: {
          type: "verified", detail: oo.detail || "verification-gate",
          source_title: oo.sourceTitle || "", source_url: oo.sourceUrl || "",
        },
      });
    }

    /* ユーザーの発言1件を観察する：明示コマンドの解釈＋候補抽出。
       戻り値は組み込み側が表示に使える通知つき。 */
    async function observeUserMessage(userText) {
      const result = { added: [], forgotten: [], notices: [] };
      const cmd = parseExplicitCommand(userText);
      if (cmd && cmd.action === "remember") {
        const r = await addExplicit(cmd.text);
        if (r.added) {
          result.added.push(r.entry);
          result.notices.push(`覚えました：${r.entry.text}`);
        } else if (r.reason === "looks-secret") {
          result.notices.push("秘密情報らしい内容は覚えられません。");
        } else if (r.reason === "duplicate") {
          result.notices.push("それは既に覚えています。");
        }
        return result; // 明示コマンドの文自体は抽出対象にしない
      }
      if (cmd && cmd.action === "forget") {
        const forgotten = await forget(cmd.text);
        result.forgotten = forgotten;
        result.notices.push(forgotten.length
          ? `忘れました：${forgotten.map((e) => e.text).join("、")}`
          : "該当する記憶が見つかりませんでした。");
        return result;
      }
      for (const cand of extractCandidates(userText)) {
        const r = await add(cand);
        if (r.added) result.added.push(r.entry);
      }
      return result;
    }

    /* --- 忘却・削除・編集（ユーザーの制御用 API） --- */

    /* 主題または本文の一部に一致する現行エントリを「忘れた」にする
       （ファイルには forgotten として残る。完全削除は remove）。 */
    async function forget(query) {
      const nq = normalize(query);
      if (!nq) return [];
      const hit = [];
      const touched = new Set();
      for (const f of FILES) {
        for (const e of state.entries[f]) {
          if (e.status !== "active") continue;
          if (normalize(e.topic).includes(nq) || normalize(e.text).includes(nq) ||
              (normalize(e.topic) && nq.includes(normalize(e.topic)))) {
            e.status = "forgotten";
            e.updated_at = nowIso();
            hit.push(e);
            touched.add(f);
          }
        }
      }
      if (touched.size) await persist([...touched]);
      return hit;
    }

    async function remove(id) {
      for (const f of FILES) {
        const before = state.entries[f].length;
        state.entries[f] = state.entries[f].filter((e) => e.id !== id);
        if (state.entries[f].length !== before) {
          await persist([f]);
          return true;
        }
      }
      return false;
    }

    /* UI からの直接編集：本文をその場で直す（updated_at を更新）。 */
    async function updateText(id, newText) {
      const text = String(newText || "").trim();
      if (!text || text.length > MAX_TEXT_LEN || looksSecret(text)) return null;
      for (const f of FILES) {
        const e = state.entries[f].find((x) => x.id === id);
        if (e) {
          e.text = text;
          e.updated_at = nowIso();
          await persist([f]);
          return e;
        }
      }
      return null;
    }

    function list(filter) {
      const fo = filter || {};
      return allEntries().filter((e) =>
        (fo.kind ? e.kind === fo.kind : true) &&
        (fo.status ? e.status === fo.status : e.status === "active") &&
        (fo.file ? fileOf(e) === fo.file : true));
    }

    function get(id) { return allEntries().find((e) => e.id === id) || null; }

    /* --- 関連選択と文脈の整形 --- */

    /* 質問との関連でエントリに点数を付ける。
       主題の一致を最重視し、本文の語の重なり、種別の基礎点、
       新しさで並べる。daily は日付・主題が合うときだけ対象。 */
    function scoreEntry(e, question, qNorm, qChunks, today) {
      let score = 0;
      const nTopic = normalize(e.topic);
      if (nTopic && qNorm.includes(nTopic)) score += 4;
      const chunks = contentChunks(e.topic + " " + e.text);
      let overlap = 0;
      for (const c of chunks) if (qChunks.has(c)) overlap += 1;
      score += Math.min(overlap, 4);
      // 種別の基礎点：ユーザー情報は「核」として常に対象に入る。
      // それ以外は語の重なりか主題一致が無いと閾値に届かない。
      if (e.kind === "user") score += 1.2;
      else if (e.kind === "preference") score += 0.4;
      else if (e.kind === "person") score += 0.2;
      if (e.kind === "daily") {
        // 日次ログは記録であって本体ではない。本体エントリの写しが
        // ある場合は本体を優先させるため、少しだけ下げる。
        score -= 0.5;
        if (e.date === today) score += 0.5;
        else if (!nTopic || !qNorm.includes(nTopic)) score -= 2;
      }
      // 同点なら新しいものを優先するための微小な加点。
      score += Math.min(0.3, (Date.parse(e.updated_at) || 0) / 1e13);
      return score;
    }

    function selectRelevant(question, options) {
      const oo = options || {};
      const maxChars = oo.maxChars || DEFAULT_MAX_CHARS;
      const maxEntries = oo.maxEntries || DEFAULT_MAX_ENTRIES;
      const minScore = oo.minScore != null ? oo.minScore : 1.0;
      const qNorm = normalize(question);
      const qChunks = contentChunks(question);
      const today = jstDateKey();
      const scored = activeEntries()
        .map((e) => ({ e, s: scoreEntry(e, question, qNorm, qChunks, today) }))
        .filter((x) => x.s >= minScore)
        .sort((a, b) => b.s - a.s);
      const picked = [];
      const pickedTexts = new Set();
      let used = 0;
      for (const { e } of scored) {
        if (picked.length >= maxEntries) break;
        // 日次ログには本体エントリの写しが入ることがある。
        // 同じ文を二重に注入しないよう、本文で重複排除する。
        const nText = normalize(e.text);
        if (pickedTexts.has(nText)) continue;
        const lineLen = e.text.length + (e.topic ? e.topic.length : 0) + 8;
        if (picked.length > 0 && used + lineLen > maxChars) continue;
        picked.push(e);
        pickedTexts.add(nText);
        used += lineLen;
      }
      return picked;
    }

    /* 注入用の短い箇条書きに整形する（モデルが小さいので簡潔に）。 */
    function formatContext(entries) {
      if (!entries || !entries.length) return "";
      const lines = entries.map((e) => `・${KIND_LABEL[e.kind] || ""}：${e.text}`);
      return "覚えていること：\n" + lines.join("\n");
    }

    /* Ultra のシステムプロンプト末尾に足す形（ChatML の system 用）。 */
    function systemSuffix(question, options) {
      const block = formatContext(selectRelevant(question, options));
      if (!block) return "";
      return `\n\n${block}\n（これはユーザーについての記憶です。関係する質問のときだけ使い、関係ない質問では触れないでください。）`;
    }

    /* 自作モデル（QA形式）向け：強い主題一致があるときだけ、
       質問文の末尾に短い参考を添える。一致が弱ければ何もしない。 */
    function decorateQuestion(question, options) {
      const oo = Object.assign({}, options || {}, { minScore: 4 });
      const picked = selectRelevant(question, oo).slice(0, 3);
      if (!picked.length) return question;
      const ref = picked.map((e) => e.text).join("；");
      const suffix = `（覚えていること：${ref}）`;
      if (suffix.length > 160) return question;
      return question + suffix;
    }

    /* 自分のことを聞かれた質問への直接参照（メモ帳レーンと同じ流儀の
       アプリ側即答用）。自己言及＋主題一致があるときだけ返す。 */
    function findDirectAnswer(question) {
      const q = String(question || "");
      if (!/(私|僕|あたし|自分|おれ|俺)/.test(q) &&
          !/(覚えて|おぼえて)/.test(q)) return null;
      const qNorm = normalize(q);
      for (const e of activeEntries()) {
        const nTopic = normalize(e.topic);
        if (nTopic && qNorm.includes(nTopic)) return e;
      }
      return null;
    }

    /* --- 入出力（設定画面・バックアップ用） --- */

    function exportAll() {
      const files = {};
      for (const f of FILES) files[f] = serializeJsonl(state.entries[f]);
      return files;
    }

    async function importAll(files) {
      for (const f of FILES) {
        if (files && typeof files[f] === "string") {
          state.entries[f] = parseJsonl(files[f]);
        }
      }
      state.loaded = true;
      await persist(FILES.slice());
      return api;
    }

    const api = {
      load, add, addExplicit, addVerified, observeUserMessage,
      forget, remove, updateText, list, get,
      selectRelevant, formatContext, systemSuffix, decorateQuestion,
      findDirectAnswer, exportAll, importAll,
      get loaded() { return state.loaded; },
      get storeKind() { return store.kind; },
      _state: state,
    };
    return api;
  }

  /* ---------------------------------------------------------------- */
  /* 公開                                                              */
  /* ---------------------------------------------------------------- */

  const CubeMemory = {
    create,
    makeMemoryStore, makeLocalStore, makeServerStore, makeHybridStore,
    // 内部部品（テスト・組み込み時の検証用に公開）
    _internal: {
      KINDS, FILES, FILE_OF_KIND, KIND_LABEL, LOCAL_KEYS,
      normalize, jstDateKey, contentChunks, looksSecret,
      parseJsonl, serializeJsonl, sanitizeEntry,
      validateCandidate, extractCandidates, parseExplicitCommand,
      inferKindForExplicit, isQuestionLike,
      DEFAULT_MAX_CHARS, DEFAULT_MAX_ENTRIES, MAX_TEXT_LEN,
    },
  };

  global.CubeMemory = CubeMemory;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = CubeMemory;
  }
})(typeof window !== "undefined" ? window : globalThis);
