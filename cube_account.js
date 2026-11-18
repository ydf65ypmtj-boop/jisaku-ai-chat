"use strict";
/*
 * Cube アカウントモジュール（cube_account.js）
 *
 * 公開版 Cube（静的版）に「アカウント方式」を与える独立モジュール。
 * 役割は2つ：
 *   1) Supabase Auth（Google）によるログイン（任意）
 *   2) チャット履歴の保存・同期（ローカル保存＋ログイン時の
 *      Supabase 同期）
 *
 * 設計の要点：
 * - ログインは任意。未ログインでも履歴はこの端末の localStorage に
 *   保存され、ログインすると Supabase（ユーザーごとの行、RLS で
 *   自分の行だけ）に同期されて端末をまたいで使える。
 * - config（Project URL + Publishable key）が空の間は、モジュール
 *   全体が無効（マネージャは disabled を返し、script.js 側は
 *   アカウント領域も履歴 UI も一切出さない＝従来挙動と完全同一）。
 * - Publishable key はブラウザに埋め込む前提の公開キー。
 *   Secret key は絶対にここへ入れない。
 * - 同期の規則（会話単位）：
 *     同じ id の会話は updated_at が新しい方を採用する。
 *     削除は deleted_at の「墓標」で伝播させる（墓標が会話の
 *     updated_at より新しければ削除が勝ち、逆なら復活）。
 *     オフライン時の正は localStorage。ログイン中は読み書きの
 *     たびにベストエフォートでクラウドと突き合わせる。
 * - プロフィール記憶（cube_memory.js）のクラウド同期は対象外
 *   （端末内保存のまま。将来の拡張候補として設計書にのみ記載）。
 *
 * 使い方（ブラウザ）：
 *   <script src="cube_account.js"></script>
 *   const mgr = CubeAccount.createManager({ config });
 * Node（テスト）：
 *   const CubeAccount = require("./cube_account.js");
 */
(function (global) {

  /* ---------------------------------------------------------------- */
  /* 定数                                                              */
  /* ---------------------------------------------------------------- */

  const SUPABASE_ESM_URL =
    "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm";
  const CONVERSATIONS_TABLE = "conversations";
  const HISTORY_LOCAL_KEY = "jisaku-ai-chat-history-v1";
  const MAX_CONVERSATIONS = 100;   // ローカル保持の上限（古い順に削除）
  const TITLE_MAX_LEN = 40;

  /* ---------------------------------------------------------------- */
  /* 小物                                                              */
  /* ---------------------------------------------------------------- */

  function isConfigured(config) {
    return !!(
      config &&
      typeof config.url === "string" && config.url.trim() &&
      typeof config.publishableKey === "string" &&
      config.publishableKey.trim()
    );
  }

  function nowIso() { return new Date().toISOString(); }

  let idCounter = 0;
  function newConversationId() {
    idCounter += 1;
    return "c_" + Date.now().toString(36) + "_" + idCounter.toString(36) +
      Math.floor(Math.random() * 1296).toString(36);
  }

  /* 会話タイトルは最初の質問から自動生成する。 */
  function titleFromQuestion(q) {
    const t = String(q || "").replace(/\s+/g, " ").trim();
    if (!t) return "無題の会話";
    return t.length > TITLE_MAX_LEN ? t.slice(0, TITLE_MAX_LEN) + "…" : t;
  }

  function normalizeUser(u) {
    if (!u) return null;
    const meta = u.user_metadata || {};
    return {
      id: u.id,
      email: u.email || "",
      name: meta.full_name || meta.name || u.email || "",
      avatarUrl: meta.avatar_url || meta.picture || "",
    };
  }

  /* ---------------------------------------------------------------- */
  /* 認証マネージャ（Supabase Auth / Google）                          */
  /* ---------------------------------------------------------------- */

  /* ブラウザ用：supabase-js を CDN から遅延読み込みしてクライアントを
     作る。テストでは clientFactory を差し替える。 */
  async function defaultClientFactory(config) {
    const mod = await import(/* @vite-ignore */ SUPABASE_ESM_URL);
    const createClient = mod.createClient || (mod.default && mod.default.createClient);
    if (!createClient) throw new Error("supabase-js createClient not found");
    return createClient(config.url, config.publishableKey);
  }

  function makeDisabledManager() {
    return {
      enabled: false,
      config: null,
      session: null,
      user: null,
      signedIn: false,
      async init() { return this; },
      async signInWithGoogle() { /* 無効時は無動作 */ },
      async signOut() { /* 無効時は無動作 */ },
      onChange() { return () => {}; },
      backend() { return null; },
      pointsBackend() { return null; },
    };
  }

  function createManager(options) {
    const opts = options || {};
    const config = opts.config || {};
    if (!isConfigured(config)) return makeDisabledManager();
    const clientFactory = opts.clientFactory || defaultClientFactory;

    let client = null;
    let clientPromise = null;
    let session = null;
    let user = null;
    const listeners = new Set();

    function notify() {
      for (const cb of [...listeners]) {
        try { cb(api); } catch { /* リスナの失敗は認証を妨げない */ }
      }
    }
    function setSession(s) {
      session = s || null;
      user = session && session.user ? normalizeUser(session.user) : null;
    }
    async function ensureClient() {
      if (client) return client;
      if (!clientPromise) {
        clientPromise = (async () => {
          client = await clientFactory(config);
          return client;
        })();
      }
      return clientPromise;
    }

    const api = {
      enabled: true,
      config,
      get session() { return session; },
      get user() { return user; },
      get signedIn() { return !!user; },

      /* セッションの復元（OAuth リダイレクトからの復帰を含む）と、
         以後の認証状態変化の購読を開始する。 */
      async init() {
        const c = await ensureClient();
        try {
          const { data } = await c.auth.getSession();
          setSession(data && data.session);
        } catch { /* セッションが無ければ未ログインとして続ける */ }
        try {
          c.auth.onAuthStateChange((_event, s) => {
            setSession(s);
            notify();
          });
        } catch { /* 購読できなくても復元済みセッションで動く */ }
        notify();
        return api;
      },

      /* Google でログイン（Supabase の OAuth 画面へ遷移する）。 */
      async signInWithGoogle() {
        const c = await ensureClient();
        let redirectTo = opts.redirectTo;
        if (!redirectTo && typeof location !== "undefined" && location.origin) {
          redirectTo = location.origin + location.pathname;
        }
        const { error } = await c.auth.signInWithOAuth({
          provider: "google",
          options: redirectTo ? { redirectTo } : {},
        });
        if (error) throw error;
      },

      async signOut() {
        const c = await ensureClient();
        try { await c.auth.signOut(); } catch { /* ローカル側は落とす */ }
        setSession(null);
        notify();
      },

      onChange(cb) {
        listeners.add(cb);
        return () => listeners.delete(cb);
      },

      /* ログイン中だけ、履歴同期用の Supabase バックエンドを返す。 */
      backend() {
        if (!client || !user) return null;
        return makeSupabaseConvBackend(client, user.id);
      },

      /* ログイン中だけ、Cubeポイント用の Supabase バックエンドを
         返す（backend() と同じ流儀）。 */
      pointsBackend() {
        if (!client || !user) return null;
        return makeSupabasePointsBackend(client, user.id);
      },
    };
    return api;
  }

  /* ---------------------------------------------------------------- */
  /* 履歴の状態とマージ（純粋関数）                                    */
  /*                                                                   */
  /* 状態の形（ローカル blob と同じ形）：                              */
  /*   { conversations: [conv...], tombstones: { id: deletedAtIso } }  */
  /* conv = { id, title, created_at, updated_at, turns: [...] }        */
  /* ---------------------------------------------------------------- */

  function emptyState() {
    return { conversations: [], tombstones: {} };
  }

  function sanitizeConversation(c) {
    if (!c || typeof c !== "object" || typeof c.id !== "string" || !c.id) {
      return null;
    }
    return {
      id: c.id,
      title: typeof c.title === "string" ? c.title : "",
      created_at: typeof c.created_at === "string" ? c.created_at : nowIso(),
      updated_at: typeof c.updated_at === "string" ? c.updated_at : nowIso(),
      turns: Array.isArray(c.turns) ? c.turns : [],
    };
  }

  function parseStateBlob(text) {
    if (!text) return emptyState();
    try {
      const j = JSON.parse(text);
      if (!j || typeof j !== "object") return emptyState();
      const conversations = [];
      for (const c of Array.isArray(j.conversations) ? j.conversations : []) {
        const s = sanitizeConversation(c);
        if (s) conversations.push(s);
      }
      const tombstones = {};
      if (j.tombstones && typeof j.tombstones === "object") {
        for (const [id, ts] of Object.entries(j.tombstones)) {
          if (typeof ts === "string") tombstones[id] = ts;
        }
      }
      return { conversations, tombstones };
    } catch {
      return emptyState();
    }
  }

  function serializeState(state) {
    return JSON.stringify({
      conversations: state.conversations,
      tombstones: state.tombstones,
    });
  }

  /*
   * ローカル状態とクラウド状態をマージする。
   * cloud = { convs: Map(id->conv), tombstones: Map(id->deletedAt) }
   * 規則：会話は updated_at の新しい方。墓標は、その時刻が会話の
   * updated_at 以上なら削除が勝ち（墓標を保持）、会話の方が新しければ
   * 会話が復活し墓標は捨てる。
   */
  function mergeConversationStates(localState, cloud) {
    const convs = new Map();
    for (const c of localState.conversations || []) convs.set(c.id, c);
    const tombs = new Map(Object.entries(localState.tombstones || {}));
    if (cloud) {
      for (const [id, cc] of cloud.convs) {
        const lc = convs.get(id);
        if (!lc || cc.updated_at > lc.updated_at) convs.set(id, cc);
      }
      for (const [id, ts] of cloud.tombstones) {
        const cur = tombs.get(id);
        if (!cur || ts > cur) tombs.set(id, ts);
      }
    }
    const outConvs = [];
    for (const [id, conv] of convs) {
      const ts = tombs.get(id);
      if (ts && ts >= conv.updated_at) continue; // 削除が勝ち
      if (ts) tombs.delete(id);                  // 会話が新しく復活
      outConvs.push(conv);
    }
    outConvs.sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
    return {
      conversations: outConvs,
      tombstones: Object.fromEntries(tombs),
    };
  }

  /* 上限を超えたら古い会話を落とす（落とした id を返す）。
     keepId（表示中の会話）は落とさない。 */
  function pruneState(state, keepId) {
    if (state.conversations.length <= MAX_CONVERSATIONS) return [];
    const sorted = [...state.conversations]
      .sort((a, b) => (a.updated_at < b.updated_at ? -1 : 1));
    const dropped = [];
    while (state.conversations.length > MAX_CONVERSATIONS &&
           sorted.length) {
      const oldest = sorted.shift();
      if (oldest.id === keepId) continue;
      state.conversations = state.conversations
        .filter((c) => c.id !== oldest.id);
      state.tombstones[oldest.id] = nowIso();
      dropped.push(oldest.id);
    }
    return dropped;
  }

  /* ---------------------------------------------------------------- */
  /* ローカル履歴ストア（localStorage）                                */
  /*                                                                   */
  /* ストア契約（ローカル／クラウド共通）：                            */
  /*   listConversations(): Promise<[{id,title,created_at,             */
  /*     updated_at,turnCount}]>（updated_at の新しい順）              */
  /*   loadConversation(id): Promise<conv|null>                        */
  /*   saveConversation(conv): Promise<void>                           */
  /*   deleteConversation(id): Promise<void>                           */
  /* ---------------------------------------------------------------- */

  function makeLocalHistoryStore(storage) {
    const ls = storage ||
      (typeof window !== "undefined" ? window.localStorage : null);

    function readState() {
      if (!ls) return emptyState();
      try {
        return parseStateBlob(ls.getItem(HISTORY_LOCAL_KEY));
      } catch {
        return emptyState();
      }
    }
    function writeState(state) {
      if (!ls) return;
      try {
        ls.setItem(HISTORY_LOCAL_KEY, serializeState(state));
      } catch { /* 保存できない環境では諦める（会話自体は続く） */ }
    }

    return {
      kind: "local-history",
      async listConversations() {
        const state = readState();
        return state.conversations
          .map((c) => ({
            id: c.id, title: c.title,
            created_at: c.created_at, updated_at: c.updated_at,
            turnCount: (c.turns || []).length,
          }))
          .sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
      },
      async loadConversation(id) {
        const state = readState();
        return state.conversations.find((c) => c.id === id) || null;
      },
      async saveConversation(conv) {
        const s = sanitizeConversation(conv);
        if (!s) return;
        const state = readState();
        const i = state.conversations.findIndex((c) => c.id === s.id);
        if (i >= 0) state.conversations[i] = s;
        else state.conversations.push(s);
        // 同じ id に墓標が残っていたら、保存＝復活として取り消す。
        if (state.tombstones[s.id] && state.tombstones[s.id] <= s.updated_at) {
          delete state.tombstones[s.id];
        }
        pruneState(state, s.id);
        writeState(state);
      },
      async deleteConversation(id) {
        const state = readState();
        state.conversations = state.conversations
          .filter((c) => c.id !== id);
        state.tombstones[id] = nowIso();
        writeState(state);
      },
      /* 同期用（クラウドストアが使う内部 API）。 */
      async _readState() { return readState(); },
      async _writeState(state) { writeState(state); },
    };
  }

  /* ---------------------------------------------------------------- */
  /* Supabase バックエンド（conversations テーブル）                   */
  /* ---------------------------------------------------------------- */

  function makeSupabaseConvBackend(client, userId) {
    return {
      kind: "supabase-conversations",
      /* 自分の行だけが返る（RLS）。墓標行は deleted_at で区別する。 */
      async fetchState() {
        const { data, error } = await client
          .from(CONVERSATIONS_TABLE)
          .select("id,title,turns,created_at,updated_at,deleted_at");
        if (error) throw new Error(`conversations fetch failed: ${error.message || error}`);
        const convs = new Map();
        const tombstones = new Map();
        for (const row of data || []) {
          if (row.deleted_at) {
            tombstones.set(row.id, row.deleted_at);
          } else {
            convs.set(row.id, {
              id: row.id,
              title: row.title || "",
              created_at: row.created_at,
              updated_at: row.updated_at,
              turns: Array.isArray(row.turns) ? row.turns : [],
            });
          }
        }
        return { convs, tombstones };
      },
      async upsertConversation(conv) {
        const { error } = await client
          .from(CONVERSATIONS_TABLE)
          .upsert({
            id: conv.id,
            user_id: userId,
            title: conv.title,
            turns: conv.turns,
            created_at: conv.created_at,
            updated_at: conv.updated_at,
            deleted_at: null,
          }, { onConflict: "id" });
        if (error) throw new Error(`conversations upsert failed: ${error.message || error}`);
      },
      /* 墓標：行があれば deleted_at を付け、無ければ墓標行を作る。 */
      async applyTombstone(id, deletedAt) {
        const { error } = await client
          .from(CONVERSATIONS_TABLE)
          .upsert({
            id,
            user_id: userId,
            title: "",
            turns: [],
            created_at: deletedAt,
            updated_at: deletedAt,
            deleted_at: deletedAt,
          }, { onConflict: "id" });
        if (error) throw new Error(`conversations tombstone failed: ${error.message || error}`);
      },
    };
  }

  /* ---------------------------------------------------------------- */
  /* Supabase バックエンド（Cubeポイント）                             */
  /*                                                                   */
  /* 残高は points_balance（RLS で自分の行だけ読める）。書き込みは    */
  /* RPC（earn_points / spend_points）経由のみで、金額・1日上限・     */
  /* 冪等はサーバー側の関数が決める。ここは呼び出しの形だけを担う。   */
  /* ポイントは付加機能：失敗しても呼び出し側が握れば会話は無事。     */
  /* ---------------------------------------------------------------- */

  const POINTS_BALANCE_TABLE = "points_balance";

  function makeSupabasePointsBackend(client, userId) {
    return {
      kind: "supabase-points",
      /* 自分の残高。行がまだ無ければ 0。 */
      async fetchBalance() {
        const { data, error } = await client
          .from(POINTS_BALANCE_TABLE)
          .select("balance")
          .eq("user_id", userId)
          .maybeSingle();
        if (error) {
          throw new Error(`points balance fetch failed: ${error.message || error}`);
        }
        return data && typeof data.balance === "number" ? data.balance : 0;
      },
      /* 付与（RPC）。新しい残高を返す。同じ冪等キーの再送は
         残高を変えず、現在残高を返す（サーバー側の仕様）。 */
      async earn(reason, idempotencyKey, metadata) {
        const { data, error } = await client.rpc("earn_points", {
          p_reason: reason,
          p_idempotency_key: idempotencyKey,
          p_metadata: metadata || {},
        });
        if (error) {
          throw new Error(`earn_points failed: ${error.message || error}`);
        }
        return typeof data === "number" ? data : null;
      },
      /* 消費（RPC）。新しい残高を返す。残高不足はエラー
         （メッセージに insufficient_balance を含む）になる。 */
      async spend(reason, amount, idempotencyKey, metadata) {
        const { data, error } = await client.rpc("spend_points", {
          p_reason: reason,
          p_amount: amount,
          p_idempotency_key: idempotencyKey,
          p_metadata: metadata || {},
        });
        if (error) {
          throw new Error(`spend_points failed: ${error.message || error}`);
        }
        return typeof data === "number" ? data : null;
      },
    };
  }

  /* ---------------------------------------------------------------- */
  /* クラウド同期ストア（ローカルを土台に、ログイン中だけ同期）        */
  /* ---------------------------------------------------------------- */

  function makeCloudHistoryStore(localStore, manager, options) {
    const opts = options || {};
    const debounceMs = opts.debounceMs != null ? opts.debounceMs : 600;
    let syncTimer = null;
    let lastSyncOk = false;

    /* クラウドと突き合わせ、ローカルを統合結果に更新し、
       ローカル側の新しい変更をクラウドへ送る。未ログイン・
       失敗時は false を返し、ローカルの内容はそのまま使う。 */
    async function sync() {
      const backend = manager && manager.backend ? manager.backend() : null;
      if (!backend) { lastSyncOk = false; return false; }
      try {
        const cloud = await backend.fetchState();
        const local = await localStore._readState();
        const merged = mergeConversationStates(local, cloud);
        await localStore._writeState(merged);
        // ローカルの方が新しい／クラウドに無い会話を送る。
        for (const conv of merged.conversations) {
          const cc = cloud.convs.get(conv.id);
          if (!cc || conv.updated_at > cc.updated_at) {
            await backend.upsertConversation(conv);
          }
        }
        // クラウドにまだ無い墓標を送る。
        for (const [id, ts] of Object.entries(merged.tombstones)) {
          if (!cloud.tombstones.has(id)) {
            await backend.applyTombstone(id, ts);
          }
        }
        lastSyncOk = true;
        return true;
      } catch {
        lastSyncOk = false;
        return false;
      }
    }

    /* 保存・削除のたびの同期はデバウンスする（連続ターンで
       リクエストをまとめ、失敗しても次回に持ち越す）。 */
    function scheduleSync() {
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(() => {
        syncTimer = null;
        void sync();
      }, debounceMs);
      if (syncTimer && typeof syncTimer.unref === "function") {
        syncTimer.unref();
      }
    }

    return {
      kind: "cloud-history",
      get lastSyncOk() { return lastSyncOk; },
      syncNow: sync,
      async flushSync() {
        if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
        return sync();
      },
      async listConversations() {
        await sync();
        return localStore.listConversations();
      },
      async loadConversation(id) {
        await sync();
        return localStore.loadConversation(id);
      },
      async saveConversation(conv) {
        await localStore.saveConversation(conv);
        scheduleSync();
      },
      async deleteConversation(id) {
        await localStore.deleteConversation(id);
        scheduleSync();
      },
      /* 同期用に土台のローカルストアも公開する。 */
      _local: localStore,
    };
  }

  /* ---------------------------------------------------------------- */
  /* 公開                                                              */
  /* ---------------------------------------------------------------- */

  const CubeAccount = {
    isConfigured,
    createManager,
    makeLocalHistoryStore,
    makeCloudHistoryStore,
    makeSupabaseConvBackend,
    makeSupabasePointsBackend,
    newConversationId,
    titleFromQuestion,
    _internal: {
      CONVERSATIONS_TABLE, POINTS_BALANCE_TABLE,
      HISTORY_LOCAL_KEY, MAX_CONVERSATIONS,
      normalizeUser, emptyState, sanitizeConversation,
      parseStateBlob, serializeState, mergeConversationStates,
      pruneState,
    },
  };

  global.CubeAccount = CubeAccount;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = CubeAccount;
  }
})(typeof window !== "undefined" ? window : globalThis);
