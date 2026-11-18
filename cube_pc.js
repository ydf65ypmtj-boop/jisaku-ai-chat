"use strict";
/*
 * Cube「PC操作を見せる機能」— クライアントパネル（cube_pc.js）
 *
 * 独立モジュール。ページ本体（script.js）への依存は無く、組み込みは
 * INTEGRATION.md の最小フック（script タグ＋ボタン＋コンテナ＋初期化
 * 1行）だけで済む。他機能（記憶・アカウント・通常検索・音声）とは
 * ファイル単位で分離してあり、マージ可能な形にしてある。
 *
 * 設計（DESIGN.md）：
 * - 仮想PCの画面をステップごとに表示し、操作を平易な日本語で見せる。
 * - 成功の表示はサーバー側の外部検証が pass のときだけ。モデルの
 *   自己申告だけでは成功にしない（結果文面はサーバーが決めた
 *   result.message をそのまま出す）。
 *
 * 使い方（ブラウザ）：
 *   <script src="cube_pc.js"></script>
 *   const panel = CubePc.createPanel({
 *     container: document.getElementById("cube-pc-panel"),
 *     baseUrl: CUBE_SERVER_BASE_URL,   // 例: https://imac...ts.net:8443
 *   });
 *   panel.mount();
 * Node（テスト）：const CubePc = require("./cube_pc.js");
 */
(function (global) {

  /* 対応カテゴリ（サーバー pc_bridge.js の SUPPORTED と同期させる） */
  const CATEGORIES = [
    { id: "open_page", label: "ページを開く", example: "example.org を開いて" },
    { id: "browser_search", label: "ブラウザで検索", example: "『富士山の高さ』を検索して" },
    { id: "editor_save", label: "メモ帳に入力して保存", example: "メモ帳に『会議は午後3時からです』と書いて memo.txt に保存して" },
    { id: "editor_input", label: "メモ帳に入力", example: "メモ帳に『今日は良い天気です』と入力して" },
    { id: "folder_create", label: "フォルダ作成", example: "『shiryo-2026』フォルダを作って" },
    { id: "terminal", label: "端末でコマンド実行", example: "端末で echo OK123 を実行して" },
  ];

  /*
   * 入力時の先行判定（サーバーの classifyTask の軽量ミラー）。
   * あくまで即時フィードバック用で、最終判定はサーバーが行う。
   */
  function classifyPreview(task) {
    const t = (task || "").trim();
    if (!t) return { supported: false, reason: "empty", detail: "" };
    if (/パスワード|暗証|ログイン|サインイン|認証コード/.test(t)) {
      return { supported: false, reason: "password", detail: "パスワード入力やログインが要る操作は、今は受け付けていません。" };
    }
    if (/削除|消して|送信して|送って|インストール|アンインストール/.test(t)) {
      return { supported: false, reason: "destructive", detail: "削除・送信・インストールのような操作は、今は受け付けていません。" };
    }
    if (/フォルダ/.test(t) && /(作|作成|つく)/.test(t)) return { supported: true, category: "folder_create" };
    if (/echo\s+\S+/.test(t)) return { supported: true, category: "terminal" };
    if (/保存/.test(t) && /\.(txt|md)/.test(t)) return { supported: true, category: "editor_save" };
    if (/(メモ帳|Mousepad|エディタ)/i.test(t) && /(書い|入力|打っ)/.test(t)) return { supported: true, category: "editor_input" };
    if (/(検索|しらべ|調べ)/.test(t)) return { supported: true, category: "browser_search" };
    if (/[a-zA-Z0-9\-]+\.(com|net|org|jp|io|dev)/.test(t) && /(開|表示|アクセス|見せ)/.test(t)) return { supported: true, category: "open_page" };
    return { supported: false, reason: "out_of_scope", detail: "その操作はまだできません。今できるのは下の例のような操作です。" };
  }

  /* モデルの生操作 → 平易な日本語ラベル */
  function actionLabel(raw) {
    const s = String(raw || "").trim();
    let m;
    if ((m = s.match(/^CLICK\s+(\d+)\s+(\d+)/i))) return `(${m[1]}, ${m[2]}) をクリック`;
    if ((m = s.match(/^DOUBLE_CLICK\s+(\d+)\s+(\d+)/i))) return `(${m[1]}, ${m[2]}) をダブルクリック`;
    if ((m = s.match(/^RIGHT_CLICK\s+(\d+)\s+(\d+)/i))) return `(${m[1]}, ${m[2]}) を右クリック`;
    if ((m = s.match(/^TYPE\s+(.+)/i))) return `『${m[1]}』と入力`;
    if ((m = s.match(/^KEY\s+(.+)/i))) return `${m[1]} キーを押す`;
    if ((m = s.match(/^HOTKEY\s+(.+)/i))) return `${m[1].replace(/\+/g, "+")} を同時押し`;
    if ((m = s.match(/^SCROLL\s+(.+)/i))) return "スクロール";
    if ((m = s.match(/^WAIT\s+(\d+)/i))) return `${m[1]}秒待つ`;
    if (/^DONE/i.test(s)) return "完了を申告";
    if ((m = s.match(/^FAIL\s*(.*)/i))) return `できないと申告${m[1] ? "：" + m[1] : ""}`;
    return s;
  }

  const RESULT_CLASS = {
    verified_success: "cubepc-result--success",
    unverified: "cubepc-result--unverified",
    failed: "cubepc-result--failed",
    stopped: "cubepc-result--stopped",
    error: "cubepc-result--error",
  };

  function createPanel(options = {}) {
    const doc = options.document || global.document;
    const fetchImpl = options.fetchImpl || global.fetch.bind(global);
    const EventSourceImpl = options.EventSourceImpl || global.EventSource;
    const baseUrl = (options.baseUrl || "").replace(/\/$/, "");
    const container = options.container;
    if (!container) throw new Error("CubePc.createPanel: container is required");

    const state = {
      taskId: null, running: false, es: null,
      els: {},
    };

    function el(tag, className, text) {
      const e = doc.createElement(tag);
      if (className) e.className = className;
      if (text != null) e.textContent = text;
      return e;
    }

    function mount() {
      const root = el("div", "cubepc");
      const els = state.els;

      els.notice = el("p", "cubepc-notice",
        "Cube が仮想PCを操作します。成功の表示は、エージェントの自己申告ではなく、こちらで実際に確認できた場合だけ出ます。");
      root.appendChild(els.notice);

      // 例文チップ
      const chips = el("div", "cubepc-chips");
      for (const c of CATEGORIES) {
        const chip = el("button", "cubepc-chip", c.example);
        chip.setAttribute("type", "button");
        chip.addEventListener("click", () => {
          els.input.value = c.example;
          refreshPreview();
        });
        chips.appendChild(chip);
      }
      root.appendChild(chips);

      const form = el("div", "cubepc-form");
      els.input = el("input", "cubepc-input");
      els.input.setAttribute("type", "text");
      els.input.setAttribute("placeholder", "やってほしいことを書いてください");
      els.input.addEventListener("input", refreshPreview);
      els.run = el("button", "cubepc-run", "実行");
      els.run.setAttribute("type", "button");
      els.run.addEventListener("click", () => submit());
      els.stop = el("button", "cubepc-stop", "止める");
      els.stop.setAttribute("type", "button");
      els.stop.disabled = true;
      els.stop.addEventListener("click", () => stop());
      form.appendChild(els.input);
      form.appendChild(els.run);
      form.appendChild(els.stop);
      root.appendChild(form);

      els.preview = el("p", "cubepc-preview", "");
      root.appendChild(els.preview);

      els.state = el("p", "cubepc-state", "まだ実行していません");
      root.appendChild(els.state);

      els.screen = el("img", "cubepc-screen");
      els.screen.setAttribute("alt", "仮想PCの画面");
      root.appendChild(els.screen);

      els.log = el("ol", "cubepc-log");
      root.appendChild(els.log);

      els.result = el("div", "cubepc-result");
      els.result.hidden = true;
      root.appendChild(els.result);

      container.appendChild(root);
      state.root = root;
      return panel;
    }

    function refreshPreview() {
      const v = classifyPreview(state.els.input.value);
      state.els.preview.textContent = v.supported ? "" : (v.detail || "");
      return v;
    }

    function setRunning(running) {
      state.running = running;
      state.els.run.disabled = running;
      state.els.stop.disabled = !running;
      state.els.input.disabled = running;
    }

    function logLine(text) {
      const li = el("li", "cubepc-logline", text);
      state.els.log.appendChild(li);
      return li;
    }

    async function submit() {
      const task = state.els.input.value.trim();
      const v = refreshPreview();
      if (!v.supported) {
        showResult("unsupported", v.detail || "その操作はまだできません。");
        return;
      }
      // 前回分を片付ける
      state.els.log.textContent = "";
      state.els.result.hidden = true;
      setRunning(true);
      state.els.state.textContent = "開始しています…";
      let resp;
      try {
        resp = await fetchImpl(`${baseUrl}/pc/tasks`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task }),
        });
      } catch (e) {
        setRunning(false);
        showResult("error", "サーバーに接続できませんでした。");
        return;
      }
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 422) {
        setRunning(false);
        showResult("unsupported", data.detail || "その操作はまだできません。");
        return;
      }
      if (resp.status === 409) {
        setRunning(false);
        showResult("error", data.detail || "別のタスクを実行中です。");
        return;
      }
      if (!resp.ok || !data.taskId) {
        setRunning(false);
        showResult("error", "開始できませんでした。");
        return;
      }
      state.taskId = data.taskId;
      openEvents(data.taskId);
    }

    function openEvents(taskId) {
      if (state.es) { try { state.es.close(); } catch (_) {} }
      const es = new EventSourceImpl(`${baseUrl}/pc/tasks/${taskId}/events`);
      state.es = es;
      es.addEventListener("step", (ev) => {
        const d = JSON.parse(ev.data);
        state.els.state.textContent = `ステップ ${d.step} を実行中…`;
        for (const a of d.actions || []) logLine(`ステップ${d.step}：${actionLabel(a.raw)}`);
      });
      es.addEventListener("screenshot", (ev) => {
        const d = JSON.parse(ev.data);
        state.els.screen.src = `${baseUrl}${d.url}`;
      });
      es.addEventListener("verification", (ev) => {
        const d = JSON.parse(ev.data);
        if (d.phase === "done_reject") {
          logLine(`確認：まだ完了していないため差し戻しました（${d.evidence || ""}）`);
        }
      });
      es.addEventListener("finished", (ev) => {
        const d = JSON.parse(ev.data);
        finish(d.result);
      });
      es.addEventListener("error", () => {
        // SSE の一時切断。サーバー側はスナップショットで復元できる
        // 設計だが、自動再接続は EventSource の標準動作に任せる。
        if (state.running) state.els.state.textContent = "接続が切れました。再接続を試みています…";
      });
    }

    function finish(result) {
      setRunning(false);
      if (state.es) { try { state.es.close(); } catch (_) {} state.es = null; }
      state.els.state.textContent = "終了しました";
      showResult(result.status, result.message || "");
    }

    function showResult(kind, message) {
      const r = state.els.result;
      r.hidden = false;
      r.className = "cubepc-result " + (RESULT_CLASS[kind] || "");
      r.textContent = message;
    }

    async function stop() {
      if (!state.taskId) return;
      try {
        await fetchImpl(`${baseUrl}/pc/tasks/${state.taskId}/stop`, { method: "POST" });
        state.els.state.textContent = "止めています…";
      } catch (_) { /* 終了イベント側で確定する */ }
    }

    function destroy() {
      if (state.es) { try { state.es.close(); } catch (_) {} }
      if (state.root && state.root.parentNode) {
        state.root.parentNode.removeChild(state.root);
      }
    }

    const panel = { mount, submit, stop, destroy, _state: state };
    return panel;
  }

  const api = { createPanel, classifyPreview, actionLabel, CATEGORIES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CubePc = api;
})(typeof window !== "undefined" ? window : globalThis);
