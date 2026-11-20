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
 * - ライブ画面と乗っ取り（2026-11-20 追加）：実行中は実行側が
 *   ライブ画面（step "live" のスクリーンショット）を流し続ける。
 *   乗っ取りは専用の「操作する」ボタンからだけ行う（画面タップ
 *   では切り替わらない）。ボタンを押すと AI を止めてユーザーが
 *   直接操作でき、操作中は画面タップがクリックとして送られる。
 *   操作モードの正本はサーバーの control イベントで、手入力は
 *   mode:"user" の間だけ橋渡しが通す（AI と競合させない）。
 * - 実行前確認（2026-11-20 追加）：削除・送信は実行側が直前で
 *   止まり、confirm イベントで確認を求める。パネルは要約と
 *   「実行する／やめる」ボタンを出し、応答は input 経由で返す。
 *   確定は confirm_resolved イベントが正本で、応答が無ければ
 *   実行側は実行せずに終わる（安全側）。
 * - ログイン（金庫方式、2026-11-20 追加）：専用フォーム（URL／
 *   ID／パスワード欄、パスワードは password 型）に本人が入力し、
 *   「ログインを実行」を押す行為そのものを承認として
 *   POST /pc/login で開始する。パスワードは送信直後に欄から
 *   消し、チャットやタスク文には一切書かせない。実行はモデル
 *   ではなく実行側ドライバが直接行い、成否はログイン後の
 *   ページ状態の外部確認で決まる。
 * - 文書作成（お題だけ、2026-11-20 追加）：お題・目的だけを渡すと、
 *   文面はサーバー側で Ultra が作り（PC専門モデルは作文しない）、
 *   仮想PCに .txt で保存される。生成文は結果に表示される。
 *   お題が読み取れないときは聞き返しの文面を出す。
 * - プログラミング（2026-11-20 追加）：作るプログラムの内容だけを
 *   渡すと、コードはサーバー側で Ultra が作り（PC専門モデルは
 *   コーディングしない）、仮想PCの専用フォルダに保存して実行され、
 *   コードと実行結果が結果に表示される。対応は Python のみ。
 *   内容が読み取れないときは聞き返しの文面を出す。
 * - 軽いWebアプリ（2026-11-20 追加）：作るアプリの内容だけを渡すと、
 *   HTML・CSS・JavaScript を1ファイルに収めた軽いアプリをサーバー
 *   側で Ultra が作り（PC専門モデルはコーディングしない）、仮想PC
 *   の専用フォルダに保存してブラウザで開き、表示を確認する。
 *   サーバーが要るアプリは対象外と正直に返す。内容が読み取れない
 *   ときは聞き返しの文面を出す。
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

  /* 対応カテゴリ（サーバー pc_bridge.js の SUPPORTED と同期させる）。
     login はタスク文では実行できない（金庫方式の専用フォーム
     からのみ）ため formOnly とし、チップはフォームへの案内に使う。 */
  const CATEGORIES = [
    { id: "open_page", label: "ページを開く", example: "example.org を開いて" },
    { id: "browser_search", label: "ブラウザで検索", example: "『富士山の高さ』を検索して" },
    { id: "editor_save", label: "メモ帳に入力して保存", example: "メモ帳に『会議は午後3時からです』と書いて memo.txt に保存して" },
    { id: "doc_compose", label: "お題だけで文書作成", example: "『地域の祭り』についての文書を作って" },
    { id: "code_run", label: "プログラム作成・実行", example: "『1から100までの合計を計算する』プログラムを作って" },
    { id: "web_app", label: "軽いWebアプリ作成", example: "『クリックカウンター』のWebアプリを作って" },
    { id: "editor_input", label: "メモ帳に入力", example: "メモ帳に『今日は良い天気です』と入力して" },
    { id: "folder_create", label: "フォルダ作成", example: "『shiryo-2026』フォルダを作って" },
    { id: "terminal", label: "端末でコマンド実行", example: "端末で echo OK123 を実行して" },
    { id: "file_delete", label: "ファイルを削除（実行前に確認）", example: "『memo.txt』を削除して" },
    { id: "form_send", label: "フォームを送信（実行前に確認）", example: "example.com のフォームに『よろしくお願いします』と入力して送信して" },
    { id: "login", label: "ログイン（金庫方式）", example: "", formOnly: true },
  ];

  /* 実画面の大きさ（体の Xvfb は 1280x800 固定） */
  const SCREEN_W = 1280;
  const SCREEN_H = 800;

  /*
   * 表示中の画面要素の矩形とタップ位置から、実画面（1280x800）の
   * 座標へ換算する。矩形が取れない／大きさゼロなら null。
   * 換算は橋渡し側の検証範囲（x:0〜1279、y:0〜799）に丸める。
   */
  function toScreenCoords(rect, clientX, clientY) {
    if (!rect || !rect.width || !rect.height) return null;
    const clamp = (v, max) => Math.min(max, Math.max(0, v));
    return {
      x: clamp(Math.round((clientX - rect.left) / rect.width * SCREEN_W), SCREEN_W - 1),
      y: clamp(Math.round((clientY - rect.top) / rect.height * SCREEN_H), SCREEN_H - 1),
    };
  }

  /*
   * ライブ操作の入力を橋渡しへ送る（POST /pc/tasks/:id/input）。
   * 戻り値： {status, ok, data}。fetch 自体の失敗は throw する
   * （呼び出し側で「接続できませんでした」と正直に出すため）。
   */
  async function postInput(baseUrl, taskId, cmd, fetchImpl) {
    const f = fetchImpl || global.fetch.bind(global);
    const base = (baseUrl || "").replace(/\/$/, "");
    const resp = await f(`${base}/pc/tasks/${taskId}/input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd),
    });
    const data = await resp.json().catch(() => ({}));
    return { status: resp.status, ok: resp.ok, data };
  }

  /* 乗っ取り中の基本キーのボタン定義（表示ラベル → 橋渡しのキー名） */
  const KEY_BUTTONS = [
    ["Enter", "Return"], ["Backspace", "BackSpace"], ["Tab", "Tab"],
    ["Esc", "Escape"], ["↑", "Up"], ["↓", "Down"], ["←", "Left"],
    ["→", "Right"], ["Space", "space"],
  ];

  /* 削除対象のファイル名（サーバー extractDeleteTarget の軽量ミラー） */
  function deleteNamePreview(t) {
    const safe = (n) => /^[A-Za-z0-9_.\-]+$/.test(n) && n !== "." && n !== ".." && !n.includes("..");
    const full = t.match(/\/home\/user\/([A-Za-z0-9_.\-]+)/);
    if (full) {
      const rest = t.slice(t.indexOf(full[0]) + full[0].length);
      if (rest.startsWith("/")) return null;
      return safe(full[1]) ? full[1] : null;
    }
    const q = t.match(/[『「]([^』」]+)[』」]/);
    if (q && safe(q[1])) return q[1];
    const bare = t.match(/([A-Za-z0-9_.\-]+\.(?:txt|md))/);
    return bare && safe(bare[1]) ? bare[1] : null;
  }

  /* 文書作成のお題（サーバー extractDocTopic の軽量ミラー）。
     取れなければ null。「『…』という文章を書いて」のような本文
     そのものを指す言い方はお題として扱わない。 */
  function docTopicPreview(t) {
    let topic = null;
    if (!/[』」]という/.test(t)) {
      const q = t.match(/[『「]([^『」]{1,60})[』」]/);
      if (q) topic = q[1];
    }
    if (!topic) {
      const m = t.match(/(.{1,60}?)(?:について|に関する|をテーマに)(?:の)?(?:文書|レポート|文章)/);
      if (m) topic = m[1];
    }
    if (!topic) {
      const m = t.match(/(.{1,60}?)の(?:文書|レポート|文章)を/);
      if (m) topic = m[1];
    }
    if (!topic) return null;
    topic = topic
      .replace(/^[『「]/, "").replace(/[』」]$/, "")
      .replace(/(について|に関する|をテーマに|の)+$/, "")
      .replace(/^(メモ帳に|仮想PCで|PCで|Cubeに)/, "")
      .trim();
    if (!topic || topic.length > 60) return null;
    if (/(文書|レポート|文章)$/.test(topic)) return null;
    if (/(作って|作成して|書いて|作る|作成|書く)$/.test(topic)) return null;
    return topic;
  }

  /* プログラムの仕様（サーバー extractCodeSpec の軽量ミラー）。
     取れなければ null。他言語の明示は "other" を返す。 */
  function codeSpecPreview(t) {
    if (!/Python|パイソン/i.test(t) &&
        /JavaScript|ジャバスクリプト|TypeScript|Java(?!Script)|C\+\+|C#|Ruby|ルビー|PHP|Go言語|Rust|Swift|Kotlin|Perl/i.test(t)) {
      return "other";
    }
    const q = t.match(/[『「]([^』」]+)[』」]/);
    if (q) return q[1].trim() || null;
    let s = t.replace(/^(仮想PCで|PCで|パソコンで|Cubeに|プログラムで)\s*/, "");
    s = s.replace(/(の)?(プログラム|コード|スクリプト)(を)?(作って|書いて|作成して|作る|書く|つくって|作成|作製).*$/, "");
    s = s.replace(/(プログラム|コード|スクリプト)(で|を)?\s*$/, "");
    s = s.replace(/^(Pythonで|パイソンで)\s*/i, "");
    s = s.replace(/(ください|お願いします)。?\s*$/, "");
    s = s.trim();
    if (!s || s.length > 200) return null;
    if (/^(プログラム|コード|スクリプト)$/.test(s)) return null;
    return s;
  }

  /* Webアプリの仕様（サーバー extractWebAppSpec の軽量ミラー）。
     取れなければ null。サーバーが要るアプリは "server" を返す。 */
  function webAppSpecPreview(t) {
    if (/サーバー|データベース|オンライン対戦|オンライン共有|掲示板|会員登録|決済|課金/.test(t)) {
      return "server";
    }
    const q = t.match(/[『「]([^』」]+)[』」]/);
    if (q) return q[1].trim() || null;
    let s = t.replace(/^(仮想PCで|PCで|パソコンで|Cubeに)\s*/, "");
    s = s.replace(/^(軽い)?(Webアプリ|ウェブアプリ|Webページ|ウェブページ)(で|として)\s*/, "");
    s = s.replace(/(の)?(軽い)?(Webアプリ|ウェブアプリ|Webページ|ウェブページ|ブラウザアプリ|ホームページ|アプリ|ページ)(を)?(作って|書いて|作成して|作る|書く|つくって|作成|作製).*$/, "");
    s = s.replace(/(Webアプリ|ウェブアプリ|アプリ)(で|を)?\s*$/, "");
    s = s.replace(/(を)?(作って|書いて|作成して|つくって)(ください)?。?\s*$/, "");
    s = s.replace(/(ください|お願いします)。?\s*$/, "");
    s = s.trim();
    if (!s || s.length > 200) return null;
    if (/^(Webアプリ|ウェブアプリ|アプリ|ページ)$/i.test(s)) return null;
    return s;
  }

  /*
   * 入力時の先行判定（サーバーの classifyTask の軽量ミラー）。
   * あくまで即時フィードバック用で、最終判定はサーバーが行う。
   * 削除・送信は実行前確認つきで対応（2026-11-20）。対象や宛先が
   * 特定できないときは、サーバーと同じく聞き返しの文面を出す。
   */
  function classifyPreview(task) {
    const t = (task || "").trim();
    if (!t) return { supported: false, reason: "empty", detail: "" };
    if (/パスワード|暗証|ログイン|サインイン|認証コード/.test(t)) {
      return { supported: false, reason: "password", detail: "パスワードやログインは、文章からは受け付けません。下のログインフォーム（金庫方式）から実行してください。チャットにパスワードを書かないでください。" };
    }
    if (/インストール|アンインストール/.test(t)) {
      return { supported: false, reason: "destructive", detail: "インストールのような操作は、今は受け付けていません。" };
    }
    if (/削除|消して|消す/.test(t)) {
      if (/フォルダ/.test(t)) {
        return { supported: false, reason: "destructive", detail: "フォルダの削除は受け付けていません。削除できるのは仮想PCの /home/user にあるファイルだけです。" };
      }
      if (!deleteNamePreview(t)) {
        return { supported: false, reason: "need_detail", detail: "削除するファイル名を指定してください（例：『memo.txt』を削除して）。実行の前に確認画面が出ます。" };
      }
      return { supported: true, category: "file_delete" };
    }
    if (/送信|投稿|送って/.test(t)) {
      if (!/(https?:\/\/|localhost|[a-zA-Z0-9\-]+\.(?:com|net|org|jp|io|dev|co\.jp|ne\.jp))/.test(t)) {
        return { supported: false, reason: "need_detail", detail: "送信先のページ（URL）を指定してください。送信ボタンを押す前に確認画面が出ます。" };
      }
      return { supported: true, category: "form_send" };
    }
    if (/フォルダ/.test(t) && /(作|作成|つく)/.test(t)) return { supported: true, category: "folder_create" };
    if (/(Webアプリ|ウェブアプリ|Webページ|ウェブページ|ブラウザアプリ|ホームページ)/i.test(t) && /(作っ|作成|書い|作る|書く|つくっ)/.test(t) &&
        !(/(文書|レポート|文章)/.test(t) && !/(Webアプリ|ウェブアプリ)(を)?(作っ|作成|つくっ)/.test(t))) {
      if (/作り方|とは|使い方|やり方|方法/.test(t)) {
        return { supported: false, reason: "out_of_scope", detail: "Webアプリの作り方の説明は扱いません。作ってほしい軽いWebアプリの内容を指定してください（例：『クリックカウンター』のWebアプリを作って）。" };
      }
      const wspec = webAppSpecPreview(t);
      if (wspec === "server") {
        return { supported: false, reason: "needs_server", detail: "サーバーやデータベースが要るアプリは今は作れません。作れるのは、ブラウザだけで動く軽いWebアプリ（HTML・CSS・JavaScript を1ファイルに収めたもの）だけです。" };
      }
      if (wspec) return { supported: true, category: "web_app" };
      return { supported: false, reason: "need_detail", detail: "作るWebアプリの内容を指定してください（例：『クリックカウンター』のWebアプリを作って）。アプリはUltraが書き、仮想PCに保存してブラウザで開き、表示を確認します。" };
    }
    if (/(プログラム|コード|スクリプト)/.test(t) && /(作っ|書い|作成|作る|書く|つくっ)/.test(t) &&
        !(/メモ帳|Mousepad|エディタ/i.test(t) && /保存/.test(t) && /[A-Za-z0-9_.\-]+\.(?:txt|md)/.test(t))) {
      if (/作り方|とは|使い方|やり方|方法/.test(t)) {
        return { supported: false, reason: "out_of_scope", detail: "プログラムの作り方の説明は扱いません。作って実行してほしいプログラムの内容を指定してください（例：『1から100までの合計を計算する』プログラムを作って）。" };
      }
      const spec = codeSpecPreview(t);
      if (spec === "other") {
        return { supported: false, reason: "unsupported_lang", detail: "プログラムの作成・実行は今は Python だけ対応しています。Python でよければ、作るプログラムの内容を指定してください。" };
      }
      if (spec) return { supported: true, category: "code_run" };
      return { supported: false, reason: "need_detail", detail: "作るプログラムの内容を指定してください（例：『1から100までの合計を計算する』プログラムを作って）。プログラムはUltraが書き、仮想PCに保存して実行し、結果を表示します。" };
    }
    if (/(文書|レポート|文章)/.test(t) && /(作っ|作成|書い|作る|書く|つくっ)/.test(t)) {
      if (docTopicPreview(t)) return { supported: true, category: "doc_compose" };
      return { supported: false, reason: "need_detail", detail: "お題を指定してください（例：『地域の祭り』についての文書を作って）。文書はUltraが書き、仮想PCに保存します。" };
    }
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
      // ライブ操作：操作モードはサーバーの control イベントが正本。
      // "ai"（AI操作中）／"user"（ユーザー操作中・AI停止）。
      controlMode: "ai", controlBusy: false,
      // 実行前確認（削除・送信）：サーバーの confirm イベントが正本。
      confirm: null, confirmBusy: false,
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

      // 例文チップ（login は専用フォームへの案内チップ）
      const chips = el("div", "cubepc-chips");
      for (const c of CATEGORIES) {
        const chip = el("button", "cubepc-chip", c.formOnly ? c.label : c.example);
        chip.setAttribute("type", "button");
        chip.addEventListener("click", () => {
          if (c.formOnly) {
            if (els.loginUrl) els.loginUrl.focus();
            return;
          }
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

      // ログインフォーム（金庫方式）：本人がここに入力して
      // 「ログインを実行」を押す行為そのものが実行の承認になる。
      // パスワードは password 型の欄で受け、送信直後に欄から消す。
      // チャットやタスク文にパスワードを書く経路は作らない。
      const loginBox = el("div", "cubepc-vault");
      const loginTitle = el("p", "cubepc-vault-title", "ログイン（金庫方式）");
      loginBox.appendChild(loginTitle);
      const loginNote = el("p", "cubepc-vault-note",
        "パスワードはAIに見せず、実行側が直接入力します。" +
        "チャットにパスワードを書かないでください。" +
        "「ログインを実行」を押すと、そのままログインを実行します。");
      loginBox.appendChild(loginNote);
      const loginFields = el("div", "cubepc-vault-fields");
      els.loginUrl = el("input", "cubepc-vault-input");
      els.loginUrl.setAttribute("type", "text");
      els.loginUrl.setAttribute("placeholder", "サイトのURL（https://…）");
      els.loginUrl.setAttribute("autocomplete", "url");
      els.loginUser = el("input", "cubepc-vault-input");
      els.loginUser.setAttribute("type", "text");
      els.loginUser.setAttribute("placeholder", "ID・メールアドレス");
      els.loginUser.setAttribute("autocomplete", "username");
      els.loginPass = el("input", "cubepc-vault-input");
      els.loginPass.setAttribute("type", "password");
      els.loginPass.setAttribute("placeholder", "パスワード");
      els.loginPass.setAttribute("autocomplete", "current-password");
      els.loginRun = el("button", "cubepc-vault-run", "ログインを実行");
      els.loginRun.setAttribute("type", "button");
      els.loginRun.addEventListener("click", () => submitLogin());
      loginFields.appendChild(els.loginUrl);
      loginFields.appendChild(els.loginUser);
      loginFields.appendChild(els.loginPass);
      loginFields.appendChild(els.loginRun);
      loginBox.appendChild(loginFields);
      root.appendChild(loginBox);

      els.state = el("p", "cubepc-state", "まだ実行していません");
      root.appendChild(els.state);

      els.screen = el("img", "cubepc-screen");
      els.screen.setAttribute("alt", "仮想PCの画面");
      els.screen.addEventListener("click", (ev) => onScreenTap(ev));
      root.appendChild(els.screen);

      // ライブ操作（乗っ取り）コントロール
      const controls = el("div", "cubepc-controls");
      els.takeover = el("button", "cubepc-takeover", "操作する");
      els.takeover.setAttribute("type", "button");
      els.takeover.addEventListener("click", () => requestTakeover());
      els.release = el("button", "cubepc-release", "AIに戻す");
      els.release.setAttribute("type", "button");
      els.release.hidden = true;
      els.release.addEventListener("click", () => requestRelease());
      els.controlState = el("span", "cubepc-control-state", "");
      controls.appendChild(els.takeover);
      controls.appendChild(els.release);
      controls.appendChild(els.controlState);
      root.appendChild(controls);

      const keys = el("div", "cubepc-keys");
      keys.hidden = true;
      els.keys = keys;
      for (const [label, key] of KEY_BUTTONS) {
        const b = el("button", "cubepc-key", label);
        b.setAttribute("type", "button");
        b.addEventListener("click", () => sendInput({ type: "key", key }));
        keys.appendChild(b);
      }
      root.appendChild(keys);

      // 実行前確認（削除・送信）の確認ボックス
      const confirmBox = el("div", "cubepc-confirm");
      confirmBox.hidden = true;
      els.confirmBox = confirmBox;
      els.confirmText = el("p", "cubepc-confirm-text", "");
      confirmBox.appendChild(els.confirmText);
      const confirmBtns = el("div", "cubepc-confirm-btns");
      els.confirmOk = el("button", "cubepc-confirm-ok", "実行する");
      els.confirmOk.setAttribute("type", "button");
      els.confirmOk.addEventListener("click", () => answerConfirm("approve"));
      els.confirmCancel = el("button", "cubepc-confirm-cancel", "やめる");
      els.confirmCancel.setAttribute("type", "button");
      els.confirmCancel.addEventListener("click", () => answerConfirm("deny"));
      confirmBtns.appendChild(els.confirmOk);
      confirmBtns.appendChild(els.confirmCancel);
      confirmBox.appendChild(confirmBtns);
      root.appendChild(confirmBox);

      const typeRow = el("div", "cubepc-type-row");
      typeRow.hidden = true;
      els.typeRow = typeRow;
      els.typeInput = el("input", "cubepc-type-input");
      els.typeInput.setAttribute("type", "text");
      els.typeInput.setAttribute("placeholder", "入力する文字");
      const typeBtn = el("button", "cubepc-type-btn", "入力する");
      typeBtn.setAttribute("type", "button");
      typeBtn.addEventListener("click", () => {
        const text = els.typeInput.value;
        if (text) {
          sendInput({ type: "type", text });
          els.typeInput.value = "";
        }
      });
      typeRow.appendChild(els.typeInput);
      typeRow.appendChild(typeBtn);
      root.appendChild(typeRow);

      els.log = el("ol", "cubepc-log");
      root.appendChild(els.log);

      els.result = el("div", "cubepc-result");
      els.result.hidden = true;
      root.appendChild(els.result);

      container.appendChild(root);
      state.root = root;
      updateControlUi();
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
      if (state.els.loginRun) {
        state.els.loginRun.disabled = running;
        state.els.loginUrl.disabled = running;
        state.els.loginUser.disabled = running;
        state.els.loginPass.disabled = running;
      }
      updateControlUi();
    }

    /* 前回分の表示を片付けて、新しい実行の開始状態にする。 */
    function resetRunUi() {
      state.els.log.textContent = "";
      state.els.result.hidden = true;
      state.els.confirmBox.hidden = true;
      state.confirm = null;
      state.confirmBusy = false;
      state.controlMode = "ai";
      state.controlBusy = false;
    }

    /*
     * ログイン（金庫方式）の開始。フォームの値を POST /pc/login
     * へ送る。パスワード欄は送信を開始した直後に空にする（結果を
     * 待たない）。開始できたら通常タスクと同じ SSE 表示に繋ぐ。
     */
    async function submitLogin() {
      if (state.running) return;
      const els = state.els;
      const url = els.loginUrl.value.trim();
      const userId = els.loginUser.value;
      const password = els.loginPass.value;
      if (!url || !userId || !password) {
        showResult("unsupported", "URL・ID・パスワードを全部入力してください。");
        return;
      }
      resetRunUi();
      setRunning(true);
      els.state.textContent = "ログインを開始しています…";
      let respPromise;
      try {
        respPromise = fetchImpl(`${baseUrl}/pc/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, userId, password }),
        });
      } catch (e) {
        els.loginPass.value = "";
        setRunning(false);
        showResult("error", "サーバーに接続できませんでした。");
        return;
      }
      // パスワードは送信を開始したら直ちに欄から消す。
      els.loginPass.value = "";
      let resp;
      try {
        resp = await respPromise;
      } catch (e) {
        setRunning(false);
        showResult("error", "サーバーに接続できませんでした。");
        return;
      }
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 400 || resp.status === 422) {
        setRunning(false);
        showResult("unsupported", data.detail || "入力を確認してください。");
        return;
      }
      if (resp.status === 409) {
        setRunning(false);
        showResult("error", data.detail || "別のタスクを実行中です。");
        return;
      }
      if (!resp.ok || !data.taskId) {
        setRunning(false);
        showResult("error", "ログインを開始できませんでした。");
        return;
      }
      state.taskId = data.taskId;
      openEvents(data.taskId);
    }

    /* ---- ライブ操作（乗っ取り） ---- */

    function updateControlUi() {
      const els = state.els;
      if (!els.takeover) return;
      const mode = state.controlMode;
      els.takeover.hidden = mode === "user";
      els.takeover.disabled = !state.running || mode === "user";
      els.release.hidden = mode !== "user";
      els.release.disabled = !state.running;
      els.keys.hidden = mode !== "user";
      els.typeRow.hidden = mode !== "user";
      els.controlState.textContent = !state.running ? "" :
        mode === "user"
          ? "あなたが操作中です（AIは止まっています）。画面をタップするとクリックできます。"
          : state.controlBusy
            ? "AIを止めています…"
            : "AIが操作中です。自分で操作したいときは「操作する」を押してください。";
    }

    async function sendInput(cmd) {
      if (!state.taskId) return;
      try {
        const r = await postInput(baseUrl, state.taskId, cmd, fetchImpl);
        if (!r.ok) {
          state.els.controlState.textContent =
            (r.data && r.data.detail) || "入力を受け付けられませんでした。";
        }
      } catch (e) {
        state.els.controlState.textContent = "サーバーに接続できませんでした。";
      }
    }

    async function requestTakeover() {
      if (!state.taskId || state.controlMode === "user") return;
      state.controlBusy = true;
      updateControlUi();
      await sendInput({ type: "takeover" });
      // 実際の停止は実行側の control イベントで確定する。一定時間
      // 来なければ「切り替え中」を解除する（固まらせない）。
      const t = setTimeout(() => {
        if (state.controlMode !== "user") {
          state.controlBusy = false;
          updateControlUi();
        }
      }, 10000);
      if (t && typeof t.unref === "function") t.unref();
    }

    async function requestRelease() {
      if (!state.taskId) return;
      await sendInput({ type: "release" });
    }

    function onScreenTap(ev) {
      if (!state.running || !state.taskId) return;
      if (!ev || typeof ev.clientX !== "number") return;
      const rect = typeof state.els.screen.getBoundingClientRect === "function"
        ? state.els.screen.getBoundingClientRect() : null;
      const pt = toScreenCoords(rect, ev.clientX, ev.clientY);
      if (!pt) return;
      if (state.controlMode === "user") {
        sendInput({ type: "click", x: pt.x, y: pt.y });
      }
      // AI操作中のタップは何もしない（乗っ取りは「操作する」
      // ボタンからだけ。誤タップで AI を止めないため）
    }

    function onControlEvent(d) {
      if (!d || (d.mode !== "user" && d.mode !== "ai")) return;
      state.controlMode = d.mode;
      state.controlBusy = false;
      updateControlUi();
    }

    /* ---- 実行前確認（削除・送信のゲート） ---- */

    function onConfirmEvent(d) {
      if (!d || !d.confirmId) return;
      state.confirm = d;
      state.confirmBusy = false;
      const els = state.els;
      els.confirmText.textContent =
        (d.summary || "この操作を実行しますか？") +
        (d.detail ? `（${d.detail}）` : "");
      els.confirmOk.disabled = false;
      els.confirmCancel.disabled = false;
      els.confirmBox.hidden = false;
      els.state.textContent = "確認待ちです。内容を見て「実行する」か「やめる」を選んでください。";
    }

    function onConfirmResolved(d) {
      state.confirm = null;
      state.confirmBusy = false;
      state.els.confirmBox.hidden = true;
      const decision = d && d.decision;
      logLine(decision === "approve"
        ? "確認：実行します"
        : decision === "deny"
          ? "確認：キャンセルしました（実行しません）"
          : decision === "timeout"
            ? "確認：返事がなかったため実行しませんでした"
            : "確認：終了しました");
    }

    async function answerConfirm(decision) {
      if (!state.taskId || !state.confirm || state.confirmBusy) return;
      state.confirmBusy = true;
      state.els.confirmOk.disabled = true;
      state.els.confirmCancel.disabled = true;
      try {
        const r = await postInput(baseUrl, state.taskId, {
          type: "confirm",
          confirmId: state.confirm.confirmId,
          decision,
        }, fetchImpl);
        if (!r.ok) {
          state.confirmBusy = false;
          state.els.confirmOk.disabled = false;
          state.els.confirmCancel.disabled = false;
          state.els.confirmText.textContent =
            (r.data && r.data.detail) || "確認の応答を受け付けられませんでした。";
        }
        // 成功時は実行側の confirm_resolved イベントでボックスを閉じる
      } catch (e) {
        state.confirmBusy = false;
        state.els.confirmOk.disabled = false;
        state.els.confirmCancel.disabled = false;
        state.els.confirmText.textContent = "サーバーに接続できませんでした。";
      }
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
      resetRunUi();
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
        // ライブ画面は URL が同じまま中身が差し替わるため、
        // キャッシュ避けのクエリを付ける（ステップ画面は URL が変わる）。
        const bust = d.step === "live" ? `?t=${Date.now()}` : "";
        state.els.screen.src = `${baseUrl}${d.url}${bust}`;
      });
      es.addEventListener("control", (ev) => {
        onControlEvent(JSON.parse(ev.data));
      });
      es.addEventListener("confirm", (ev) => {
        onConfirmEvent(JSON.parse(ev.data));
      });
      es.addEventListener("confirm_resolved", (ev) => {
        onConfirmResolved(JSON.parse(ev.data));
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
      state.confirm = null;
      state.confirmBusy = false;
      state.els.confirmBox.hidden = true;
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

    const panel = { mount, submit, submitLogin, stop, destroy, _state: state };
    return panel;
  }

  const api = {
    createPanel, classifyPreview, actionLabel, CATEGORIES,
    toScreenCoords, postInput, deleteNamePreview, codeSpecPreview,
    webAppSpecPreview,
    SCREEN_W, SCREEN_H,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CubePc = api;
})(typeof window !== "undefined" ? window : globalThis);
