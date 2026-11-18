"use strict";
/*
 * Cube 電話デモモジュール (cube_call_demo.js)
 *
 * Cube の画面内で「Cube が電話をかける実演」を見せる独立モジュール。
 * 相手は実際の電話回線ではなく、台本シナリオの店員役 (シミュレーション)。
 * その旨はパネル上部に常時表示する (デモであることを隠さない)。
 *
 * 流れ:
 *   シナリオ選択 -> サーバー橋渡し (call_demo_bridge.py) が専門モデル +
 *   店員シミュレータで通話を実行 -> ターンをライブ表示 -> 終了後に
 *   要約・確定スロット・外部採点 (測定と同一の score_run) の結果を表示。
 *   成否はモデルの自己申告ではなく、サーバー側の外部採点の結果をそのまま出す。
 *
 * 使い方 (ブラウザ):
 *   <script src="cube_call_demo.js"></script>
 *   CubeCallDemo.createPanel(document.getElementById('call-demo'), {
 *     baseUrl: CubeServerBaseUrl,          // script.js の CUBE_SERVER_BASE_URL
 *     voice: { speakTurn(who, text) {} },  // 任意: 音声会話モジュールへの橋渡し
 *   });
 * Node (テスト):
 *   const CubeCallDemo = require("./cube_call_demo.js");
 */
(function (global) {

  /* ---------------------------------------------------------------- */
  /* 定数                                                              */
  /* ---------------------------------------------------------------- */

  var HONESTY_NOTICE =
    "これはデモ (実演) です。実際の電話回線には発信していません。" +
    "Cube が、台本シナリオの店員役 (シミュレーション) に電話をかける様子を、" +
    "この画面の中で再現しています。";
  var SCORE_NOTE =
    "成否は、電話専門モデルの測定と同じ外部採点 (score_run) による判定です。" +
    "Cube の自己申告ではありません。失敗した項目もそのまま表示します。";
  var CHECK_LABELS = {
    intro: "名乗り", ivr: "IVR対応", reask: "聞き返し", slots: "用件の取り違えなし",
    readback: "復唱確認", phone: "電話番号を作らない", offscript: "台本外対応",
    summary: "要約の正確さ", converge: "通話の収束"
  };
  var DEFAULT_POLL_MS = 900;

  var CSS = [
    ".ccd-root{font-family:inherit;border:1px solid #d8dee9;border-radius:12px;padding:14px;background:#fbfcfe;color:#1c2733}",
    ".ccd-notice{background:#fff7e0;border:1px solid #e6c65c;border-radius:8px;padding:8px 10px;font-size:13px;line-height:1.5}",
    ".ccd-scenarios{display:grid;gap:8px;margin-top:12px}",
    ".ccd-card{border:1px solid #d8dee9;border-radius:10px;padding:10px 12px;background:#fff;text-align:left;width:100%}",
    ".ccd-card h4{margin:0 0 4px;font-size:15px}",
    ".ccd-card p{margin:0 0 8px;font-size:13px;color:#43536a;line-height:1.5}",
    ".ccd-card button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:7px 14px;font-size:14px;cursor:pointer}",
    ".ccd-card button:disabled{background:#9db8e8;cursor:default}",
    ".ccd-status{margin:12px 0 6px;font-size:13px;color:#43536a}",
    ".ccd-transcript{display:grid;gap:6px;max-height:46vh;overflow-y:auto;padding:4px}",
    ".ccd-turn{max-width:88%;padding:8px 10px;border-radius:10px;font-size:14px;line-height:1.55;white-space:pre-wrap;word-break:break-word}",
    ".ccd-turn.ai{justify-self:end;background:#dbeafe}",
    ".ccd-turn.staff{justify-self:start;background:#eceff3}",
    ".ccd-turn .ccd-who{display:block;font-size:11px;color:#5b6b82;margin-bottom:2px}",
    ".ccd-result{margin-top:12px;border-top:1px solid #d8dee9;padding-top:10px}",
    ".ccd-result h4{margin:8px 0 4px;font-size:14px}",
    ".ccd-result p{font-size:14px;line-height:1.6;margin:0}",
    ".ccd-slots{border-collapse:collapse;font-size:14px}",
    ".ccd-slots th,.ccd-slots td{border:1px solid #d8dee9;padding:4px 10px;text-align:left}",
    ".ccd-slots th{background:#eef2f7;font-weight:600}",
    ".ccd-badge{display:inline-block;border-radius:999px;padding:3px 12px;font-size:13px;font-weight:700}",
    ".ccd-badge.ok{background:#dcfce7;color:#166534}",
    ".ccd-badge.ng{background:#fee2e2;color:#991b1b}",
    ".ccd-fail{font-size:13px;color:#991b1b;margin:4px 0 0}",
    ".ccd-note{font-size:12px;color:#5b6b82;line-height:1.5}",
    ".ccd-error{background:#fee2e2;border:1px solid #f0a3a3;border-radius:8px;padding:8px 10px;font-size:13px}"
  ].join("\n");

  /* ---------------------------------------------------------------- */
  /* 小物 (純粋関数: テスト対象)                                        */
  /* ---------------------------------------------------------------- */

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function slotRows(slots) {
    if (!slots) return [];
    return Object.keys(slots).map(function (k) { return [k, String(slots[k])]; });
  }

  function checksView(result) {
    if (!result) return { ok: false, label: "", failures: [] };
    var failures = (result.failures || []).slice();
    return {
      ok: result.success === true,
      label: result.success === true ? "全項目合格" : "未達の項目があります",
      failures: failures
    };
  }

  /*
   * イベント列をパネルの状態に畳み込む reducer。
   * state: {phase: 'idle'|'dialing'|'in_call'|'done'|'error',
   *         turns: [{who, text, stage}], result, error, title}
   */
  function reduceEvents(state, events) {
    var next = {
      phase: state.phase, turns: state.turns.slice(),
      result: state.result, error: state.error, title: state.title
    };
    (events || []).forEach(function (ev) {
      if (ev.type === "call_started") {
        next.phase = "in_call";
        next.title = ev.title || "";
      } else if (ev.type === "turn") {
        next.turns.push({ who: ev.who, text: ev.text, stage: ev.stage });
      } else if (ev.type === "call_ended") {
        next.phase = "done";
        next.result = ev.result || null;
      } else if (ev.type === "error") {
        next.phase = "error";
        next.error = ev.message || "エラーが発生しました";
      }
    });
    return next;
  }

  function initialState() {
    return { phase: "idle", turns: [], result: null, error: null, title: "" };
  }

  /* ---------------------------------------------------------------- */
  /* パネル                                                            */
  /* ---------------------------------------------------------------- */

  function ensureStyle(doc) {
    if (doc.getElementById("ccd-style")) return;
    var st = doc.createElement("style");
    st.id = "ccd-style";
    st.textContent = CSS;
    doc.head.appendChild(st);
  }

  function createPanel(root, opts) {
    opts = opts || {};
    var baseUrl = String(opts.baseUrl || "").replace(/\/+$/, "");
    var fetchFn = opts.fetchFn || (global.fetch ? global.fetch.bind(global) : null);
    var pollMs = opts.pollMs || DEFAULT_POLL_MS;
    var voice = opts.voice || null;
    var doc = root.ownerDocument || global.document;
    ensureStyle(doc);

    var state = initialState();
    var cursor = 0;
    var sessionId = null;
    var timer = null;
    var seenTurns = 0;

    root.innerHTML =
      '<div class="ccd-root">' +
      '<div class="ccd-notice">' + escapeHtml(HONESTY_NOTICE) + "</div>" +
      '<div class="ccd-scenarios"></div>' +
      '<div class="ccd-status">シナリオを選んでください。</div>' +
      '<div class="ccd-transcript"></div>' +
      '<div class="ccd-result-slot"></div>' +
      "</div>";
    var elScenarios = root.querySelector(".ccd-scenarios");
    var elStatus = root.querySelector(".ccd-status");
    var elTranscript = root.querySelector(".ccd-transcript");
    var elResult = root.querySelector(".ccd-result-slot");

    function statusText() {
      if (state.phase === "dialing") return "発信中… (シミュレーション)";
      if (state.phase === "in_call") return "通話中: " + state.title;
      if (state.phase === "done") return "通話が終わりました。";
      if (state.phase === "error") return "エラーで終了しました。";
      return "シナリオを選んでください。";
    }

    function render() {
      elStatus.textContent = statusText();
      var html = "";
      state.turns.forEach(function (t) {
        html += '<div class="ccd-turn ' + (t.who === "ai" ? "ai" : "staff") + '">' +
          '<span class="ccd-who">' + (t.who === "ai" ? "Cube" : "店員 (シミュレーション)") +
          "</span>" + escapeHtml(t.text) + "</div>";
      });
      elTranscript.innerHTML = html;
      elTranscript.scrollTop = elTranscript.scrollHeight;

      // 音声フック: 新しく届いた Cube の発話だけを渡す (cube_voice.js 連携用)
      if (voice && typeof voice.speakTurn === "function") {
        for (var i = seenTurns; i < state.turns.length; i++) {
          if (state.turns[i].who === "ai") {
            try { voice.speakTurn("ai", state.turns[i].text); } catch (e) { /* 音声は任意 */ }
          }
        }
      }
      seenTurns = state.turns.length;

      if (state.phase === "error") {
        elResult.innerHTML = '<div class="ccd-error">' + escapeHtml(state.error) + "</div>";
      } else if (state.phase === "done" && state.result) {
        var r = state.result;
        var view = checksView(r);
        var rows = slotRows(r.slots).map(function (kv) {
          return "<tr><th>" + escapeHtml(kv[0]) + "</th><td>" + escapeHtml(kv[1]) + "</td></tr>";
        }).join("");
        elResult.innerHTML =
          '<div class="ccd-result">' +
          "<h4>通話の要約 (Cube の報告)</h4><p>" + escapeHtml(r.summary || "") + "</p>" +
          "<h4>確定した内容 (店員側の記録から機械的に抽出)</h4>" +
          '<table class="ccd-slots">' + rows + "</table>" +
          "<h4>外部採点</h4>" +
          '<span class="ccd-badge ' + (view.ok ? "ok" : "ng") + '">' + escapeHtml(view.label) + "</span>" +
          (view.failures.length
            ? '<p class="ccd-fail">未達: ' + escapeHtml(view.failures.join(" / ")) + "</p>" : "") +
          '<p class="ccd-note">' + escapeHtml(SCORE_NOTE) + "</p>" +
          "</div>";
      } else {
        elResult.innerHTML = "";
      }
      var busy = state.phase === "dialing" || state.phase === "in_call";
      Array.prototype.forEach.call(elScenarios.querySelectorAll("button"), function (b) {
        b.disabled = busy;
      });
    }

    function poll() {
      if (!sessionId) return;
      fetchFn(baseUrl + "/call/status?session=" + encodeURIComponent(sessionId) +
              "&since=" + cursor)
        .then(function (res) { return res.json(); })
        .then(function (data) {
          var events = data.events || [];
          if (events.length) {
            cursor = events[events.length - 1].seq + 1;
            state = reduceEvents(state, events);
            render();
          }
          if (data.done) {
            stopPolling();
            if (data.result && state.phase !== "done") {
              state = reduceEvents(state, [{ type: "call_ended", result: data.result }]);
              render();
            }
          }
        })
        .catch(function () {
          // 一時的な接続失敗は次のポーリングで回復を試みる
        });
    }

    function startPolling() {
      stopPolling();
      timer = setInterval(poll, pollMs);
      poll();
    }
    function stopPolling() {
      if (timer) { clearInterval(timer); timer = null; }
    }

    function startCall(scenarioId) {
      state = initialState();
      state.phase = "dialing";
      cursor = 0;
      seenTurns = 0;
      render();
      fetchFn(baseUrl + "/call/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: scenarioId })
      })
        .then(function (res) {
          return res.json().then(function (j) { return { status: res.status, body: j }; });
        })
        .then(function (out) {
          if (out.status !== 200) {
            state.phase = "error";
            state.error = (out.body && out.body.error) || ("開始できませんでした (" + out.status + ")");
            render();
            return;
          }
          sessionId = out.body.session_id;
          startPolling();
        })
        .catch(function () {
          state.phase = "error";
          state.error = "サーバーに接続できませんでした。";
          render();
        });
    }

    function loadScenarios() {
      if (!fetchFn) {
        elStatus.textContent = "この環境では通信が使えません。";
        return;
      }
      fetchFn(baseUrl + "/call/scenarios")
        .then(function (res) { return res.json(); })
        .then(function (data) {
          var html = "";
          (data.scenarios || []).forEach(function (s) {
            html += '<div class="ccd-card"><h4>' + escapeHtml(s.title) + "</h4><p>" +
              escapeHtml(s.description) + '</p><button data-scn="' +
              escapeHtml(s.id) + '">この内容で電話をかける (デモ)</button></div>';
          });
          elScenarios.innerHTML = html;
          Array.prototype.forEach.call(elScenarios.querySelectorAll("button"), function (b) {
            b.addEventListener("click", function () { startCall(b.getAttribute("data-scn")); });
          });
        })
        .catch(function () {
          elStatus.textContent = "サーバーに接続できません。時間をおいて再度開いてください。";
        });
    }

    loadScenarios();
    render();

    return {
      startCall: startCall,
      getState: function () { return state; },
      destroy: function () { stopPolling(); }
    };
  }

  /* ---------------------------------------------------------------- */

  var api = {
    createPanel: createPanel,
    HONESTY_NOTICE: HONESTY_NOTICE,
    SCORE_NOTE: SCORE_NOTE,
    CHECK_LABELS: CHECK_LABELS,
    _internals: {
      escapeHtml: escapeHtml,
      slotRows: slotRows,
      checksView: checksView,
      reduceEvents: reduceEvents,
      initialState: initialState
    }
  };

  global.CubeCallDemo = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
