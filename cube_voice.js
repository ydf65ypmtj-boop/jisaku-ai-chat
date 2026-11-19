"use strict";
/*
 * Cube 音声会話モジュール（cube_voice.js）
 *
 * 公開版 Cube に「音声会話モード」（通話風の全画面で、喋って質問し、
 * 答えを声で聞く）を与えるための独立モジュール。記憶
 * （cube_memory.js）・アカウント（cube_account.js）と同じ方式で、
 * ページ本体への依存は最小限のフックだけに閉じ込める：
 *
 *   依存するのはホスト（通常は script.js の App）の次の3点だけ。
 *     host.ask(text)        … 既存チャット経路へ質問を送る
 *     host.onAssistant(cb)  … 表示された回答（pushAssistant の msg）を
 *                             受け取る
 *     host.isBusy()         … 回答生成中かどうか
 *   回答の生成・検索・出典検証はすべて既存経路の仕事であり、この
 *   モジュールは「表示された最終テキスト」を読み上げるだけ。
 *   検証ルールには一切触れない（出典・注記・バッジは読み上げない）。
 *
 * 設計の要点：
 * - 音声認識は Web Speech API、言語は ja-JP 固定が既定（英会話
 *   アプリで en-US 固定だった失敗を繰り返さない）。
 * - 読み上げ（TTS）は2段構え：
 *     本線 … cube-server の /voice 経由で VOICEVOX が合成した WAV
 *     予備 … ブラウザ内蔵 speechSynthesis（サーバー未接続時）
 *   セッション開始時に /voice/health を短い締切で確認し、使える方を
 *   選ぶ。途中で本線が落ちたら予備に切り替えて会話は続ける。
 * - 回答は文単位に割って順次合成・再生する（次の文を再生中に先行
 *   合成）。文が画面の字幕にもなる。
 * - AI が喋り終わったら自動で聞き取りを再開する（自動マイク、
 *   既定 ON。ユーザーの英会話アプリでの指定どおり）。
 * - バージイン（割り込み）：発話中も聞き取りを回し（対応環境のみ）、
 *   AI 自身の声のエコーと見分けがつく発話を検出したら再生を止めて
 *   聞き取りに移る。iOS Safari ではエコー誤認の危険が高いため既定で
 *   無効にし、画面タップ／ボタンでの割り込みにする。
 * - 状態遷移：idle → listening → thinking → speaking → listening …
 *   エラーは error、終了は ended。遷移はすべて session の単一の
 *   経路を通るので、モックでそのまま検証できる。
 * - 2Dキャラの口パク：音声会話画面にキャラ（C案）を表示し、
 *   VOICEVOX 本線の再生に合わせて口を動かす。タイミングは
 *   サーバーの /voice/schedule（合成に使う audio_query から作る
 *   モーラタイミング）が文ごとに返し、再生中は
 *   audio.currentTime を追って母音の口に切り替える。待機中は
 *   閉じ口＋瞬きループ。方式は 2026-11-18 の口パク試作の踏襲。
 *   キャラやタイミングが用意できない環境では玉（オーブ）の
 *   表示のまま会話を続ける（フェイルソフト）。
 *
 * 使い方（ブラウザ）：
 *   <script src="cube_voice.js"></script>   （script.js の後）
 *   あとはモジュールが自分で入口ボタンと通話画面を差し込み、
 *   グローバルの App をホストとして会話を始める。
 *   明示的に作る場合：
 *     const session = CubeVoice.createSession({ ... });
 *     session.start();
 * Node（テスト）：
 *   const CubeVoice = require("./cube_voice.js");
 */
(function (global) {

  /* ---------------------------------------------------------------- */
  /* 定数                                                              */
  /* ---------------------------------------------------------------- */

  const DEFAULT_LANG = "ja-JP";
  const DEFAULT_SERVER_BASE = "https://imac.tail9ed76a.ts.net:8443";
  const HEALTH_TIMEOUT_MS = 2500;   // /voice/health の締切
  const SYNTH_TIMEOUT_MS = 20000;   // 1文の合成の締切
  const REPLY_DEBOUNCE_MS = 200;    // 回答の分割push を束ねる待ち（回答は通常1回のpush で届くため、束ねに必要な最小限の待ちに短縮）
  const THINKING_TIMEOUT_MS = 150000; // 回答が来ないままの上限
  const MAX_SPOKEN_SENTENCES = 15;  // 1回答で読み上げる文の上限
  const PREFETCH_AHEAD = 2;         // TTS先行合成の窓（再生中の文＋先行2文まで。同時合成の競合を避ける）
  const TRUNCATION_NOTICE = "（続きは画面で確認できます）";
  const MIC_RESTART_LIMIT = 5;      // 聞き取りの連続再起動の上限

  const STATES = ["idle", "listening", "thinking", "speaking", "error", "ended"];

  /* ---------------------------------------------------------------- */
  /* 小物                                                              */
  /* ---------------------------------------------------------------- */

  function noop() {}

  function clampText(s, n) {
    const t = String(s || "");
    return t.length > n ? t.slice(0, n) : t;
  }

  /* 読み上げ用にテキストを整える。表示テキスト（検証を通った回答
     本文）だけが対象で、意味は変えない：URL・コード・装飾記号を
     落とし、空白をならす。出典や注記はそもそも渡ってこない。 */
  function sanitizeForSpeech(text) {
    let t = String(text || "");
    t = t.replace(/```[\s\S]*?```/g, "（コードは画面で確認できます）");
    t = t.replace(/https?:\/\/[^\s）)]+/g, "（リンクは画面で確認できます）");
    t = t.replace(/`([^`]*)`/g, "$1");
    t = t.replace(/[*_#>|]+/g, "");
    t = t.replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "");
    t = t.replace(/[ \t]+/g, " ");
    t = t.replace(/\n{2,}/g, "\n");
    return t.trim();
  }

  /* 文単位に割る。日本語の句読点（。！？）と改行を境界にし、
     短すぎる断片は前の文にくっつける（「はい。そうです。」の
     ような細切れを防ぐ）。 */
  function splitSentences(text) {
    const t = String(text || "");
    const raw = [];
    let cur = "";
    for (let i = 0; i < t.length; i += 1) {
      const ch = t[i];
      cur += ch;
      if ("。！？".includes(ch) || ch === "\n") {
        if (cur.trim()) raw.push(cur.trim());
        cur = "";
      }
    }
    if (cur.trim()) raw.push(cur.trim());
    const out = [];
    for (const s of raw) {
      // 「はい。」のような3文字以下の相づち断片だけを前の文に
      // くっつける（「一です。二です。」のような列挙は分ける）。
      if (out.length && s.length <= 3 && !/[0-9０-９]/.test(s)) {
        out[out.length - 1] += s;
      } else {
        out.push(s);
      }
    }
    return out.filter((s) => s.length > 0);
  }

  /* エコー判定用に正規化する（空白・句読点を落として比べる）。 */
  function normalizeForCompare(text) {
    return String(text || "")
      .replace(/[\s。、！？・「」『』（）()!?,.．，]/g, "")
      .toLowerCase();
  }

  /* 聞き取り結果が「今読み上げている文」のエコーらしいか。
     正規化した聞き取りが、読み上げ中テキストの正規化に含まれる
     （またはその逆）ならエコーとみなす。 */
  function looksLikeEcho(heard, speakingText) {
    const h = normalizeForCompare(heard);
    const s = normalizeForCompare(speakingText);
    if (!h || !s) return false;
    if (h.length < 3) return true; // 短い相づち程度はエコー扱いで無視
    return s.includes(h) || h.includes(s);
  }

  function isIosLike(ua) {
    const s = String(ua || "");
    return /iPad|iPhone|iPod/.test(s) ||
      (s.includes("Macintosh") && s.includes("Mobile"));
  }

  /* ---------------------------------------------------------------- */
  /* 音声認識（Web Speech API の薄いラッパ）                          */
  /* ---------------------------------------------------------------- */

  /* session が使う認識の形：
     { available, lang, start(), stop(),
       onInterim, onFinal, onError, onEnd }（コールバックは代入式） */
  function createWebSpeechRecognizer(env) {
    const e = env || {};
    const Ctor = e.SpeechRecognition || e.webkitSpeechRecognition || null;
    const rec = {
      available: !!Ctor,
      lang: e.lang || DEFAULT_LANG,
      onInterim: noop, onFinal: noop, onError: noop, onEnd: noop,
      _impl: null,
      _running: false,
      start() {
        if (!Ctor || rec._running) return false;
        const impl = new Ctor();
        impl.lang = rec.lang;
        impl.interimResults = true;
        impl.continuous = false;
        impl.maxAlternatives = 1;
        impl.onresult = (ev) => {
          let interim = "";
          for (let i = ev.resultIndex; i < ev.results.length; i += 1) {
            const r = ev.results[i];
            const txt = (r[0] && r[0].transcript) || "";
            if (r.isFinal) rec.onFinal(txt.trim());
            else interim += txt;
          }
          if (interim) rec.onInterim(interim.trim());
        };
        impl.onerror = (ev) => { rec.onError(ev && ev.error ? ev.error : "unknown"); };
        impl.onend = () => {
          rec._running = false;
          rec._impl = null;
          rec.onEnd();
        };
        rec._impl = impl;
        rec._running = true;
        try { impl.start(); } catch { rec._running = false; rec._impl = null; return false; }
        return true;
      },
      stop() {
        if (rec._impl) {
          try { rec._impl.onend = null; rec._impl.stop(); } catch { /* noop */ }
          rec._impl = null;
        }
        rec._running = false;
      },
    };
    return rec;
  }

  /* ---------------------------------------------------------------- */
  /* TTS エンジン                                                      */
  /* ---------------------------------------------------------------- */

  /* エンジンの形：
     { kind, prepare(sentence)->Promise<token>,
       play(token, sentence, {onStart})->Promise<void>, cancel() }
     VOICEVOX 側は prepare が合成（fetch）、play が再生。加えて
     unlock()（自動再生制限の解除）を持ち、再生要素は1個に固定する。
     ブラウザ側は prepare が即時解決、play が utterance の再生。 */

  /* 無音 WAV（自動再生制限の解除用）。unlock() が永続要素で再生する。 */
  const SILENT_WAV_URL =
    "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";

  function createVoicevoxEngine(env) {
    const e = env || {};
    const fetchFn = e.fetch;
    const base = String(e.serverBaseUrl || DEFAULT_SERVER_BASE).replace(/\/+$/, "");
    const timeoutMs = e.synthTimeoutMs || SYNTH_TIMEOUT_MS;
    /* 再生要素は1個だけを使い回す（env.audioElement があればそれを
       共有し、無ければ自分で作る）。iOS Safari では、ユーザー操作で
       再生を許可されたことのない新しい Audio 要素の play() は
       NotAllowedError で拒否されるため、文ごとの新規生成では実機で
       再生できない。入口クリックで unlock() しておいた同じ要素なら、
       src を差し替えての再生が通る。 */
    let audioEl = e.audioElement || null;
    let currentUrl = null;

    function ensureAudio() {
      if (!audioEl) audioEl = new e.Audio();
      return audioEl;
    }

    function stopCurrent() {
      if (e.lipSync) {
        try { e.lipSync.stopPlayback(); } catch { /* noop */ }
      }
      if (audioEl) {
        audioEl.onended = null;
        audioEl.onerror = null;
        try { audioEl.pause(); } catch { /* noop */ }
      }
      if (currentUrl) {
        try { e.URL.revokeObjectURL(currentUrl); } catch { /* noop */ }
        currentUrl = null;
      }
    }

    /* 口パク用のモーラタイミングを /voice/schedule から取る。
       取れなくても読み上げは続ける（口は動かないだけ）ため、
       失敗はすべて null に倒す。 */
    async function fetchSchedule(sentence) {
      const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
      try {
        const res = await fetchFn(`${base}/voice/schedule`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentence, speaker: e.speaker ?? null }),
          signal: ctrl ? ctrl.signal : undefined,
        });
        if (!res || !res.ok) return null;
        const j = await res.json().catch(() => null);
        if (!j || !Array.isArray(j.schedule)) return null;
        const segs = j.schedule.filter((s) => s &&
          typeof s.start === "number" && typeof s.end === "number" &&
          typeof s.vowel === "string");
        return segs.length ? segs : null;
      } catch {
        return null;
      } finally {
        if (timer) clearTimeout(timer);
      }
    }

    return {
      kind: "voicevox",
      async prepare(sentence) {
        const schedP = fetchSchedule(sentence); // 失敗しても null（投げない）
        const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
        const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
        try {
          const res = await fetchFn(`${base}/voice/synthesize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: sentence, speaker: e.speaker ?? null }),
            signal: ctrl ? ctrl.signal : undefined,
          });
          if (!res || !res.ok) throw new Error(`voice http ${res && res.status}`);
          const blob = await res.blob();
          const schedule = await schedP;
          return { blob, schedule };
        } finally {
          if (timer) clearTimeout(timer);
        }
      },
      /* ユーザー操作（入口ボタンのクリック）の文脈で呼び、永続要素で
         無音を一度再生して自動再生制限の許可を得ておく。以後の文の
         再生は同じ要素の src 差し替えなので許可が効く。文の再生中は
         何もしない。 */
      unlock() {
        if (currentUrl) return;
        try {
          const audio = ensureAudio();
          audio.src = SILENT_WAV_URL;
          const p = audio.play();
          if (p && typeof p.catch === "function") p.catch(() => {});
        } catch { /* noop */ }
      },
      play(token) {
        return new Promise((resolve, reject) => {
          stopCurrent();
          const url = e.URL.createObjectURL(token.blob);
          currentUrl = url;
          const audio = ensureAudio();
          audio.src = url;
          audio.onended = () => { stopCurrent(); resolve(); };
          audio.onerror = () => { stopCurrent(); reject(new Error("audio playback failed")); };
          const p = audio.play();
          if (p && typeof p.catch === "function") {
            p.catch((err) => { stopCurrent(); reject(err); });
          }
          if (token && token.schedule && e.lipSync) {
            try {
              e.lipSync.play(token.schedule, () => audio.currentTime);
            } catch { /* 口パクが無くても再生は続ける */ }
          }
        });
      },
      cancel() { stopCurrent(); },
    };
  }

  function createBrowserTtsEngine(env) {
    const e = env || {};
    const synth = e.speechSynthesis || null;
    let currentUtter = null;
    return {
      kind: "browser",
      available: !!synth,
      async prepare(sentence) { return { text: sentence }; },
      play(token) {
        return new Promise((resolve, reject) => {
          if (!synth) { reject(new Error("speechSynthesis unavailable")); return; }
          const u = new e.SpeechSynthesisUtterance(token.text);
          u.lang = e.lang || DEFAULT_LANG;
          const voices = (synth.getVoices ? synth.getVoices() : []) || [];
          const ja = voices.find((v) => /^ja([-_]|$)/i.test(v.lang || ""));
          if (ja) u.voice = ja;
          currentUtter = u;
          u.onend = () => { currentUtter = null; resolve(); };
          u.onerror = (ev) => {
            currentUtter = null;
            if (ev && ev.error === "interrupted") resolve();
            else reject(new Error(`tts ${ev && ev.error}`));
          };
          synth.speak(u);
        });
      },
      cancel() {
        if (synth) { try { synth.cancel(); } catch { /* noop */ } }
        currentUtter = null;
      },
    };
  }

  /* ---------------------------------------------------------------- */
  /* 2Dキャラの口パク（音素タイミング駆動）                            */
  /* ---------------------------------------------------------------- */

  const LIP_VOWELS = ["a", "i", "u", "e", "o"];

  /* 母音 → フレームキー。母音以外（"none"・促音・撥音など）は
     閉じ口 "base"。 */
  function frameKeyForVowel(vowel) {
    return LIP_VOWELS.includes(vowel) ? vowel : "base";
  }

  /* スケジュール [{start, end, vowel}] の時刻 t（秒）の母音。
     区間外は "none"（閉じ口）。試作と同じ線形走査。 */
  function vowelAtTime(schedule, t) {
    if (!Array.isArray(schedule)) return "none";
    for (const seg of schedule) {
      if (seg && t >= seg.start && t < seg.end) return seg.vowel;
    }
    return "none";
  }

  /* 口パク制御。描画は deps.show(frameKey) に委ね、タイマ・rAF・
     乱数は注入する（Node のモックで検証できる形）。試作の方式：
       - 瞬きは 2.5〜5 秒のランダム間隔で約150ms、口より優先表示
       - 再生中は rAF で clock（audio.currentTime）を追い、
         その時刻の母音の口に切り替える
       - 停止・終了では閉じ口に戻る */
  function createLipSync(deps) {
    const d = deps || {};
    const show = typeof d.show === "function" ? d.show : noop;
    const setT = d.setTimeout || setTimeout;
    const clearT = d.clearTimeout || clearTimeout;
    const raf = d.requestAnimationFrame ||
      (typeof requestAnimationFrame !== "undefined" ? requestAnimationFrame : null);
    const caf = d.cancelAnimationFrame ||
      (typeof cancelAnimationFrame !== "undefined" ? cancelAnimationFrame : null);
    const random = d.random || Math.random;

    let mouth = "base";      // 今の口形（フレームキー）
    let blinking = false;
    let shownKey = null;     // 最後に show したキー
    let running = false;     // 瞬きループが回っているか
    let blinkWaitTimer = null;
    let blinkEndTimer = null;
    let playRaf = null;
    let playClock = null;    // () => 秒。null を返したら再生終了扱い
    let playSchedule = null;

    function render() {
      const key = blinking ? "blink" : mouth;
      if (key === shownKey) return;
      shownKey = key;
      show(key);
    }
    function setMouth(vowel) {
      const next = frameKeyForVowel(vowel);
      if (next !== mouth) mouth = next;
      render();
    }
    function scheduleBlink() {
      if (!running) return;
      blinkWaitTimer = setT(() => {
        blinkWaitTimer = null;
        if (!running) return;
        blinking = true;
        render();
        blinkEndTimer = setT(() => {
          blinkEndTimer = null;
          blinking = false;
          render();
          scheduleBlink();
        }, 150);
      }, 2500 + random() * 2500);
    }
    function tick() {
      if (!playClock) { playRaf = null; return; }
      const t = playClock();
      if (t === null || t === undefined || Number.isNaN(t)) {
        stopPlayback();
        return;
      }
      setMouth(vowelAtTime(playSchedule, t));
      playRaf = raf ? raf(tick) : null;
    }
    function stopPlayback() {
      if (playRaf !== null && caf) caf(playRaf);
      playRaf = null;
      playClock = null;
      playSchedule = null;
      if (mouth !== "base") mouth = "base";
      render();
    }
    return {
      start() {
        if (running) return;
        running = true;
        render();
        scheduleBlink();
      },
      stop() {
        running = false;
        if (blinkWaitTimer) { clearT(blinkWaitTimer); blinkWaitTimer = null; }
        if (blinkEndTimer) { clearT(blinkEndTimer); blinkEndTimer = null; }
        blinking = false;
        stopPlayback();
        render();
      },
      play(schedule, clock) {
        stopPlayback();
        if (!Array.isArray(schedule) || !schedule.length ||
            typeof clock !== "function" || !raf) return;
        playSchedule = schedule;
        playClock = clock;
        playRaf = raf(tick);
      },
      stopPlayback,
      get shownKey() { return shownKey; },
    };
  }

  /* /voice/health を見て本線が使えるか判定する。 */
  async function probeVoicevox(env) {
    const e = env || {};
    if (typeof e.fetch !== "function") return false;
    const base = String(e.serverBaseUrl || DEFAULT_SERVER_BASE).replace(/\/+$/, "");
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), e.healthTimeoutMs || HEALTH_TIMEOUT_MS) : null;
    try {
      const res = await e.fetch(`${base}/voice/health`, {
        method: "GET",
        signal: ctrl ? ctrl.signal : undefined,
      });
      if (!res || !res.ok) return false;
      const j = await res.json().catch(() => null);
      return !!(j && j.ok);
    } catch {
      return false;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /* ---------------------------------------------------------------- */
  /* 読み上げキュー（文単位・先行合成つき）                            */
  /* ---------------------------------------------------------------- */

  function createSpeechQueue(env) {
    const e = env || {};
    let engine = e.engine || null;
    const queue = {
      onSentenceStart: noop,   // (sentence, index, total)
      onDrained: noop,         // キューが空になって再生も終わった
      onError: noop,           // (err, sentence)
      _items: [],
      _playing: false,
      _generation: 0,
      _nextSeq: 0,   // 次に積む文の通し番号
      _playSeq: 0,   // 再生中（または次に再生する）文の通し番号。先行合成の窓の基準
      setEngine(next) { engine = next; },
      getEngine() { return engine; },
      /* 先行合成は「再生中＋先行 PREFETCH_AHEAD 文まで」の窓に限る。
         全文を一斉に合成すると、第1文の合成が後続文と合成エンジン
         （VOICEVOX）内で競合して再生開始が遅れうるため。窓は再生が
         進むにつれて前へずらす。 */
      _startPrepare(item) {
        if (item._prepareStarted || !item.engine) return;
        item._prepareStarted = true;
        item.prepared = item.engine.prepare(item.text).then(
          (token) => ({ token }),
          (err) => ({ err }),
        );
      },
      _fillWindow() {
        for (const item of queue._items) {
          if (item.seq > queue._playSeq + PREFETCH_AHEAD) break;
          queue._startPrepare(item);
        }
      },
      enqueueSentences(sentences) {
        const gen = queue._generation;
        for (const text of sentences) {
          // エンジンは文ごとに固定する（途中で本線→予備に切り替えても、
          // 既に合成した文のトークンと再生器が食い違わないように）。
          const item = { text, engine, prepared: null, seq: queue._nextSeq, _prepareStarted: false };
          queue._nextSeq += 1;
          queue._items.push(item);
        }
        queue._fillWindow();
        void queue._pump(gen);
        return queue._items.length;
      },
      async _pump(gen) {
        if (queue._playing) return;
        queue._playing = true;
        let index = 0;
        try {
          while (queue._items.length) {
            if (gen !== queue._generation) break;
            const item = queue._items.shift();
            queue._playSeq = item.seq;
            queue._startPrepare(item);
            queue._fillWindow();
            if (!item.engine) continue;
            queue._currentEngine = item.engine;
            let token = null;
            if (item.prepared) {
              const r = await item.prepared;
              if (gen !== queue._generation) break;
              if (r.err) { queue.onError(r.err, item.text); continue; }
              token = r.token;
            } else {
              try { token = await item.engine.prepare(item.text); }
              catch (err) { queue.onError(err, item.text); continue; }
            }
            if (gen !== queue._generation) break;
            queue.onSentenceStart(item.text, index, index + queue._items.length + 1);
            index += 1;
            try {
              await item.engine.play(token, item.text);
            } catch (err) {
              if (gen !== queue._generation) break;
              queue.onError(err, item.text);
            }
          }
        } finally {
          queue._playing = false;
          queue._currentEngine = null;
          if (gen === queue._generation && !queue._items.length) {
            queue._playSeq = queue._nextSeq; // 窓の基準を次の文へ
            queue.onDrained();
          }
        }
      },
      cancel() {
        queue._generation += 1;
        queue._items = [];
        queue._playSeq = queue._nextSeq; // 窓の基準を次の文へ
        if (queue._currentEngine) queue._currentEngine.cancel();
        if (engine) engine.cancel();
      },
      get busy() { return queue._playing || queue._items.length > 0; },
    };
    return queue;
  }

  /* ---------------------------------------------------------------- */
  /* 会話セッション（状態機械）                                        */
  /* ---------------------------------------------------------------- */

  function createSession(options) {
    const opts = options || {};
    const host = opts.host || null;
    const recognizer = opts.recognizer || null;
    const queue = opts.queue || createSpeechQueue({ engine: opts.engine || null });
    const setT = opts.setTimeout || setTimeout;
    const clearT = opts.clearTimeout || clearTimeout;
    const debounceMs = opts.replyDebounceMs ?? REPLY_DEBOUNCE_MS;
    const thinkingTimeoutMs = opts.thinkingTimeoutMs ?? THINKING_TIMEOUT_MS;
    const maxSentences = opts.maxSentences ?? MAX_SPOKEN_SENTENCES;

    const listeners = { state: [], subtitle: [], user: [], notice: [], engine: [] };
    function emit(kind, payload) {
      for (const cb of listeners[kind] || []) {
        try { cb(payload); } catch { /* noop */ }
      }
    }

    const session = {
      state: "idle",
      autoMic: opts.autoMic !== false,          // 既定 ON（ユーザー指定）
      bargeIn: !!opts.bargeIn,                  // 対応環境でのみ ON にする
      engineKind: queue.getEngine() ? queue.getEngine().kind : null,
      _micLive: false,
      _micRestarts: 0,
      _pendingTexts: [],
      _debounceTimer: null,
      _thinkingTimer: null,
      _unsubHost: null,
      _speakingText: "",
      _ended: false,

      on(kind, cb) { (listeners[kind] = listeners[kind] || []).push(cb); return session; },

      _setState(next) {
        if (!STATES.includes(next)) return;
        if (session.state === next) return;
        session.state = next;
        emit("state", next);
      },

      /* ---- 聞き取り ---- */
      _startMic() {
        if (!recognizer || !recognizer.available) {
          emit("notice", "このブラウザでは音声認識が使えません。文字で入力してください。");
          return false;
        }
        if (session._micLive) return true;
        const ok = recognizer.start();
        if (ok) { session._micLive = true; session._micRestarts = 0; }
        return ok;
      },
      _stopMic() {
        if (recognizer && session._micLive) recognizer.stop();
        session._micLive = false;
      },

      /* ---- 開始・終了 ---- */
      start() {
        if (session.state !== "idle" && session.state !== "ended" &&
            session.state !== "error") return false;
        session._ended = false;
        session._micRestarts = 0;
        if (host && typeof host.onAssistant === "function" && !session._unsubHost) {
          session._unsubHost = host.onAssistant((msg) => session._onAssistant(msg));
        }
        session._setState("listening");
        if (session.autoMic) session._startMic();
        return true;
      },
      end() {
        session._ended = true;
        session._stopMic();
        queue.cancel();
        if (session._debounceTimer) { clearT(session._debounceTimer); session._debounceTimer = null; }
        if (session._thinkingTimer) { clearT(session._thinkingTimer); session._thinkingTimer = null; }
        session._pendingTexts = [];
        if (session._unsubHost) { session._unsubHost(); session._unsubHost = null; }
        session._setState("ended");
      },

      /* ---- ユーザーの発話（認識の最終結果） ---- */
      submitUtterance(text) {
        const q = String(text || "").trim();
        if (!q || session._ended) return false;
        if (session.state === "speaking") session.interrupt();
        if (session.state !== "listening") return false;
        session._stopMic();
        emit("user", q);
        session._setState("thinking");
        session._armThinkingTimeout();
        let accepted = false;
        try {
          accepted = host ? host.ask(q) !== false : false;
        } catch { accepted = false; }
        if (!accepted) {
          session._disarmThinkingTimeout();
          emit("notice", "前の回答を待っています。少し待ってからもう一度どうぞ。");
          session._backToListening();
        }
        return accepted;
      },

      _armThinkingTimeout() {
        session._disarmThinkingTimeout();
        session._thinkingTimer = setT(() => {
          if (session.state !== "thinking") return;
          emit("notice", "回答が返ってきませんでした。もう一度聞いてみてください。");
          session._backToListening();
        }, thinkingTimeoutMs);
      },
      _disarmThinkingTimeout() {
        if (session._thinkingTimer) { clearT(session._thinkingTimer); session._thinkingTimer = null; }
      },

      _backToListening() {
        if (session._ended) return;
        session._setState("listening");
        if (session.autoMic) session._startMic();
      },

      /* ---- 回答の到着 ---- */
      _onAssistant(msg) {
        if (session._ended || !msg || typeof msg.text !== "string") return;
        if (session.state !== "thinking" && session.state !== "speaking" &&
            session.state !== "listening") return;
        session._disarmThinkingTimeout();
        session._pendingTexts.push(msg.text);
        if (session._debounceTimer) clearT(session._debounceTimer);
        session._debounceTimer = setT(() => {
          session._debounceTimer = null;
          session._flushSpeech();
        }, debounceMs);
      },
      _flushSpeech() {
        if (!session._pendingTexts.length || session._ended) return;
        const joined = session._pendingTexts.join("\n");
        session._pendingTexts = [];
        let sentences = splitSentences(sanitizeForSpeech(joined));
        if (!sentences.length) { session._backToListening(); return; }
        let truncated = false;
        if (sentences.length > maxSentences) {
          sentences = sentences.slice(0, maxSentences);
          truncated = true;
        }
        session._setState("speaking");
        queue.enqueueSentences(sentences);
        if (truncated) emit("notice", TRUNCATION_NOTICE);
        if (session.bargeIn && recognizer && recognizer.available) {
          session._startMic(); // バージイン用に聞き取りを並走させる
        }
      },

      /* ---- 割り込み ---- */
      interrupt() {
        if (session.state !== "speaking") return false;
        queue.cancel();
        session._speakingText = "";
        session._backToListening();
        return true;
      },

      /* ---- 自動マイクの ON/OFF ---- */
      setAutoMic(v) {
        session.autoMic = !!v;
        if (!v) session._stopMic();
        else if (session.state === "listening" && !session._ended) session._startMic();
      },

      /* ---- 手動で聞き取りを始める（自動マイク OFF のとき） ---- */
      startListening() {
        if (session.state !== "listening" || session._ended) return false;
        return session._startMic();
      },

      /* ---- TTS エンジンの差し替え（本線→予備の劣化など） ---- */
      setEngine(engine) {
        queue.setEngine(engine);
        session.engineKind = engine ? engine.kind : null;
        emit("engine", session.engineKind);
      },
    };

    /* キューの出来事を状態遷移につなぐ */
    let consecutiveTtsErrors = 0;
    queue.onSentenceStart = (sentence) => {
      consecutiveTtsErrors = 0;
      session._speakingText = sentence;
      emit("subtitle", sentence);
    };
    queue.onDrained = () => {
      session._speakingText = "";
      if (session.state === "speaking") session._backToListening();
    };
    queue.onError = (err) => {
      emit("notice", `読み上げでエラーがありました：${err && err.message ? err.message : err}`);
      // 本線（VOICEVOX）が続けて失敗したら予備（ブラウザの声）へ
      // 切り替えて会話を続ける。切り替えは次の文から効く。
      consecutiveTtsErrors += 1;
      const cur = queue.getEngine();
      if (consecutiveTtsErrors >= 2 && opts.fallbackEngine &&
          cur && cur.kind !== opts.fallbackEngine.kind) {
        session.setEngine(opts.fallbackEngine);
        consecutiveTtsErrors = 0;
        emit("notice", "音声サーバーに繋がらないため、ブラウザの声に切り替えました。");
      }
    };

    /* 認識の出来事を状態遷移につなぐ */
    if (recognizer) {
      recognizer.onInterim = (text) => {
        if (session.state === "speaking" && session.bargeIn) {
          if (!looksLikeEcho(text, session._speakingText)) {
            session.interrupt();
          }
          return;
        }
        if (session.state === "listening") emit("subtitle", text);
      };
      recognizer.onFinal = (text) => {
        if (session.state === "speaking") return; // 発話中の確定は捨てる
        session._micLive = false;
        session.submitUtterance(text);
      };
      recognizer.onError = (kind) => {
        session._micLive = false;
        if (kind === "not-allowed" || kind === "service-not-allowed") {
          session._setState("error");
          emit("notice", "マイクの使用が許可されませんでした。ブラウザの設定で許可してから、もう一度お試しください。");
          return;
        }
        if (session.state === "listening" && session.autoMic && !session._ended) {
          session._micRestarts += 1;
          if (session._micRestarts <= MIC_RESTART_LIMIT) session._startMic();
        }
      };
      recognizer.onEnd = () => {
        const wasLive = session._micLive;
        session._micLive = false;
        if (!wasLive || session._ended) return;
        // 発話が無いまま認識が終わった：聞き取り継続が望ましい状態
        // なら自動で張り直す（自動マイク ON の listening、または
        // バージイン並走中の speaking）。
        if (session.state === "listening" && session.autoMic) {
          session._micRestarts += 1;
          if (session._micRestarts <= MIC_RESTART_LIMIT) session._startMic();
          else session._micRestarts = 0;
        } else if (session.state === "speaking" && session.bargeIn) {
          session._startMic();
        }
      };
    }

    return session;
  }

  /* ---------------------------------------------------------------- */
  /* ブラウザ用ホスト（App への最小フック）                           */
  /* ---------------------------------------------------------------- */

  /* script.js の App をホストにする。pushAssistant を包んで「表示
     された回答」を購読するだけで、本体の処理には触れない。
     履歴の再生も pushAssistant を通るが、通話画面が開いている間は
     履歴を開けない（全画面モーダル）ため、再生が読み上げられる
     ことはない。 */
  function createBrowserHost(app) {
    if (!app || typeof app.ask !== "function" || typeof app.pushAssistant !== "function") {
      return null;
    }
    const callbacks = [];
    const original = app.pushAssistant;
    app.pushAssistant = function (msg) {
      const r = original.call(app, msg);
      for (const cb of callbacks.slice()) {
        try { cb(msg); } catch { /* noop */ }
      }
      return r;
    };
    return {
      ask(text) { void app.ask(text); return true; },
      isBusy() { return app.phase !== "idle"; },
      onAssistant(cb) {
        callbacks.push(cb);
        return () => {
          const i = callbacks.indexOf(cb);
          if (i >= 0) callbacks.splice(i, 1);
        };
      },
      unwrap() { app.pushAssistant = original; },
    };
  }

  /* ---------------------------------------------------------------- */
  /* ブラウザ UI（入口ボタン＋通話風の全画面）                         */
  /* ---------------------------------------------------------------- */

  const STYLE_TEXT = `
.cv-entry-btn { flex: none; height: 40px; padding: 0 14px; border-radius: 999px;
  border: 1.5px solid var(--accent); background: var(--surface);
  color: var(--accent); font-size: 14px; cursor: pointer; }
.cv-entry-btn:hover { background: var(--accent-tint); }
.cv-overlay { position: fixed; inset: 0; z-index: 200; background: var(--bg);
  color: var(--text); display: flex; flex-direction: column; align-items: center;
  padding: max(16px, env(safe-area-inset-top)) 20px max(16px, env(safe-area-inset-bottom)); }
.cv-overlay[hidden] { display: none; }
.cv-top { width: 100%; max-width: 560px; display: flex; justify-content: space-between;
  align-items: baseline; }
.cv-title { font-weight: 700; font-size: 17px; }
.cv-engine { font-size: 12px; color: var(--dim); }
.cv-credit { font-size: 11px; color: var(--dim); margin: 10px 0 0; }
.cv-orb-wrap { flex: 1; display: flex; align-items: center; justify-content: center; }
.cv-orb { width: 148px; height: 148px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, var(--grad-2), var(--grad-1));
  box-shadow: 0 12px 40px rgba(27, 80, 200, 0.35); }
.cv-char { position: relative; width: min(66vw, 288px); aspect-ratio: 1 / 1; flex: none; }
.cv-char[hidden] { display: none; }
.cv-char img { position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: cover; border-radius: 28px; opacity: 0;
  box-shadow: 0 12px 40px rgba(27, 80, 200, 0.25); }
.cv-char img.on { opacity: 1; }
.cv-overlay[data-state="listening"] .cv-orb { animation: cv-pulse 2.2s ease-in-out infinite; }
.cv-overlay[data-state="speaking"] .cv-orb { animation: cv-talk 0.9s ease-in-out infinite; }
.cv-overlay[data-state="thinking"] .cv-orb { animation: cv-pulse 1.1s ease-in-out infinite; opacity: 0.75; }
.cv-overlay[data-state="error"] .cv-orb { background: var(--warn); animation: none; }
.cv-overlay[data-state="listening"] .cv-char { animation: cv-pulse 2.2s ease-in-out infinite; }
.cv-overlay[data-state="speaking"] .cv-char { animation: cv-talk 0.9s ease-in-out infinite; }
.cv-overlay[data-state="thinking"] .cv-char { animation: cv-pulse 1.1s ease-in-out infinite; }
@keyframes cv-pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.06); } }
@keyframes cv-talk { 0%,100% { transform: scale(1); } 50% { transform: scale(1.13); } }
@media (prefers-reduced-motion: reduce) { .cv-orb, .cv-char { animation: none !important; } }
.cv-status { font-size: 15px; color: var(--dim); min-height: 22px; margin: 4px 0 8px; }
.cv-subtitles { width: 100%; max-width: 560px; min-height: 120px; }
.cv-sub-user { color: var(--dim); font-size: 15px; margin: 0 0 6px; }
.cv-sub-ai { font-size: 20px; line-height: 1.6; margin: 0; }
.cv-notice { width: 100%; max-width: 560px; color: var(--warn); font-size: 13px;
  min-height: 18px; margin: 8px 0 0; }
.cv-controls { width: 100%; max-width: 560px; display: flex; flex-wrap: wrap; gap: 10px;
  align-items: center; justify-content: center; margin-top: 10px; }
.cv-main-btn { height: 46px; padding: 0 26px; border-radius: 999px; border: none;
  background: var(--accent); color: #fff; font-size: 16px; cursor: pointer; }
.cv-main-btn:disabled { background: var(--surface-alt); color: var(--dim); cursor: default; }
.cv-end-btn { height: 46px; padding: 0 20px; border-radius: 999px;
  border: 1.5px solid var(--border); background: var(--surface); color: var(--text);
  font-size: 15px; cursor: pointer; }
.cv-toggle { font-size: 14px; color: var(--dim); display: flex; gap: 6px; align-items: center; }
.cv-speaker { height: 40px; border-radius: 10px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); font-size: 14px; max-width: 200px; }
.cv-textform { width: 100%; max-width: 560px; display: flex; gap: 8px; margin-top: 12px; }
.cv-textform[hidden] { display: none; }
.cv-textinput { flex: 1; height: 42px; border-radius: 10px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); padding: 0 12px; font-size: 15px; }
.cv-textform button { height: 42px; padding: 0 16px; border-radius: 10px; border: none;
  background: var(--accent); color: #fff; font-size: 15px; cursor: pointer; }
`;

  const STATUS_TEXT = {
    listening: "聞いています…",
    thinking: "考え中…",
    speaking: "話しています",
    error: "エラーがありました",
    idle: "",
    ended: "",
  };

  /* ブラウザに組み込む。App（script.js）が無い環境では何もしない。 */
  function mountBrowserUi(env) {
    const e = env || {};
    const doc = e.document || (typeof document !== "undefined" ? document : null);
    const win = e.window || (typeof window !== "undefined" ? window : null);
    if (!doc || !win) return null;
    const app = e.app || (typeof App !== "undefined" ? App : null);
    if (!app) return null;
    if (doc.querySelector(".cv-overlay")) return null; // 二重組み込み防止

    const serverBase = e.serverBaseUrl ||
      (typeof CUBE_SERVER_BASE_URL !== "undefined" && CUBE_SERVER_BASE_URL) ||
      DEFAULT_SERVER_BASE;
    const lang = e.lang || DEFAULT_LANG;
    const ua = (win.navigator && win.navigator.userAgent) || "";

    const host = createBrowserHost(app);
    if (!host) return null;
    const recognizer = createWebSpeechRecognizer({
      SpeechRecognition: win.SpeechRecognition,
      webkitSpeechRecognition: win.webkitSpeechRecognition,
      lang,
    });
    const browserEngine = createBrowserTtsEngine({
      speechSynthesis: win.speechSynthesis,
      SpeechSynthesisUtterance: win.SpeechSynthesisUtterance,
      lang,
    });
    const voiceState = { speaker: null, voicevoxReady: false };
    /* 2Dキャラ（口パク）の状態。フレームは遅延読み込みし、
       用意できるまでは玉（オーブ）表示のままにする。エンジンへは
       代理オブジェクトを渡し、キャラが組み上がってからでも
       口パクが効くようにする。 */
    const charState = { lipSync: null, loading: false, ready: false };
    const lipSyncProxy = {
      play(schedule, clock) {
        if (charState.lipSync) charState.lipSync.play(schedule, clock);
      },
      stopPlayback() {
        if (charState.lipSync) charState.lipSync.stopPlayback();
      },
    };
    /* VOICEVOX エンジンの再生要素は、エンジンを作り直しても同じ
       1個を使い回す。iOS の自動再生制限の解除は要素単位で効くため、
       入口クリックで解除する要素と、実際に文を再生する要素を
       一致させる必要がある。 */
    let sharedAudioEl = null;
    let voicevoxEngine = null;
    const makeVoicevox = () => {
      if (!sharedAudioEl) {
        try { sharedAudioEl = new win.Audio(); } catch { sharedAudioEl = null; }
      }
      voicevoxEngine = createVoicevoxEngine({
        fetch: win.fetch.bind(win),
        URL: win.URL,
        Audio: win.Audio,
        audioElement: sharedAudioEl,
        serverBaseUrl: serverBase,
        speaker: voiceState.speaker,
        lipSync: lipSyncProxy,
      });
      return voicevoxEngine;
    };
    const queue = createSpeechQueue({ engine: browserEngine });
    const session = createSession({
      host, recognizer, queue,
      autoMic: true,
      bargeIn: !isIosLike(ua),
      fallbackEngine: browserEngine,
    });

    /* ---- DOM ---- */
    const style = doc.createElement("style");
    style.textContent = STYLE_TEXT;
    doc.head.appendChild(style);

    const entry = doc.createElement("button");
    entry.type = "button";
    entry.className = "cv-entry-btn";
    entry.textContent = "🎙 音声会話";
    const sendBtn = doc.getElementById("send-btn");
    if (sendBtn && sendBtn.parentElement) sendBtn.parentElement.insertBefore(entry, sendBtn);
    else {
      const form = doc.getElementById("ask-form");
      if (form) form.appendChild(entry);
      else doc.body.appendChild(entry);
    }

    const overlay = doc.createElement("div");
    overlay.className = "cv-overlay";
    overlay.hidden = true;
    overlay.dataset.state = "idle";
    overlay.innerHTML = `
      <div class="cv-top">
        <span class="cv-title">Cube 音声会話</span>
        <span class="cv-engine mono"></span>
      </div>
      <div class="cv-orb-wrap">
        <div class="cv-orb" role="img" aria-label="Cube"></div>
        <div class="cv-char" hidden aria-hidden="true">
          <img data-frame="base" alt="" />
          <img data-frame="a" alt="" />
          <img data-frame="i" alt="" />
          <img data-frame="u" alt="" />
          <img data-frame="e" alt="" />
          <img data-frame="o" alt="" />
          <img data-frame="blink" alt="" />
        </div>
      </div>
      <p class="cv-status" aria-live="polite"></p>
      <div class="cv-subtitles">
        <p class="cv-sub-user"></p>
        <p class="cv-sub-ai"></p>
      </div>
      <p class="cv-notice" role="alert"></p>
      <div class="cv-controls">
        <button type="button" class="cv-main-btn"></button>
        <label class="cv-toggle"><input type="checkbox" class="cv-automic" checked /> 自動マイク</label>
        <select class="cv-speaker" hidden aria-label="声の種類"></select>
        <button type="button" class="cv-end-btn">終了</button>
      </div>
      <form class="cv-textform" hidden>
        <input class="cv-textinput" type="text" placeholder="文字で質問する…" />
        <button type="submit">送信</button>
      </form>
      <p class="cv-credit">音声合成：VOICEVOX Nemo</p>`;
    doc.body.appendChild(overlay);

    const q = (sel) => overlay.querySelector(sel);
    const orbEl = q(".cv-orb");
    const charEl = q(".cv-char");
    const charImgs = {};
    charEl.querySelectorAll("img[data-frame]").forEach((img) => {
      charImgs[img.dataset.frame] = img;
    });
    const statusEl = q(".cv-status");
    const subUser = q(".cv-sub-user");
    const subAi = q(".cv-sub-ai");
    const noticeEl = q(".cv-notice");
    const mainBtn = q(".cv-main-btn");
    const endBtn = q(".cv-end-btn");
    const automicBox = q(".cv-automic");
    const speakerSel = q(".cv-speaker");
    const engineEl = q(".cv-engine");
    const textForm = q(".cv-textform");
    const textInput = q(".cv-textinput");

    /* キャラのフレーム（cube_character_frames.js）を遅延読み込み
       して口パクを組み立てる。失敗したら玉表示のまま（会話は
       続けられる）。 */
    function ensureCharacter() {
      if (charState.ready || charState.loading) return;
      charState.loading = true;
      const build = () => {
        const frames = win.CubeCharFrames || null;
        if (!frames) { charState.loading = false; return; }
        let pending = 0;
        let done = false;
        const finish = () => {
          if (done) return;
          done = true;
          if (!charImgs.base || !charImgs.base.dataset.ok) {
            charState.loading = false; // 土台が読めなければ玉のまま
            return;
          }
          charState.lipSync = createLipSync({
            show(key) {
              const target = (charImgs[key] && charImgs[key].dataset.ok) ? key : "base";
              for (const k of Object.keys(charImgs)) {
                charImgs[k].classList.toggle("on", k === target);
              }
            },
            requestAnimationFrame: win.requestAnimationFrame
              ? win.requestAnimationFrame.bind(win) : null,
            cancelAnimationFrame: win.cancelAnimationFrame
              ? win.cancelAnimationFrame.bind(win) : null,
          });
          charEl.hidden = false;
          orbEl.style.display = "none";
          charState.ready = true;
          charState.loading = false;
          if (!overlay.hidden) charState.lipSync.start();
        };
        for (const key of Object.keys(charImgs)) {
          const src = frames[key];
          if (!src) continue;
          pending += 1;
          const img = charImgs[key];
          const settle = (ok) => {
            if (ok) {
              img.dataset.ok = "1";
              if (img.decode) img.decode().catch(() => {});
            }
            if (--pending === 0) finish();
          };
          img.onload = () => settle(true);
          img.onerror = () => settle(false);
          img.src = src;
        }
        if (pending === 0) { charState.loading = false; return; }
        setTimeout(finish, 4000); // 読み込みが固まった場合の保険
      };
      if (win.CubeCharFrames) { build(); return; }
      const script = doc.createElement("script");
      script.src = e.charFramesUrl || "cube_character_frames.js";
      script.onload = build;
      script.onerror = () => { charState.loading = false; };
      doc.head.appendChild(script);
    }

    function renderEngine() {
      engineEl.textContent = session.engineKind === "voicevox" ? "声：VOICEVOX Nemo" : "声：ブラウザ内蔵";
    }
    function renderMain() {
      const s = session.state;
      mainBtn.disabled = false;
      if (s === "speaking") mainBtn.textContent = "割り込む";
      else if (s === "thinking") { mainBtn.textContent = "考え中…"; mainBtn.disabled = true; }
      else if (s === "error") mainBtn.textContent = "もう一度";
      else if (s === "listening") {
        if (session.autoMic) { mainBtn.textContent = "聞き取り中…"; mainBtn.disabled = true; }
        else mainBtn.textContent = "話す";
      } else mainBtn.textContent = "はじめる";
    }
    session.on("state", (s) => {
      overlay.dataset.state = s;
      statusEl.textContent = s === "listening" && !session.autoMic
        ? "タップして話してください" : (STATUS_TEXT[s] || "");
      if (s === "listening") subUser.textContent = "";
      renderMain();
      maybeUpgradeEngine();
    });
    session.on("subtitle", (text) => {
      if (session.state === "speaking") subAi.textContent = text;
      else if (session.state === "listening") subUser.textContent = text;
    });
    session.on("user", (text) => { subUser.textContent = text; subAi.textContent = ""; });
    session.on("notice", (text) => { noticeEl.textContent = text; });
    session.on("engine", renderEngine);

    mainBtn.addEventListener("click", () => {
      const s = session.state;
      if (s === "speaking") session.interrupt();
      else if (s === "error") session.start();
      else if (s === "listening" && !session.autoMic) session.startListening();
    });
    overlay.querySelector(".cv-orb-wrap").addEventListener("click", () => {
      if (session.state === "speaking") session.interrupt();
    });
    endBtn.addEventListener("click", () => {
      session.end();
      if (charState.lipSync) charState.lipSync.stop();
      overlay.hidden = true;
    });
    automicBox.addEventListener("change", () => session.setAutoMic(automicBox.checked));
    speakerSel.addEventListener("change", () => {
      voiceState.speaker = speakerSel.value === "" ? null : Number(speakerSel.value);
      if (session.engineKind === "voicevox" && !queue.busy) session.setEngine(makeVoicevox());
    });
    textForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const v = textInput.value.trim();
      if (!v) return;
      textInput.value = "";
      if (session.state === "idle" || session.state === "ended") session.start();
      session.submitUtterance(v);
    });
    if (!recognizer.available) textForm.hidden = false;

    /* VOICEVOX が使えるなら本線へ（再生中でないときだけ差し替える） */
    function maybeUpgradeEngine() {
      if (!voiceState.voicevoxReady) return;
      if (session.engineKind === "voicevox" || queue.busy) return;
      session.setEngine(makeVoicevox());
      renderEngine();
      void loadSpeakers();
    }
    async function loadSpeakers() {
      if (!speakerSel.hidden) return;
      try {
        const res = await win.fetch(`${serverBase}/voice/speakers`);
        if (!res.ok) return;
        const j = await res.json();
        const list = (j && j.speakers) || [];
        if (!list.length) return;
        speakerSel.innerHTML = "";
        for (const sp of list) {
          const opt = doc.createElement("option");
          opt.value = String(sp.id);
          opt.textContent = sp.style ? `${sp.name}（${sp.style}）` : sp.name;
          speakerSel.appendChild(opt);
        }
        if (voiceState.speaker != null) speakerSel.value = String(voiceState.speaker);
        else if (j.defaultSpeaker != null) {
          speakerSel.value = String(j.defaultSpeaker);
          voiceState.speaker = Number(j.defaultSpeaker);
        }
        speakerSel.hidden = false;
      } catch { /* 声一覧が無くても会話はできる */ }
    }

    entry.addEventListener("click", () => {
      overlay.hidden = false;
      noticeEl.textContent = "";
      ensureCharacter();
      if (charState.lipSync) charState.lipSync.start();
      // iOS 等の自動再生制限の解除：ユーザー操作のうちに、VOICEVOX
      // エンジンの永続 Audio 要素で無音を一度再生して許可を得る。
      // 使い捨ての要素で解除しても文の再生に使う要素には効かず、
      // NotAllowedError になる（実機で確認済みの不具合）。要素は
      // エンジン側で1個に固定してあるので、ここでその要素を unlock
      // すれば以後の src 差し替え再生が通る。
      try {
        const eng = voicevoxEngine || makeVoicevox();
        if (eng && typeof eng.unlock === "function") eng.unlock();
      } catch { /* noop */ }
      session.start();
      renderEngine();
      renderMain();
      if (!voiceState.voicevoxReady) {
        void probeVoicevox({ fetch: win.fetch.bind(win), serverBaseUrl: serverBase })
          .then((ok) => { voiceState.voicevoxReady = ok; maybeUpgradeEngine(); });
      } else {
        maybeUpgradeEngine();
      }
    });

    return { session, overlay, entry };
  }

  if (typeof document !== "undefined" && typeof window !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => mountBrowserUi());
    } else {
      mountBrowserUi();
    }
  }

  /* ---------------------------------------------------------------- */
  /* 公開                                                            */
  /* ---------------------------------------------------------------- */

  const CubeVoice = {
    createSession,
    createSpeechQueue,
    createWebSpeechRecognizer,
    createVoicevoxEngine,
    createBrowserTtsEngine,
    createBrowserHost,
    probeVoicevox,
    mountBrowserUi,
    createLipSync,
    vowelAtTime,
    frameKeyForVowel,
    splitSentences,
    sanitizeForSpeech,
    _internal: {
      normalizeForCompare,
      looksLikeEcho,
      isIosLike,
      DEFAULT_SERVER_BASE,
      MAX_SPOKEN_SENTENCES,
    },
  };

  global.CubeVoice = CubeVoice;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = CubeVoice;
  }
})(typeof window !== "undefined" ? window : globalThis);
