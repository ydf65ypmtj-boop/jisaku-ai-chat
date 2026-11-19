"use strict";
/*
* Cube「健康相談（試作）」— クライアントモジュール（cube_doctor.js）
*
* 設計メモ ~/workspace/goals/ai-employee-company/files/doctor-ai-v1-design.md
* 厳守の試作。診断はしない。働きは次の4つだけ：
* 1) 症状の構造化ヒアリング（知識ベースの質問を順に聞く）
* 2) 受診目安の案内（知識ベースの4区分。レッドフラグはルール側で即判定）
* 3) 出典つき一般情報（知識ベースにある内容だけ、出典を必ず併記）
* 4) 相談メモの生成（医師に見せる形式）
*
* 安全の原則（設計メモの核心）：
* - 根拠は知識ベース（CubeDoctorKB）だけ。載っていない症状・病気・薬には
* 断言せず「分かりません。医療機関に相談してください」で引く。
* - レッドフラグ判定と禁止表現チェックは生成文に任せずルール側で実装する。
* 禁止表現チェックは全ての出力文に機械的にかける。引っかかった文は
* 安全な固定文に差し替える。
* - 受診目安の区分判定もルール側：回答と知識ベースの区分基準文との
* 文字バイグラム重なりで機械的に決める。一致が無ければ保守側
* （今日中に受診）に倒す（医療過疎地向けの保守バイアス、設計メモ）。
* - 健康相談の内容は通常のチャット履歴・知識メモ帳とは分離し、
* 端末の保存領域にもサーバーにも一切書かない。セッションは
* メモリ上だけに持ち、ページを閉じる／最初からやり直すと消える。
* - 妊娠中の人には一般情報を出さず、かかりつけ産科への相談を既定にする。
* 薬の個別判断（何を・どれだけ・続ける／やめる）には踏み込まない。
*
* 知識ベースは cube_doctor_kb.js（生成物）が globalThis.CubeDoctorKB に
* 置く。このモジュール単体では読み込まず、ブラウザでは初回表示時に
* 遅延読み込みする（失敗してもチャット本体には影響しない）。
*
* 使い方（ブラウザ）：<script src="cube_doctor.js"></script> を置けば
* #doctor-root に自動で組み立てる。
* Node（テスト）：const CubeDoctor = require("./cube_doctor.js");
* const kb = require("./cube_doctor_kb.js");
*/
(function (global) {

var VERSION = 1;

/* 画面に常時出す注意書き（設計メモ：誤認防止のため常時明示） */
var ALWAYS_NOTICE =
"これは診断ではありません。医師の診察の代わりにはなりません。";
var PROTOTYPE_NOTICE =
"試作・監修前：この健康相談は内部試作です。医療専門家の監修を" +
"まだ受けていません。急な症状や判断に迷う場合は、医療機関や" +
"救急相談窓口（#7119、実施地域のみ）に相談してください。";

/* 禁止表現チェックに引っかかった文の差し替え先（安全な固定文） */
var SAFE_FALLBACK =
"すみません、安全確認のため、この返答は表示できませんでした。" +
"症状がつらい場合や判断に迷う場合は、医療機関や救急相談窓口" +
"（#7119、実施地域のみ）に相談してください。" +
"この目安は診断ではありません。医師の診察の代わりにはなりません。";

var ABSTAIN_MSG =
"そのことについては分かりません。推測でお答えすることは" +
"できません。医療機関に相談してください。判断に迷う場合は、" +
"救急相談窓口（#7119、実施地域のみ）も利用できます。" +
"この目安は診断ではありません。医師の診察の代わりにはなりません。";

var PREGNANCY_MSG =
"妊娠中の方の症状については、この相談では一般情報を出さない" +
"決まりにしています。かかりつけの産科に相談してください。" +
"急な腹痛や出血、息苦しさなどがある場合は、ためらわず" +
"119番に電話してください。" +
"この目安は診断ではありません。医師の診察の代わりにはなりません。";

var MEDICATION_MSG =
"薬についての個別の判断（どの薬を使うか、量や回数、続けるか" +
"やめるか）は、この相談ではお答えできません。処方した医師や" +
"薬剤師に相談してください。飲んでいる薬は、自己判断で" +
"止めないでください。" +
"この目安は診断ではありません。医師の診察の代わりにはなりません。";

var STOP_MED_NOTICE =
"薬を続けるかやめるかの判断は、この相談ではできません。" +
"自己判断で止めず、処方した医師や薬剤師に相談してください。";

var ENDED_MSG =
"この相談は終了しています。別の症状を相談したいときは、" +
"「最初から」ボタンを押してください。急な症状がある場合は、" +
"ためらわず119番に電話してください。";

/* ------------------------------------------------------------ */
/* テキスト処理（ルール判定の道具） */
/* ------------------------------------------------------------ */

function normalize(text) {
return String(text || "")
.replace(/[\s　]/g, "")
.replace(/[！-～]/g, function (c) {
return String.fromCharCode(c.charCodeAt(0) - 0xfee0);
})
.toLowerCase();
}

function bigramSet(text) {
var s = normalize(text);
var set = new Set();
for (var i = 0; i + 2 <= s.length; i++) set.add(s.slice(i, i + 2));
return set;
}

/* a のバイグラムのうち b にも含まれる割合（0..1） */
function bigramCoverage(aText, bText) {
var a = bigramSet(aText);
if (a.size === 0) return 0;
var b = bigramSet(bText);
var hit = 0;
a.forEach(function (g) { if (b.has(g)) hit++;});
return hit / a.size;
}

/* ------------------------------------------------------------ */
/* 禁止表現チェック（banned_phrases.json の機械適用） */
/* ------------------------------------------------------------ */

function checkBanned(text, banned) {
var hits = [];
if (!banned ||!banned.categories) return hits;
var t = String(text || "");
banned.categories.forEach(function (cat) {
(cat.patterns || []).forEach(function (pat) {
var re;
try { re = new RegExp(pat);} catch (e) { return;}
if (re.test(t)) hits.push({ category: cat.id, pattern: pat});
});
});
return hits;
}

/* ------------------------------------------------------------ */
/* レッドフラグ判定（ルール側・広めに取る） */
/* */
/* パターンは部分一致。ただし「息苦しさはありません」のような */
/* 否定の回答を緊急と誤判定しないため、パターンが現れた文節の */
/* 残り（パターン自体を除いた部分）に否定語があれば、その一致は */
/* 否定されたものとして扱う。自傷の訴え（crisis_support）は */
/* 否定でも拾う（「死にたくない」も支援案内を出す側に倒す）。 */
/* ------------------------------------------------------------ */

var NEGATION_TOKENS = [
"ありません", "ないです", "ではない", "じゃありません",
"ません", "なし", "無い", "ない", "大丈夫", "平気",
];

function segmentAround(text, idx, len) {
var start = 0;
var delims = "、。！？\n";
for (var i = idx - 1; i >= 0; i--) {
if (delims.indexOf(text[i]) >= 0) { start = i + 1; break;}
}
var end = text.length;
for (var j = idx + len; j < text.length; j++) {
if (delims.indexOf(text[j]) >= 0) { end = j; break;}
}
return text.slice(start, idx) + text.slice(idx + len, end);
}

function findRedFlag(text, redFlags) {
if (!redFlags ||!redFlags.rules) return null;
var t = String(text || "");
for (var r = 0; r < redFlags.rules.length; r++) {
var rule = redFlags.rules[r];
var pats = rule.patterns || [];
for (var p = 0; p < pats.length; p++) {
var idx = t.indexOf(pats[p]);
if (idx < 0) continue;
if (rule.action && rule.action.type!== "crisis_support") {
var rest = segmentAround(t, idx, pats[p].length);
var negated = NEGATION_TOKENS.some(function (n) {
return rest.indexOf(n) >= 0;
});
if (negated) continue;
}
return rule;
}
}
return null;
}

/* ------------------------------------------------------------ */
/* 症状の照合（知識ベースの title / aliases のみを使う） */
/* 完全な部分一致を優先し、無ければバイグラムの重なりで判定。 */
/* 基準に届かなければ「該当なし」＝棄権（推測で埋めない）。 */
/* ------------------------------------------------------------ */

var BIGRAM_MATCH_THRESHOLD = 0.75;

function matchSymptom(text, entries) {
var t = String(text || "");
var exact = [];
entries.forEach(function (e) {
var names = [e.title].concat(e.aliases || []);
var hit = names.some(function (n) { return n && t.indexOf(n) >= 0;});
if (hit) exact.push(e);
});
if (exact.length === 1) return { entry: exact[0], candidates: []};
if (exact.length > 1) return { entry: null, candidates: exact};

var scored = entries.map(function (e) {
var names = [e.title].concat(e.aliases || []).join(" ");
return { entry: e, score: bigramCoverage(t, names)};
});
scored.sort(function (a, b) { return b.score - a.score;});
if (
scored.length &&
scored[0].score >= BIGRAM_MATCH_THRESHOLD &&
(scored.length === 1 || scored[0].score > scored[1].score)
) {
return { entry: scored[0].entry, candidates: []};
}
return { entry: null, candidates: []};
}

/* ------------------------------------------------------------ */
/* 受診目安の区分判定（ルール側） */
/* 回答全体と各区分の基準文とのバイグラム重なりで決める。 */
/* 同点・無一致のときは保守側（今日中に受診）に倒す。 */
/* 発熱の「41℃以上」のように、基準文に℃つき数値がある場合は */
/* 回答中の体温がそれ以上なら少なくとも「今日中」にする。 */
/* ------------------------------------------------------------ */

var TRIAGE_ORDER = ["emergency_119", "today", "within_days", "watchful_waiting"];

function determineTriage(entry, answers) {
var triage = entry.triage || {};
var joined = (answers || []).map(function (a) { return a.answer;}).join(" ");
var scores = {};
TRIAGE_ORDER.forEach(function (key) {
var lv = triage[key];
if (!lv) return;
var criteriaText = (lv.criteria || []).join(" ");
scores[key] = bigramCoverage(criteriaText, joined);
});
var best = null;
var bestScore = 0;
TRIAGE_ORDER.forEach(function (key) {
if (!(key in scores)) return;
if (scores[key] > bestScore) { best = key; bestScore = scores[key];}
});
var defaulted = false;
if (!best || bestScore === 0) {
best = triage.today? "today": TRIAGE_ORDER.find(function (k) { return triage[k];});
defaulted = true;
}
/* 体温の数値ルール（基準文に「N℃」がある場合だけ働く） */
var todayCriteria = triage.today? (triage.today.criteria || []).join(" "): "";
var m = todayCriteria.match(/(\d+(?:\.\d+)?)℃/);
if (m && best!== "emergency_119") {
var limit = parseFloat(m[1]);
var nums = joined.match(/(\d+(?:\.\d+)?)\s*(?:℃|度)/g) || [];
var over = nums.some(function (n) { return parseFloat(n) >= limit;});
if (over && (best === "within_days" || best === "watchful_waiting")) {
best = "today";
}
}
return { levelKey: best, level: triage[best], scores: scores, defaulted: defaulted};
}

/* ------------------------------------------------------------ */
/* セッション（状態機械。保存は一切しない） */
/* ------------------------------------------------------------ */

function createSession(kb, opts) {
opts = opts || {};
var nowFn = opts.now || function () { return new Date();};
var session = {
state: "choosing", /* choosing | interview | done | emergency */
entry: null,
qIndex: 0,
answers: [], /* {question, answer} */
candidates: null,
triageResult: null,
startedAt: null,
};

function msg(text, extra) {
var m = { role: "bot", text: String(text)};
if (extra) Object.assign(m, extra);
/* 全ての出力に禁止表現チェックを機械的にかける */
var hits = checkBanned(m.text, kb.banned_phrases);
if (hits.length) {
m.text = SAFE_FALLBACK;
m.bannedHits = hits;
}
return m;
}

function sourceList(sourceIds, entry) {
var byId = {};
(entry.sources || []).forEach(function (s) { byId[s.source_id] = s;});
return (sourceIds || []).map(function (id) {
var s = byId[id];
return s? { id: id, name: s.name, url: s.url, publisher: s.publisher}: null;
}).filter(Boolean);
}

function emergencyResponse(rule) {
var lines = [rule.action.message_template];
if (rule.action.type === "call_119") {
lines.push("自分で運転して受診しないでください。近くに人がいる場合は、その人にも知らせてください。");
}
var srcs = (rule.sources || []).map(function (s) {
return { id: s.source_id, name: s.name, url: s.url, publisher: s.publisher};
});
session.state = "emergency";
return [msg(lines.join("\n"), { kind: "emergency", ruleId: rule.id, sources: srcs})];
}

function beginInterview(entry) {
session.entry = entry;
session.qIndex = 0;
session.answers = [];
session.state = "interview";
session.startedAt = nowFn();
var intro =
"「" + entry.title + "」についてお聞きします。" +
"全部で" + entry.interview_questions.length + "つ質問します。" +
"分かる範囲で答えてください。途中でつらい症状が出たら、" +
"質問の途中でもそのまま書いてください。";
var q = entry.interview_questions[0];
return [msg(intro), msg(q.question, { kind: "question", questionId: q.id})];
}

function buildResult() {
var entry = session.entry;
var tri = determineTriage(entry, session.answers);
session.triageResult = tri;
var out = [];

/* 受診目安 */
var lv = tri.level;
var lines = ["" + lv.label];
(lv.criteria || []).forEach(function (c) { lines.push("・" + c);});
if (lv.departments && lv.departments.length) {
lines.push("受診先の目安：" + lv.departments.join("、"));
}
if (tri.levelKey === "watchful_waiting") {
if (lv.worsening_signs && lv.worsening_signs.length) {
lines.push("次のような悪化のサインが出たら、区分を上げて受診してください。");
lv.worsening_signs.forEach(function (c) { lines.push("・" + c);});
}
if (lv.note) lines.push(lv.note);
}
if (tri.defaulted) {
lines.push(entry.conservative_bias || "迷う場合は、受診する側に倒して案内します。");
}
lines.push("この区分は、お聞きした内容と目安の基準を照らし合わせて機械的に出したものです。");
out.push(msg(lines.join("\n"), { kind: "triage", levelKey: tri.levelKey}));

/* 一般情報（出典つき。出典の無い情報は KB に入っていない） */
if (entry.general_info && entry.general_info.length) {
var infoLines = [""];
var allSrcs = [];
entry.general_info.forEach(function (gi) {
var srcs = sourceList(gi.source_ids, entry);
var names = srcs.map(function (s) { return s.name;}).join("／");
infoLines.push("・" + gi.text + (names? "（出典：" + names + "）": ""));
srcs.forEach(function (s) {
if (!allSrcs.some(function (x) { return x.id === s.id;})) allSrcs.push(s);
});
});
out.push(msg(infoLines.join("\n"), { kind: "info", sources: allSrcs}));
}

/* 全区分の一覧＋締め（KB の文面どおり） */
var all = [""];
TRIAGE_ORDER.forEach(function (key) {
var l = entry.triage[key];
if (!l) return;
all.push("■" + l.label);
(l.criteria || []).forEach(function (c) { all.push("・" + c);});
});
all.push(entry.closing_note);
out.push(msg(all.join("\n"), { kind: "closing"}));

session.state = "done";
return out;
}

session.start = function () {
var names = kb.entries.map(function (e) { return e.title;}).join("、");
var text =
"健康相談（試作）です。症状に合わせて質問をし、受診の目安と" +
"一般情報をお伝えします。\n" + ALWAYS_NOTICE + "\n" +
"相談できる症状は次のとおりです：" + names + "。\n" +
"症状のボタンを押すか、症状をそのまま書いてください。" +
"急な強い症状がある場合は、この相談を使わず119番に" +
"電話してください。";
return [msg(text, { kind: "start"})];
};

session.selectEntry = function (entryId) {
var entry = kb.entries.find(function (e) { return e.id === entryId;});
if (!entry) return [msg(ABSTAIN_MSG, { kind: "abstain"})];
session.candidates = null;
return beginInterview(entry);
};

session.input = function (rawText) {
var text = String(rawText || "").trim();
if (!text) return [];

/* 1) レッドフラグは状態を問わず最優先で判定する */
var rule = findRedFlag(text, kb.red_flags);
if (rule) return emergencyResponse(rule);

/* 2) 妊娠中の人は一般情報を出さず産科相談を既定にする */
if (/妊娠/.test(text)) {
session.state = "done";
return [msg(PREGNANCY_MSG, { kind: "pregnancy"})];
}

/* 3) 薬の個別判断の質問には踏み込まない */
var asksMedication =
/薬/.test(text) &&
/(飲|やめ|止め|量|回数|効|処方|市販|買)/.test(text) &&
/(？|\?|ですか|いい|べき|たい|教えて)/.test(text);
if (asksMedication) {
var out = [msg(MEDICATION_MSG, { kind: "medication"})];
if (session.state === "interview" && session.entry) {
var q = session.entry.interview_questions[session.qIndex];
out.push(msg("質問に戻ります。" + q.question, { kind: "question", questionId: q.id}));
}
return out;
}

if (session.state === "done" || session.state === "emergency") {
return [msg(ENDED_MSG, { kind: "ended"})];
}

if (session.state === "choosing") {
/* 候補が提示済みなら、その中から照合する */
var pool = session.candidates || kb.entries;
var m = matchSymptom(text, pool);
if (m.entry) {
session.candidates = null;
return beginInterview(m.entry);
}
if (m.candidates && m.candidates.length) {
session.candidates = m.candidates;
var names = m.candidates.map(function (e) { return e.title;}).join("、");
return [msg(
"当てはまる症状が複数あります：" + names + "。" +
"どちらの相談ですか。ボタンを押すか、症状の名前を書いてください。",
{ kind: "choose"}
)];
}
return [msg(ABSTAIN_MSG, { kind: "abstain"})];
}

if (session.state === "interview") {
var outMsgs = [];
if (/薬/.test(text) && /(やめたい|止めたい|やめよう|止めよう)/.test(text)) {
outMsgs.push(msg(STOP_MED_NOTICE, { kind: "medication"}));
}
var cur = session.entry.interview_questions[session.qIndex];
session.answers.push({ question: cur.question, answer: text});
session.qIndex++;
if (session.qIndex < session.entry.interview_questions.length) {
var nq = session.entry.interview_questions[session.qIndex];
outMsgs.push(msg(nq.question, { kind: "question", questionId: nq.id}));
return outMsgs;
}
return outMsgs.concat(buildResult());
}

return [msg(ABSTAIN_MSG, { kind: "abstain"})];
};

/* 相談メモ（医師に見せる形式）。保存はせず、表示とコピーだけ。 */
session.buildMemo = function () {
if (!session.entry || session.state!== "done") return null;
var lines = [];
lines.push("医師に見せるためのメモ");
lines.push("日時：" + nowFn().toLocaleString("ja-JP"));
lines.push("相談した症状：" + session.entry.title);
lines.push("聞き取り内容：");
session.answers.forEach(function (a, i) {
lines.push((i + 1) + ". " + a.question);
lines.push("　→ " + a.answer);
});
if (session.triageResult && session.triageResult.level) {
lines.push("案内した受診の目安：" + session.triageResult.level.label +
"（目安であり、診断ではありません）");
}
lines.push("※このメモは健康相談（試作・監修前）の聞き取りを整理したものです。");
lines.push(ALWAYS_NOTICE);
var memo = lines.join("\n");
if (checkBanned(memo, kb.banned_phrases).length) return SAFE_FALLBACK;
return memo;
};

return session;
}

  /* ------------------------------------------------------------ */
  /* パネル UI（DOM は textContent のみで組み立てる）              */
  /* ------------------------------------------------------------ */

  function createPanel(container, opts) {
    opts = opts || {};
    var doc = opts.document ||
      (typeof document !== "undefined" ? document : null);
    var kb = opts.kb || global.CubeDoctorKB;
    if (!doc || !container) return null;
    if (!kb) {
      var err = doc.createElement("p");
      err.className = "doctor-error";
      err.textContent = "知識ベースを読み込めませんでした。通信の状態を確認して、ページを開き直してください。";
      container.appendChild(err);
      return null;
    }

    var root = doc.createElement("div");
    root.className = "doctor-root";
    container.appendChild(root);

    /* 試作バナー（常時表示） */
    var banner = doc.createElement("div");
    banner.className = "doctor-banner";
    banner.textContent = PROTOTYPE_NOTICE + " " + ALWAYS_NOTICE;
    root.appendChild(banner);

    /* 操作行：大きな文字トグル／最初から */
    var controls = doc.createElement("div");
    controls.className = "doctor-controls";
    var largeBtn = doc.createElement("button");
    largeBtn.type = "button";
    largeBtn.className = "doctor-btn";
    largeBtn.textContent = "文字を大きく";
    largeBtn.setAttribute("aria-pressed", "false");
    var restartBtn = doc.createElement("button");
    restartBtn.type = "button";
    restartBtn.className = "doctor-btn";
    restartBtn.textContent = "最初から";
    controls.appendChild(largeBtn);
    controls.appendChild(restartBtn);
    root.appendChild(controls);

    /* 症状ボタン */
    var symptomRow = doc.createElement("div");
    symptomRow.className = "doctor-symptoms";
    kb.entries.forEach(function (e) {
      var b = doc.createElement("button");
      b.type = "button";
      b.className = "doctor-symptom-btn";
      b.textContent = e.title;
      b.addEventListener("click", function () {
        appendUser(e.title);
        render(session.selectEntry(e.id));
      });
      symptomRow.appendChild(b);
    });
    root.appendChild(symptomRow);

    /* 会話ログ */
    var log = doc.createElement("div");
    log.className = "doctor-log";
    log.setAttribute("aria-live", "polite");
    root.appendChild(log);

    /* 相談メモ表示域 */
    var memoWrap = doc.createElement("div");
    memoWrap.className = "doctor-memo-wrap";
    memoWrap.hidden = true;
    var memoBtn = doc.createElement("button");
    memoBtn.type = "button";
    memoBtn.className = "doctor-btn";
    memoBtn.textContent = "相談メモを作る（医師に見せる用）";
    var memoPre = doc.createElement("pre");
    memoPre.className = "doctor-memo";
    memoPre.hidden = true;
    var copyBtn = doc.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "doctor-btn";
    copyBtn.textContent = "メモをコピー";
    copyBtn.hidden = true;
    memoWrap.appendChild(memoBtn);
    memoWrap.appendChild(memoPre);
    memoWrap.appendChild(copyBtn);
    root.appendChild(memoWrap);

    /* 入力フォーム */
    var form = doc.createElement("form");
    form.className = "doctor-form";
    var input = doc.createElement("textarea");
    input.className = "doctor-input";
    input.rows = 2;
    input.placeholder = "症状や、質問への答えを書いてください";
    var sendBtn = doc.createElement("button");
    sendBtn.type = "submit";
    sendBtn.className = "doctor-btn doctor-send";
    sendBtn.textContent = "送る";
    form.appendChild(input);
    form.appendChild(sendBtn);
    root.appendChild(form);

    var session = createSession(kb, opts.sessionOpts || {});

    function appendBubble(role, text, kind) {
      var div = doc.createElement("div");
      div.className = "doctor-msg doctor-msg--" + role +
        (kind ? " doctor-msg--" + kind : "");
      div.textContent = text;
      log.appendChild(div);
      return div;
    }

    function appendSources(parent, sources) {
      if (!sources || !sources.length) return;
      var box = doc.createElement("div");
      box.className = "doctor-sources";
      var label = doc.createElement("span");
      label.textContent = "出典：";
      box.appendChild(label);
      sources.forEach(function (s, i) {
        if (i > 0) box.appendChild(doc.createTextNode("／"));
        var a = doc.createElement("a");
        a.href = s.url;
        a.textContent = s.name;
        a.target = "_blank";
        a.rel = "noopener";
        box.appendChild(a);
      });
      parent.appendChild(box);
    }

    function appendUser(text) {
      appendBubble("user", text);
    }

    function render(messages) {
      (messages || []).forEach(function (m) {
        var div = appendBubble("bot", m.text, m.kind);
        appendSources(div, m.sources);
      });
      memoWrap.hidden = !(session.state === "done" && session.entry);
      try { log.scrollTop = log.scrollHeight; } catch (e) { /* noop */ }
    }

    form.addEventListener("submit", function (ev) {
      if (ev && ev.preventDefault) ev.preventDefault();
      var text = input.value.trim();
      if (!text) return;
      input.value = "";
      appendUser(text);
      render(session.input(text));
    });

    largeBtn.addEventListener("click", function () {
      var large = root.className.indexOf("doctor-large") < 0;
      root.className = large ? "doctor-root doctor-large" : "doctor-root";
      largeBtn.setAttribute("aria-pressed", String(large));
      largeBtn.textContent = large ? "文字を元に戻す" : "文字を大きく";
    });

    restartBtn.addEventListener("click", function () {
      session = createSession(kb, opts.sessionOpts || {});
      log.textContent = "";
      memoPre.hidden = true;
      copyBtn.hidden = true;
      render(session.start());
    });

    memoBtn.addEventListener("click", function () {
      var memo = session.buildMemo();
      if (!memo) return;
      memoPre.textContent = memo;
      memoPre.hidden = false;
      copyBtn.hidden = false;
    });

    copyBtn.addEventListener("click", function () {
      var memo = memoPre.textContent;
      try {
        if (typeof navigator !== "undefined" && navigator.clipboard) {
          navigator.clipboard.writeText(memo);
        }
      } catch (e) { /* コピー失敗時もメモ表示自体は残す */ }
    });

    render(session.start());

    return {
      root: root,
      getSession: function () { return session; },
      elements: {
        banner: banner, log: log, form: form, input: input,
        largeBtn: largeBtn, restartBtn: restartBtn,
        memoWrap: memoWrap, memoBtn: memoBtn, memoPre: memoPre,
      },
    };
  }

  /* ------------------------------------------------------------ */
  /* ブラウザ起動：#doctor-root があれば組み立てる。               */
  /* 知識ベースは初回表示時にだけ遅延読み込みする。                */
  /* ------------------------------------------------------------ */

  function boot() {
    var root = document.getElementById("doctor-root");
    if (!root || root.dataset.doctorBooted) return;
    root.dataset.doctorBooted = "1";
    if (global.CubeDoctorKB) {
      createPanel(root, {});
      return;
    }
    var src = "";
    var scripts = document.getElementsByTagName("script");
    for (var i = 0; i < scripts.length; i++) {
      var s = scripts[i].src || "";
      if (/cube_doctor\.js($|\?)/.test(s)) {
        src = s.replace(/cube_doctor\.js($|\?.*)/, "cube_doctor_kb.js");
        break;
      }
    }
    if (!src) src = "cube_doctor_kb.js";
    var tag = document.createElement("script");
    tag.src = src;
    tag.onload = function () { createPanel(root, {}); };
    tag.onerror = function () {
      var p = document.createElement("p");
      p.className = "doctor-error";
      p.textContent = "健康相談の知識ベースを読み込めませんでした。ページを開き直してください。";
      root.appendChild(p);
    };
    document.head.appendChild(tag);
  }

  var api = {
    VERSION: VERSION,
    ALWAYS_NOTICE: ALWAYS_NOTICE,
    PROTOTYPE_NOTICE: PROTOTYPE_NOTICE,
    SAFE_FALLBACK: SAFE_FALLBACK,
    ABSTAIN_MSG: ABSTAIN_MSG,
    checkBanned: checkBanned,
    findRedFlag: findRedFlag,
    matchSymptom: matchSymptom,
    determineTriage: determineTriage,
    bigramCoverage: bigramCoverage,
    createSession: createSession,
    createPanel: createPanel,
  };

  global.CubeDoctor = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;

  if (typeof document !== "undefined" && typeof window !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot);
    } else {
      boot();
    }
  }
})(typeof window !== "undefined" ? window : globalThis);
