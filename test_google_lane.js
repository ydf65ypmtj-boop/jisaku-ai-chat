#!/usr/bin/env node
"use strict";
/* Google連携レーン（cube-server の /google・/gmail・/gcal）の単体テスト。
*
* - parseGmailSendQuery：送信の意図があるときだけ発火し、宛先は
* 明示されたアドレスだけを拾う。名前だけの場合は missing に
* 宛先が入り、絶対に推測しないこと。
* - mergeGmailDraft：聞き返しへの返信で不足分だけが埋まり、
* 別の話題（質問形）は下書きに吸い込まれないこと。
* - parseGmailReadQuery：新着確認・一覧・最新を読む依頼だけが
* 発火し、送信の依頼は拾わないこと。
* - parseGcalQuery：確認・作成・変更・削除の切り分けと、日付・
* 時刻・予定名の抽出。作成で揃わないものは missing に入ること。
* - fetch 系：JWT つき POST で叩き、ok:false は失敗として区別、
* 通信失敗は null になること。送信は /gmail/send に宛先・件名・
* 本文をそのまま渡すこと。
* - ask() の配線：Google レーンは PC操作レーンより前にあり、
* バッジ gmail / gcal が定義されていること。アカウント欄の
* 連携 UI（initGoogleLinkUi）が script.js 内にあること。
*
* Usage: node test_google_lane.js [path-to-script.js]
*/
const fs = require("fs");
const path = require("path");

const scriptPath = path.resolve(process.argv[2] || path.join(__dirname, "script.js"));
const store = new Map();
global.localStorage = {
getItem: (k) => (store.has(k)? store.get(k): null),
setItem: (k, v) => store.set(k, String(v)),
removeItem: (k) => store.delete(k),
clear: () => store.clear(),
};
let lastFetch = null;
global.fetch = async (url, opts) => {
lastFetch = { url, opts};
throw new TypeError("no-net-in-test");
};

const Cube = require(scriptPath);

let failures = 0;
function check(name, cond, detail) {
if (cond) {
console.log(`PASS ${name}`);
} else {
failures += 1;
console.log(`FAIL ${name}${detail? `: ${detail}`: ""}`);
}
}

async function main() {
/* --- parseGmailSendQuery：発火と抽出 --- */
const full = Cube.parseGmailSendQuery(
"youzhenshengu54@gmail.comに、件名：明日の会議、本文：10時に集合です、とメールを送って");
check("gmail送信 発火: 全部入り",
!!full && full.to === "youzhenshengu54@gmail.com" &&
full.subject === "明日の会議" && full.body === "10時に集合です" &&
full.missing.length === 0, JSON.stringify(full));
const quoted = Cube.parseGmailSendQuery(
"test@example.comに『テスト』という件名で『これはテストです』とメールを送って");
check("gmail送信 発火: 引用形の件名・本文",
!!quoted && quoted.to === "test@example.com" &&
quoted.subject === "テスト" && quoted.body === "これはテストです" &&
quoted.missing.length === 0, JSON.stringify(quoted));
const nameOnly = Cube.parseGmailSendQuery("お母さんにメールを送って");
check("gmail送信: 名前だけは宛先を推測しない",
!!nameOnly && nameOnly.to === "" &&
nameOnly.missing.includes("宛先のメールアドレス"),
JSON.stringify(nameOnly));
const addrOnly = Cube.parseGmailSendQuery("test@example.comにメールを送って");
check("gmail送信: アドレスだけは件名・本文が missing",
!!addrOnly && addrOnly.to === "test@example.com" &&
addrOnly.missing.includes("件名") && addrOnly.missing.includes("本文"),
JSON.stringify(addrOnly));

/* --- parseGmailSendQuery：不発 --- */
const SEND_NO_FIRE = [
"メールの書き方を教えて",
"メール来てる？",
"メールとは何ですか",
"明日の天気は？",
"",
];
for (const q of SEND_NO_FIRE) {
check(`gmail送信 不発: ${q || "(空)"}`,
Cube.parseGmailSendQuery(q) === null,
JSON.stringify(Cube.parseGmailSendQuery(q)));
}

/* --- mergeGmailDraft --- */
const mergedBody = Cube.mergeGmailDraft(
{ to: "a@b.co", subject: "件名", body: ""}, "明日10時に集合です");
check("gmail下書き: 素の返信は不足が1つのとき本文になる",
mergedBody.body === "明日10時に集合です", JSON.stringify(mergedBody));
const mergedAddr = Cube.mergeGmailDraft(
{ to: "", subject: "s", body: "b"}, "a@b.co です");
check("gmail下書き: アドレスの返信で宛先が埋まる",
mergedAddr.to === "a@b.co", JSON.stringify(mergedAddr));
const mergedTopic = Cube.mergeGmailDraft(
{ to: "", subject: "", body: ""}, "明日の天気は？");
check("gmail下書き: 不足が複数のとき別の話題を吸い込まない",
mergedTopic.to === "" && mergedTopic.subject === "" &&
mergedTopic.body === "", JSON.stringify(mergedTopic));
const mergedMarkers = Cube.mergeGmailDraft(
{ to: "", subject: "", body: ""}, "件名：テスト、本文：本文です");
check("gmail下書き: 件名・本文マーカーは常に取り込む",
mergedMarkers.subject === "テスト" && mergedMarkers.body === "本文です",
JSON.stringify(mergedMarkers));

/* --- parseGmailReadQuery --- */
const readNew = Cube.parseGmailReadQuery("メール来てる？");
check("gmail読み取り 発火: 新着確認",
!!readNew && readNew.q === "" && readNew.openFirst === false,
JSON.stringify(readNew));
const readFirst = Cube.parseGmailReadQuery("最新のメールを読んで");
check("gmail読み取り 発火: 最新を読む",
!!readFirst && readFirst.openFirst === true, JSON.stringify(readFirst));
const readFrom = Cube.parseGmailReadQuery("お母さんからのメールある？");
check("gmail読み取り 発火: 差出人で絞る",
!!readFrom && readFrom.q === "from:お母さん", JSON.stringify(readFrom));
const readUnread = Cube.parseGmailReadQuery("未読メールを見せて");
check("gmail読み取り 発火: 未読だけ",
!!readUnread && readUnread.q === "is:unread", JSON.stringify(readUnread));
check("gmail読み取り 不発: 送信の依頼は拾わない",
Cube.parseGmailReadQuery("test@example.comにメールを送って") === null);
check("gmail読み取り 不発: 知識質問",
Cube.parseGmailReadQuery("メールの書き方を教えて") === null);

/* --- parseGcalQuery --- */
const todayKey = Cube.gcalNextDateKey(
Cube.parseGcalDate("今日の予定").key); // 翌日キーの形だけ使う
const listTomorrow = Cube.parseGcalQuery("明日の予定は？");
check("gcal確認 発火: 明日は1日分",
!!listTomorrow && listTomorrow.kind === "list" &&
listTomorrow.label === "明日" &&
listTomorrow.timeMin.endsWith("T00:00:00+09:00") &&
listTomorrow.timeMax > listTomorrow.timeMin,
JSON.stringify(listTomorrow));
const listDefault = Cube.parseGcalQuery("予定を教えて");
check("gcal確認 発火: 日付なしは今後1週間",
!!listDefault && listDefault.kind === "list" &&
listDefault.label === "今後1週間", JSON.stringify(listDefault));
const create = Cube.parseGcalQuery("明日の10時に『歯医者』の予定を入れて");
check("gcal作成 発火: 引用の予定名",
!!create && create.kind === "create" && create.title === "歯医者" &&
create.times[0] === "10:00" && create.missing.length === 0 &&
!!create.date, JSON.stringify(create));
const createBare = Cube.parseGcalQuery("明日10時から11時まで会議の予定を追加して");
check("gcal作成 発火: 裸の予定名と時刻範囲",
!!createBare && createBare.title === "会議" &&
createBare.times[0] === "10:00" && createBare.times[1] === "11:00",
JSON.stringify(createBare));
const createMissing = Cube.parseGcalQuery("予定を入れて");
check("gcal作成: 何も無ければ全部 missing",
!!createMissing && createMissing.kind === "create" &&
createMissing.missing.length === 3, JSON.stringify(createMissing));
const createNoDate = Cube.parseGcalQuery("10時に会議の予定を入れて");
check("gcal作成: 日付が無いと missing に日付",
!!createNoDate && createNoDate.missing.includes("日付") &&
createNoDate.title === "会議", JSON.stringify(createNoDate));
const del = Cube.parseGcalQuery("会議の予定を消して");
check("gcal削除 発火: 名前の手がかり",
!!del && del.kind === "delete" && del.hint === "会議",
JSON.stringify(del));
const delOrd = Cube.parseGcalQuery("2番目の予定を消して");
check("gcal削除 発火: 番号指定",
!!delOrd && delOrd.kind === "delete" && delOrd.ordinal === 1,
JSON.stringify(delOrd));
const upd = Cube.parseGcalQuery("会議の予定を明日の15時に変更して");
check("gcal変更 発火: 新しい日時つき",
!!upd && upd.kind === "update" && upd.hint === "会議" &&
upd.times[0] === "15:00" &&!!upd.date && upd.date.label === "明日",
JSON.stringify(upd));
const GCAL_NO_FIRE = [
"明日の天気は？",
"文化の日とは？",
"今日は祝日？",
"",
];
for (const q of GCAL_NO_FIRE) {
check(`gcal 不発: ${q || "(空)"}`,
Cube.parseGcalQuery(q) === null,
JSON.stringify(Cube.parseGcalQuery(q)));
}

/* --- 時刻トークン --- */
check("gcal時刻: 午後と範囲",
JSON.stringify(Cube.gcalTimeTokens("午後3時30分から17時まで")) ===
JSON.stringify(["15:30", "17:00"]),
JSON.stringify(Cube.gcalTimeTokens("午後3時30分から17時まで")));
check("gcal時刻: 所要時間（〜時間）は拾わない",
Cube.gcalTimeTokens("3時間かかる").length === 0);
check("gcal時刻: HH:MM 形式",
JSON.stringify(Cube.gcalTimeTokens("10:30集合")) ===
JSON.stringify(["10:30"]));

/* --- 表示テキスト --- */
const evText = Cube.gcalListText("明日", [{
id: "1", summary: "歯医者",
start: "2026-11-21T10:00:00+09:00", end: "2026-11-21T11:00:00+09:00",
location: "", description: "", htmlLink: "",
}]);
check("gcal表示: 日時と曜日つき",
evText.includes("11月21日（土） 10:00〜11:00：歯医者"), evText);
check("gcal表示: 予定なし",
Cube.gcalListText("明日", []) === "明日の予定はありません。");
const mailText = Cube.gmailListText([{
id: "1", from: "A <a@b.co>", subject: "テスト", date: "",
snippet: "", unread: true,
}]);
check("gmail表示: 未読つき一覧",
mailText.includes("テスト（A <a@b.co>）"), mailText);
check("gmail表示: 0件",
Cube.gmailListText([]) === "条件に合うメールはありませんでした。");

/* --- fetch 系（スタブ） --- */
const okJson = (obj) => async (url, opts) => {
lastFetch = { url, opts};
return { ok: true, status: 200, json: async () => obj};
};
global.fetch = okJson({ ok: true, configured: true, linked: true, google_email: "a@b.co"});
const st = await Cube.fetchGoogleStatus("JWT");
check("fetchGoogleStatus: 連携済みが返る",
!!st && st.ok === true && st.linked === true &&
st.googleEmail === "a@b.co", JSON.stringify(st));
check("fetchGoogleStatus: JWT が Authorization に載る",
!!lastFetch && lastFetch.opts.headers.Authorization === "Bearer JWT" &&
lastFetch.url.endsWith("/google/status"), JSON.stringify(lastFetch));
global.fetch = okJson({ ok: true, messageId: "m1", log: { ts: "t", to: "a@b.co", subject: "s", messageId: "m1"}});
const sent = await Cube.sendGmail("JWT", { to: "a@b.co", subject: "s", body: "b"});
check("sendGmail: 成功と送信ログが返る",
!!sent && sent.ok === true && sent.messageId === "m1" &&
!!sent.log && sent.log.to === "a@b.co", JSON.stringify(sent));
const sentBody = JSON.parse(lastFetch.opts.body);
check("sendGmail: 宛先・件名・本文をそのまま渡す",
lastFetch.url.endsWith("/gmail/send") && sentBody.to === "a@b.co" &&
sentBody.subject === "s" && sentBody.body === "b",
JSON.stringify(sentBody));
global.fetch = okJson({ ok: false, error: "not_linked"});
const ng = await Cube.sendGmail("JWT", { to: "a@b.co", subject: "s", body: "b"});
check("sendGmail: ok:false は失敗として返る",
!!ng && ng.ok === false && ng.error === "not_linked");
global.fetch = async () => { throw new TypeError("down");};
check("sendGmail: 通信失敗は null",
(await Cube.sendGmail("JWT", { to: "a@b.co", subject: "s", body: "b"})) === null);
check("fetchGoogleStatus: トークン無しは null",
(await Cube.fetchGoogleStatus("")) === null);
global.fetch = okJson({ ok: true, events: [{ id: "e1", summary: "会議", start: "2026-11-21T10:00:00+09:00", end: "2026-11-21T11:00:00+09:00"}]});
const evs = await Cube.fetchGcalEvents("JWT", "2026-11-21T00:00:00+09:00", "2026-11-22T00:00:00+09:00");
check("fetchGcalEvents: 予定が整形されて返る",
!!evs && evs.ok === true && evs.events.length === 1 &&
evs.events[0].summary === "会議", JSON.stringify(evs));

/* --- ask() の配線（ソース確認） --- */
const src = fs.readFileSync(scriptPath, "utf8");
const askStart = src.indexOf("async ask(rawQuestion)");
const askSrc = src.slice(askStart, askStart + 60000);
const iGmail = askSrc.indexOf("runGmailSendLane(gmailSendParsed)");
const iGcal = askSrc.indexOf("runGcalLane(gcalParsed)");
const iPc = askSrc.indexOf("detectPcQuery(question)");
const iMeter = askSrc.indexOf("2.5) 時間制メーターの関門");
check("配線: Gmail送信レーンは PC操作レーンより前",
iGmail >= 0 && iPc > iGmail, `gmail=${iGmail} pc=${iPc}`);
check("配線: カレンダーレーンは PC操作レーンより前",
iGcal >= 0 && iPc > iGcal, `gcal=${iGcal} pc=${iPc}`);
check("配線: Googleレーンはメーター関門より前",
iGmail >= 0 && iMeter > iGmail);
check("配線: バッジ gmail / gcal が定義されている",
src.includes("gmail: {") && src.includes('className: "badge-gmail"') &&
src.includes("gcal: {") && src.includes('className: "badge-gcal"'));
check("配線: 連携 UI（initGoogleLinkUi）と送信ログ通知がある",
src.includes("initGoogleLinkUi") &&
src.includes("google-link-btn") &&
src.includes("checkSendLogNotices") &&
src.includes("/gmail/sendlog"));
void todayKey;

console.log(failures? `\n${failures} FAILURES`: "\nALL PASS");
process.exit(failures? 1: 0);
}

main().catch((e) => {
console.log(`FAIL harness error: ${e && e.stack || e}`);
process.exit(1);
});
