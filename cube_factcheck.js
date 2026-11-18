/* cube_factcheck.js — 「嘘チェックの可視化」モジュール
 *
 * 役割は2つ：
 *  1) 回答の検証可視化：検証ゲート（script.js の verifyModelAnswer）が
 *     内部で計算した文単位の判定を、ユーザーに見える「検証の内訳」
 *     パネルに変える。判定そのものはゲートの返り値をそのまま使う。
 *  2) 嘘チェックモード：貼られた文章を文に割り、各主張文を証拠レーン
 *     （Wikipedia 索引＋通常検索レーン）と照合して、文単位に
 *     「裏づけあり／食い違い／裏づけなし」を返す。
 *
 * 原則（ゲートと同じ）：
 *  - 本モジュールは判定基準・閾値を一切持たない。裏づけの判定は全て
 *    ホスト（script.js）の既存関数 verifySentence / verifyModelAnswer /
 *    isClaimSentence 等に委ね、ここでは取り回しと表示だけを行う。
 *  - 証拠が無ければ「裏づけなし（確認できませんでした）」と言うだけ
 *    で、「嘘」とは断定しない。
 *
 * ホストとの接続：script.js のトップレベル関数を globalThis から
 * 遅延解決する（本ファイルは script.js より前に読み込まれるため、
 * 読み込み時点ではまだ存在しない）。Node テストでは host を明示的に
 * 注入できる（依存性注入）。script.js が無い環境では、内訳パネルも
 * 嘘チェックも静かに無効になる（チャットの邪魔をしない）。
 *
 * 公開側のグローバル名：CubeFactCheck（cube_memory.js の CubeMemory、
 * cube_account.js の CubeAccount と同じ方式）。
 */
(function () {
  "use strict";

  const VERSION = 1;
  /* 嘘チェックで照合する主張文の上限（証拠レーンの負荷の上限）。 */
  const MAX_CLAIMS = 12;
  /* 貼り付け文章の長さの上限（文字数）。 */
  const MAX_TEXT_LENGTH = 4000;

  /* ---------------------------------------------------------------- */
  /* ホスト関数の解決                                                   */
  /* ---------------------------------------------------------------- */

  function globalFn(name) {
    try {
      const f = globalThis[name];
      return typeof f === "function" ? f : null;
    } catch {
      return null;
    }
  }

  /**
   * ホスト（ゲート関数群）を解決する。引数 host があればそれを優先し、
   * 無いものは globalThis から拾う。必須関数が欠けていれば null。
   */
  function resolveHost(host) {
    const h = host || {};
    const pick = (name) => h[name] || globalFn(name);
    const resolved = {
      splitForGate: pick("splitForGate"),
      isClaimSentence: pick("isClaimSentence"),
      isAbstentionSentence: pick("isAbstentionSentence"),
      verifySentence: pick("verifySentence"),
      searchWikipedia: pick("searchWikipedia"),
      fetchArticle: pick("fetchArticle"),
      extractFact: pick("extractFact"),
      sanitizeFact: pick("sanitizeFact"),
      scopeToSubject: pick("scopeToSubject"),
      modelAgreesWithFact: pick("modelAgreesWithFact"),
      /* 通常検索レーン（未デプロイの間は存在しない＝接続口だけ）。 */
      searchGeneral: h.searchGeneral || globalFn("searchGeneralWeb"),
    };
    if (
      !resolved.splitForGate ||
      !resolved.isClaimSentence ||
      !resolved.verifySentence
    ) {
      return null;
    }
    return resolved;
  }

  /* 証拠レーンの予算を質問単位で張り直す（ask() と同じ作法）。
     IndexLane は script.js の字句スコープにあるため、ブラウザでは
     裸の識別子で触る。無い環境（Node・単体テスト）では何もしない。 */
  function beginEvidenceLane() {
    try {
      // eslint-disable-next-line no-undef
      if (typeof IndexLane !== "undefined" && IndexLane.beginQuestion) {
        // eslint-disable-next-line no-undef
        IndexLane.beginQuestion({ maxMs: 120000, maxRequests: 60000 });
      }
    } catch {
      /* レーンが無い環境では素通り */
    }
  }

  /* ---------------------------------------------------------------- */
  /* 1) 回答の検証内訳（buildBreakdown）                                */
  /* ---------------------------------------------------------------- */

  const VIA_LABELS = {
    fact: "種類つき事実",
    window: "記事本文の照合",
    definition: "定義の照合",
    superlative: "最上級の裏づけ",
    tautology: "同語反復のため対象外",
    degenerate: "崩壊出力のため対象外",
  };

  /**
   * ゲートの返り値から表示用の内訳を作る。
   * input: { question, gate, fact, results }
   *   gate は verifyModelAnswer の返り値そのもの。文単位の verified は
   *   ゲートの perSentence をそのまま使う（再判定しない）。出典は、
   *   ゲートが内部で使ったのと同じ verifySentence を同じ入力で呼び、
   *   文ごとの出典を取り戻す（判定は変わらない：同じ関数・同じ入力）。
   * 返り値：JSON 化できるプレーンなオブジェクト（履歴保存用）。
   */
  function buildBreakdown(input, host) {
    const h = resolveHost(host);
    const gate = input?.gate;
    if (!gate || !Array.isArray(gate.perSentence)) return null;
    const question = input.question ?? "";
    const ctx = { fact: input.fact ?? null, results: input.results ?? [] };
    const sentences = [];
    const sourceMap = new Map();
    let claims = 0;
    let verifiedClaims = 0;
    for (const ps of gate.perSentence) {
      const entry = {
        text: String(ps.sentence ?? ""),
        kind: ps.kind ?? "claim",
        verdict: "nonclaim",
        via: ps.via ?? null,
        source: null,
        displayed: true,
      };
      if (ps.kind === "abstention") {
        entry.verdict = "abstention";
      } else if (ps.kind === "claim") {
        claims += 1;
        entry.displayed = !!ps.verified;
        if (ps.verified) {
          verifiedClaims += 1;
          entry.verdict = "verified";
          /* 文ごとの出典の取り戻し（ゲートと同じ入力での再実行）。 */
          if (h?.verifySentence) {
            try {
              const v = h.verifySentence(entry.text, question, ctx);
              if (v?.source) {
                entry.source = {
                  title: v.source.title ?? "",
                  url: v.source.url ?? "",
                };
                if (v.via) entry.via = v.via;
              }
            } catch {
              /* 出典が取り戻せなくても内訳自体は出す */
            }
          }
          if (!entry.source && gate.source) {
            entry.source = {
              title: gate.source.title ?? "",
              url: gate.source.url ?? "",
            };
          }
        } else {
          entry.verdict = "unverified";
        }
      }
      if (entry.source && entry.source.url) {
        sourceMap.set(entry.source.url, entry.source);
      }
      sentences.push(entry);
    }
    return {
      v: VERSION,
      kind: "answer",
      status: gate.status ?? null,
      counts: {
        claims,
        verified: verifiedClaims,
        unverified: claims - verifiedClaims,
        sources: sourceMap.size,
      },
      sources: [...sourceMap.values()],
      sentences,
    };
  }

  /* ---------------------------------------------------------------- */
  /* 内訳パネルの描画（renderPanel）                                    */
  /* ---------------------------------------------------------------- */

  function viaLabel(via) {
    return via ? VIA_LABELS[via] || via : null;
  }

  function el(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function sourceLink(doc, source) {
    const a = doc.createElement("a");
    a.href = source.url;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = source.title || source.url;
    return a;
  }

  /**
   * 内訳パネル（<details>）を作る。既定では閉じている（答えの邪魔を
   * しない）。document は通常はグローバルのもの。テストでは注入できる。
   */
  function renderPanel(breakdown, doc) {
    const d = doc || globalThis.document;
    if (!d || !breakdown) return null;
    const { counts } = breakdown;
    const details = el(d, "details", "fc-panel");
    const summary = el(
      d,
      "summary",
      "fc-summary mono",
      `検証の内訳：主張${counts.claims}文中 ${counts.verified}文に裏づけ` +
        (counts.unverified > 0 ? `／${counts.unverified}文は未確認` : "") +
        (counts.sources > 0 ? `（出典${counts.sources}件）` : ""),
    );
    details.appendChild(summary);
    if (breakdown.status === "none") {
      details.appendChild(
        el(
          d,
          "p",
          "fc-note",
          "裏づけの取れた文が無かったため、モデルの答えは表示していません。",
        ),
      );
    }
    const ul = el(d, "ul", "fc-list");
    for (const s of breakdown.sentences) {
      const li = el(d, "li", `fc-row fc-${s.verdict}`);
      const mark =
        s.verdict === "verified"
          ? "✓"
          : s.verdict === "unverified"
            ? "！"
            : "−";
      li.appendChild(el(d, "span", "fc-mark", mark));
      const body = el(d, "span", "fc-sentence");
      body.appendChild(el(d, "span", "fc-text", s.text));
      const meta = el(d, "span", "fc-meta");
      if (s.verdict === "verified") {
        meta.append("裏づけあり");
        if (s.source && s.source.url) {
          meta.append("：");
          meta.appendChild(sourceLink(d, s.source));
        }
        const vl = viaLabel(s.via);
        if (vl) meta.append(`（${vl}）`);
      } else if (s.verdict === "unverified") {
        const vl = viaLabel(s.via);
        meta.textContent = vl
          ? `${vl}（答えには表示していません）`
          : "未確認のため、答えには表示していません";
      } else if (s.verdict === "abstention") {
        meta.textContent = "「分かりません」という表明（主張ではないため判定対象外）";
      } else {
        meta.textContent = "主張ではない文（判定対象外）";
      }
      body.appendChild(meta);
      li.appendChild(body);
      ul.appendChild(li);
    }
    details.appendChild(ul);
    if (breakdown.sources.length > 0) {
      const foot = el(d, "p", "fc-sources");
      foot.append("出典一覧：");
      breakdown.sources.forEach((src, i) => {
        if (i > 0) foot.append("、");
        foot.appendChild(sourceLink(d, src));
      });
      details.appendChild(foot);
    }
    return details;
  }

  /* ---------------------------------------------------------------- */
  /* 2) 嘘チェックモード（checkText）                                   */
  /* ---------------------------------------------------------------- */

  /* 証拠の重複除去（タイトル正規化＋URL）。通常検索レーンの
     mergeEvidenceResults と同じ上限 15 件。 */
  function mergeResults(base, extra) {
    const out = [...(base || [])];
    const seenUrl = new Set(out.map((r) => r?.url).filter(Boolean));
    const seenTitle = new Set(
      out.map((r) => String(r?.title ?? "").replace(/\s+/g, "")).filter(Boolean),
    );
    for (const r of extra || []) {
      if (!r) continue;
      const t = String(r.title ?? "").replace(/\s+/g, "");
      if (r.url && seenUrl.has(r.url)) continue;
      if (t && seenTitle.has(t)) continue;
      if (r.url) seenUrl.add(r.url);
      if (t) seenTitle.add(t);
      out.push(r);
      if (out.length >= 15) break;
    }
    return out.slice(0, 15);
  }

  /* 1文分の証拠集め：ask() の作法（記事→事実の抽出と点検、検索結果の
     対象絞り）を、文自身をクエリ文脈として再利用する。 */
  async function gatherEvidence(sentence, h) {
    let article = null;
    let results = [];
    let fact = null;
    if (h.fetchArticle) {
      try {
        article = await h.fetchArticle(sentence);
      } catch {
        article = null;
      }
    }
    if (h.searchWikipedia) {
      try {
        results = (await h.searchWikipedia(sentence)) || [];
      } catch {
        results = [];
      }
    }
    let generalConnected = false;
    if (h.searchGeneral) {
      generalConnected = true;
      try {
        const extra = await h.searchGeneral(sentence);
        if (Array.isArray(extra) && extra.length > 0) {
          results = mergeResults(results, extra);
        }
      } catch {
        /* 通常検索レーンの失敗は Wikipedia の証拠だけで続行 */
      }
    }
    if (h.scopeToSubject) {
      try {
        results = h.scopeToSubject(sentence, results);
      } catch {
        /* 絞りに失敗しても証拠列はそのまま使う */
      }
    }
    if (article && h.extractFact) {
      try {
        fact = h.extractFact(sentence, article);
      } catch {
        fact = null;
      }
      if (fact && h.sanitizeFact) {
        try {
          fact = h.sanitizeFact(fact, article, sentence);
        } catch {
          fact = null;
        }
      }
    }
    return { article, results, fact, generalConnected };
  }

  /**
   * 貼られた文章を文単位に照合する。
   * opts: { host, maxClaims, onProgress(done, total) }
   * 返り値：{ v, kind:"check", counts, lanes:{wikipedia, general}, sentences }
   *   sentence verdict: verified | mismatch | unverified |
   *                     abstention | nonclaim | skipped
   * 「食い違い」は、文自身を文脈に抽出した種類つき事実と文が食い違う
   * ときだけ付ける（ask() の食い違い判定と同じ modelAgreesWithFact）。
   * それ以外で裏づけが取れなければ「裏づけなし」であり、「嘘」とは
   * 言わない。
   */
  async function checkText(text, opts = {}) {
    const h = resolveHost(opts.host);
    if (!h) return null;
    const maxClaims = opts.maxClaims ?? MAX_CLAIMS;
    const raw = String(text ?? "").slice(0, MAX_TEXT_LENGTH);
    const parts = h.splitForGate(raw);
    const sentences = [];
    let claimSeen = 0;
    let generalConnected = !!h.searchGeneral;
    beginEvidenceLane();
    for (const s of parts) {
      const entry = { text: s, verdict: "nonclaim", via: null, source: null, fact: null };
      sentences.push(entry);
      if (h.isAbstentionSentence && h.isAbstentionSentence(s, "")) {
        entry.verdict = "abstention";
        continue;
      }
      if (!h.isClaimSentence(s, "")) {
        entry.verdict = "nonclaim";
        continue;
      }
      claimSeen += 1;
      if (claimSeen > maxClaims) {
        entry.verdict = "skipped";
        continue;
      }
      const ev = await gatherEvidence(s, h);
      if (ev.generalConnected) generalConnected = true;
      const ctx = { fact: ev.fact, article: ev.article, results: ev.results };
      let v = null;
      try {
        v = h.verifySentence(s, "", ctx);
      } catch {
        v = null;
      }
      if (v) {
        entry.verdict = "verified";
        entry.via = v.via ?? null;
        entry.source = v.source
          ? { title: v.source.title ?? "", url: v.source.url ?? "" }
          : null;
      } else if (
        ev.fact &&
        h.modelAgreesWithFact &&
        !h.modelAgreesWithFact(s, ev.fact)
      ) {
        entry.verdict = "mismatch";
        entry.fact = {
          answer: ev.fact.answer ?? "",
          source: {
            title: ev.fact.sourceTitle ?? "",
            url: ev.fact.sourceUrl ?? "",
          },
        };
      } else {
        entry.verdict = "unverified";
      }
      if (opts.onProgress) {
        try {
          opts.onProgress(claimSeen, Math.min(claimSeen, maxClaims));
        } catch {
          /* 進捗表示の失敗は照合を止めない */
        }
      }
    }
    const counts = { claims: 0, verified: 0, mismatch: 0, unverified: 0, skipped: 0, other: 0 };
    for (const s of sentences) {
      if (s.verdict === "verified") { counts.claims += 1; counts.verified += 1; }
      else if (s.verdict === "mismatch") { counts.claims += 1; counts.mismatch += 1; }
      else if (s.verdict === "unverified") { counts.claims += 1; counts.unverified += 1; }
      else if (s.verdict === "skipped") { counts.claims += 1; counts.skipped += 1; }
      else counts.other += 1;
    }
    return {
      v: VERSION,
      kind: "check",
      counts,
      lanes: { wikipedia: true, general: generalConnected },
      sentences,
    };
  }

  /* ---------------------------------------------------------------- */
  /* 嘘チェック結果の描画                                               */
  /* ---------------------------------------------------------------- */

  const CHECK_VERDICT_LABELS = {
    verified: "裏づけあり",
    mismatch: "食い違い",
    unverified: "裏づけなし",
    abstention: "主張なし",
    nonclaim: "主張なし",
    skipped: "未判定（上限）",
  };

  function renderCheckResults(result, doc) {
    const d = doc || globalThis.document;
    if (!d || !result) return null;
    const wrap = el(d, "div", "fc-check-results");
    const c = result.counts;
    wrap.appendChild(
      el(
        d,
        "p",
        "fc-check-summary mono",
        `主張${c.claims}文：裏づけあり ${c.verified}／食い違い ${c.mismatch}／裏づけなし ${c.unverified}` +
          (c.skipped > 0 ? `／未判定 ${c.skipped}` : ""),
      ),
    );
    if (!result.lanes.general) {
      wrap.appendChild(
        el(
          d,
          "p",
          "fc-note",
          "通常検索レーンは未接続のため、Wikipedia の証拠だけで判定しています。",
        ),
      );
    }
    const ul = el(d, "ul", "fc-list");
    for (const s of result.sentences) {
      const li = el(d, "li", `fc-row fc-${s.verdict}`);
      li.appendChild(
        el(d, "span", `fc-chip fc-chip-${s.verdict}`, CHECK_VERDICT_LABELS[s.verdict] || s.verdict),
      );
      const body = el(d, "span", "fc-sentence");
      body.appendChild(el(d, "span", "fc-text", s.text));
      const meta = el(d, "span", "fc-meta");
      if (s.verdict === "verified") {
        meta.append("裏づけ：");
        if (s.source && s.source.url) meta.appendChild(sourceLink(d, s.source));
        const vl = viaLabel(s.via);
        if (vl) meta.append(`（${vl}）`);
        body.appendChild(meta);
      } else if (s.verdict === "mismatch" && s.fact) {
        meta.append(`ネットの事実では「${s.fact.answer}」`);
        if (s.fact.source && s.fact.source.url) {
          meta.append("（出典：");
          meta.appendChild(sourceLink(d, s.fact.source));
          meta.append("）");
        }
        body.appendChild(meta);
      } else if (s.verdict === "unverified") {
        meta.textContent = "証拠が見つからなかったため、確認できませんでした（＝「嘘」という意味ではありません）";
        body.appendChild(meta);
      } else if (s.verdict === "skipped") {
        meta.textContent = `主張文の上限（${MAX_CLAIMS}文）を超えたため判定していません`;
        body.appendChild(meta);
      }
      li.appendChild(body);
      ul.appendChild(li);
    }
    wrap.appendChild(ul);
    return wrap;
  }

  /* ---------------------------------------------------------------- */
  /* 嘘チェック画面（view-factcheck の中身を組み立てる）                 */
  /* ---------------------------------------------------------------- */

  function initFactCheckView(root) {
    const d = globalThis.document;
    if (!d || !root || root.dataset.fcInit) return;
    root.dataset.fcInit = "1";
    root.appendChild(el(d, "h2", "fc-title", "嘘チェック"));
    root.appendChild(
      el(
        d,
        "p",
        "fc-desc",
        "他のAIの回答やネットの文章を貼ると、文ごとに証拠と照合して「裏づけあり／食い違い／裏づけなし」を返します。「裏づけなし」は「嘘」という意味ではなく、「確認できませんでした」という意味です。",
      ),
    );
    const ta = d.createElement("textarea");
    ta.id = "fc-input";
    ta.rows = 6;
    ta.maxLength = MAX_TEXT_LENGTH;
    ta.placeholder = "チェックしたい文章を貼ってください";
    root.appendChild(ta);
    const row = el(d, "div", "fc-run-row");
    const btn = d.createElement("button");
    btn.type = "button";
    btn.className = "primary-btn";
    btn.textContent = "チェックする";
    row.appendChild(btn);
    const status = el(d, "span", "fc-status mono", "");
    row.appendChild(status);
    root.appendChild(row);
    const resultsBox = el(d, "div", "fc-results-box");
    root.appendChild(resultsBox);
    root.appendChild(
      el(
        d,
        "p",
        "fc-note",
        `判定できるのは、Wikipedia などの証拠で裏づけられる主張だけです。意見・感想・言い換えの大きい文は「裏づけなし」になりやすく、将来のことや証拠の無い話題は判定できません。主張文は最大${MAX_CLAIMS}文まで照合します。`,
      ),
    );
    btn.addEventListener("click", async () => {
      const text = ta.value.trim();
      if (!text) {
        status.textContent = "文章を貼ってください。";
        return;
      }
      btn.disabled = true;
      status.textContent = "照合中…";
      resultsBox.replaceChildren();
      try {
        const result = await checkText(text, {
          onProgress: (done) => {
            status.textContent = `照合中…（${done}文目）`;
          },
        });
        if (!result) {
          status.textContent = "検証の仕組みが読み込めていないため、チェックできませんでした。";
          return;
        }
        status.textContent = "チェック完了";
        const node = renderCheckResults(result);
        if (node) resultsBox.appendChild(node);
      } catch {
        status.textContent = "チェック中にエラーが起きました。";
      } finally {
        btn.disabled = false;
      }
    });
  }

  const api = {
    VERSION,
    MAX_CLAIMS,
    MAX_TEXT_LENGTH,
    buildBreakdown,
    renderPanel,
    checkText,
    renderCheckResults,
    initFactCheckView,
    _resolveHost: resolveHost,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof globalThis !== "undefined") {
    globalThis.CubeFactCheck = api;
  }
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => {
      const root = document.getElementById("fc-root");
      if (root) initFactCheckView(root);
    });
  }
})();
