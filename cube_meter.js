/* cube_meter.js — Cube の時間制メーター（サーバーモデル利用時間）
 *
 * 役割：iMac サーバーで動くモデル（Cube Twin / Pro / Max / Ultra）の
 * 利用時間を「1日あたりの無料枠（秒）＋ポイントで買った追加時間」
 * として数え、残量の表示と消費を担う。端末内モデル（Nano / Flash）
 * は対象外（メーターなしで使える）。
 *
 * 設計（既存実装との整合）：
 *  - 単位は「%」ではなく秒で持つ。ただしポイントのトップアップは
 *    既存実装が「メーター追加 +%」（POINTS_TOPUP_MENU / 端末保存キー
 *    cube-points-meter-bonus-v1）で積み上げているため、その % を
 *    「基準単位 BONUS_UNIT_SEC（= 無料枠1日分 600秒）に対する割合」
 *    として秒に換算する。+10% = 60秒、+100% = 600秒。
 *    ポイント側の保存値（累積 %）は一切変更しない（読み取り専用）。
 *    追加分の消費は本モジュールの状態（usedBonusSec）で数えるので、
 *    買うたびに残量はその場で増え、使い切るまで日をまたいで残る。
 *  - 無料枠は JST の日付で日次リセット（ポイントの1日判定と同方式）。
 *    追加分（ボーナス）は日次リセットしない（買った時間は消えない）。
 *  - 消費順序は「その日の無料枠 → ポイント追加分」。
 *  - 状態は端末の localStorage（キー cube-meter-v1）に
 *    {day, usedBaseSec, usedBonusSec} で保持する。サーバー保存は
 *    しない（ACCOUNT_DESIGN の将来拡張 plan_usage が本命で、
 *    本モジュールは端末側の先行実装）。
 *
 * プランと安全弁（設定として表現）：
 *  - plans: free / pro / max / student_max の1日あたり基準秒数。
 *    student_max は baseSecPerDay: null ＝通常は無制限（学生は
 *    Max 無料・2026-11-18 決定）。ただし congestion.enabled の間は
 *    全プランに capSecPerDay の上限がかかる（混雑時の安全弁。
 *    学生 Max にも上限が残るという決定をこの形で表現する）。
 *  - 既定プランは free。課金・学生判定の仕組みは未決のため、
 *    プランの切り替えは設定（window.__CUBE_METER_CONFIG__ または
 *    createMeter の config 引数）でのみ行う。
 *  - 既定値：free 600秒/日（10分）。生成1回は数秒程度なので、
 *    通常の質問なら1日100回以上に相当する。数値は設定で変更可。
 *
 * 公開側のグローバル名：CubeMeter（cube_memory.js の CubeMemory、
 * cube_account.js の CubeAccount と同じ方式）。モジュールが無い
 * 環境では script.js 側がメーターなし（従来挙動）で動く。
 */
(function () {
  "use strict";

  const VERSION = 1;

  /* メーター状態の保存キー（端末内）。 */
  const STATE_KEY = "cube-meter-v1";
  /* ポイント追加分（累積 %）の保存キー。script.js の
     POINTS_BONUS_KEY と同じ値。ポイント側が所有するキーなので、
     本モジュールは読み取りだけを行う。 */
  const BONUS_KEY = "cube-points-meter-bonus-v1";

  /* ポイントの「+%」を秒へ換算する基準単位：+100% = 600秒。
     無料枠1日分（free の baseSecPerDay）と同値にそろえてある
     （「1日分まるごと追加」が +100% の意味になる）。 */
  const BONUS_UNIT_SEC = 600;

  const DEFAULT_CONFIG = {
    /* 適用プラン（plans のキー）。 */
    plan: "free",
    plans: {
      free: { baseSecPerDay: 600 },
      pro: { baseSecPerDay: 1800 },
      max: { baseSecPerDay: 7200 },
      /* 学生 Max 無料：通常は無制限（null）。混雑時のみ
         congestion.capSecPerDay が上限になる。 */
      student_max: { baseSecPerDay: null },
    },
    /* 混雑時の安全弁：有効な間は全プランの1日枠をこの秒数で
       頭打ちにする。 */
    congestion: { enabled: false, capSecPerDay: 600 },
    bonusUnitSec: BONUS_UNIT_SEC,
  };

  /* JST（Asia/Tokyo）の日付キー "YYYY-MM-DD"。ポイント側
     （pointsJstDateKey）と同方式：RPC の1日判定とそろえるため
     端末のタイムゾーンではなく JST で数える。 */
  function jstDateKey(date) {
    const d = date || new Date();
    try {
      const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: "Asia/Tokyo",
        year: "numeric", month: "2-digit", day: "2-digit",
      }).formatToParts(d);
      const pick = (t) => (parts.find((p) => p.type === t) || {}).value || "";
      const y = pick("year"), m = pick("month"), dd = pick("day");
      if (y && m && dd) return `${y}-${m}-${dd}`;
    } catch { /* Intl が使えない環境では端末ローカルの日付で代替 */ }
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd2 = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${mm}-${dd2}`;
  }

  /* 設定の解決：既定値に上書き（部分指定可）を重ねる。 */
  function resolveConfig(override) {
    const o = override && typeof override === "object" ? override : {};
    const cfg = {
      plan: typeof o.plan === "string" ? o.plan : DEFAULT_CONFIG.plan,
      plans: {},
      congestion: {
        enabled: !!(o.congestion && o.congestion.enabled),
        capSecPerDay:
          o.congestion && Number.isFinite(o.congestion.capSecPerDay)
            ? o.congestion.capSecPerDay
            : DEFAULT_CONFIG.congestion.capSecPerDay,
      },
      bonusUnitSec: Number.isFinite(o.bonusUnitSec)
        ? o.bonusUnitSec
        : DEFAULT_CONFIG.bonusUnitSec,
    };
    for (const key of Object.keys(DEFAULT_CONFIG.plans)) {
      const base = DEFAULT_CONFIG.plans[key].baseSecPerDay;
      const over = o.plans && o.plans[key];
      cfg.plans[key] = {
        baseSecPerDay:
          over && (over.baseSecPerDay === null ||
            Number.isFinite(over.baseSecPerDay))
            ? over.baseSecPerDay
            : base,
      };
    }
    if (o.plans) {
      for (const key of Object.keys(o.plans)) {
        if (!cfg.plans[key] && o.plans[key] &&
            (o.plans[key].baseSecPerDay === null ||
              Number.isFinite(o.plans[key].baseSecPerDay))) {
          cfg.plans[key] = { baseSecPerDay: o.plans[key].baseSecPerDay };
        }
      }
    }
    if (!cfg.plans[cfg.plan]) cfg.plan = DEFAULT_CONFIG.plan;
    return cfg;
  }

  /* 適用プランの「今日の枠」総量（秒）。無制限なら Infinity。
     混雑時は capSecPerDay で頭打ち（安全弁）。 */
  function effectiveBaseSec(config) {
    const cfg = config || DEFAULT_CONFIG;
    const plan = cfg.plans[cfg.plan] || cfg.plans.free;
    let base = plan.baseSecPerDay === null
      ? Infinity
      : Math.max(0, plan.baseSecPerDay || 0);
    if (cfg.congestion && cfg.congestion.enabled) {
      base = Math.min(base, Math.max(0, cfg.congestion.capSecPerDay || 0));
    }
    return base;
  }

  /* 残量表示用の整形：Infinity は「無制限」、60秒以上は「M分S秒」
     （秒が0なら「M分」）、未満は「S秒」。小数秒は切り捨て。 */
  function formatDuration(sec) {
    if (!Number.isFinite(sec)) return "無制限";
    const s = Math.max(0, Math.floor(sec));
    if (s >= 60) {
      const m = Math.floor(s / 60);
      const r = s % 60;
      return r ? `${m}分${r}秒` : `${m}分`;
    }
    return `${s}秒`;
  }

  /**
   * メーター本体を作る。
   * opts:
   *  - storage: {getItem, setItem}（localStorage 互換。必須に近いが、
   *    無ければメモリ内だけで動く）
   *  - now: () => ms（テスト用に差し替え可能）
   *  - config: resolveConfig 済み、または上書きオブジェクト
   *  - getBonusPct: () => 累積 %（ポイント追加分）。未指定なら
   *    storage の BONUS_KEY から読む。
   *  - onChange: (snapshot) => void（消費・リセットのたびに呼ぶ）
   */
  function createMeter(opts) {
    const o = opts || {};
    const storage = o.storage || null;
    const now = typeof o.now === "function" ? o.now : () => Date.now();
    const config = o.config && o.config.plans
      ? o.config
      : resolveConfig(o.config);
    const onChange = typeof o.onChange === "function" ? o.onChange : null;

    const readBonusPct = () => {
      try {
        if (typeof o.getBonusPct === "function") {
          const v = Number(o.getBonusPct());
          return Number.isFinite(v) && v > 0 ? v : 0;
        }
        if (storage) {
          const v = Number(storage.getItem(BONUS_KEY));
          return Number.isFinite(v) && v > 0 ? v : 0;
        }
      } catch { /* 読めなければ追加分なしとして扱う */ }
      return 0;
    };

    let state = { day: "", usedBaseSec: 0, usedBonusSec: 0 };
    function load() {
      try {
        const raw = storage ? storage.getItem(STATE_KEY) : null;
        if (raw) {
          const rec = JSON.parse(raw);
          if (rec && typeof rec === "object") {
            state = {
              day: typeof rec.day === "string" ? rec.day : "",
              usedBaseSec: Number.isFinite(rec.usedBaseSec)
                ? Math.max(0, rec.usedBaseSec) : 0,
              usedBonusSec: Number.isFinite(rec.usedBonusSec)
                ? Math.max(0, rec.usedBonusSec) : 0,
            };
            return;
          }
        }
      } catch { /* 壊れた記録は初期状態で上書きする */ }
      state = { day: "", usedBaseSec: 0, usedBonusSec: 0 };
    }
    function save() {
      try {
        if (storage) storage.setItem(STATE_KEY, JSON.stringify(state));
      } catch { /* 保存できなくても計量は続ける */ }
    }
    /* JST の日付が変わっていたら、その日の使用量だけを戻す。
       ポイント追加分の使用量は日をまたいで保持する。 */
    function rollover() {
      const today = jstDateKey(new Date(now()));
      if (state.day !== today) {
        state.day = today;
        state.usedBaseSec = 0;
        save();
        return true;
      }
      return false;
    }
    load();
    rollover();

    function baseTotalSec() {
      return effectiveBaseSec(config);
    }
    function baseRemainingSec() {
      rollover();
      const total = baseTotalSec();
      if (!Number.isFinite(total)) return Infinity;
      return Math.max(0, total - state.usedBaseSec);
    }
    function bonusTotalSec() {
      return Math.floor((readBonusPct() / 100) * config.bonusUnitSec);
    }
    function bonusRemainingSec() {
      return Math.max(0, bonusTotalSec() - state.usedBonusSec);
    }
    function remainingSec() {
      const b = baseRemainingSec();
      if (!Number.isFinite(b)) return Infinity;
      return b + bonusRemainingSec();
    }
    function canUse() {
      return remainingSec() > 0;
    }
    function snapshot() {
      return {
        plan: config.plan,
        congestion: !!(config.congestion && config.congestion.enabled),
        baseTotalSec: baseTotalSec(),
        baseRemainingSec: baseRemainingSec(),
        bonusTotalSec: bonusTotalSec(),
        bonusRemainingSec: bonusRemainingSec(),
        remainingSec: remainingSec(),
        day: state.day,
      };
    }
    function notify() {
      if (onChange) {
        try { onChange(snapshot()); } catch { /* 表示側の失敗は無視 */ }
      }
    }

    /* sec 秒を消費する（無料枠→追加分の順）。残量を超えた分は
       切り捨て（0未満にはしない）。実際に消費した内訳を返す。 */
    function consume(sec) {
      const want = Number.isFinite(sec) ? Math.max(0, sec) : 0;
      rollover();
      let rest = want;
      let fromBase = 0;
      let fromBonus = 0;
      const baseRem = baseRemainingSec();
      if (Number.isFinite(baseRem) && rest > 0) {
        fromBase = Math.min(baseRem, rest);
        state.usedBaseSec += fromBase;
        rest -= fromBase;
      } else if (!Number.isFinite(baseRem)) {
        /* 無制限枠：消費は記録しない（残量は無制限のまま）。 */
        fromBase = rest;
        rest = 0;
      }
      if (rest > 0) {
        fromBonus = Math.min(bonusRemainingSec(), rest);
        state.usedBonusSec += fromBonus;
        rest -= fromBonus;
      }
      if (fromBase > 0 || fromBonus > 0) {
        save();
        notify();
      }
      return { fromBase, fromBonus, remainingSec: remainingSec() };
    }

    /* 非同期処理 fn の実行時間を計って消費する。fn が失敗しても
       使った時間は消費する（サーバーは実際に動いたため）。 */
    async function consumeAround(fn) {
      const start = now();
      try {
        return await fn();
      } finally {
        const elapsed = Math.max(0, (now() - start) / 1000);
        if (elapsed > 0) consume(elapsed);
      }
    }

    return {
      config,
      baseTotalSec,
      baseRemainingSec,
      bonusTotalSec,
      bonusRemainingSec,
      remainingSec,
      canUse,
      consume,
      consumeAround,
      snapshot,
      get state() { return { ...state }; },
    };
  }

  const api = {
    VERSION,
    STATE_KEY,
    BONUS_KEY,
    BONUS_UNIT_SEC,
    DEFAULT_CONFIG,
    jstDateKey,
    resolveConfig,
    effectiveBaseSec,
    formatDuration,
    createMeter,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof globalThis !== "undefined") {
    globalThis.CubeMeter = api;
  }
})();
