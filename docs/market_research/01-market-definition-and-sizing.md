# 1. Market Definition & Sizing

> **Evidence status for this section:** Demand-side *counts* are **`[VERIFIED]`** (primary sources).
> Dollar TAM/SAM/SOM and CAGR figures were **`[UNVERIFIED]`** by the research — the market-report
> sources fetched (Grand View, Fortune Business Insights, IntelMarket, MarketGrowthReports) did not
> yield claims that survived verification, so any dollar figure below is flagged as a working
> assumption requiring primary confirmation.

---

## 1.1 Market definition — decide what you're sizing

"Soccer video analysis" is not one market. Your product is **one cell** in a grid. Sizing the wrong
cell is the classic error (the "carve-out trap", §1.4).

**Segment by customer (who pays / who uses):**

| Tier | User | Buyer | Payer | Fit for your product |
|------|------|-------|-------|----------------------|
| Individual player / parent | Player | Parent | Parent | **Core B2C** — but lowest WTP, highest privacy sensitivity |
| Grassroots club / rec team | Players | Coach/manager | Club fees | **Core B2C / B2B2C** — best CAC amortisation |
| Youth academy | Players | Academy | Academy | Adjacent — wants development tracking |
| School / college | Players | Coach/AD | Institution | Adjacent — US recruitment culture strong |
| Semi-pro | Players | Coach | Club | Edge of core — wants accuracy you may not have |
| Pro club / federation | Analysts | Club | Club | **Not your market** — owned by Hudl/Wyscout/Stats Perform |
| Scout / agent | Scout | Scout | Scout/club | Niche — recruitment-grade accuracy demanded |

**Segment by product type:** capture hardware · cloud analysis software · AI auto-tagging ·
GPS/wearables · human-analyst services · data feeds/APIs. *You are "cloud analysis software +
AI auto-tagging, software-only".* This is the most commoditizable cell (§4) and the one most exposed
to "needs-their-hardware" competitors who bundle capture.

**Segment by job-to-be-done:** highlights & sharing · recruitment tape · tactical coaching ·
individual development · stat tracking · live streaming. *Your headline JTBD is "stat tracking +
individual development"* — note this is the **least forgiving of accuracy errors** and the one most
dependent on the unsolved per-player attribution problem.

**Segment by geography:** US (recruitment/scholarship culture, COPPA), UK/EU (GDPR, FA safeguarding),
Australia (current participation data, Privacy Act reform). Verified demand evidence here is
**AU/UK-skewed** (§2).

**→ Your niche, stated precisely:** *consumer/amateur, software-only, upload-and-get-team-stats,
JTBD = per-player stat tracking + development, soccer-first.* That single cell — not "sports analytics"
— is what the sizing below must address.

---

## 1.2 TAM / SAM / SOM — framework

Size **two ways** and trust the bottom-up for a consumer niche.

### Top-down `[UNVERIFIED — placeholders, confirm before use]`
The original brief cited "sports analytics ~$X billion" headline markets. **None of the dollar/CAGR
figures survived verification in this research.** Before using any top-down number:
- Pull and *cite* a specific report (Grand View / MarketsandMarkets / Mordor / Fortune Business
  Insights) — and immediately **carve out** the pro/broadcast/betting share (§1.4), which typically
  dominates these headline numbers.
- Capture the **CAGR** and **maturity stage** from the same report.
- Treat the residual "amateur/grassroots video-analysis software" slice as your top-down SAM — it will
  be a *small fraction* of the headline.

> ⚠️ Do not put a headline "sports analytics = $Xbn, therefore our TAM is huge" number in a deck. It is
> the single most common credibility-killer for this category. The verified evidence below is the
> defensible foundation; the dollar figure is decoration until you cite it.

### Bottom-up `[VERIFIED inputs]` — the defensible method
**Formula:** `addressable teams (or players) × attach rate × annual price = revenue potential`.

Verified population inputs you can build on:

| Input | Value | Source | Note |
|-------|-------|--------|------|
| Global players (all) | **265M** (2006) | FIFA Big Count `[VERIFIED 3-0]` | Headline — *do not* use as base |
| Global **registered** players | **38M** (2006) | FIFA Big Count `[VERIFIED 3-0]` | **This is your global top-of-funnel** |
| Global youth (<18) | **22M** (21.5M registered) | FIFA Big Count `[VERIFIED 3-0]` | USA 3.9M, Germany 2.08M lead |
| Australia total participation | **1,912,493** (2024, +11% YoY) | Football Australia 2024 `[VERIFIED 3-0]` | Includes one-off/school |
| Australia **outdoor affiliated** | **640,234** across **2,205 clubs** (+10% YoY) | Football Australia 2024 `[VERIFIED 3-0]` | **The realistic per-player base** |

**Worked bottom-up illustration (Australia, illustrative — your attach rate & price are unverified):**

```
Teams as unit:  ~2,205 clubs → assume ~5–15 teams/club → ~11k–33k teams
  × attach rate (e.g. 5%)        → ~550–1,650 paying teams
  × annual price (e.g. A$300/team) → ~A$165k–500k SOM in one country, early

Players as unit: 640,234 affiliated
  × attach rate (e.g. 2% individual B2C) → ~12,800 paying users
  × annual price (e.g. A$60/player)        → ~A$770k ARR ceiling, one country
```
*Every multiplier above except the population is a placeholder.* The point: even generous attach rates
against a *verified* 640K base in one mid-size football nation yield a **low-single-digit-$M SOM
early** — which is fine for a wedge, but confirms this is a **volume/geography-expansion** business,
not a high-ACV one. The economics therefore live or die on **CAC, churn and gross margin** (§3, all
**`[UNVERIFIED]`** and the top open question).

---

## 1.3 Maturity & growth

- **Consumer end is fragmenting / early** (many small hardware+app players: Veo, Trace, XbotGo,
  Pixellot, etc. — *positioning unverified here*), while the **top is consolidating** (Hudl has rolled
  up Wyscout and InStat — *the StatsBomb tie-up appears in our source list but specifics unverified*).
- Participation is **growing** where measured: Australia **+11% YoY total / +10% affiliated** in 2024
  `[VERIFIED]`. That is a genuine tailwind for a grassroots product.
- The global FIFA Big Count is **~20 years old and never repeated** `[VERIFIED caveat]` — so global
  absolute sizing is *stale*; lean on current national federation data (like Football Australia's)
  for live numbers.

---

## 1.4 The carve-out trap (apply this everywhere)

The verified FIFA split is the cleanest illustration of the trap:

```
265M players (headline)  ←  DO NOT size against this
   └─ 226M unregistered/occasional  ←  not a per-player-analytics buyer
   └─  38M registered               ←  candidate base (still over-inclusive)
        └─ minus very-young MiniRoos, casual social players, non-payers
            └─ realistic SOM = a small fraction of this
```

Football Australia's *own report* separates "Outdoor Affiliated Football" from
schools/community/promotional participation — the federation itself validates the carve-out. Use the
**affiliated/registered** line, then haircut further for age and seriousness. **Never** let a
pro-dominated "$X billion sports analytics" figure stand in for your addressable market.

---

### Section 1 takeaways
- ✅ A **real, growing, registered grassroots base** exists and is countable from primary federation data.
- ⚠️ **Dollar TAM/SAM/SOM and CAGR remain unverified** — build them bottom-up and cite every report.
- 📐 Early SOM is **small and volume-driven**, so the model's viability is a **unit-economics** question
  (§3), not a market-size question.
