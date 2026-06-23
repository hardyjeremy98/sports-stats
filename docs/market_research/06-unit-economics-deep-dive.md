# 06 — Unit Economics Deep-Dive: Cost-to-Serve & Pricing

**Scope.** Follow-up to the market-research dossier, narrowed to two questions: (1) what does it
cost to process **one amateur soccer match** through a CV pipeline (detection + multi-object tracking
+ player re-ID + event detection), and (2) what can we **charge** and still clear a healthy SaaS
gross margin and a viable LTV:CAC. Every external number is tagged **`[VERIFIED]`** (survived 3-vote
adversarial verification, primary/secondary source cited) or **`[UNVERIFIED]`** (our own modelling
estimate — treat as a working assumption, not a fact). The worked model in §4 chains VERIFIED price
inputs with UNVERIFIED throughput/storage assumptions; its outputs are therefore **directional**.

---

## Bottom line up front

On a **GPU-only, no-human-in-the-loop** basis, the marginal compute+storage+bandwidth cost to process
one ~90–105-minute match lands at roughly **$1.80 (low) / $2.10 (expected) / $4.70 (high)** per match.
That is small enough that the per-match COGS is **not the thing that kills the business** — at a
plausible consumer price of **$8–15/match** (or a season/team bundle), gross margin is a healthy
**74–86%**, comfortably above the 80%-classic / mid-60s-AI-loaded SaaS bands. The binding constraints
are instead (a) **whether the pipeline actually works on phone footage** (see
[`07-technology-maturity-deep-dive.md`](07-technology-maturity-deep-dive.md)),
(b) **per-payer LTV**, which for consumer fitness/sports apps tops out near **~$36 Year-1**
`[VERIFIED]`, and (c) **seasonal churn** — sports subscriptions are structurally churn-prone because
viewing/usage collapses in the off-season `[VERIFIED]`. **The price floor is ~$3–5/match** (covers
worst-case compute with margin); the real question is not cost-to-serve but **CAC vs a thin per-payer
LTV under seasonal churn.** If you add **human QA/tagging**, the economics flip entirely: Hudl's human
service is **$700/team/season** `[VERIFIED]` (~$25–40/match), an order of magnitude above automated
compute — so any human step must be priced as a premium tier, not baked into the base.

---

## 1. Compute cost to process one match

### 1.1 Cloud GPU hourly pricing (inference GPUs)

| GPU | Provider / instance | Rate | Tag |
|---|---|---|---|
| **NVIDIA T4** (16 GB) | AWS `g4dn.xlarge` on-demand, us-east-1 | **$0.526/hr** | `[VERIFIED]` 3-0 (Vantage) |
| **NVIDIA T4** (16 GB) | AWS `g4dn.xlarge` **spot** | **$0.254/hr** (~½ on-demand) | `[VERIFIED]` 3-0 (Vantage) |
| **NVIDIA L4** (24 GB) | RunPod Pods | **$0.39/hr** | `[VERIFIED]` 3-0 (runpod.io) |
| **NVIDIA A100** (80 GB) | RunPod Pods PCIe / SXM | **$1.39 / $1.49/hr** | `[VERIFIED]` 3-0 (runpod.io) |
| **NVIDIA A100** (80 GB) | RunPod **Serverless** | **$2.72/hr** | `[VERIFIED]` 3-0 (runpod.io) |

The L4 is the sweet-spot inference GPU here: purpose-built for inference, 24 GB VRAM, and at $0.39/hr
on RunPod it is **~half** the AWS list price for the same silicon (`g6.xlarge` ≈ $0.80/hr) `[VERIFIED]`.
The A100 is the upper bound for heavy re-ID/OCR stages. **Caveat:** RunPod rates are secure-cloud
on-demand and **capacity-variable**; spot/community rates drift. (Several cheaper-GPU claims — Vast.ai
L40S ~$0.47, A100 ~$0.58, Spheron A100 ~$1.07 — were **`[REFUTED]`** in verification, so we do **not**
rely on sub-$0.39 figures.)

### 1.2 Processing-time-to-video-time ratio `[UNVERIFIED]`

We could **not** find a verified throughput figure for a *full* stacked pipeline (detection + MOT +
re-ID + jersey OCR + event detection) on match-length video — so the ratios below are **our estimates**
and the single largest source of uncertainty in the model. Detection (YOLO) alone runs faster than
real time on an L4/A100; **re-ID and jersey-OCR are the heavy, often offline stages** that push the
end-to-end ratio above real time. We model proc:video ratios of **0.3–4.0×** depending on GPU class.

| GPU / price `[VERIFIED]` | proc:video (low/exp/high) `[UNVERIFIED]` | GPU $/match (low/exp/high) |
|---|---|---|
| T4 spot $0.254/hr | 1.0 / 2.0 / 4.0× | $0.44 / $0.89 / $1.78 |
| T4 on-demand $0.526/hr | 1.0 / 2.0 / 4.0× | $0.92 / $1.84 / $3.68 |
| L4 $0.39/hr | 0.7 / 1.5 / 3.0× | $0.48 / $1.02 / $2.05 |
| A100 80GB $1.39/hr | 0.3 / 0.7 / 1.5× | $0.73 / $1.70 / $3.65 |

(Assumes ~105 min of video = 90-min match + warmup buffer `[UNVERIFIED]`.) **Takeaway:** across every
GPU choice, GPU compute for one match sits in a **~$0.45–$3.70 band** — the GPU choice matters less
than the throughput ratio.

### 1.3 Storage, bandwidth, transcode `[UNVERIFIED]`

A 1080p match at ~8 Mbps for 105 min is **~6.3 GB**. Using standard list rates (S3 Standard
~$0.023/GB/mo; AWS egress ~$0.09/GB — well-known but not separately verified here):

- **Storage, 3-month retention:** ~$0.43
- **Egress, one customer download:** ~$0.57
- **Transcode:** ~$0.05 (negligible if done on the same GPU box)
- **Storage + bandwidth subtotal: ~$1.05/match**

### 1.4 Human-in-the-loop QA (the expensive option)

If any frames need human review/tagging, the cost reference is **Hudl Assist = $700/team/season**
`[VERIFIED]` for purely human tagging — roughly **$25–40/match** at ~20–28 games/season. That is
**10–20× the automated compute cost**, so a human step cannot live in the base price; it must be a
premium SKU.

### 1.5 Net COGS per match (automated only)

| Scenario | Compute | +Storage/BW | **Net COGS/match** |
|---|---|---|---|
| **Low** | A100 fast (0.3×) | +$1.05 | **~$1.78** |
| **Expected** | L4 ~1.5× real time | +$1.05 | **~$2.08** |
| **High** | T4 on-demand 4× real time | +$1.05 | **~$4.73** |

**Defensible per-match cost-to-serve: ~$1.80 / $2.10 / $4.70** (low/expected/high), automated, no human QA.

---

## 2. Pricing benchmarks (what competitors charge)

| Product | Model | Price | Tag |
|---|---|---|---|
| **Hudl Assist** (human tagging, club soccer) | per team / season | **$700** (your games) / **$1,100** (+scout games), Standard Priority | `[VERIFIED]` 3-0 |
| **Hudl Assist** Express Priority (faster) | per team / season | **$1,000** / **$1,600** (+scout) | `[VERIFIED]` 3-0 |
| **Hudl Assist+** (player-level + athlete ID) | per team / season | **$1,100** (your games), Standard Priority | `[VERIFIED]` 3-0 |
| **Trace** PlayerFocus **Pro** (AI, all games, family of 4) | per player / year | **$300/yr** | `[VERIFIED]` 3-0 |
| **Trace** PlayerFocus **Basic** (last 5 games, family of 2) | per player / year | **$180/yr** | `[VERIFIED]` 3-0 |
| **Veo** (grassroots) | hardware cam + tiered **annual subscription** (no per-match) | discounts at 6-mo/yearly/2-yr commit; grassroots bundle ≈ $1,799 | `[VERIFIED]` 3-0 |
| **Veo Analytics / Player Spotlight** | paid add-ons, **soccer only** | add-on to subscription | `[VERIFIED]` 3-0 |

**Anchors that matter:**
- **Per-player annual ceiling (AI, self-serve):** Trace at **$180–$300/player/yr** `[VERIFIED]` is the
  closest direct comparable to an automated per-player-stats product.
- **Human-tagging ceiling:** Hudl Assist **$700–$1,100/team/season** `[VERIFIED]` is what the market
  pays when a human does the work — leaving large headroom for an automated product priced well below it.
- **Hardware-bundled subscription (Veo) does not sell per-match** `[VERIFIED]` — so a clean per-match
  or per-season **software-only** price is a genuine open position in the market.

---

## 3. SaaS unit-economics benchmarks

- **Gross-margin erosion from variable compute (COGS):** classic SaaS targets ~80% GM; adding direct
  variable cost (e.g. AI inference) compresses it — illustratively, **~$15 of variable cost on an
  $80 seat drops GM from ~80% toward ~65%** `[VERIFIED]` (note: on a literal $80 seat the arithmetic is
  ~61%; the "65%" holds on a $100 base — the verifier flagged this; treat as **illustrative**). For our
  case the analogy is reassuring: per-match COGS (~$2) is **tiny relative to a $8–15 price**, so we do
  **not** suffer the AI-COGS margin collapse the benchmark warns about.
- **LTV:CAC target:** **3:1 is the standard venture target with CAC payback < 12 months**; network-effect
  apps can run profitably at 2:1 `[VERIFIED]` 2-1.
- **Consumer/fitness LTV ceiling:** **Year-1 realized LTV per payer ≈ $35.64 for Health & Fitness apps**
  (North America all-category median ~$32) `[VERIFIED]` — a sobering ceiling for a pure-B2C prosumer play.
- **Seasonal churn:** sports streaming/subscriptions are **structurally churn-prone due to seasonal
  cycles** — usage and willingness-to-pay collapse off-season `[VERIFIED]`. (Specific fitness churn
  rates, the "January effect," and OTT 30% annual churn figures were all **`[REFUTED]`** in
  verification — do **not** cite those.)

---

## 4. Synthesis: worked model, price floor, break-even

### 4.1 Margin at candidate per-match prices (using **expected** COGS ≈ $2.08)

| Price/match | COGS | Gross margin |
|---|---|---|
| $5 | $2.08 | **58%** |
| **$8** | $2.08 | **74%** |
| **$10** | $2.08 | **79%** |
| **$15** | $2.08 | **86%** |
| $20 | $2.08 | 90% |

### 4.2 Price floor & break-even processing cost

- **Break-even processing cost** (automated, expected): **~$2.08/match**; worst case **~$4.73/match**.
- **Price floor:** **~$3–5/match** to cover worst-case compute and leave margin. Below ~$3/match you risk
  negative gross margin in the high-throughput-ratio / on-demand-T4 scenario.
- **Recommended list price:** **$8–15/match** (74–86% GM), or bundle as a **season/team** plan
  (see below) to smooth seasonality and raise LTV.

### 4.3 Season / team bundle and LTV vs CAC

At ~22 games/season `[UNVERIFIED]` and expected COGS:

| Price/match | Season revenue | Season gross profit |
|---|---|---|
| $8 | $176 | $130 |
| $10 | $220 | $174 |
| $15 | $330 | $284 |

These season figures sit **between Trace ($180–$300/yr/player)** and **Hudl Assist ($700/team/season)**
`[VERIFIED]` — a defensible position: cheaper than human tagging, comparable to AI per-player products,
but sold per-team/per-season to amortise CAC.

**LTV vs CAC verdict:**
- If sold **B2C per-payer**, LTV is capped near the **~$36 Year-1 fitness ceiling** `[VERIFIED]`. To hit
  3:1 `[VERIFIED]`, **CAC must stay under ~$12** — hard for a niche prosumer product. A pure-B2C,
  per-player model is **LTV-constrained, not COGS-constrained.**
- If sold **per-team/per-season** ($176–$330 gross/season), one purchaser covers a whole roster, so the
  effective LTV per acquisition is far higher and **3:1 is reachable at much larger CAC budgets** —
  *provided* seasonal churn is managed (annual/season commitment, off-season pause, multi-sport to fill
  the calendar). Seasonal churn is the documented structural risk `[VERIFIED]`.

### 4.4 The honest caveat

Compute cost-to-serve is **comfortably affordable** and is **not** the constraint. The constraints are
**(1) does the CV actually produce accurate per-player stats on phone footage** (per
[`07-technology-maturity-deep-dive.md`](07-technology-maturity-deep-dive.md), this is **research-grade,
not solved** — GS-HOTA ~64 even on *broadcast*) — if accuracy is
low you may be **forced into human QA at ~$25–40/match**, which destroys the automated margin; and
**(2) acquiring per-payer LTV above CAC under seasonal churn.** Price the base tier automated at
**$8–15/match or a season bundle**, keep any human QA as a clearly-priced premium SKU above Trace and
below Hudl Assist, and **sell per-team to escape the thin B2C per-payer LTV ceiling.**

---

## Source notes & time-sensitivity

- **GPU prices** (RunPod L4/A100; AWS T4 on-demand/spot) are **live as of June 2026** and are
  **capacity- and tier-variable** — re-quote before any commitment.
- **Throughput ratios, video size, retention window, games/season** are **`[UNVERIFIED]` modelling
  assumptions** — the single biggest swing factor is the proc:video ratio of the full re-ID/OCR stack.
- **Storage/egress** use standard list rates, not separately verified; committed-use or alternative
  object stores (R2 zero-egress) would lower the ~$1.05 storage/BW subtotal.
- **Refuted figures** (cheaper sub-$0.39 GPUs, specific fitness churn %, OTT 30% churn, fitness CAC
  $3–20) are **deliberately excluded** — see the dossier claim ledger.
