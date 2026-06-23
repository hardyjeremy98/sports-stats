# 5. Evidence Ledger & Methodology

This appendix is the audit trail. Every claim used in sections 1–4 traces here, with its adversarial
vote and source. **Confirmed ≠ true** — it means it survived a 3-vote skeptic panel needing 2/3 to kill.
**Refuted** claims are listed because *what failed verification is itself a finding.*

---

## Method

**First pass (2026-06-21) — full dossier:**
```
5 search angles  →  24 sources fetched  →  104 claims extracted
   →  top 25 claims sent to a 3-vote adversarial panel (need 2/3 "refute" to kill)
      →  15 CONFIRMED · 10 KILLED  →  7 merged findings after dedup
```

**Search angles:** (1) market sizing & segmentation · (2) competitive landscape & pricing ·
(3) academic/technical CV capability & unsolved problems · (4) voice-of-customer & adoption barriers ·
(5) regulatory, privacy & funding landscape.

**Second pass (2026-06-23) — supply & competition / commercial white-space** (drilling into §3, framed
to judge the white-space *independent of technical feasibility* — "if the tech worked, is the position
unoccupied?"):
```
6 search angles  →  26 sources fetched  →  120 claims extracted
   →  top 25 claims sent to the same 3-vote adversarial panel
      →  22 CONFIRMED · 3 KILLED  →  5 merged findings after dedup
```
**Search angles:** (1) competitor capability & hardware dependency · (2) pricing & revenue models ·
(3) emerging hardware-free AI entrants 2024–2026 · (4) grassroots/youth analytics market sizing ·
(5) funding & M&A down-market signals · (6) pro/data-feed providers' input requirements.
**Findings are in [§3](03-supply-and-competition.md); the new confirmed/refuted claims are in §F below.**
**Headline:** the target cell is **no longer "verified empty"** — mpact.ai already claims it commercially.

**Confidence labels** used in this dossier:
- **`[VERIFIED]`** — survived the panel (vote shown, e.g. 3-0); cited to a source below.
- **`[REFUTED]`** — killed by the panel; do not rely on it (often itself informative).
- **`[UNVERIFIED]`** — asserted in the brief or framing but *not* tested/confirmed here → treat as a
  hypothesis requiring primary research.
- **`[INFERRED]`** — a logical implication drawn from verified facts, labelled as such.

---

## A. Confirmed findings (high confidence)

| # | Finding (abbreviated) | Vote | Source |
|---|------------------------|------|--------|
| C1 | **FIFA Big Count (2006):** 265M players; **38M registered**, 226M unregistered; 22M youth <18 (21.5M reg.); USA 3.9M, Germany 2.08M. *Caveat: ~20 yrs old, never repeated → stale for 2026 absolute sizing.* | 3-0 | FIFA Big Count PDF |
| C2 | **Football Australia (2024):** total participation **1,912,493 (+11% YoY)**; **640,234 outdoor affiliated** across **2,205 clubs (+10% YoY)**. Report separates affiliated from casual → validates the carve-out. | 3-0 | Football Australia 2024 Participation Report |
| C3 | **StepOut positioning:** spans clubs/academies/leagues/federations/agents/organisers **+ coaches/analysts/individual players**; marquee logos (Real Madrid, Ajax, Rayo Vallecano, Bengaluru FC, 60+ teams); raised **$1.5M pre-Series A** (Rainmatter 2025). *Caveat: marquee "clients" are pilots/accelerator relationships, not paid.* | 3-0 | stepout.ai (+ YourStory/Prodwrks corroboration) |
| C4 | **Core CV premise unsolved:** unique per-player ID ambiguous when jersey numbers not visible; MOT of players/ball "far from being solved" under fast motion/occlusion (2022→2024-25). Both studies use **broadcast** video — phone footage is worse. | 3-0 | SoccerNet GSR (CVPR 2024); SoccerNet-Tracking (CVPR 2022) |
| C5 | **Sports re-ID > surveillance re-ID in difficulty:** near-identical kits, few samples/identity, low/variable resolution, occlusion, fast motion; off-the-shelf surveillance re-ID **degrades without modification**. → Moat is domain adaptation, not a free model. | 3-0 | Sports Re-ID (2022) |
| C6 | **Team-ID tractable, but broadcast-only:** ~**91.5%** (93.7% post-proc.) via CIELAB k-means; validated on broadcast HLS/M3U8 feeds + metadata APIs, **not phone footage**; degrades with similar kits/poor light. | 3-0 | PlayerTV (2024) |
| C7 | **Minors' footage = real friction (UK FA):** written parental consent before footage enters public domain; consent+disclosure for coaching-aid filming; clubs must run badge/wristband/check-in "do-not-film" systems. *Scope: public-domain use & organised filming, not every incidental frame.* | 3-0 | FA Guidance Notes 8.3 (July 2020) |

---

## B. Refuted claims (killed by the panel — informative failures)

| Claim | Vote | Why it matters |
|-------|------|----------------|
| StepOut **accepts ground-level phone footage** | 0-3 | Even the closest funded competitor doesn't clearly cover your hardest use case. |
| StepOut **auto-generates 400+ datapoints with no manual tagging** | 1-2 | The fully-automated claim is unproven. |
| Single-camera reconstruction "achieves only **22.26% GS-HOTA**" | 0-3 | Don't quote this number — the *specific* framing wasn't supported (but the broader "unsolved" point stands via C4). |
| **Jersey-number OCR** is "the dominant bottleneck" / "peaks ~30%" | 1-2 / 0-3 | Be precise: OCR is *a* hard part, not provably *the* capped bottleneck. |
| SoccerNet re-ID "tops out at **85-86 mAP**" | 0-3 | Don't cite this specific ceiling. |
| Deterministic decision-tree "**95%+ event detection**" from tracking | 0-3 / 1-2 | The 95% results assume **tracking is already solved as input** — they don't solve raw-video → events. |
| UK: consent required before **any** capture of under-16s / only **trained personnel** may film | 0-3 / 0-3 | The minors'-privacy friction is **narrower** than the maximalist version (see C7). |

> **Pattern in the refutations:** the *alarmist-precise* tech-pessimism claims and the *maximalist*
> privacy claims both failed — while the *structural* "it's unsolved on phone footage" and "consent is a
> real friction" claims survived. Calibrate accordingly: the problem is **hard, not hopeless**; the
> regulation is **real, not absolute.**

---

## C. Open questions (NOT answered — require primary research) {#open-questions}

These are the gaps between "interesting" and "investable":

1. **Top-down dollar market & CAGR** for the *grassroots/amateur carve-out specifically* (excluding pro
   clubs, broadcasters, betting) — with each report cited. **🟡 STILL OPEN** — the second pass fetched
   sizing reports (MarketsandMarkets, GlobalGrowthInsights, BusinessResearchInsights youth-sports) but
   **no dollar/CAGR claim survived verification.**
2. ~~**Pricing**~~ **✅ RESOLVED (pricing) by 2nd pass** — Hudl ($400/$1,000/$1,600/yr video; $700–$1,600
   Assist), Trace ($180–$300/yr per family), Veo (Cam 3 + ~$33/mo add-ons; Veo Go ~$29/mo) now
   `[VERIFIED]` (§3.3, §F). **Unit economics** (CAC/LTV/churn/margin) modelled in [§6](06-unit-economics-deep-dive.md).
3. ~~**Competitor capability on phone footage**~~ **✅ RESOLVED by 2nd pass** — all automated incumbents
   (Trace, Veo, Spiideo, zone14, Pixellot, XbotGo) are **hardware/managed-capture gated**; Hudl Assist is
   **human-in-the-loop + structured capture**; **mpact.ai** claims automated any-phone stats. `[VERIFIED]`
   (§3.1–3.2, §F). *Remaining sub-gap:* XbotGo, Wyscout/InStat, Stats Perform/Opta, StatsBomb, SkillCorner
   had **no surviving claims on their stance toward the target cell** — undetermined.
4. **Current SOTA accuracy** for end-to-end per-player event attribution from one ground-level amateur
   camera at a consumer price point. *(Bracketed by the white-space framing; see §4/§7.)*
5. **Binding compliance beyond UK FA** — US COPPA, EU/UK GDPR biometric provisions, Australian Privacy
   Act, plus image/likeness rights and data-ownership defaults under self-serve upload.
6. **Funding/M&A landscape** beyond StepOut's $1.5M — rounds, valuations, exit comps (Crunchbase/
   PitchBook) to read investor appetite. **🟡 STILL OPEN** — 2nd pass surfaced **directional** points
   (Veo raised $25M in 2021; Hudl acquisition activity) but **no funding/valuation claim survived
   verification**, and whether pro incumbents are moving down-market is undetermined.
7. **Voice-of-customer** — app-store/Reddit/forum review mining + survey + 10–20 interviews + fake-door
   sign-up/pay test (§2.5).
8. **mpact.ai (and peers') real traction & accuracy** — is the target cell occupied only on paper, or
   does mpact.ai have paying users, retention, and validated per-player accuracy from phone footage?
   *(New, raised by the 2nd pass — the decisive competitive unknown.)*

---

## D. Coverage map (where evidence is strong vs. thin)

| Research goal | Coverage | Notes |
|---------------|----------|-------|
| §1 Market sizing — **counts** | 🟢 Strong | FIFA, Football Australia (primary) |
| §1 Market sizing — **$ TAM/CAGR** | 🔴 Thin | No verified dollar figures |
| §2 Demand — **counts** | 🟢 Strong | Verified; extend to more federations |
| §2 Demand — **WTP / VOC** | 🔴 Thin | Primary research not done |
| §3 Competition — **StepOut** | 🟢 Strong | Positioning + funding + key refutations |
| §3 Competition — **competitor capability / hardware-vs-phone** | 🟢 Strong | 2nd pass: Trace, Veo, Spiideo, zone14, Hudl Assist, mpact.ai all verified |
| §3 Competition — **pricing** | 🟢 Strong | 2nd pass: Hudl, Trace, Veo verified (§3.3) |
| §3 Competition — **commercial white-space** | 🟢 Strong | 2nd pass: target cell **occupied by claim** (mpact.ai) |
| §3 Competition — **unit econ (CAC/LTV/churn)** | 🟡 Modelled | See [§6](06-unit-economics-deep-dive.md) |
| §3 Competition — **funding / M&A** | 🔴 Thin | 2nd pass: no funding claim survived (directional only) |
| §4 Technology capability | 🟢 Strong | Primary peer-reviewed papers |
| §4 Privacy (UK FA) | 🟢 Strong | Primary governing-body doc |
| §4 Wider regulation / patents / funding | 🔴 Thin | Mostly unverified |

**Other caveats:** Geographic skew toward **Australia + UK**; US/EU/AU privacy specifics unverified.
FIFA global data is **~20 years stale**. Some sources were vendor self-marketing (StepOut) or blogs
(competitor comparisons) — weighted accordingly.

---

## E. Source list

**Primary (highest weight):**
- FIFA Big Count 2006 — `https://digitalhub.fifa.com/m/55621f9fdc8ea7b4/original/mzid0qmguixkcmruvema-pdf.pdf`
- Football Australia 2024 National Participation Report — `https://footballaustralia.com.au/sites/ffa/files/2025-02/21307_FA_Participation%20Reports_2024_High%20Res_FULL.pdf`
- StepOut — `https://www.stepout.ai/en`
- SoccerNet Game State Reconstruction (CVPR 2024) — `https://arxiv.org/html/2404.11335v1`
- SoccerNet-Tracking (CVPR 2022) — `https://arxiv.org/pdf/2204.06918`
- Sports Re-Identification (2022) — `https://arxiv.org/pdf/2206.02373`
- PlayerTV / team-ID (2024) — `https://arxiv.org/html/2407.16076v1`
- Event detection from tracking data (2022) — `https://link.springer.com/article/10.1007/s12283-022-00381-6`
- FA Safeguarding 8.3 — Photographing & Filming Children — `https://www.thefa.com/-/media/thefacom-new/files/rules-and-regulations/safeguarding/section-8/8-3-photographing-and-filming-children-colour-version.ashx`
- CPSU — Photography & filming in sport — `https://thecpsu.org.uk/help-advice/topics/photography-and-filming-in-sports-and-activities/`
- Hudl × StatsBomb press release — `https://www.hudl.com/blog/hudl-statsbomb-press-release-en`

**Secondary (market reports / trade press — claims here did NOT survive verification; cite originals):**
- Grand View Research — sports analytics market — `https://www.grandviewresearch.com/industry-analysis/sports-analytics-market`
- Fortune Business Insights — sports analytics — `https://www.fortunebusinessinsights.com/sports-analytics-market-102217`
- IntelMarket Research — global football analysis — `https://www.intelmarketresearch.com/Global-Football-Analysis%20-922`
- MarketGrowthReports — youth sports software — `https://www.marketgrowthreports.com/market-reports/youth-sports-software-market-113852`
- Sportico — Pixellot — `https://www.sportico.com/business/tech/2022/pixellot-israeli-sports-tech-1234682156/`

**Blogs / vendor comparisons (low weight — directional only):**
- XbotGo — Trace vs Veo — `https://xbotgo.com/blogs/buying-guide/trace-vs-veo`
- Veo — camera systems compared — `https://www.veo.com/article/sports-video-camera-systems-compared`
- Hudl Assist pricing (soccer) — `https://www.hudl.com/pricing/club/assist/soccer`
- RecordingSports — Veo/Trace review — `https://www.recordingsports.com/post/veo-trace-soccer-camera-review`
- The Gadgeteer — XbotGo Chameleon review — `https://the-gadgeteer.com/2025/10/09/xbotgo-chameleon-review-your-personal-ai-sports-cameraman/`
- LearnOpenCV — YOLOv7-pose vs MediaPipe — `https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/`

**Flagged unreliable (excluded from findings):** a COPPA-2026 blog and an Australian facial-recognition
law blog returned 0 usable verified claims — the underlying *statutes* must be checked directly.

---

## F. Second-pass claim ledger (2026-06-23) — supply, competition & commercial white-space

This pass treated the white-space question **independent of technical feasibility** ("if the tech worked,
is the competitive position unoccupied?"). It therefore verifies **what competitors sell and claim**, not
whether their AI is accurate. Feeds [§3](03-supply-and-competition.md).

### F.1 Confirmed findings (high confidence)

| # | Finding (abbreviated) | Vote | Source |
|---|------------------------|------|--------|
| C8 | **mpact.ai occupies the exact target cell (by claim):** markets "100% automated" on- and off-ball per-player/team stats from "Any Camera… Even Your Phone!", **no proprietary hardware**, free-match consumer offer. *Caveat: unverified vendor marketing; commercial occupancy by claim, not demonstrated product/traction.* | 3-0 | mpact.ai (+ 2025 press release) |
| C9 | **All automated stat producers are hardware-gated:** Trace (TraceCam), Veo Analytics/Player Spotlight (Cam 3), Spiideo AutoData (fixed cams), zone14 (panoramic), Pixellot, XbotGo — none takes arbitrary single-phone footage as analytics input. **Veo Go** uses iPhones but requires a **two-phone managed rig**. | 3-0 | traceup, veo.com, spiideo, zone14, veo camera-comparison |
| C10 | **Hudl Assist is human-in-the-loop, not automated:** "trained analysts manually tag every play"; soccer is human-tagged (AI only in some sports, e.g. volleyball/Balltime); "AI summaries" beta sits *over* human tags. | 3-0 | hudl.com Assist FAQ + soccer pages |
| C11 | **Hudl Assist gates capture:** requires a **prescribed full-field recording workflow** on supported devices — a phone is allowed only via this gate, not as arbitrary footage. ~24h turnaround, staffed 24/7. | 3-0 | hudl.com Assist FAQ + compare-plans |
| C12 | **Verified pricing:** Hudl club video **$400 / $1,000 / $1,600 per team/yr** (Bronze/Silver/Gold); Hudl Assist **$700–$1,600/team/season**; Trace **$180 (2-seat) / $300 (4-seat) per family/yr** + TraceCam ($25/mo + $99 or free w/ yearly); Veo Cam 3 + **~$33/mo** add-ons; **Veo Go ~$29/mo + ~$50** kit. | 3-0 (tiers 2-1) | hudl, traceup, veo pricing pages |
| C13 | **Segment split:** grassroots/youth reached via **hardware** vendors (Trace → families/coaches; Veo Go → families/youth); **Spiideo AutoData targets pro/elite/academy** (Utah women's, NWSL, Swedish academies). The automated-software-only-phone path to grassroots is the thinly-served gap. | 3-0 | traceup, spiideo, veo |
| C14 | **Dominant revenue models = hardware-sale + SaaS, or human-tagging SaaS** — confirming pure software-only/any-phone/consumer is the under-occupied contrarian lane. | 2-1/3-0 | hudl, traceup, veo |

### F.2 Refuted claims (killed by the panel — informative)

| Claim | Vote | Why it matters |
|-------|------|----------------|
| **zone14 is hardware-dependent** (analytics tied to a proprietary zone14 camera) | 0-3 | zone14 is **more capture-flexible** than assumed — don't classify it as hardware-locked. |
| **Veo Go is single-phone software-only** (one iPhone, no hardware) | 1-2 | Veo Go needs **two iPhones + a rig** — a managed capture system, not arbitrary single-phone upload. |
| **Veo is entirely hardware-locked** (camera unusable without subscription; never accepts phone) | 0-3 | Overreach — the absolute "cannot be used at all" framing wasn't supported. |

> **Pattern in the 2nd pass:** the **structural** claims survived (automated ⇒ controls capture;
> Hudl Assist = human + structured capture; pricing), while **maximalist** capture claims (zone14 / Veo
> "totally locked") were refuted. The single decisive new fact: **the target cell is occupied by claim
> (mpact.ai), not verified-empty** — a commercial reality the first pass missed by inferring "empty" from
> the *technology* argument rather than a direct competitive scan.

### F.3 Second-pass source list

**Primary (vendor pages — highest weight for *positioning/pricing* claims, self-marketing for capability):**
- mpact.ai (Impact Soccer) — `https://mpact.ai/` · `https://mpact.ai/faqs`
- Hudl Assist FAQ — `https://www.hudl.com/en_gb/products/assist/faq` · soccer — `https://www.hudl.com/products/assist/soccer` · compare-plans — `https://www.hudl.com/products/assist/soccer/compare-plans` · club pricing — `https://www.hudl.com/pricing/club/soccer`
- Trace — `https://traceup.com/pricing` · `https://traceup.com/academy/how-trace-subscriptions-basic-vs-pro-plans`
- Veo — `https://www.veo.com/en-us/pricing` · `https://www.veo.com/en-us/product/player-spotlight` · `https://www.veo.com/en-us/product/veo-go` · `https://www.veo.com/article/soccer-camera-systems-compared`
- Spiideo AutoData (soccer) — `https://www.spiideo.com/autodata/autodata-soccer/`
- zone14 — `https://zone14.ai/en/football-tracking-without-gps/`
- playvista.ai — `https://playvista.ai/` *(emerging entrant; directional)*
- SkillCorner — `https://medium.com/skillcorner/...` · `https://skillcorner.com/articles/data-on-demand`

**Secondary (trade press / market reports — claims here mostly did NOT survive; cite originals):**
- SCMP — AI talent ID from phone-recorded video — `https://www.scmp.com/sport/football/article/3355669/...`
- MarketsandMarkets — sports analytics — `https://www.marketsandmarkets.com/Market-Reports/sports-analytics-market-35276513.html`
- GlobalGrowthInsights — football analysis software — `https://www.globalgrowthinsights.com/market-reports/football-analysis-software-market-114831`
- BusinessResearchInsights — youth sports software — `https://www.businessresearchinsights.com/market-reports/youth-sports-software-market-122833`
- TechCrunch — Veo $25M raise (2021) — `https://techcrunch.com/2021/01/06/veo-raises-25m-...`
- IndianStartupTimes — StepOut $1.5M (Rainmatter) — `https://www.indianstartuptimes.com/.../rainmatter-doubles-down-on-stepout-with-1-5m-pre-series-a-...`
- The Upside — Hudl acquisition special report — `https://www.theupside.us/p/upside-special-report-hudls-acquisition`

> **2nd-pass caveats:** the decisive "cell is occupied" finding rests on a **single entrant (mpact.ai)**
> whose claims are **unverified vendor marketing** with no independent accuracy benchmark. All pricing/
> feature claims are **June 2026 vendor pages and will drift**; some Hudl URLs 404 and were confirmed on
> equivalent live pages. **Angles 4 (market sizing) and 5 (funding/M&A) returned no surviving claims** —
> they remain open questions (§C).

---

*Workflow stats — pass 1: 5 angles · 24 sources fetched · 104 claims extracted · 25 verified · 15 confirmed /
10 killed · 106 agent calls. Compiled 2026-06-21.*
*Workflow stats — pass 2 (supply & competition): 6 angles · 26 sources fetched · 120 claims extracted ·
25 verified · 22 confirmed / 3 killed · 109 agent calls. Compiled 2026-06-23.*
