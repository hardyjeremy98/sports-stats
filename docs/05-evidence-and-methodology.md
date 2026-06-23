# 5. Evidence Ledger & Methodology

This appendix is the audit trail. Every claim used in sections 1–4 traces here, with its adversarial
vote and source. **Confirmed ≠ true** — it means it survived a 3-vote skeptic panel needing 2/3 to kill.
**Refuted** claims are listed because *what failed verification is itself a finding.*

---

## Method

```
5 search angles  →  24 sources fetched  →  104 claims extracted
   →  top 25 claims sent to a 3-vote adversarial panel (need 2/3 "refute" to kill)
      →  15 CONFIRMED · 10 KILLED  →  7 merged findings after dedup
```

**Search angles:** (1) market sizing & segmentation · (2) competitive landscape & pricing ·
(3) academic/technical CV capability & unsolved problems · (4) voice-of-customer & adoption barriers ·
(5) regulatory, privacy & funding landscape.

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
   clubs, broadcasters, betting) — with each report cited.
2. **Pricing & unit economics** — actual price points and **CAC / LTV / seasonal churn / gross margin**
   for Veo, Hudl Assist, Trace, StepOut. *(The brief's Veo/Hudl pricing anchors are unverified.)*
3. **Competitor capability on phone footage** — do Hudl/Veo/Trace/Pixellot/Spiideo *accept* ground-level
   single-phone video, or require proprietary fixed/auto-follow hardware? *(None verified here.)*
4. **Current SOTA accuracy** for end-to-end per-player event attribution from one ground-level amateur
   camera at a consumer price point.
5. **Binding compliance beyond UK FA** — US COPPA, EU/UK GDPR biometric provisions, Australian Privacy
   Act, plus image/likeness rights and data-ownership defaults under self-serve upload.
6. **Funding/M&A landscape** beyond StepOut's $1.5M — rounds, valuations, exit comps (Crunchbase/
   PitchBook) to read investor appetite.
7. **Voice-of-customer** — app-store/Reddit/forum review mining + survey + 10–20 interviews + fake-door
   sign-up/pay test (§2.5).

---

## D. Coverage map (where evidence is strong vs. thin)

| Research goal | Coverage | Notes |
|---------------|----------|-------|
| §1 Market sizing — **counts** | 🟢 Strong | FIFA, Football Australia (primary) |
| §1 Market sizing — **$ TAM/CAGR** | 🔴 Thin | No verified dollar figures |
| §2 Demand — **counts** | 🟢 Strong | Verified; extend to more federations |
| §2 Demand — **WTP / VOC** | 🔴 Thin | Primary research not done |
| §3 Competition — **StepOut** | 🟢 Strong | Positioning + funding + key refutations |
| §3 Competition — **all others / pricing / unit econ** | 🔴 Thin | Not independently verified |
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

*Workflow stats: 5 angles · 24 sources fetched · 104 claims extracted · 25 verified · 15 confirmed /
10 killed · 106 agent calls. Compiled 2026-06-21.*
