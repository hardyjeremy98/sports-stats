# 3. Supply — Competition & How Money Is Made

> **Evidence status (updated 2026-06-23 by a second deep-research pass focused on this section):**
> The competitive map is **now largely verified.** A dedicated 6-angle / 26-source / 25-claim adversarial
> pass (see [§5](05-evidence-and-methodology.md)) confirmed each major competitor's business model,
> segment, and — critically — **whether it accepts ordinary phone footage or requires its own hardware.**
> **Pricing is now `[VERIFIED]`** for Hudl, Trace and Veo (replacing the old brief anchors). The two
> things still **`[UNVERIFIED]`** are **dollar TAM/CAGR** and the **funding/M&A landscape** — both
> returned no surviving claims and remain open questions. Unit economics are modelled in
> [§6](06-unit-economics-deep-dive.md).
>
> **Headline change vs. the first pass:** the target cell is **no longer "verified empty."** Judged on
> *commercial* grounds alone, **at least one entrant (mpact.ai) already claims the exact position** —
> automated + any-phone + consumer-priced. The white-space is **thinly contested, not vacant.** See §3.2.

---

## 3.0 How to read this section — two different "white-space" questions

The brief asked whether a gap exists. That actually decomposes into **two independent questions**, and the
first pass blurred them. This section keeps them apart:

1. **The commercial question (this section):** *If the technology worked, is the competitive position
   unoccupied?* This is answerable today by mapping what competitors **sell and claim**, regardless of
   whether their AI is any good. **Answer below: the position is thinly contested — one early entrant
   already claims it — not the virgin territory the brief assumed.**
2. **The technology question ([§4](04-enabling-environment.md), [§7](07-technology-maturity-deep-dive.md)):**
   *Can the thing actually be built on phone footage?* That is where the hard "MOT + re-ID + attribution
   is unsolved on ground-level video" evidence lives.

The first pass collapsed these into "the cell is empty *because* the tech isn't solved." That conflation
hid the commercially decisive fact: **competitors are already moving into the cell on a marketing basis,
so you cannot assume you'd arrive first.** Treat §4/§7 as the *feasibility* gate and this section as the
*occupancy* map.

---

## 3.1 Competitive landscape `[NOW VERIFIED]`

Each competitor was checked against three axes: **business model** (hardware / software-only /
human-in-the-loop), **segment** (grassroots-consumer / academy / pro), and **capture requirement** (needs
their hardware / structured-capture-on-phone / accepts any phone footage).

### A. The fully-automated stat producers — all hardware-gated `[VERIFIED 3-0]`
Every competitor that produces **automated** per-player/team stats does so by **controlling the capture
with proprietary or managed hardware**. None accepts arbitrary single-phone footage as its analytics input.

| Competitor | Model | Segment | Capture requirement | Note |
|------------|-------|---------|---------------------|------|
| **Trace** | Hardware + SaaS | **Grassroots / youth families** | **Proprietary TraceCam** (16-ft tripod bundle); phone "MultiCam" is a *supplementary* angle, **not** the analytics input | Auto-edits per-player highlights for both teams. `[VERIFIED 3-0]` |
| **Veo** (Analytics / Player Spotlight) | Hardware + SaaS add-ons | Grassroots → pro | **Veo Cam 3** (proprietary 4K). Per-player tracking (soccer-only) is a **paid add-on** on top of camera+base plan | `[VERIFIED 3-0]` |
| **Veo Go** | Managed two-phone rig | Families / youth | Uses the user's iPhone — **but requires *two* iPhones + a rig**, a managed capture system, not arbitrary single-phone upload | `[VERIFIED 3-0]`; the "Veo Go = single-phone software-only" reading was **`[REFUTED 1-2]`** |
| **Spiideo** (AutoData) | Cloud + fixed cameras | **Pro / elite / academy** | Fully machine-generated ("no human data collectors") — **but requires a Spiideo fixed-camera install** | Case studies: Utah women's, NWSL, Swedish academies. `[VERIFIED 3-0]` |
| **zone14** (STATS) | AI camera + software | Clubs / academy | Automated without GPS/wearables/manual tagging; built around **panoramic** AI-camera footage but **not strictly locked** to one proprietary camera | The "zone14 is hardware-dependent" framing was **`[REFUTED 0-3]`** — it is more capture-flexible than assumed. `[VERIFIED 2-1]` that it produces automated stats |
| **Pixellot** | Hardware/AI cameras | Clubs / venues / broadcast | Proprietary panoramic cameras (Air NXT etc.) | `[VERIFIED 3-0]` (trade press) |
| **XbotGo** (Falcon / Chameleon) | AI auto-follow phone mount / camera | Consumer / grassroots | Proprietary auto-follow hardware | `[VERIFIED 3-0]` (trade press) |

### B. The software-accepting incumbent — but human-in-the-loop & structured capture `[VERIFIED 3-0]`
- **Hudl Assist** is the closest incumbent that takes **user-supplied** footage and returns per-player
  soccer stats — **but it is not automated and not "any footage":**
  - **Human-in-the-loop:** "the core of Hudl Assist relies on a team of **trained analysts who manually
    tag every play, event, and statistic**"; AI tagging exists only in *some* sports (e.g. volleyball via
    Balltime) — **soccer remains human-tagged.** "AI performance summaries" (beta) is a summarization
    layer *over* human-tagged events, not CV event detection. `[VERIFIED 3-0]`
  - **Structured capture, not arbitrary upload:** users must "follow the required recording workflow for
    your sport," capturing **the entire playing surface** on Hudl Focus or a supported device. A phone is
    allowed **only through this gate**, not as arbitrary ground-level footage. `[VERIFIED 3-0]`
  - **~24h turnaround, staffed 24/7.** This is a **services** business wearing a SaaS skin — which is why
    it is priced like one (§3.3) and why an *automated* entrant could undercut it structurally.

### C. The emerging phone-footage entrants — the cell is being entered `[VERIFIED 3-0 / directional]`
- **★ mpact.ai (Impact Soccer)** `[VERIFIED 3-0, marketing-claim basis]` — **occupies the exact target
  cell.** Homepage: *"100% Automated… Your Entire Match Analyzed in Hours,"* *"the only solution that
  provides 100% automated both on- and off-the-ball stats for every match,"* and *"Any Camera, Any
  Time… **Even Your Phone!**"* (compatible with Apple, Android, YouTube). The FAQ confirms **no
  proprietary hardware** ("phone-on-tripod gets the same results"), with a **free-match** consumer offer.
  A 2025 press release corroborates automatic ball/player tracking "without any human intervention" in
  4–6 hours.
  - **⚠️ Critical caveat:** these are **unverified vendor marketing claims** with **no independent
    accuracy benchmark.** The white-space question brackets *feasibility*, so this establishes
    **commercial occupancy by claim, not a demonstrated working product or market traction.** But its mere
    existence **refutes the "verified empty" thesis** — you would not be first to *pitch* this.
- **playvista.ai** and others surfaced as additional phone-footage AI entrants (directional, not
  independently verified); an SCMP report describes AI using **phone-recorded video** to surface football
  talent. The signal: **multiple parties are now aiming at the phone-footage lane.**

### D. Off-map for the consumer wedge (pro/data layer) `[context]`
- **Hudl** (rolled up **Wyscout, InStat**, partnered with **StatsBomb**), **Stats Perform/Opta**,
  **SkillCorner** — pro/scouting/broadcast data feeds and video-tracking. SkillCorner does *video-based*
  tracking but for **broadcast/pro** feeds. These are **not your segment**, but they are the most credible
  **down-market threat** (§4) — though the research found **no evidence any of them has shipped an
  automated phone-footage consumer product.** Their positioning relative to the target cell is
  **undetermined** (open question, §5).
- **StepOut** `[VERIFIED 3-0, first pass]` — software-only, broadly positioned (clubs→individuals), raised
  **$1.5M pre-Series A** (Rainmatter, 2025). But **`[REFUTED 0-3]`** that it accepts ground-level phone
  footage, and **`[REFUTED 1-2]`** the fully-automated "400+ datapoints, no manual tagging" claim. It is
  built around **broadcast-grade input** — so it sits one column *left* of the target. Marquee logos
  (Real Madrid, Ajax) are **accelerator/pilot relationships, not paid clients.**

---

## 3.2 Positioning map `[NOW VERIFIED]`

Two axes separate winners from losers here: **how automated** the analysis is (vertical) and **how
constrained the capture** must be (horizontal). **Left → right = easier for the customer to supply
footage; top → bottom = more human labour in the loop.**

| **Automation ↓ · Capture →** | **Needs their hardware**<br/>(own camera / fixed rig / managed multi-phone) | **Software-only, but needs<br/>broadcast-/structured-grade footage** | **Software-only, accepts<br/>any single-phone footage** |
|---|---|---|---|
| **Fully automated**<br/>(no human tagging) | Trace · Veo Analytics/Player Spotlight · Spiideo AutoData · zone14 · Pixellot · XbotGo · **Veo Go** (2-phone rig) `[VERIFIED 3-0]` | **StepOut** `[VERIFIED]`<br/>*(needs broadcast-grade footage)* | **★ mpact.ai — OCCUPIED (by claim)** ⁱ<br/>*+ emerging: playvista.ai, others* |
| **Human-in-the-loop**<br/>(manual / paid tagging) | — | **Hudl Assist** `[VERIFIED 3-0]`<br/>*(prescribed full-field recording workflow)* | manual tagging · spreadsheets · the coach's eye |

<sub>ⁱ **`[VERIFIED 3-0]`** that mpact.ai *markets and sells* an automated, any-phone, consumer-priced
product — **commercial occupancy.** **Not** verified: that it works, or has traction. Accuracy is
bracketed by the white-space question; see §4/§7 for the feasibility gate.</sub>

**What changed and why it matters:**

- **The top-right cell is occupied — by claim, by an early entrant — not verified-empty.** The first pass
  marked it `[VERIFIED] empty` on the strength of §4's *technology* argument. That was a **category
  error**: a tech limitation is not a commercial scan. A direct commercial scan finds **mpact.ai sitting
  in the exact cell**, and others (playvista.ai) aiming at it. The honest statement is: *no **entrenched,
  proven** incumbent owns this cell, but it is **actively being entered**, so first-mover advantage is
  **not** available for the taking.*
- **Hudl Assist moved.** The first pass put it in the **bottom-right** ("accepts any uploaded footage").
  Verification shows it belongs in the **bottom-*middle*** — it requires a **prescribed full-field
  recording workflow**, not arbitrary phone footage. So even the human-in-the-loop incumbent **gates its
  capture**. The bottom-right ("any phone footage" + human tagging) is occupied only by **DIY substitutes**
  (spreadsheets, the coach's eye) — a real but unmonetised substitute pressure.
- **Every automated competitor still controls capture via hardware** (or a managed multi-phone rig).
  That remains the single most important structural pattern: **controlling capture is how they make the CV
  work.** "Upload any footage" is therefore a position you must **earn with hard tech** (§4) — and now
  also one you must **win against mpact.ai**, not merely invent.

---

## 3.3 Business models & pricing `[NOW VERIFIED]`

The old brief anchors (Veo "$799 + $42/mo", Hudl Assist "$1,200–$5,000") are **superseded** by verified
pricing below. The dominant revenue models are **(a) hardware-sale + SaaS** and **(b) human-tagging SaaS**
— confirming a **pure software-only / any-phone / consumer** model is the **under-occupied contrarian
lane.**

| Product | Model | Verified price | Tag |
|---|---|---|---|
| **Hudl** Club Soccer (video SaaS) | per team / year | Bronze **$400** (~20 games) · Silver **$1,000** (~65) · Gold **$1,600** (~130) | `[VERIFIED]` (tiers 2-1) |
| **Hudl Assist** (human tagging) | per team / season | **$700–$1,600** | `[VERIFIED 3-0]` |
| **Trace** PlayerFocus | per **family** / year | Basic **$180/yr** (2 seats) · Pro **$300/yr** (4 seats) | `[VERIFIED 3-0]` |
| **Trace** hardware (TraceCam) | lease | free with yearly sub, **or $25/mo + $99** one-time | `[VERIFIED 3-0]` |
| **Veo** | hardware + add-on SaaS | Cam 3 free on Performance Bundle; Analytics / Player Spotlight **~$33/mo** add-ons | `[VERIFIED 3-0]` |
| **Veo Go** | managed 2-phone SaaS | **~$29/mo + ~$50** accessory kit | `[VERIFIED 3-0]` |

**Anchors that matter for your pricing:**
- **Automated per-player ceiling (self-serve):** Trace at **$180–$300 per *family* per year** is the
  closest comparable to an automated per-player-stats consumer product. Note it is priced **per family,
  not per team** — a materially different unit than the per-team SaaS above.
- **Human-tagging ceiling:** Hudl Assist **$700–$1,600/team/season** is what the market pays when a
  **human** does the work — leaving large headroom for an automated product priced well below it.
- **No incumbent monetises pure software-only any-phone automated stats at a consumer price** — except the
  unproven mpact.ai. That is the open commercial lane (with the §4 feasibility gate attached).

---

## 3.4 Unit economics → see [§6](06-unit-economics-deep-dive.md)

The first pass flagged unit economics as the top open question; the **§6 deep-dive now models it** with
verified GPU pricing. Headline: automated per-match COGS is **~$1.80–$4.70**, *not* the constraint;
the binding constraints are **per-payer LTV** (≈$36 Year-1 consumer ceiling `[VERIFIED]`) and **seasonal
churn** `[VERIFIED]`. The strategic implication for *this* section is unchanged: **low price × high volume
→ CAC and churn decide viability**, which points to **B2B2C + virality**, not paid acquisition.

---

## 3.5 Go-to-market & distribution

Channels that work in this category (the virality loop is the standout):
- **D2C web + app stores** — high CAC, exposed to seasonal churn; this is the lane mpact.ai's free-match
  offer is testing.
- **Club / league / federation B2B2C** — *best fit*: amortises CAC over a roster, **fixes capture**
  (directly attacking the structural hardware-vs-phone tension above), and folds parental consent (§2.4)
  into the club's membership flow. Solves three risks at once. Note Trace's grassroots traction
  (US Youth Soccer, ECNL partnerships) is exactly this motion.
- **Coach referral & tournaments** — concentrated, trust-based.
- **Retail** — only if you ship hardware (which the software-only thesis rejects).
- **🟢 Highlight-sharing virality loop** — the growth engine Veo/Trace lean on: a player shares a
  highlight → teammates/parents/opponents see it → they want their own. This loop is **forgiving of
  imperfect stats** (it sells clips, not Opta data), aligning with the safer highlights/development wedge
  from §2.2 and driving CAC toward zero.

---

### Section 3 takeaways
- 🟥 **The headline correction: the target cell is occupied by claim, not verified-empty.** **mpact.ai**
  already markets automated + any-phone + consumer-priced per-player stats; **playvista.ai** and others
  are aiming at the same lane. On *commercial* grounds the position is **thinly contested, not vacant** —
  you would not be first to pitch it. `[VERIFIED 3-0, marketing-claim basis]`
- 🧭 **The dominant structural pattern holds and is now verified:** **every automated incumbent controls
  capture via hardware** (Trace, Veo, Spiideo, Pixellot, XbotGo) or a managed multi-phone rig (Veo Go).
  Going software-only on "any phone footage" rejects the very lever that makes their CV work.
- ✅ **Hudl Assist is human-in-the-loop and gates capture** (full-field recording workflow) — it is a
  **services** business, priced $700–$1,600/team/season, *not* an automated any-footage product. The
  automated lane below it is genuinely open on cost.
- 💸 **Pricing is now verified** (Hudl, Trace, Veo) and **§6 models unit economics** — the contest is on
  **CAC × seasonal churn**, won via **B2B2C + the highlight virality loop**.
- ⚠️ **Still unverified:** **dollar TAM/CAGR** and the **funding/M&A landscape** (incl. whether pro
  incumbents are moving down-market) — both returned no surviving claims and remain the primary open
  questions (§5).
