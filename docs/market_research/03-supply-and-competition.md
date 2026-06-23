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
>
> **Update — 2026-06-24 (third pass, deliberately *cross-sport* — 5 angles / 99 agents / 17 sources / 25
> claims adversarially verified):** a deeper sweep that refused to limit itself to soccer surfaced **six
> materially new, verified entrants** (§3.1.C). The decisive one is **SportsVisio** — a **$9M+-funded**,
> software-only, **any-phone** product that, in **basketball/3×3/volleyball, appears to actually *deliver*
> genuine per-player box scores** (not just market them). This is the **strongest challenge yet** to the
> "claimed-but-not-filled" thesis — but it lands in **court sports, not soccer.** The honest synthesis:
> **the feasibility wall (§4) is sport-dependent.** In bounded court sports with legible jerseys
> (basketball, volleyball, cricket) automated per-player from phone footage is **already being shipped**;
> in soccer's large-field / 22-player / wide-low-res setting it **remains empty** (mpact still ships only
> team stats). Two more verified per-player-from-phone entrants — **Fulltrack AI** (cricket) and **SportAI**
> (racket sports) — confirm the pattern. A recurring trap was re-confirmed: most "AI sports" branding masks
> **highlight editors, auto-filming hardware, or human scorekeeping**, not CV per-player engines (and
> **TheStats.ai was refuted outright, 0-3**).

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
| **TwinPlay.ai** *(new 2026-06-24)* | Proprietary AI camera + SaaS | Basketball — shooting analytics | **Own 12MP/30fps AI camera** (built-in CV electronics, tripod); needs mains power + internet; **no phone-footage path** | Produces genuine **per-player** shooting maps / %s by shot type / streaks / release time. `[VERIFIED 3-0]` — but **single primary source, no disclosed pricing or funding** |

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

### C. The emerging phone-footage entrants — *claimed* in soccer, *delivered* in court sports `[VERIFIED — multi-sport deep-dive 2026-06-24]`

> **The cross-sport correction.** The 2026-06-23 pass concluded the phone-footage per-player cell was
> "claimed but not filled" — true **for soccer**, where mpact.ai still ships only team stats. But a
> deliberately cross-sport sweep (2026-06-24) found the cell is **already being filled in *court* sports**,
> where the §4 attribution problem is far more tractable (bounded space, ~5–12 visible players, legible
> jerseys). **SportsVisio is the proof.** So the precise statement is now: *the soccer cell is claimed-not-
> filled; the basketball/volleyball/cricket cells are **being delivered** by funded entrants.*

**C.1 — Soccer (still claimed, not filled):**
- **★ mpact.ai (Impact Soccer)** — **markets the exact target pitch, but a four-angle verification
  (own site · press · third-party · adversarial) found the per-player-stats claim is not substantiated;
  its own fine print walks it back.** Homepage markets *"100% automated… on- and off-the-ball stats,"*
  *"Any Camera… **Even Your Phone!**"*, no hardware, free-match offer. **But:**
  - **What actually ships is *team-level*.** All **15 enumerated metrics are team stats** (goals, possession,
    total shots, passes, etc.). The "individual" deliverable is a **player card that stores routed
    *highlight clips***, not a per-player numeric stat line. FAQ: *"route the match highlights **and team
    statistics** to certain players… **when it can read their jersey number clearly**."* `[VERIFIED 3-0]`
  - **Genuine per-player numeric stats were a *roadmap* item, not shipped.** Both Oct & Nov 2025 press
    releases: *"In the coming weeks, Impact Soccer **will add** individual player statistics."* (All press
    is **self-issued EIN Presswire** syndicated to TV affiliates — vendor copy, not journalism.) `[VERIFIED 3-0]`
  - **The fine print concedes the hard problem isn't solved.** Per-player needs **1080p/4K, close enough
    to OCR the jersey** (contradicting "Even Your Phone!"), and when it can't read the number,
    *"**you'll have to add it to your player's statistics**"* — a **manual fallback.** Its flagship
    *"Impact IQ™"* metric scores **zones/space, not individuals** — a way to sell "AI analytics" while
    sidestepping attribution. `[VERIFIED 3-0]` This is the **exact §4 wall**, conceded in their own FAQ.
  - **Early-stage, unsubstantiated traction.** Founder **Josh Konowe**; **founded 2024, 2–10 employees,
    no disclosed funding** ("AWS invested" = cloud credits); inconsistent user claims (500 vs 2,300 vs
    2,500); **zero independent demos/reviews** (the only "review" site is a discredited fake-review
    aggregator). `[VERIFIED 3-0]`
  - **⚠️ Net read:** mpact **does the easy half** (team stats from phone footage) and **punts on the hard
    half** (per-player attribution → jersey-OCR-when-possible + manual fallback + roadmap). So the target
    cell is **claimed in marketing, not filled in substance.** It **refutes "nobody is pitching this"** —
    but, under the "assume the tech works" premise, the *genuine* automated per-player-stats-from-casual-
    footage position **remains open.** mpact is evidence of competitive *intent*, not of a solved product.
- **playvista.ai** and others surfaced as additional phone-footage AI entrants (directional, not
  independently verified); an SCMP report describes AI using **phone-recorded video** to surface football
  talent. The signal: **multiple parties are now aiming at the phone-footage lane.**

**C.2 — Court & racket sports (the cell is being *delivered*, not just claimed) `[VERIFIED 3-0 each]`:**

| Entrant | Sport(s) | Capture | Output | Traction / pricing | Verdict |
|---|---|---|---|---|---|
| **★★ SportsVisio** | Basketball, 3×3, volleyball (baseball planned) | **Software-only — "footage from any device… if you can film the game, the AI can read it"; no hardware** | **Genuine per-player box scores** (pts/reb/ast/stl/blk/TO/fouls, shot charts, TS%; kills/digs/blocks for volleyball) **+ per-player highlights + "Coach Mode"**, ~24h turnaround | **$9M+ raised** ($3.2M Jun 2025, Sony Innovation Fund in); **150+ leagues/clubs/teams, ~16k users, 16 countries**; ~**$750 / 20-game pack (~$3/player/game)** | **The most material new threat.** Appears to genuinely *deliver* the target promise — but in **court sports, not soccer**, and at a **semi-pro price**. ⚠️ 95% event / 92% attribution accuracy is **vendor-claimed, un-benchmarked**; reviews report processing delays. |
| **Fulltrack AI** (Maiden AI) | Cricket | **Software-only — phone on a generic tripod, no proprietary hardware** | **Automated 3D ball tracking, pitch maps, genuine per-player numeric stats** (speed, swing, spin, wagon wheels, shot distribution) | Founded 2020 (Seattle), **$420k seed**; "1000+ international/IPL/BBL/County/state players"; tiers ~£17–£720 / ~$9.99/mo | Real per-player delivery. ⚠️ The **only entrant with independent academic validation** — which found it **systematically *overestimates* ball speed** (~2.6–3.9 km/h, up to ~10 mph), "software refinement required." A cautionary data point on *everyone's* un-benchmarked accuracy claims. |
| **SportAI** | Tennis, padel, pickleball | **Camera-agnostic — "single camera source," no extra hardware** (but in practice often court-camera/broadcast partners, e.g. Save My Play, Wingfield) | **Technique analysis, real-time per-player game stats, ball speed, 3D ball tracking, tactical data, highlights** — B2B API + B2C "SportAI Open" | Founded late 2023; **$3M raise Nov 2025** (oversubscribed; backer Casper Ruud); acquired **Padelytics** (Aug 2025); *TIME* Best Invention 2025 | Leading new racket-sports analytics platform. ⚠️ "Casual handheld phone, any angle" is **not** explicitly substantiated — real input leans on court-camera partners, so its capture bar may be higher than "any phone." |

**⚠️ Net read (C.2):** in **court/racket sports** the per-player-from-phone position is **no longer merely
claimed — it is being shipped** by a funded field (SportsVisio especially). This **narrows** the soccer-
specific whitespace thesis to exactly that: **soccer.** The transferable lesson for a football-first product
is sobering — *the moment you move into basketball/volleyball/cricket to escape soccer's feasibility wall,
you meet funded incumbents who are already there.*

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

### E. The "AI sports" decoys — adjacent, but NOT per-player stat engines `[VERIFIED 3-0 each, 2026-06-24]`
A recurring trap: "AI sports analytics" branding repeatedly masks something that is **not** a CV per-player
stat engine. Verification disqualified the following as direct competitors to the target product — useful
to name precisely so the competitive map isn't inflated:
- **Auto-follow filming mounts — no stats at all.** **Pivo Pod** (motorised AI face/body-tracking mount,
  BYO phone, ~$100–250), **MOVE'N SEE Pix4Team 2** (AI camera robot, ~18 amateur team sports each with a
  "dedicated AI", launched Jan 2024), and **Pixem 2** (beacon-based single-athlete tracking) are **filming
  devices only** — they auto-frame the action and **offload analysis to third-party software** (Hudl,
  Wyscout, Veo). Pix4Team's own FAQ: *"it follows the action, it does not follow any specific player or the
  ball."* These compete on **capture**, not analytics. (Same category as XbotGo, OBSBOT, SoloShot.)
- **BallerCam (BallerTV, launched Jul 2025)** — turns an **iPhone 13 Pro+** into a fixed 4K 180° camera via
  a **proprietary patent-pending ultra-wide lens** (so: hardware-required, not arbitrary phone footage).
  Basketball/volleyball/soccer/futsal, youth/grassroots, **$299 device / $399 bundle**. Output is
  **livestream/replay/highlights**; per-player stats exist but are **human-entered by a scorekeeper**, not
  CV-derived — auto stats are a **Spring-2026 roadmap** promise.
- **Athlete AI (by MWM)** — multi-sport phone app (basketball, football, soccer, baseball, volleyball,
  tennis, lacrosse…), no extra gear — **but it is an AI *highlight/reel editor*, not a stat generator.**
  Its developer's own page states it *"does not automatically generate numeric per-player statistics from
  video"*; stats are a **user-entered profile add-on**. A substitute on the *highlights* axis, not on stats.
- **TheStats.ai — `[REFUTED 0-3]`.** Marketed "drop in broadcast footage → full box score" across 9+
  sports with persistent career histories. **Both central claims failed adversarial verification** (no
  substantiation of the multi-sport box-score engine or the persistent player histories). **Excluded as an
  unsubstantiated / vaporware candidate.**

---

## 3.2 Positioning maps `[NOW VERIFIED]`

No single 2×2 captures this market: the same competitor can share your cell on one pair of axes and be a
completely different business on another. mpact.ai is the proof — it shares the automation×capture cell
(Map A) with the target, yet diverges sharply once you add **price, segment, output-depth and
footage-quality**. So the whitespace is mapped **five ways** below; each plots the same competitor set
against a different pair of axes. Read them together.

### Map A — Automation × Capture *(how they make the CV work)*

Two axes: **how automated** the analysis is (vertical) and **how constrained the capture** must be
(horizontal). **Left → right = easier for the customer to supply footage; top → bottom = more human labour
in the loop.**

| **Automation ↓ · Capture →** | **Needs their hardware**<br/>(own camera / fixed rig / managed multi-phone) | **Software-only, but needs<br/>broadcast-/structured-grade footage** | **Software-only, accepts<br/>any single-phone footage** |
|---|---|---|---|
| **Fully automated**<br/>(no human tagging) | Trace · Veo Analytics/Player Spotlight · Spiideo AutoData · zone14 · Pixellot · XbotGo · **TwinPlay** (basketball) · **Veo Go** (2-phone rig) `[VERIFIED 3-0]`<br/>*(+ filming-only mounts: Pivo · Pix4Team 2 · Pixem · BallerCam)* | **StepOut** · **SportAI** (racket; court-camera in practice) `[VERIFIED]`<br/>*(needs broadcast-/structured-grade footage)* | **★★ SportsVisio — DELIVERED** (basketball/volleyball) ⁱⁱ<br/>**Fulltrack AI — DELIVERED** (cricket)<br/>**★ mpact.ai — CLAIMED, not filled** (soccer) ⁱ<br/>*+ emerging: playvista.ai, others* |
| **Human-in-the-loop**<br/>(manual / paid tagging) | **BallerCam** (human-tagged stats) | **Hudl Assist** `[VERIFIED 3-0]`<br/>*(prescribed full-field recording workflow)* | **Athlete AI** (highlight editor; stats user-entered) · manual tagging · spreadsheets · the coach's eye |

<sub>ⁱ **`[VERIFIED — 4-angle, 2026-06-23]`** that mpact.ai *markets* an automated, any-phone, consumer-priced
product — but verification found it **ships team-level stats + highlight-clip routing**, while *genuine
per-player numeric stats* are a **roadmap item gated on 1080p/4K + readable jersey, with a manual
fallback** (its own FAQ). So the **soccer** cell is **claimed, not substantively filled.** See §3.1.C and §5.F.2.<br/>
ⁱⁱ **`[VERIFIED 3-0 — cross-sport pass, 2026-06-24]`** SportsVisio (basketball/3×3/volleyball) and Fulltrack AI
(cricket) are software-only, any-phone, and **appear to genuinely deliver per-player numeric stats** — so the
top-right cell is **filled in court sports**, *not* empty. It remains empty for **soccer**. Accuracy is
**vendor-claimed and un-benchmarked** (the one independently-tested entrant, Fulltrack, overestimates ball
speed). See §3.1.C.2.</sub>

**What changed and why it matters:**

- **The top-right cell splits by sport — and that split is the whole story now.** The 2026-06-23 pass
  read this cell as "claimed but not filled" on the strength of mpact.ai (soccer). The cross-sport pass
  shows that conclusion was **soccer-specific.** In **court/racket sports the cell is being *delivered***:
  **SportsVisio** (basketball/volleyball, **$9M+ funded**) and **Fulltrack AI** (cricket) ship genuine
  per-player numbers from phone footage today; **SportAI** does the same for racket sports (with a higher
  capture bar). In **soccer**, mpact still **does the easy half (team stats) and punts on per-player
  attribution** (jersey-OCR-when-possible + manual fallback + roadmap). The honest statement is now
  **sport-dependent:** *in soccer, no one has demonstrably **delivered** automated per-player stats from
  casual footage — the position is contested in marketing but open in substance. In basketball, volleyball
  and cricket, funded entrants **already deliver it** — so a football-first product cannot treat
  "multi-sport expansion" as escape into open water; the court sports are where the competition already
  lives.* (Accuracy everywhere is vendor-claimed and largely un-benchmarked — see §3.1.C.2.)
- **Hudl Assist moved.** The first pass put it in the **bottom-right** ("accepts any uploaded footage").
  Verification shows it belongs in the **bottom-*middle*** — it requires a **prescribed full-field
  recording workflow**, not arbitrary phone footage. So even the human-in-the-loop incumbent **gates its
  capture**. The bottom-right ("any phone footage" + human tagging) is occupied only by **DIY substitutes**
  (spreadsheets, the coach's eye) — a real but unmonetised substitute pressure.
- **The hardware-control pattern holds in *soccer*, but the cross-sport data breaks it.** Among the
  soccer/large-field incumbents, every automated competitor still controls capture via hardware (or a
  managed multi-phone rig) — controlling capture is how they make the CV work. But **SportsVisio, Fulltrack
  and SportAI broke that pattern in court/racket sports** by going software-only on phone footage. Why?
  Because **bounded courts, ~5–12 visible players and legible jerseys make attribution tractable** without
  controlling the camera. That is the deepest cross-sport lesson: *the hardware moat is a function of the
  sport's geometry, not an iron law.* "Upload any footage" is still a position you must **earn with hard
  tech** in soccer (§4) — and one you must now **win against funded incumbents** the moment you enter
  basketball/volleyball/cricket.

### Map B — Price ceiling × Customer seriousness *(the post-mpact whitespace)*

Map A says mpact shares your cell; this map says it **doesn't share your customer.** Plotting price against
seriousness shows mpact entered from the **semi-pro** side at a **semi-pro price** (~½ a tank of gas/match,
≈$25–40), leaving the **casual/youth × true-consumer-price** corner open.

| **Price ↓ · Seriousness →** | **Casual / rec** | **Youth / academy** | **Semi-pro** | **Pro / elite** |
|---|---|---|---|---|
| **Enterprise / custom** | — | — | — | Stats Perform/Opta · StatsBomb · SkillCorner · Spiideo · StepOut · SportAI (B2B API) |
| **$700+/season** | — | **SportsVisio** (~$750/20-game) | Hudl Assist · Hudl video · **SportsVisio** | Hudl Assist |
| **~$25–40/match · ~$180–300/yr** | — | Trace† · Veo Go† · XbotGo† | **mpact.ai** | — |
| **<$15/match (consumer)** | **Fulltrack** (~$10/mo, cricket) · **SportAI Open** (B2C) | **★ TARGET — EMPTY** (soccer) · Fulltrack (cricket) | — | — |
| **Free** | DIY · coach's eye · spreadsheets | — | — | — |

<sub>Reveals: in **soccer**, the automated-per-player position at a **true consumer price (<$15/match) for
casual→youth** is still held only by **free DIY** — mpact priced itself one tier up, into semi-pro, and
**SportsVisio** sits in the **$700+/20-game (semi-pro/youth)** band, *not* true consumer. The genuine
sub-$15 consumer prize in court sports is partly taken by **Fulltrack** (cricket, ~$10/mo). †Bundles/needs
hardware.</sub>

### Map C — Output depth × Footage quality required *(the safe wedge vs. the trap)*

| **Output ↓ · Footage needed →** | **Needs 4K / legible numbers** | **Elevated / structured** | **Any casual phone** |
|---|---|---|---|
| **Per-player + off-ball** | mpact (per-player layer)ⁱ · StepOut (broadcast) | Hudl Assist (human) · SportAI (court-camera) | **★ TARGET / TRAP — EMPTY in soccer** |
| **Per-player numeric** | mpact (gated)ⁱ | Veo · Trace (own camera) | **★ soccer: EMPTY** · **court sports: SportsVisio + Fulltrack** ⁱⁱ |
| **Team-level stats** | — | — | mpact (what it *ships*) · **reachable** |
| **Highlights / clips only** | — | — | **★ SAFE WEDGE** — XbotGo · Veo · Trace · Athlete AI clips |

<sub>ⁱ mpact's *per-player* output is gated on **1080p/4K + readable jersey numbers** (its FAQ) — so even
the one entrant in the *soccer* cell can't deliver per-player depth on *casual* footage. <br/>ⁱⁱ **The
2026-06-24 correction:** the per-player-numeric × any-casual-phone cell is **no longer empty in court
sports** — **SportsVisio** (basketball/volleyball) and **Fulltrack** (cricket) sit in it. It stays empty in
**soccer** (the §4 trap is hardest there). Reveals: **rich output × casual footage is a *solved* problem in
court sports and an *open trap* in soccer**, while **highlights/team-level × any footage is reachable in any
sport** — the defensible on-ramp.</sub>

### Map D — Distribution / buyer × Segment *(GTM whitespace)*

| **Channel ↓ · Segment →** | **Casual** | **Youth** | **Semi-pro** | **Pro** |
|---|---|---|---|---|
| **D2C consumer** | Fulltrack (cricket) · SportAI Open | XbotGo · Veo Go · Trace · BallerCam | mpact.ai · Trace · Fulltrack | — |
| **B2B2C club / league** | *(thin)* | Veo · Trace · Hudl · zone14 · Pixellot · **SportsVisio** | Veo · Hudl · **SportsVisio** | — |
| **B2B pro / data feed** | — | — | — | Spiideo · Stats Perform · Opta · SkillCorner · StepOut · SportAI (API) |

<sub>Reveals: **casual** is served almost entirely by **D2C + hardware**; reaching casual via the
CAC-efficient **B2B2C-club** channel with automated per-player is **thin** — clubs skew youth/serious. Note
**SportsVisio** reaches youth/semi-pro via exactly this **B2B2C-league** motion (150+ leagues/clubs) — i.e.
the cleanest GTM (§3.5) is a partial whitespace **only in soccer**; in court sports a funded player already
runs that play.</sub>

### Map E — Output value × Capture effort *("worth the hassle")*

| **Value ↓ · Capture effort →** | **Low (just upload a phone clip)** | **Medium** | **High (buy / rig a camera)** |
|---|---|---|---|
| **High output** | **SportsVisio · Fulltrack** (court sports) · mpact (soccer: claims; needs good footage) | SportAI (court-camera) | Veo · Trace · Spiideo · Pixellot · TwinPlay |
| **Medium output** | **★ TARGET (soccer)** | XbotGo · Pivo · Pix4Team (auto-follow rigs) | — |
| **Low output** | DIY · watch the game · Athlete AI (highlights) | BallerCam (livestream) | — |

<sub>Reveals the value proposition in one frame: **in soccer, every high-output incumbent demands high
capture effort** (their own camera) — the low-effort × high-output corner is held only by mpact-by-claim.
**But the 2026-06-24 pass placed SportsVisio and Fulltrack squarely in low-effort × high-output for court
sports** — proof the quadrant is *occupiable*, and *occupied* outside soccer. The bet remains the
under-served **low-effort × decent-output** quadrant — but it is **contested wherever the sport's geometry
makes it tractable.**</sub>

> **Reading the five together:** the picture is now **two-tier by sport.** *In soccer,* mpact occupies your
> cell on **Map A only**; on **B** it's a pricier semi-pro product, on **C** its per-player depth needs good
> footage, on **D** it's D2C not B2B2C, on **E** it's high-output-but-footage-dependent — so the precise
> soccer whitespace (**automated, decent per-player/team output, from *casual low-res* phone footage, at a
> *true consumer price*, reachable via *B2B2C clubs***) is **still empty on every map.** *In court/racket
> sports,* however, **SportsVisio, Fulltrack and SportAI already occupy most of those cells** — so the
> whitespace is **soccer-(and large-field-)specific, not category-wide.** That sport-by-sport precision is
> the single most important output of this pass.

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
| **SportsVisio** *(new)* | software-only SaaS (any phone) | **~$750 / 20-game pack (~$3/player/game)** | `[VERIFIED 3-0]` |
| **Fulltrack AI** *(new)* | software-only app (any phone) | tiers **~£17–£720** / **~$9.99/mo** | `[VERIFIED 3-0]` |
| **BallerCam** *(new)* | hardware (iPhone+lens) + storage | **$299 device / $399 bundle** · storage from **$99/yr** | `[VERIFIED 3-0]` |
| **TwinPlay.ai** *(new)* | proprietary AI camera + SaaS | **not disclosed** (free demo only) | `[VERIFIED 3-0]` claim; price `[UNVERIFIED]` |
| **SportAI** *(new)* | B2B API + B2C "Open" | **not publicly disclosed** | `[VERIFIED 3-0]` model; price `[UNVERIFIED]` |

**Anchors that matter for your pricing:**
- **The first *automated software-only* per-player anchor is now SportsVisio at ~$3/player/game
  (~$750/20-game).** Materially: an automated competitor *does* now price per-player stats from phone
  footage — but at a **semi-pro/youth** point, **not** a true-consumer one, and **in court sports.** It
  validates willingness-to-pay for the category while leaving the sub-$15 soccer-consumer point open.
- **Fulltrack proves a true-consumer price is viable on the automated side — at ~$10/mo** — but only in
  cricket, where ball-tracking (not multi-player attribution) is the core output.
- **Automated per-player ceiling (self-serve):** Trace at **$180–$300 per *family* per year** is the
  closest comparable to an automated per-player-stats consumer product. Note it is priced **per family,
  not per team** — a materially different unit than the per-team SaaS above.
- **Human-tagging ceiling:** Hudl Assist **$700–$1,600/team/season** is what the market pays when a
  **human** does the work — leaving large headroom for an automated product priced well below it.
- **In *soccer*, no incumbent monetises pure software-only any-phone automated *per-player* stats at a
  consumer price** — mpact.ai *markets* it but (per §3.1.C) ships only team stats + clip-routing, with
  per-player on a roadmap/manual basis. That soccer lane is **claimed but commercially open** (with the §4
  feasibility gate attached). **In court sports the lane is *taken*:** SportsVisio (basketball/volleyball)
  and Fulltrack (cricket) already monetise software-only any-phone per-player stats — so "no one charges for
  this" is **only true for soccer/large-field sports.**

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
- 🟥 **The 2026-06-24 headline correction: the whitespace is *soccer-specific*, not category-wide.** A
  deliberately cross-sport pass (99 agents, 25 claims verified) found that the "claimed-but-not-filled"
  thesis is **true only in soccer.** In soccer, **mpact.ai** still ships only team stats + clip-routing
  (per-player is roadmap/manual, jersey-OCR-gated) — so the soccer cell is **contested in marketing but
  open in substance.** But in **court/racket sports the cell is being *delivered*:** **★ SportsVisio**
  (basketball/volleyball, **$9M+ funded**, 150+ leagues, ~$3/player/game) appears to genuinely ship
  per-player box scores from **any-phone** footage; **Fulltrack AI** (cricket, ~$10/mo) and **SportAI**
  (racket) do likewise. So a football-first product **cannot treat multi-sport expansion as escape into
  open water** — the court sports are where funded competition already lives. `[VERIFIED 3-0 each, 2026-06-24]`
- 🧭 **The hardware-control pattern is real *but sport-dependent.*** Every automated *soccer/large-field*
  incumbent controls capture via hardware (Trace, Veo, Spiideo, Pixellot, XbotGo) or a managed multi-phone
  rig (Veo Go) — but SportsVisio, Fulltrack and SportAI **broke that pattern in bounded court sports**,
  where legible jerseys + few visible players make attribution tractable *without* owning the camera. The
  moat is a function of **the sport's geometry**, not an iron law.
- 🔬 **Accuracy is the unaudited soft spot across the whole field.** Every per-player accuracy figure here
  is **vendor-self-reported** (SportsVisio's 95%/92%, TwinPlay's, SportAI's). The **one** entrant with
  independent academic testing — **Fulltrack** — was found to **systematically overestimate ball speed**,
  which should temper trust in everyone else's un-benchmarked claims.
- 🪤 **"AI sports" branding repeatedly masks non-stat products.** Verification disqualified **auto-follow
  filming mounts** (Pivo, MOVE'N SEE Pix4Team 2, Pixem — no stats, analysis offloaded), **BallerCam**
  (hardware + human-tagged stats), and **Athlete AI** (highlight editor, stats user-entered) — and
  **refuted TheStats.ai outright (0-3)**. Naming these precisely keeps the competitive map honest.
- ✅ **Hudl Assist is human-in-the-loop and gates capture** (full-field recording workflow) — it is a
  **services** business, priced $700–$1,600/team/season, *not* an automated any-footage product. The
  automated lane below it is genuinely open on cost.
- 💸 **Pricing is now verified** (Hudl, Trace, Veo) and **§6 models unit economics** — the contest is on
  **CAC × seasonal churn**, won via **B2B2C + the highlight virality loop**.
- ⚠️ **Still unverified:** **dollar TAM/CAGR** stays open. The **funding/M&A landscape** is now
  *partially* lit — verified rounds for **SportsVisio ($9M+, Sony Innovation Fund), SportAI ($3M, Nov 2025,
  + Padelytics acquisition) and Fulltrack/Maiden AI ($420k seed)** show **fresh capital flowing into
  phone-footage CV analytics** — but a systematic map (and whether pro incumbents are moving down-market)
  remains incomplete (§5).
