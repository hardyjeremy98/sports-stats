# Sports Video Analytics SaaS — Market Research

Research dossier for a proposed **B2C/B2B sports video analytics SaaS**: upload amateur/grassroots
footage (including ground-level phone video) → get **automated player-by-player performance stats**
(shots, shots on target, interceptions, passes, missed passes, sport-dependent extras). Soccer first,
other sports later. B2C = players/teams/coaches/parents; B2B = orgs and rec centres that resell.

> **How to read this dossier.** Findings are tagged by evidence strength. Each claim that survived
> adversarial fact-checking is marked **`[VERIFIED]`** with its vote (e.g. 3-0) and a citation.
> Anything the research could **not** confirm — including several headline numbers from the original
> brief — is marked **`[UNVERIFIED]`** or **`[REFUTED]`** and must be treated as a working assumption,
> not a fact. See [`05-evidence-and-methodology.md`](docs/05-evidence-and-methodology.md) for the full
> claim ledger and source list.

## Contents

| # | Section | What's inside |
|---|---------|---------------|
| — | [**Executive Summary & Verdict**](docs/00-executive-summary.md) | The bottom line, read this first |
| 1 | [Market Definition & Sizing](docs/01-market-definition-and-sizing.md) | Segmentation, TAM/SAM/SOM, the carve-out trap |
| 2 | [Demand — The Customer](docs/02-demand.md) | Personas, JTBD, participation counts, willingness-to-pay, barriers |
| 3 | [Supply — Competition & Money](docs/03-supply-and-competition.md) | Competitive map, pricing, business models, GTM |
| 4 | [Enabling Environment](docs/04-enabling-environment.md) | CV capability, what's unsolved, Five Forces, privacy/IP, funding |
| 5 | [Evidence & Methodology](docs/05-evidence-and-methodology.md) | Verified claims, refuted claims, open questions, sources |
| 6 | [Unit Economics Deep-Dive](docs/06-unit-economics-deep-dive.md) | Cost-to-process-a-match, pricing benchmarks, worked margin/LTV:CAC model |
| 7 | [Technology Maturity Deep-Dive](docs/07-technology-maturity-deep-dive.md) | Per-component CV benchmarks (HOTA/mAP/GS-HOTA), years-to-product |

> **Sections 6–7 are follow-up deep-dives** commissioned after the main dossier, drilling into the two
> halves of the verdict that section 0 flagged as the binding constraints: *unit economics* and
> *technology maturity*.

## The one-paragraph answer

A **real grassroots demand base exists** (FIFA: 265M players globally, but only ~38M *registered*;
Australia alone has **640K club-registered outdoor players** as of 2024) — so the bottom-up addressable
market is genuine, *if* far smaller than the "sports analytics is $X billion" headlines that are
dominated by pro clubs, broadcasters and betting. But the **core technical premise** — clean self-serve
upload of a *single ground-level phone camera* yielding *accurate full-team per-player stats* — sits
directly on CV problems the peer-reviewed literature shows are **not solved**: multi-object tracking
under occlusion, player re-identification with near-identical kits, and per-player event attribution
when jersey numbers aren't visible. The leading academic pipelines *and* the closest funded broadcast-grade
competitor (StepOut) are built for **professional broadcast feeds, not phone footage**. Add
**minors'-footage regulation** (the FA mandates written parental consent and operational "do-not-film"
systems) and the *technical* verdict is: **the white-space is real precisely because the technology and
unit economics do not yet robustly support it at a consumer price.** This is a hard-tech bet disguised as
a SaaS bet. **But the *commercial* white-space is narrower than that makes it sound:** a focused
competitive scan (§3, 2026-06-23) found **mpact.ai already marketing automated per-player stats from "any
phone" at a consumer price** — but a four-angle deep-dive found it ships only **team stats + highlight-clip
routing**, with genuine *per-player* stats on a **roadmap/manual, jersey-OCR-gated** basis. So the cell is
**claimed, not filled** — the real question is execution (a *working* product) and traction, not novelty.

## Coverage warning

The research returned **strong, primary-sourced** evidence on (a) bottom-up demand counts,
(b) the technology-capability and minors'-privacy questions, and — after a **second pass (2026-06-23)** —
(c) **competitor capability, pricing, and the commercial white-space** (§3). That second pass found the
target cell is **claimed but not filled**: **mpact.ai** markets automated per-player stats from "any
phone" but (verified four ways) ships only team stats + clip-routing, with per-player on a roadmap/manual
basis; meanwhile every *automated* incumbent (Trace, Veo, Spiideo, zone14, Pixellot,
XbotGo) is **hardware-gated** and **Hudl Assist** is **human-in-the-loop + structured capture**.
It still returned **thin or no verified** evidence on: **top-down dollar TAM/CAGR** and the
**funding/M&A landscape** (whether pro incumbents like Stats Perform are moving down-market is
undetermined), plus willingness-to-pay/VOC. Treat those as **open questions requiring primary research** —
listed explicitly in [section 5](docs/05-evidence-and-methodology.md#open-questions).

---
*Generated by a fan-out / adversarial-verification research workflow: 5 search angles → 24 sources
fetched → 104 claims extracted → 25 verified by 3-vote adversarial panel → 15 confirmed, 10 killed.
Last updated 2026-06-21.*
