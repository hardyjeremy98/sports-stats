# Executive Summary & Verdict

*Read this first. Detailed sections 1–4 follow; the evidence ledger is in section 5.*

---

## The verdict in three sentences

1. **Demand is real but smaller than the headlines.** A large grassroots football population exists,
   but the per-player-analytics base is the *registered/club* cohort (≈38M globally per FIFA; 640K
   outdoor-affiliated in Australia), not the 265M "anyone who kicks a ball" figure. **`[VERIFIED]`**
2. **The product's core technical premise is the unsolved part.** Accurate full-team per-player stats
   from a *single ground-level phone camera* depends on multi-object tracking, player re-ID, and
   per-player attribution that the literature says are **not solved** — and the validated pipelines
   (academic and commercial) are built for **broadcast video, not phone footage**. **`[VERIFIED]`**
3. **Regulation adds real friction where your users are youngest.** Filming minors triggers written
   parental-consent and operational "do-not-film" obligations (UK FA), which complicate a frictionless
   self-serve consumer-upload model. **`[VERIFIED]`**

> **Bottom line:** The white-space you spotted (clean upload → full-team per-player stats at a consumer
> price) is real but **narrower on two fronts than the brief assumed.** *Technically*, it's a gap that
> exists **because the CV and unit economics don't yet robustly support it** — a **hard-tech / applied-CV
> bet**, not a pure-SaaS bet. *Commercially*, a focused competitive scan (§3, updated 2026-06-23) found
> the position is **no longer vacant: at least one entrant (mpact.ai) already markets the exact pitch —
> automated per-player stats from "any phone," consumer-priced.** So it is investable, but only if you can
> (a) constrain the capture problem, (b) build a domain-adaptation data moat, (c) be honest that "any
> phone footage" is the hardest possible starting point, **and (d) beat mpact.ai/peers to a *working*
> product — you would not be first to *claim* the cell.**

---

## Why the gap exists (the central tension)

Your thesis has two halves. The research validates one and challenges the other:

| Half of the thesis | Verdict | Why |
|---|---|---|
| **"There's a large, underserved grassroots audience"** | ✅ Supported | Millions of registered players/teams worldwide; incumbents skew expensive/pro/hardware-bound (section 2–3). |
| **"We can deliver accurate per-player stats from any phone footage at consumer price"** | ⚠️ Challenged | The CV stack required (MOT + re-ID + jersey OCR + event attribution) is unsolved on *broadcast* video and degrades further on *amateur ground-level* video (section 4). |

The gap is not "nobody thought of this." The gap is **"the input you've chosen (single phone camera,
upload-and-go) is the regime where the technology is weakest and the per-unit compute cost is real,
against a consumer price ceiling and seasonal churn."**

---

## Kill-risks, ranked

| # | Risk | Severity | Evidence basis |
|---|------|----------|----------------|
| 1 | **Accuracy at a consumer price point** — per-player attribution from one ground-level camera isn't reliably solved; users punish wrong stats harder than missing ones | **Critical** | `[VERIFIED]` MOT "far from solved"; re-ID harder than surveillance; broadcast-only validation |
| 2 | **Capture friction** — someone still has to film a wide, elevated, legible shot; "any phone footage" makes the CV *harder*, not the product *easier* | **High** | `[VERIFIED]` pipelines need broadcast-grade input; `[INFERRED]` ground-level worsens occlusion |
| 3 | **CV commoditization vs. moat** — YOLO/DeepSORT/MediaPipe are commodity, so a model alone is no moat; but sports re-ID needs domain adaptation, so it's also not free | **High** | `[VERIFIED]` surveillance re-ID degrades on sports without modification |
| 4 | **Unit economics / seasonal churn** — per-match compute cost + off-season cancellation vs. consumer WTP | **High** | `[UNVERIFIED]` — *no CAC/LTV/churn data confirmed; must be modelled* |
| 5 | **Minors' privacy regulation** — consent + do-not-film systems complicate frictionless upload | **Medium** | `[VERIFIED]` FA consent + wristband obligations (UK-specific) |
| 6 | **A competitor already claims the cell** — mpact.ai markets automated per-player stats from "any phone" at consumer price; funded incumbents (StepOut, $1.5M) could also extend down-market | **Medium→High** | `[VERIFIED]` mpact.ai occupies the target cell *by claim* (§3, 2026-06-23); StepOut funded & broadly positioned but `[REFUTED]` it accepts phone footage *today* |

---

## Where to position if you proceed (white-space)

The defensible openings are **not** "best CV model." They are:

- **Constrain the capture problem instead of pretending it doesn't exist.** Ship a cheap capture spec
  (tripod height, corner placement, a guided in-app framing overlay, or a low-cost clip-on wide lens).
  The winners in this space (Veo, Trace, XbotGo) effectively *sell the capture solution*. "Upload any
  footage" is a marketing promise that fights your own accuracy.
- **Own a proprietary, labelled, grassroots-footage dataset.** That — not the architecture — is the
  moat, because generic re-ID degrades on sports and sports data is scarce. Every customer match
  becomes training data; design consent and labelling into the funnel.
- **Pick a wedge JTBD that tolerates imperfect attribution.** Team-level + highlight clips + "good
  enough" individual stats for *development and sharing* (not recruitment-grade truth) is far more
  forgiving than promising Opta-accurate event data to semi-pro scouts.
- **B2B2C through clubs/leagues to solve capture + consent + distribution at once.** A club fixes the
  camera, handles parental consent in its membership flow, and amortises CAC across a roster — directly
  attacking risks #2, #4 and #5 together.

---

## What you still need to find out before committing

These were **not** answered by the research and are the difference between "interesting" and
"investable" (full list in [section 5](docs/05-evidence-and-methodology.md#open-questions)):

1. **Real unit economics**: per-match GPU/inference cost vs. consumer price; CAC, LTV, and *measured*
   seasonal churn. *(Now modelled in [§6](docs/06-unit-economics-deep-dive.md): COGS ~$1.80–$4.70/match;
   the binding constraints are per-payer LTV and seasonal churn, not compute.)*
2. ~~Competitor capability on phone footage~~ **✅ now verified (§3, 2026-06-23):** all *automated*
   incumbents (Trace/Veo/Spiideo/zone14/Pixellot/XbotGo) are **hardware-gated**; **Hudl Assist** is
   **human-in-the-loop + structured capture**; **mpact.ai** claims automated any-phone stats.
   **New unknown:** does **mpact.ai have real traction/accuracy**, or is the cell occupied only on paper?
3. **Current SOTA accuracy** for end-to-end per-player event attribution from one amateur camera.
4. **Binding compliance** beyond UK FA: US COPPA, EU/UK GDPR biometric rules, Australian Privacy Act,
   and image/likeness + data-ownership defaults under a self-serve upload model.
5. **Dollar TAM/CAGR** and the **funding/M&A landscape** — both still unverified after two passes (no
   surviving claims); directional only (Veo raised $25M in 2021; StepOut $1.5M in 2025).

> Next step recommendation: run a **technical de-risking spike** (can you hit acceptable per-player
> accuracy on *real grassroots phone footage* you collect yourself?) **in parallel with** a
> **fake-door + 15-interview demand test**. Both are cheap; both attack the two halves of the thesis
> that this research left genuinely open.
