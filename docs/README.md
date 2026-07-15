# Documentation Map

This page defines where project knowledge belongs and which source wins when documents disagree.

## Precedence

Use this order for product and engineering decisions:

1. **Accepted decision records** in [`decisions/`](decisions/).
2. **Canonical current vision documents**, currently
   [`player-identity-vision.md`](player-identity-vision.md).
3. **Factual implementation inventory** in
   [`implementation-status.md`](implementation-status.md), verified against code.
4. **Repository operating guidance** in [`../CLAUDE.md`](../CLAUDE.md).
5. **Current architecture overview** in [`../README.md`](../README.md).
6. **Research and historical recommendations** in [`../technology/`](../technology/).
7. **Market research and other supporting material** under `docs/`.

This is not a general claim that an ADR overrides code reality. Decision records describe intended
constraints; `implementation-status.md` describes what currently exists. When those differ, record
the implementation as incomplete rather than rewriting the decision to match it.

## Canonical documents

| Document | Responsibility | Update when |
|---|---|---|
| [`player-identity-vision.md`](player-identity-vision.md) | Product objective, terminology, identity strategy, evaluation principles | Product goals, hard constraints, or strategic priorities change |
| [`implementation-status.md`](implementation-status.md) | Implemented/prototype/stub/planned inventory | Code, tests, evaluation, or Lab capability changes |
| [`decisions/`](decisions/) | Durable choices and their consequences | A major choice is accepted, superseded, or reversed |
| [`../CLAUDE.md`](../CLAUDE.md) | Concise instructions needed in every AI coding session | Commands, architecture invariants, canonical links, or maintenance rules change |
| [`../README.md`](../README.md) | New-contributor and product overview | Public architecture, setup, or top-level product description changes |

## Supporting documents

- `technology/` is a dated research dossier. It contains evidence, candidates, and historical
  recommendations. It is not implementation truth.
- `docs/canvases/` contains version-controlled source for interactive Cursor analytical artifacts.
- `docs/market_research/` contains market evidence and commercial context.
- `docs/roadmap.md` is early business commentary, not the current engineering roadmap.

## Documentation maintenance rules

When changing code:

1. Update `implementation-status.md` if capability status or behavior changed.
2. Update the relevant canonical vision only if the intended direction changed.
3. Add or supersede an ADR when changing a durable constraint or architecture decision.
4. Update `README.md` when setup, public behavior, or the top-level pipeline changes.
5. Update `CLAUDE.md` only with concise, repeatedly useful instructions.
6. Link measured claims to an experiment report, run set, dataset split, and code/model revision.
7. Preserve dates and status labels on research and canonical documents.

When adding research:

1. Put paper/model findings in `technology/` or a dated experiment report.
2. Distinguish sourced facts from engineering hypotheses.
3. State dataset, footage domain, metric, date, licensing, and model availability.
4. Do not promote a model to `implementation-status.md` until runnable code exists.
5. Do not turn a research recommendation into policy without updating the canonical vision or an ADR.

## Resolving contradictions

Do not silently average conflicting documents.

1. Identify whether the conflict is about intent, implementation, or evidence.
2. Apply the precedence rules above.
3. Correct stale lower-precedence text or add a supersession note.
4. If the higher-precedence decision itself changed, supersede its ADR explicitly.
5. Keep historical evidence when useful, but label it as historical.
