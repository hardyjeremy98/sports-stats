# Architecture Decision Records

Decision records capture durable choices that should survive implementation changes and new agent
sessions.

| ADR | Decision | Status |
|---|---|---|
| [001](001-identity-without-ocr.md) | Player identity must work without jersey OCR | Accepted |
| [002](002-tracklet-level-global-inference.md) | Infer identity from tracklets over the complete match | Accepted |
| [003](003-quality-gated-multimodal-evidence.md) | Gate and fuse identity evidence by modality quality | Accepted |
| [004](004-semantic-identity-evaluation.md) | Evaluate semantic identity separately from tracking | Accepted |

## Adding or changing a decision

Use the next sequential number and include:

- Status and date.
- Context and the problem being decided.
- The explicit decision.
- Consequences and trade-offs.
- Evidence that would justify reconsideration.

Do not edit an accepted ADR to reverse its meaning. Mark it superseded and add a new ADR that links
back to the old one.
