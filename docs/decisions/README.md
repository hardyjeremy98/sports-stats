# Architecture Decision Records

Decision records capture durable choices that should survive implementation changes and new agent
sessions.

| ADR | Decision | Status |
|---|---|---|
| [001](001-identity-without-ocr.md) | Player identity must work without jersey OCR | Accepted |
| [002](002-tracklet-level-global-inference.md) | Infer identity from tracklets over the complete match | Accepted |
| [003](003-quality-gated-multimodal-evidence.md) | Gate and fuse identity evidence by modality quality | Accepted |
| [004](004-semantic-identity-evaluation.md) | Evaluate semantic identity separately from tracking | Accepted |
| [005](005-capped-marginal-naming-balance.md) | Naming belief balance is capped-marginal (unbalanced OT), not doubly-stochastic Sinkhorn | Superseded by 006 |
| [006](006-no-naming-balance.md) | The naming decoder has no balancing step | Accepted |
| [007](007-roster-slot-identity-for-attribution-benchmarks.md) | Roster-slot identity substitutes for jersey number in attribution benchmarks | Accepted |
| [008](008-role-slots-are-not-roster-slots.md) | Tactical role slots are not roster slots; the mapping is per-half and time-varying | Accepted |
| [009](009-job-table-orchestration.md) | Orchestration lives in the job table; cloud services supply workers, never logic | Accepted |

## Adding or changing a decision

Use the next sequential number and include:

- Status and date.
- Context and the problem being decided.
- The explicit decision.
- Consequences and trade-offs.
- Evidence that would justify reconsideration.

Do not edit an accepted ADR to reverse its meaning. Mark it superseded and add a new ADR that links
back to the old one.
