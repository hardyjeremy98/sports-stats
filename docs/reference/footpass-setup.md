# FOOTPASS acquisition and layout

**FOOTPASS** (Footovision Play-by-Play Action Spotting in Soccer) is the dataset behind the
SoccerNet 2026 **Player-Centric Ball-Action Spotting** challenge: 54 complete top-European
matches (2023–24), full-length broadcast video plus per-player tactical data with track-level
identity, tactical role, pitch position and velocity.

Paper: Ochin, Chekroun, Stanciulescu & Manitsaris, *FOOTPASS: A multi-modal multi-agent
tactical context dataset for play-by-play action spotting in soccer broadcast videos*, CVIU
269:104790 (2026). Baselines and loaders: `github.com/JeremieOchin/FOOTPASS`.

## Why this tier exists in MatchLab

Filed initially as a **B4** (action attribution) asset — it is the substrate for reproducing
TAAD → TAAD+GNN → TAAD+DST. It is also, unexpectedly, the strongest available substrate for
**B2 position evidence**, for three reasons that no other tier in `configs/datasets/` provides:

1. **Full-length matches with track-level identity.** The GT-tracklet re-ID harness
   (`stages/track/oracle.py`, SPO-85) fragments GT tracks at their natural gaps to get exact
   merge labels. On SoccerNet that runs over 30-second / 750-frame clips. FOOTPASS runs the
   same construction over 90 minutes, which is the only way role/zone occupancy evidence
   becomes measurable at all — a positional footprint needs minutes, not seconds.
2. **Ground-truth pitch positions and velocities**, so the position-evidence channel can be
   developed with calibration error factored out entirely, then re-tested with real
   calibration afterwards. Same isolation principle as the oracle-team / oracle-detection
   conditions used throughout the re-ID work.
3. **Ground-truth tactical roles** (13 per team) — supervised targets for the role/zone
   occupancy prior, and the 2 × 13 = 26 role-slot encoding that DST's `teamvec` uses is
   directly the representation the global roster-assignment layer wants.

Caveats that bound its use — these are the same ones recorded on the B4 Experiment Log entry
and they still hold: FOOTPASS **supplies** tracking, jersey and role as inputs, so it can
never be a benchmark for *producing* identity, and its challenge metric requires jersey-number
match at a fixed high-recall threshold (τ = 0.15), which is the opposite end of the curve from
MatchLab's abstention-first design. It is broadcast footage, so the amateur-phone-footage
domain gap remains. Use it as a **development and isolation substrate**, not as a do-no-harm
gate — the same standing caution that applies to the GT-fragment harness.

## Access

Two separate gates, and you need both:

1. **Hugging Face access.** The data lives at
   `huggingface.co/datasets/SoccerNet/SN-PCBAS-2026`, which is a **gated** repo. Two distinct
   things are required and they fail differently — a **read** token authenticates you, and a
   per-account **access grant** on the dataset page authorizes you:
   - `401` — no token, or a fine-grained token missing the *"Read access to contents of all
     public gated repos you can access"* scope. A classic **Read** token avoids this trap;
     write scope is never needed.
   - `403` — token valid, but the account is not on the dataset's authorized list. Fix by
     visiting the dataset page while logged in and requesting access. Access was granted for
     account `JDHdsa` on 2026-07-27; a `403` recurring later means the grant was revoked or a
     different account's token is in use, not a scope problem.
2. **The SoccerNet NDA password**, obtained by submitting SoccerNet's NDA form. It cannot clear
   the Hugging Face gate — it is only for decrypting archives.
   **Measured 2026-07-27: the three `tactical_data_*.zip` archives are NOT encrypted** (checked
   via `zipfile.ZipInfo.flag_bits & 0x1` — all `False`), so no password is needed for this tier.
   Beware the false positive here: `unzip -P <anything>` succeeds silently on an unencrypted
   archive, so a successful extraction does **not** confirm the password is right. The video
   zips are presumably the encrypted ones the upstream README refers to, but that is
   **unverified** — none have been downloaded.

**Neither secret belongs in this repo.** Both live in the gitignored `.env` as `HF_TOKEN` and
`SOCCERNET_NDA_PASSWORD`; keep them out of docs, configs and commits.

Check both gates before starting a multi-GB pull:

```bash
set -a; . ./.env; set +a
curl -sS -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_TOKEN" \
  -L https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026/resolve/main/tactical_data_format.txt
```

`200` means both gates are clear. To confirm *which* account a token belongs to (a common
cause of a stubborn `403` is requesting access on one account and issuing the token on
another): `curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2`.

## File inventory (upstream, 2026-07-27)

| File | Size | Needed for |
|---|---:|---|
| `tactical_data_TRAIN.zip` | 2.46 GB | B2 position/role work, B4 DST |
| `tactical_data_VAL.zip` | 0.16 GB | " |
| `tactical_data_CHALLENGE.zip` | 0.15 GB | " |
| `tactical_data_format.txt` | — | schema note (gated) |
| `videos_352x640_TRAIN.zip` | 22.8 GB | B4 TAAD training (the resolution the baselines use) |
| `videos_352x640_VAL.zip` | 1.36 GB | " |
| `videos_352x640_CHALLENGE.zip` | 1.55 GB | " |
| `videos_fullHD_TRAIN_01..05.zip` | 183.5 GB | full-resolution crops (B2 appearance on full matches) |
| `videos_fullHD_VAL.zip` | 11.3 GB | " |
| `videos_fullHD_CHALLENGE.zip` | 11.5 GB | " |
| **Total** | **~235 GB** | |

**Acquire in tiers.** The tactical data is 2.77 GB and delivers everything B2 needs for the
position-evidence and role/zone work — no video at all. The 352×640 tier (25.7 GB) is what the
FOOTPASS baselines train on and is the right second step for B4. The full-HD tier is 206 GB
and only earns its disk if B2 appearance work moves onto full matches; note that player crops
from 352×640 broadcast frames are far too small for re-ID embeddings, so full-HD is the only
video tier that would serve B2.

## Layout in this repo

`data/` is gitignored; nothing below is committed.

```
data/footpass/
  tactical_data_{TRAIN,VAL,CHALLENGE}.zip   # 2.77 GB, kept for re-extraction; safe to delete
  tactical/            # PRESENT: {train,val,challenge}_tactical_data.h5 (9.3 GB)
  videos_352x640/      # empty — low-res broadcast video (B4 baselines)
  videos_fullHD/       # empty — full-resolution broadcast video (only if B2 needs crops)
```

`TAAD_sample_list.json` and `playbyplay_GT/*.json` are **not** in the tactical archives — they
ship in the baseline GitHub repo (`JeremieOchin/FOOTPASS`), which is where a B4 reproduction
would get them.

The FOOTPASS baseline repo expects `<repo>/data/` and `<repo>/videos/`; if those baselines are
run, do it in a sibling checkout (the `external-trackers/` / `external-calibrators/` /
`external-spotters/` isolation pattern) and point it at these paths rather than duplicating
the data.

## Tactical data format

One HDF5 file per split, keyed by game id (`game_18_H1`, `game_18_H2`, … — one key per half).
Each value is a 2-D array of per-player-per-frame rows.

**TRAIN and VAL carry 14 columns; TEST/CHALLENGE carries only 13 — it has no `class` column,
because the challenge action labels are withheld for the evaluation server.** Any loader must
branch on split rather than assuming a fixed width. Note this does *not* limit the B2 use of the
challenge split: identity, role, position, velocity and boxes are all present; only the action
labels are missing.

Column order (upstream `tactical_data_format.txt`, matching `utils/TAAD_Dataset.py:178`):

| # | Column | Notes |
|---:|---|---|
| 0 | `FRAME` | 0-based frame index, 25 fps |
| 1 | `PLAYER_ID` | track-level identity — the B2 ground truth |
| 2 | `LEFT_TO_RIGHT` | **pitch side**, 0 = left / 1 = right — *not* a club key, see below |
| 3 | `SHIRT_NUMBER` | jersey number |
| 4 | `ROLE_ID` | tactical role, 1–13 (see below) |
| 5 | `X_POS` | pitch x, normalised (baselines mirror with `1 - x`) |
| 6 | `Y_POS` | pitch y, normalised |
| 7 | `X_SPEED` | velocity x |
| 8 | `Y_SPEED` | velocity y |
| 9 | `ROI_X` | bbox left; **NaN when the player is not visible in frame** — the baselines use `~isnan(ROI_X)` as the observability flag |
| 10 | `ROI_Y` | bbox top |
| 11 | `ROI_WIDTH` | bbox width (full-HD pixel scale; the baselines divide by 3 for 352×640) |
| 12 | `ROI_HEIGHT` | bbox height |
| 13 | `CLS` | action class at this frame, 0 = none — **absent in TEST/CHALLENGE** |

**Identity keys — `LEFT_TO_RIGHT` is not a club.** It encodes which end a player is
defending, and it is 0-on-the-left in *every* half of the val split. What changes at
half-time is the **club↔side binding**: verified 18/18, 21/21 and 21/21 of the players
present in both halves take the opposite `LEFT_TO_RIGHT` value in H2 (`game_18`, `game_24`,
`game_47`). The stable keys are `PLAYER_ID` for a player and **`PLAYER_ID // 100` ∈ {1, 2}**
for a club. Shirt numbers also collide across the two teams within a half (7 collisions in
`game_24_H2`), so `(LEFT_TO_RIGHT, SHIRT_NUMBER)` is safe only as a per-half join key, never
as an aggregation key. Attacking direction follows directly: side 0 attacks +x, side 1
attacks −x, and normalising to a common direction is a **180° rotation** (`x → 1−x` *and*
`y → 1−y`) — the `1 − x` mirror noted in the `X_POS` row above is the baselines' own
convention and reflects the pitch, swapping the left and right flanks.

**Roles (1–13):** 1 Goalkeeper · 2 Left Back · 3 Left Central Back · 4 Mid Central Back ·
5 Right Central Back · 6 Left Midfielder · 7 Right Midfielder · 8 Defensive Midfielder ·
9 Attacking Midfielder · 10 Left Winger · 11 Right Winger · 12 Central Forward · 13 Right Back.

**Action classes (`CLS`, 8 non-zero):** 1 Drive · 2 Pass · 3 Cross · 4 Throw-in · 5 Shot ·
6 Header · 7 Tackle · 8 Block. Events are instantaneous (one frame each).

> **Corrected 2026-08-05.** This list previously read "Drive, Pass, Cross, **Shot, Header,
> Throw-in**, Tackle, Block", transposing classes 4–6. The data settles it: class 4 has 86.6%
> of its events within 5% of a touchline and class 5 has 0% (throw-in vs shot), and class 5
> concentrates 64% near a goal end. Anything built against the old order labelled every shot
> a throw-in. Verified on all 6 val halves; the tactical `CLS` stream and
> `playbyplay_val.json` agree exactly on the full `(frame, team, shirt, class)` tuple set
> (6070 events, zero symmetric difference).

**The `replay` flag exists only in `playbyplay_{split}.json`** (column 5), never in the
tactical HDF5. **10.1% of val events (614/6070) are broadcast replays** that duplicate live
play. Any event-derived statistic built from the h5 alone silently double-counts them and
fabricates possession changes at replay boundaries — see
`matchlab_train.datasets.footpass_events`, which joins the two files on
`(frame, team, shirt, class)`.

**Role-slot encoding.** DST builds `slot = LEFT_TO_RIGHT * 13 + (ROLE_ID - 1)`, giving 26
slots, then a `(5, 26, T)` tensor of `[x, y, vx, vy, observed]`. That is exactly the
representation proposed for B2's global roster-assignment layer, and it is worth reusing
rather than reinventing.

## Download commands

**Executed 2026-07-27** for the tactical tier; downloaded byte sizes matched the upstream
listing exactly. Credentials come from the gitignored `.env`, never inline:

```bash
set -a; . ./.env; set +a

# Tactical data only (2.77 GB compressed, 9.3 GB extracted) — everything B2 needs, no video.
BASE=https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026/resolve/main
for f in tactical_data_VAL.zip tactical_data_CHALLENGE.zip tactical_data_TRAIN.zip; do
  curl -sS --fail -L -C - -H "Authorization: Bearer $HF_TOKEN" -o "data/footpass/$f" "$BASE/$f"
done

# No password required for this tier — the tactical zips are not encrypted.
for f in data/footpass/tactical_data_*.zip; do
  unzip -o -q "$f" -d data/footpass/tactical
done
```

`--fail` matters: without it a `403` writes the JSON error body into the `.zip` and the failure
only surfaces later as a corrupt archive.

## Verified contents (2026-07-27)

Read back from the extracted HDF5, not assumed:

| Split | File | Extracted | Keys | Matches | Columns |
|---|---|---:|---:|---:|---:|
| train | `train_tactical_data.h5` | 8.2 GB | 96 | 48 | 14 |
| val | `val_tactical_data.h5` | 530 MB | 6 | 3 | 14 |
| challenge | `challenge_tactical_data.h5` | 504 MB | 6 | 3 | **13** |

54 matches total, as advertised. Keys are `game_<id>_H<half>`; train alone carries **157.2M
rows**. A half is ~72,000 frames (~48 min at 25 fps) covering 22–23 players, with all 13 roles
and both teams present.

**The number that matters most for B2: per-frame bbox visibility is only 36–45%**, and
per-player visibility on a val match ranges 9% → 48% (median 37%). Players are off-camera the
*majority* of the match in broadcast framing — which is precisely the fragmentation-and-re-entry
regime the merge layer exists to handle, with ground-truth identity spanning every gap. No other
tier in `configs/datasets/` exposes that regime at match length.

## Not yet done

**No ingest adapter exists**, and no video has been downloaded. `configs/datasets/footpass.json`
is still a starter manifest with an empty `sequences` list — the same intentional placeholder
pattern as `soccernet-ball.json`.

Ingesting this tier means mapping the tactical rows onto `matchlab_core.gt.GroundTruth`
(per-track boxes keyed by frame, `PLAYER_ID` as the track identity, `SHIRT_NUMBER` as the jersey
identity the semantic-identity layer scores against), and deciding whether pitch position and
role ride alongside as a sidecar. Four things the adapter must handle, all confirmed against the
real files:

- **Split-dependent width** — 13 columns for challenge, 14 for train/val. Branch, don't assume.
- **`NaN` in `ROI_*`** means "player not visible this frame", not missing data. It is the
  observability flag, and at 36–45% density it is the common case, not the exception.
- **Halves are separate keys** (`game_18_H1` / `game_18_H2`), but their frame numbering is
  **GLOBAL, not independent** — H2 continues past the end of H1 (`game_18_H1` ends at 75307,
  `game_18_H2` starts at 75525). Verified 2026-08-06 on **all 48 TRAIN and 3 VAL matches**:
  51/51 H2 keys start after their H1 ends. This entry previously said "independent frame
  numbering", and code that trusted it applied its own H1 offset to H2, double-shifting the
  axis by a whole half — the 2026-08-05 occupancy contamination. A "match" is still two
  sequences unless explicitly stitched, but stitching must NOT re-offset H2.
- **Boxes are full-HD pixel scale**; the baselines divide by 3 for the 352×640 video tier, so
  any crop pipeline must know which video tier it is paired with.

An adapter can be written now — the schema is verified and the data is on disk. Nothing about it
depends on the video tiers.
