# Jersey OCR — what gates 1–4 need, and how to get it

Tasks 1–3 of [the plan](2026-07-30-jersey-ocr-merge-channel.md) are done and reviewed. Gates 1–4
(Tasks 4–7) are blocked on four things. Items 1 and 2 need a download; items 3 and 4 are local
commands.

All paths are relative to the repo root. The work lives in the worktree at
`.claude/worktrees/jersey-ocr` on branch `worktree-jersey-ocr`, but **data belongs in the main
tree's `data/`** — it is gitignored and shared.

---

## 1. The PARSeq jersey checkpoint (required — this is the model)

Google Drive, from the reference repo's `configuration.py`. Needs `gdown` because Drive
interstitials break plain `curl`.

```bash
uv run --with gdown python -c "
import gdown
gdown.download(
    'https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm',
    'data/weights/parseq-jersey.ckpt',
    quiet=False,
)"
```

The upstream filename is
`parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt`; we store it as
`data/weights/parseq-jersey.ckpt`, which is the path `JerseyReader` defaults to.

**If the Drive quota blocks it** (Drive rate-limits popular files), open the URL in a browser,
download manually, and move it to `data/weights/parseq-jersey.ckpt`. The filename we use is what
matters, not how it arrived.

**Verify it landed:**

```bash
ls -lh data/weights/parseq-jersey.ckpt
uv run --with torch python -c "
import torch; c = torch.load('data/weights/parseq-jersey.ckpt', map_location='cpu')
print('keys:', list(c)[:8])
print('charset:', c.get('hyper_parameters', {}).get('charset_train'))"
```

A checkpoint under ~1 MB is Drive's HTML error page, not weights.

### Correction this forces to the spec

This checkpoint is fine-tuned on **SoccerNet jersey data directly** (val_accuracy 95.60 on that
set). The spec says "hockey and SoccerNet"; the truth is narrower and worse for us, because SNMOT
is also SoccerNet-derived. **Train-adjacency to gate 2's evaluation data is therefore strong, not
incidental**, and every figure from gates 2–4 must say so.

There is a hockey-fine-tuned checkpoint at Drive id `1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE`
(`parseq_epoch=3-step=95-val_accuracy=98.7903...`). Running it as a second arm on SNMOT would be
**less** train-adjacent and therefore a cleaner measurement, at some cost in domain match. Worth
doing if gate 2's numbers look suspiciously good. Not in the current plan.

---

## 2. The SoccerNet jersey dataset, test split (required for gate 1)

No NDA, no password — unlike the tracking and FOOTPASS tiers.

```bash
uv run --with SoccerNet python -c "
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
SNdl(LocalDirectory='data/soccernet/jersey').downloadDataTask(
    task='jersey-2023', split=['test'])"
```

Test-split ground truth is public (only the challenge split is hidden), which is what makes gate 1
possible at all.

**Verify:**

```bash
ls data/soccernet/jersey/test/
python -c "
import json; d = json.load(open('data/soccernet/jersey/test/groundtruth.json'))
print(len(d), 'tracklets;', sum(v == -1 for v in d.values()), 'illegible')"
```

### Correction this forces to the plan

The real layout is `test/groundtruth.json` and `test/image/<tracklet_id>/`. **Task 4's loader in
the plan assumes `test/test_gt.json` and `test/images/<id>/`** — both names are wrong. Task 4 must
be amended to the real layout before it is implemented; its unit test uses `tmp_path` and would
otherwise pass happily against a fixture that mirrors the wrong names, which is exactly the trap
of testing your own assumption.

---

## 3. Scale the SNMOT substrate (local — no download)

`data/soccernet/tracking/test/` already holds 49 sequences; 12 are ingested as videos with GT.
The plan needs more, because 153 true re-entry pairs cannot resolve a tail.

```bash
uv run matchlab-train ingest-soccernet --split test --limit 32
```

**Verify:**

```bash
ls data/videos/soccernet/*.gt.json | wc -l   # expect 32
```

---

## 4. The runtime (required for anything that loads the model)

PARSeq needs torch + timm; the torso localiser needs rtmlib + onnxruntime. None are declared
dependencies, by the repo's isolation convention — they are supplied per invocation:

```bash
uv run --with torch --with timm --with rtmlib --with onnxruntime-gpu \
  matchlab-train run configs/train/jersey-reader-gate.yaml
```

Swap `onnxruntime-gpu` for `onnxruntime` on a CPU-only box. Gate 1 is 1,211 tracklets of small
crops and will run on CPU, slowly. Gates 2–4 decode video and want the GPU.

**Sanity check before a long run:**

```bash
uv run --with torch --with timm --with rtmlib --with onnxruntime-gpu python -c "
import torch, timm, rtmlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## What happens once these land

Gate 1 is a hard stop by design: the reader must reproduce ~87.45% tracklet accuracy on the
SoccerNet jersey test split before gates 2–4 run. A mis-mapped tokenizer still produces
plausible-looking per-tracklet output, so nothing downstream is attributable until a published
number reproduces.

The first thing to run after item 1 is the **deferred probe** (Task 3, Step 1). Task 3 shipped with
its tokenizer/EOS column mapping *unverified* against a real checkpoint — guarded by a runtime
validation that fails loudly rather than by an assumption. The probe replaces that guard with
knowledge:

```bash
uv run --with torch --with timm python - <<'PY'
import torch
ckpt = torch.load("data/weights/parseq-jersey.ckpt", map_location="cpu")
hp = ckpt.get("hyper_parameters", {})
print("charset:", hp.get("charset_train"), "max_label_length:", hp.get("max_label_length"))
model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
print("output shape:", tuple(model(torch.zeros(1, 3, 32, 128)).shape))
print("tokenizer itos:", getattr(getattr(model, "tokenizer", None), "itos", None))
PY
```

Its output is the contract for `JerseyReader._char_probs`'s column mapping. If it disagrees with
what Task 3 assumed, the fix goes there before any gate runs.
