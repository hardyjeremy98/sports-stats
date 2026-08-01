"""`yolo-local` class-map resolution (detector-swap safety).

These are pure-function tests -- no ultralytics, no weights, no GPU -- because
`resolve_class_map` is deliberately separated from model loading. The bug they
exist to prevent is silent and total: two football checkpoints that both look
like "4-class football models" order their classes differently, so a weights
swap that only edits a config path relabels every player as a goalkeeper and
every goalkeeper as a player without raising anything.
"""

from __future__ import annotations

from matchlab_core.schemas import DetectionClass
from matchlab_core.stages.detect.yolo_local import _CLASS_MAP, resolve_class_map


def test_roboflow_sports_order_resolves_by_name():
    """The incumbent checkpoint's own order, resolved from names rather than
    assumed from ids -- same result as the historical hard-coded map."""
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    cmap, keep = resolve_class_map(names)
    assert cmap == {
        0: DetectionClass.BALL,
        1: DetectionClass.GOALKEEPER,
        2: DetectionClass.PLAYER,
        3: DetectionClass.REFEREE,
    }
    assert keep == {0, 1, 2, 3}


def test_different_class_order_is_not_mislabelled():
    """A checkpoint ordering ball/player/referee/goalkeeper must NOT be read
    through the roboflow id order. Under the old hard-coded map, id 1 here is
    'player' but would have been emitted as GOALKEEPER and id 3 'goalkeeper'
    as REFEREE -- a total, silent relabelling."""
    names = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}
    cmap, _ = resolve_class_map(names)
    assert cmap[1] is DetectionClass.PLAYER
    assert cmap[3] is DetectionClass.GOALKEEPER
    assert cmap != _CLASS_MAP


def test_coco_checkpoint_keeps_only_person_and_ball():
    """A COCO-pretrained checkpoint is a usable player detector, but only if
    the other 78 classes are dropped rather than defaulted into PLAYER --
    otherwise every bench, bird and handbag becomes a football player."""
    names = {0: "person", 2: "car", 13: "bench", 14: "bird", 32: "sports ball"}
    cmap, keep = resolve_class_map(names)
    assert keep == {0, 32}
    assert cmap[0] is DetectionClass.PLAYER
    assert cmap[32] is DetectionClass.BALL


def test_case_and_whitespace_tolerant():
    cmap, _ = resolve_class_map({0: " Player ", 1: "REFEREE"})
    assert cmap == {0: DetectionClass.PLAYER, 1: DetectionClass.REFEREE}


def test_missing_names_falls_back_to_roboflow_order():
    for names in (None, {}, []):
        cmap, keep = resolve_class_map(names)
        assert cmap == _CLASS_MAP
        assert keep == set(_CLASS_MAP)


def test_unrecognisable_names_fall_back_but_keep_every_id():
    """A single-class checkpoint named '0' tells us nothing. Fall back to the
    old behaviour rather than dropping all of its detections."""
    cmap, keep = resolve_class_map({0: "0"})
    assert cmap == _CLASS_MAP
    assert keep == {0}


def test_list_names_are_accepted():
    cmap, _ = resolve_class_map(["ball", "goalkeeper", "player", "referee"])
    assert cmap[2] is DetectionClass.PLAYER
