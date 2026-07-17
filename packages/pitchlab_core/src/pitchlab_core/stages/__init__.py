"""Stage implementations. Importing this package registers every implementation
with the registry — add new modules to the list below."""

from pitchlab_core.stages.annotate import overlay  # noqa: F401
from pitchlab_core.stages.associate import (  # noqa: F401
    global_embed,
    global_reid,
    identity_fallback,
)
from pitchlab_core.stages.calibrate import (  # noqa: F401
    roboflow_keypoints,
    static,
    yolo_pitch_local,
)
from pitchlab_core.stages.detect import (  # noqa: F401
    oracle,
    roboflow,
    synthetic,
    yolo_local,
    yolox_local,
)
from pitchlab_core.stages.events import possession, spotting_stub  # noqa: F401
from pitchlab_core.stages.fuse import minimap  # noqa: F401
from pitchlab_core.stages.identity import face, none  # noqa: F401
from pitchlab_core.stages.team import kit_color, siglip  # noqa: F401
from pitchlab_core.stages.track import botsort, iou, learned_stub  # noqa: F401
