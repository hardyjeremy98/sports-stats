"""Stage implementations. Importing this package registers every implementation
with the registry — add new modules to the list below."""

from matchlab_core.stages.annotate import overlay  # noqa: F401
from matchlab_core.stages.associate import (  # noqa: F401
    global_embed,
    global_reid,
    identity_fallback,
    reid_engine,
)
from matchlab_core.stages.calibrate import (  # noqa: F401
    pnlcalib,
    roboflow_keypoints,
    static,
    yolo_pitch_local,
)
from matchlab_core.stages.detect import (  # noqa: F401
    frozen,
    oracle,
    rfdetr,
    roboflow,
    synthetic,
    yolo_local,
    yolox_local,
)
from matchlab_core.stages.events import possession, spotting_stub, tdeed  # noqa: F401
from matchlab_core.stages.fuse import minimap  # noqa: F401
from matchlab_core.stages.identity import face, none  # noqa: F401
from matchlab_core.stages.team import kit_color, siglip  # noqa: F401
from matchlab_core.stages.track import (  # noqa: F401
    botsort,
    botsort_reid,
    iou,
    learned_stub,
    ocsort,
)
from matchlab_core.stages.track.tdlp import stage as tdlp_shippable  # noqa: F401
from matchlab_core.stages.track.tdlp_full import stage as tdlp_full  # noqa: F401
