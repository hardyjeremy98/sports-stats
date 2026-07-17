"""Vendored inference-only YOLOX (MixSort variant). See README.md for lineage."""

from .boxes import postprocess


def build_yolox(depth: float, width: float, num_classes: int):
    from .yolo_head import YOLOXHead
    from .yolo_pafpn import YOLOPAFPN
    from .yolox import YOLOX

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    return YOLOX(backbone, head)


__all__ = ["build_yolox", "postprocess"]
