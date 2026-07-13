from __future__ import annotations

from pydantic import BaseModel


class Point(BaseModel):
    """A 2D point. Units depend on context: image pixels or pitch centimeters."""

    x: float
    y: float


class Box(BaseModel):
    """Axis-aligned box in image pixels, xyxy."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> Point:
        """Anchor used for pitch projection: where the player touches the ground."""
        return Point(x=(self.x1 + self.x2) / 2, y=self.y2)
