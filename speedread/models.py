from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class PageResult:
    image_path: str
    timestamp_ms: int
    pce_peak_index: int
    score_components: Dict[str, float]
    warp_applied: bool
    quad_points: Optional[List[List[int]]] = None
    ocr_text: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)
