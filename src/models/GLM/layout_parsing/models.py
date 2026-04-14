from dataclasses import dataclass, field
from typing import Optional, Literal

from ..model_base import Usage


@dataclass
class LayoutDetail:
    index: int
    label: Literal["image", "text", "formula", "table"]
    native_label: Optional[str] = None
    bbox_2d: Optional[list[int]] = None
    content: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None


@dataclass
class DataInfoPage:
    width: int
    height: int


@dataclass
class DataInfo:
    num_pages: int
    pages: list[DataInfoPage] = field(default_factory=list)


@dataclass(slots=True)
class LayoutParsing:
    id: str
    created: int
    model: str

    md_results: Optional[str] = None
    layout_details: list[list[LayoutDetail]] = field(default_factory=list)
    layout_visualization: list[str] = field(default_factory=list)
    data_info: Optional[DataInfo] = None
    usage: Optional[Usage] = None
    request_id: Optional[str] = None
