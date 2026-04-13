from dataclasses import dataclass, field
from typing import Optional, Literal

from config import config
from src.tools.session import post_with_retry


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


@dataclass
class UsagePromptTokensDetails:
    cached_tokens: int


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prompt_tokens_details: Optional[UsagePromptTokensDetails] = None
    total_tokens: Optional[int] = None


@dataclass(slots=True)
class GLMOCR:
    id: str
    created: int
    model: str

    md_results: Optional[str] = None
    layout_details: list[list[LayoutDetail]] = field(default_factory=list)
    layout_visualization: list[str] = field(default_factory=list)
    data_info: Optional[DataInfo] = None
    usage: Optional[Usage] = None
    request_id: Optional[str] = None


def glm_ocr(api_key: str, file: str, *, return_crop_images: bool = False, need_layout_visualization: bool = False,
            start_page_id: int = 1, end_page_id: int = 1, request_id: Optional[str] = None,
            user_id: Optional[str] = None) -> GLMOCR:
    """https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%A1%A3%E8%A7%A3%E6%9E%90"""

    def _postprocess(json_raw: dict) -> GLMOCR:
        response_object = GLMOCR(**json_raw)
        response_object.layout_details = [[LayoutDetail(**item) for item in page] for page in json_raw["layout_details"]]
        response_object.data_info = DataInfo(**json_raw["data_info"])
        response_object.usage = Usage(**json_raw["usage"])
        return response_object

    request_url = f"{config.base_url}/paas/v4/layout_parsing"
    header: dict = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    body = {
        "model": "glm-ocr",
        "file": file,
        "return_crop_images": return_crop_images,
        "need_layout_visualization": need_layout_visualization,
        "start_page_id": start_page_id,
        "end_page_id": end_page_id,
        "request_id": request_id,
        "user_id": user_id
    }
    response = post_with_retry(request_url, headers=header, json=body)
    response.raise_for_status()

    return _postprocess(response.json())
