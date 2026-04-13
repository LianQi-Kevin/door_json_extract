from dataclasses import dataclass
from typing import Optional, Literal

from src.tools.session import post_with_retry
from config import config


@dataclass
class LayoutDetail:
    index: int
    label: Literal["image", "text", "formula", "table"]
    bbox_2d: Optional[tuple[float, float, float, float]]
    content: Optional[str]
    height: Optional[int]
    width: Optional[int]


@dataclass
class DataInfoPage:
    width: int
    height: int


@dataclass
class DataInfo:
    num_pages: int
    pages: list[DataInfoPage]


@dataclass
class UsagePromptTokensDetails:
    cached_token: int


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: UsagePromptTokensDetails
    total_tokens: int


@dataclass(slots=True)
class GLMOCR:
    id: str
    created: int
    model: str

    md_results: Optional[str]
    layout_details: list[list[LayoutDetail]]
    layout_visualization: list[str]
    data_info: DataInfo
    usage: Usage
    request_id: str


def glm_ocr(api_key: str, file: str) -> GLMOCR:
    """
    :param api_key:  bearer token
    :param file:    image url or image base64. JPG/PNG/PDF
    """
    def _postprocess(json_raw: dict) -> GLMOCR:
        # todo: unfinished
        pass

    request_url = f"{config.base_url}/paas/v4/chat/completions"
    header: dict = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    body = {
        "model": "glm-ocr",
        "file": file
    }
    response = post_with_retry(request_url, headers=header, json=body)
    response.raise_for_status()

    return _postprocess(response.json())
