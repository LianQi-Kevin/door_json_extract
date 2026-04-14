from typing import Optional

from ..config import config
from .models import LayoutParsing, LayoutDetail, DataInfo, Usage
from src.tools.session import post_with_retry


def layout_parsing(api_key: str, file: str, *, return_crop_images: bool = False,
                   need_layout_visualization: bool = False, start_page_id: int = 1, end_page_id: int = 1,
                   request_id: Optional[str] = None, user_id: Optional[str] = None) -> LayoutParsing:
    """https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%A1%A3%E8%A7%A3%E6%9E%90"""

    def _postprocess(json_raw: dict) -> LayoutParsing:
        response_object = LayoutParsing(**json_raw)
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
