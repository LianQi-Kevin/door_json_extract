import logging
from dataclasses import asdict
from typing import cast

from ..config import config
from .models_request import Request
from .models_response import Response, ResponseChoice, ResponseVideoResult, ResponseContentFilter, ResponseWebSearch
from src.tools.session import post_with_retry
from ..model_base import Usage, UsagePromptTokensDetails


def chat_completions(api_key: str, request_body: Request) -> Response:
    """https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8"""
    def _postprocess(json_raw: dict) -> Response:
        """获取到的json字串与官网定义不完全相同, 手动撞在各字段值"""
        return Response(
            id=json_raw["id"],
            request_id=json_raw["request_id"],
            created=json_raw["created"],
            model=json_raw["model"],
            choices=[
                ResponseChoice(
                    index=choice["index"],
                    message=choice["message"],
                    finish_reason=choice["finish_reason"],
                )
                for choice in json_raw["choices"]
            ],
            usage=Usage(
                prompt_tokens=json_raw["usage"].get("prompt_tokens"),
                completion_tokens=json_raw["usage"].get("completion_tokens"),
                prompt_tokens_details=UsagePromptTokensDetails(
                  cached_tokens=json_raw["usage"].get("prompt_tokens_details").get("cached_tokens"),
                ),
                total_tokens=json_raw["usage"].get("total_tokens"),
            ),
            video_result=[
                ResponseVideoResult(
                    url=result.get("url"),
                    cover_image_url=result.get("cover_image_url"),
                )
                for result in cast(list[dict], json_raw.get("video_result"))
            ] if json_raw.get("video_result") else None,
            web_search=[
                ResponseWebSearch(
                    icon=result.get("icon"),
                    content=result.get("content"),
                    link=result.get("link"),
                    media=result.get("media"),
                    refer=result.get("refer"),
                    title=result.get("title"),
                    publish_date=result.get("publish_date"),
                )
                for result in cast(list[dict], json_raw.get("web_search"))
            ] if json_raw.get("web_search") else None,
            content_filter=[
                ResponseContentFilter(
                    role=result.get("role"),
                    level=result.get("level"),
                )
                for result in cast(list[dict], json_raw.get("content_filter"))
            ] if json_raw.get("content_filter") else None,
        )

    request_url: str = rf"{config.base_url}/paas/v4/chat/completions"
    header: dict = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = post_with_retry(
        request_url,
        headers=header,
        json=asdict(request_body)
    )
    response.raise_for_status()
    logging.debug(response.json())

    return _postprocess(response.json())
