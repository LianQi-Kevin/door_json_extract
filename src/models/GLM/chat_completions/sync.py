from .models_request import Request
from .models_response import Response


def chat_completions(api_key: str, request_body: Request) -> Response:
    """https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8"""
