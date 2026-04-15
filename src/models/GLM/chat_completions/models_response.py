from typing import Literal, Union, Optional, Any
from dataclasses import dataclass

from ..model_base import Usage


@dataclass
class ResponseChoiceMessageContentVisualization:
    type: Literal["text"]
    text: str


@dataclass
class ResponseChoiceRequestContentMessageAudio:
    id: str
    data: str
    expires_at: str


@dataclass
class ResponseChoiceRequestContentMessageToolCallFunction:
    name: str
    arguments: str


@dataclass(slots=True)
class ResponseChoiceRequestContentMessageToolCallMcpToolsInputSchema:
    type: Literal["object"]
    properties: Any
    required: list[str]
    additionalProperties: bool


@dataclass
class ResponseChoiceRequestContentMessageToolCallMcpTools:
    name: str
    description: str
    annotations: Any
    input_schema: ResponseChoiceRequestContentMessageToolCallMcpToolsInputSchema


@dataclass
class ResponseChoiceRequestContentMessageToolCallMcp:
    id: str
    type: Literal["mcp_list_tools", "mcp_call"]
    server_label: str
    error: str
    tools: list[ResponseChoiceRequestContentMessageToolCallMcpTools]
    arguments: str
    name: str
    output: object


@dataclass
class ResponseChoiceRequestContentMessageToolCall:
    function: ResponseChoiceRequestContentMessageToolCallFunction
    mcp: ResponseChoiceRequestContentMessageToolCallMcp
    id: str
    type: str


@dataclass
class ResponseChoiceMessage:
    role: str
    content: Union[str, ResponseChoiceMessageContentVisualization, None]
    reasoning_content: Optional[str] = None
    audio: Optional[ResponseChoiceRequestContentMessageAudio] = None
    tool_calls: Optional[list[ResponseChoiceRequestContentMessageToolCall]] = None


@dataclass
class ResponseChoice:
    index: int
    message: ResponseChoiceMessage
    finish_reason: str


@dataclass
class ResponseVideoResult:
    url: Optional[str]
    cover_image_url: Optional[str]


@dataclass
class ResponseWebSearch:
    icon: Optional[str]
    title: Optional[str]
    link: Optional[str]
    media: Optional[str]
    publish_date: Optional[str]
    content: Optional[str]
    refer: Optional[str]


@dataclass
class ResponseContentFilter:
    role: Optional[str]
    level: Optional[int]


@dataclass(slots=True)
class Response:
    id: str
    request_id: str
    created: int
    model: str
    choices: list[ResponseChoice]
    usage: Usage
    video_result: Optional[list[ResponseVideoResult]] = None
    web_search: Optional[list[ResponseWebSearch]] = None
    content_filter: Optional[list[ResponseContentFilter]] = None
