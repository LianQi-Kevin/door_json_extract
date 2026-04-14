from dataclasses import dataclass, field
from typing import Literal, Union, Optional

TaskModel = Literal[
    "glm-5.1", "glm-5-turbo", "glm-5", "glm-4.7", "glm-4.7-flash", "glm-4.7-flashx", "glm-4.6",
    "glm-4.5-air", "glm-4.5-airx", "glm-4.5-flash", "glm-4-flash-250414", "glm-4-flashx-250414"
]
VisualizationModel = Literal[
    "glm-5v-turbo", "glm-4.6v", "autoglm-phone", "glm-4.6v-flash", "glm-4.6v-flashx",
    "glm-4v-flash", "glm-4.1v-thinking-flashx", "glm-4.1v-thinking-flash"
]
AudioModel = Literal["glm-4-voice"]
RolePlayModel = Literal["charglm-4", "emohaa"]


@dataclass(slots=True)
class RolePlayMeta:
    user_info: str
    bot_info: str
    bot_name: str
    user_name: str


@dataclass
class RequestMessageToolCallFunction:
    name: str
    arguments: str


@dataclass
class RequestMessageToolCall:
    id: str
    type: Literal["function", "web_search", "retrieval"]
    function: Optional[RequestMessageToolCallFunction] = None
    tool_call_id: Optional[str] = None


@dataclass
class RequestMessage:
    role: Literal["user", "system", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[list[RequestMessageToolCall]] = None


@dataclass(slots=True)
class Request:
    model: Union[TaskModel, VisualizationModel, AudioModel, RolePlayModel] = "glm-5.1"
    message: list[RequestMessage] = field(default_factory=list)
    meta: Optional[RolePlayMeta] = None
    stream: Optional[bool] = False
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.6
    max_tokens: Optional[int] = 1024
    stop: Optional[list[str]] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
