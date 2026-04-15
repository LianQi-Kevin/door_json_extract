from dataclasses import dataclass, field
from typing import Literal, Union, Optional, Any, TypeAlias

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


@dataclass
class RequestMessageBase:
    role: Literal["system", "user"]
    content: str


@dataclass
class RequestMessagAssistant(RequestMessageBase):
    role: Literal["assistant"]
    content: Optional[str]
    tool_calls: Optional[list[RequestMessageToolCall]] = None


@dataclass
class RequestMessagTool(RequestMessageBase):
    role: Literal["tool"]
    tool_call_id: Optional[str]


@dataclass
class Thinking:
    type: Literal["enabled", "disabled"] = "enabled"
    clear_thinking: bool = True


@dataclass
class ToolFunctionCallFunction:
    name: str
    description: str
    parameters: Any     # JSON Schema


@dataclass
class ToolFunctionCall:
    type: Literal["function"]
    function: ToolFunctionCallFunction


@dataclass
class ToolRetrieval:
    # todo: 暂时用不到, 未完成定义, unfinished
    pass


@dataclass
class ToolWebSearc:
    # todo: 暂时用不到, 未完成定义, unfinished
    pass


@dataclass
class ToolMCP:
    # todo: 暂时用不到, 未完成定义, unfinished
    pass


@dataclass
class ResponseFormat:
    type: Literal["text", "json_object"] = "text"


@dataclass(slots=True)
class TaskRequest:
    model: TaskModel
    messages: list[Union[RequestMessageBase, RequestMessagAssistant, RequestMessagTool]]
    stream: bool = False
    thinking: Thinking = field(default_factory=Thinking)
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    tool_stream: bool = False
    tool: Optional[list[Union[ToolFunctionCall, ToolRetrieval, ToolWebSearc, ToolMCP]]] = None
    tool_choice: Optional[Literal["auto"]] = None
    stop: Optional[list[str]] = None
    response_format: Optional[ResponseFormat] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None


Request: TypeAlias = TaskRequest
