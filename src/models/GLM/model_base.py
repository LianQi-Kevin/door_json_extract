from dataclasses import dataclass
from typing import Optional


@dataclass
class UsagePromptTokensDetails:
    cached_tokens: int


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prompt_tokens_details: Optional[UsagePromptTokensDetails] = None
    total_tokens: Optional[int] = None
