from models.GLM.chat_completions.models_request import Thinking
from src.models.GLM.chat_completions import sync_chat_completions, ChatCompletionsRequest
from src.models.GLM.chat_completions.models_request import RequestMessageBase

if __name__ == '__main__':
    request_body = ChatCompletionsRequest(
        model="glm-5.1",
        messages=[
            RequestMessageBase(role="system", content="你是一个有用的AI助手。"),
            RequestMessageBase(role="user", content="请简要介绍一下人工智能的发展历程。")
        ],
        thinking=Thinking("disabled")

    )
    response = sync_chat_completions(
        "YOUR-API-KEY",
        request_body,
    )
    print(response)
