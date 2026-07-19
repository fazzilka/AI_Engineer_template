import asyncio

from app.adapters.llm import build_chat_model
from app.config import ModelBackend, get_settings
from app.domain.chat import ChatMessage, MessageRole


async def run() -> None:
    settings = get_settings().model
    if settings.backend is not ModelBackend.HUGGINGFACE:
        msg = "Set MODEL__BACKEND=huggingface and configure a pre-downloaded local model"
        raise RuntimeError(msg)
    model = build_chat_model(settings)
    try:
        await model.load()
        result = await model.generate(
            messages=(ChatMessage(role=MessageRole.USER, content="Reply with OK."),),
            system_prompt="Answer briefly.",
        )
        if not result.content.strip():
            msg = "The local model returned an empty response"
            raise RuntimeError(msg)
        print(result.content)
    finally:
        await model.aclose()


def main() -> None:
    asyncio.run(run())
