from importlib.resources import files


def _load(name: str) -> str:
    prompt = files(__package__).joinpath(name).read_text(encoding="utf-8").strip()
    if not prompt:
        msg = f"Packaged prompt {name!r} must not be empty"
        raise RuntimeError(msg)
    return prompt


def load_chat_system_prompt() -> str:
    return _load("chat_system.md")


def load_rag_system_prompt() -> str:
    return _load("rag_system.md")


load_system_prompt = load_chat_system_prompt
