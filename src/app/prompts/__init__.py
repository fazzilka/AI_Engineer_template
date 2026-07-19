from importlib.resources import files


def load_system_prompt() -> str:
    prompt = files(__package__).joinpath("system.md").read_text(encoding="utf-8").strip()
    if not prompt:
        msg = "System prompt must not be empty"
        raise RuntimeError(msg)
    return prompt
