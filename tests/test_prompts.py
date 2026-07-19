from app.prompts import load_system_prompt


def test_system_prompt_is_packaged() -> None:
    prompt = load_system_prompt()

    assert "precise and helpful" in prompt
    assert "Never expose secrets" in prompt
