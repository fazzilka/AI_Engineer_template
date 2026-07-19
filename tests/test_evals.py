from pathlib import Path

import pytest
from evals.run import EvalCase, evaluate, load_cases, score

from app.adapters.llm.fake import FakeLLMClient


def test_load_cases_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset is empty"):
        load_cases(dataset)


def test_score_reports_missing_and_forbidden_terms() -> None:
    case = EvalCase(
        id="contract",
        messages=[{"role": "user", "content": "hello"}],
        must_contain=["required"],
        must_not_contain=["secret"],
    )

    result = score(case, "This output contains a secret")

    assert result.passed is False
    assert result.missing_terms == ["required"]
    assert result.forbidden_terms == ["secret"]


@pytest.mark.asyncio
async def test_evaluate_runs_cases_with_provider_neutral_client(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"id":"echo","messages":[{"role":"user","content":"hello"}],'
        '"must_contain":["hello"],"must_not_contain":[]}\n',
        encoding="utf-8",
    )

    results = await evaluate(path=dataset, client=FakeLLMClient())

    assert len(results) == 1
    assert results[0].passed is True
