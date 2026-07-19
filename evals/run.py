import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.llm import build_llm_client
from app.application.chat import ChatService
from app.config import get_settings
from app.domain.chat import ChatMessage
from app.ports.llm import LLMClient
from app.prompts import load_system_prompt

DEFAULT_DATASET = Path(__file__).with_name("cases.jsonl")


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    messages: list[ChatMessage] = Field(min_length=1)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    passed: bool
    missing_terms: list[str]
    forbidden_terms: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        msg = f"Evaluation dataset is empty: {path}"
        raise ValueError(msg)
    return cases


def score(case: EvalCase, output: str) -> EvalResult:
    normalized_output = output.casefold()
    missing = [term for term in case.must_contain if term.casefold() not in normalized_output]
    forbidden = [term for term in case.must_not_contain if term.casefold() in normalized_output]
    return EvalResult(
        case_id=case.id,
        passed=not missing and not forbidden,
        missing_terms=missing,
        forbidden_terms=forbidden,
    )


async def evaluate(*, path: Path, client: LLMClient) -> list[EvalResult]:
    service = ChatService(llm=client, system_prompt=load_system_prompt())
    results: list[EvalResult] = []
    for case in load_cases(path):
        response = await service.reply(case.messages)
        results.append(score(case, response.content))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lexical AI regression evaluations")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--min-pass-rate", type=float, default=0.66)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if not 0 <= args.min_pass_rate <= 1:
        msg = "--min-pass-rate must be between 0 and 1"
        raise ValueError(msg)

    settings = get_settings()
    client = build_llm_client(settings.llm)
    try:
        results = await evaluate(path=args.dataset, client=client)
    finally:
        await client.aclose()

    for result in results:
        print(result.model_dump_json())

    passed = sum(result.passed for result in results)
    pass_rate = passed / len(results)
    print(json.dumps({"passed": passed, "total": len(results), "pass_rate": pass_rate}))
    return 0 if pass_rate >= args.min_pass_rate else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
