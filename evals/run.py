import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.llm.fake import FakeChatModel
from app.application.chat import ChatService
from app.bootstrap.container import ApplicationContainer, build_container
from app.config import AppSettings, QdrantMode, QdrantSettings, Settings, WebFetchSettings
from app.domain.chat import ChatMessage
from app.ports.llm import ChatModel
from app.prompts import load_chat_system_prompt

CASES_DIRECTORY = Path(__file__).with_name("cases")
DEFAULT_DATASET = CASES_DIRECTORY / "chat.jsonl"


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


class FixtureDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    content_type: str
    content: str


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fixture_documents: list[FixtureDocument]
    query: str
    expected_document_ids: list[str]
    expected_source_types: list[str]
    k: int = 5


class RagEvalCase(RetrievalEvalCase):
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expect_no_answer: bool = False


def load_cases(path: Path) -> list[EvalCase]:
    cases = _load_jsonl(path, EvalCase)
    if not cases:
        msg = f"Evaluation dataset is empty: {path}"
        raise ValueError(msg)
    return cases


def _load_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


async def evaluate(*, path: Path, client: ChatModel) -> list[EvalResult]:
    service = ChatService(model=client, system_prompt=load_chat_system_prompt())
    results: list[EvalResult] = []
    for case in load_cases(path):
        response = await service.reply(case.messages)
        results.append(score(case, response.content))
    return results


def _eval_settings() -> Settings:
    return Settings(
        app=AppSettings(environment="test", allow_fake_backends=True),
        qdrant=QdrantSettings(mode=QdrantMode.MEMORY),
        web=WebFetchSettings(enabled=False),
    )


async def _ingest_fixtures(
    fixtures: list[FixtureDocument],
) -> tuple[ApplicationContainer, dict[str, str]]:
    container = build_container(_eval_settings())
    await container.start()
    identities: dict[str, str] = {}
    for fixture in fixtures:
        result = await container.ingest_upload.ingest(
            filename=fixture.filename,
            content_type=fixture.content_type,
            content=fixture.content.encode(),
        )
        identities[fixture.id] = result.document_id
    return container, identities


async def evaluate_retrieval(path: Path) -> dict[str, float]:
    cases = _load_jsonl(path, RetrievalEvalCase)
    hits = 0
    reciprocal_rank = 0.0
    recalled = 0
    expected_total = 0
    for case in cases:
        container, identities = await _ingest_fixtures(case.fixture_documents)
        try:
            results = await container.retrieve.search(query=case.query, top_k=case.k)
            expected = {identities[value] for value in case.expected_document_ids}
            actual = [item.chunk.document_id for item in results]
            actual_types = {item.chunk.source_type.value for item in results}
            relevant = expected.intersection(actual)
            hits += int(bool(relevant) and set(case.expected_source_types) <= actual_types)
            recalled += len(relevant)
            expected_total += len(expected)
            rank = next(
                (index for index, value in enumerate(actual, start=1) if value in expected),
                None,
            )
            reciprocal_rank += 0.0 if rank is None else 1 / rank
        finally:
            await container.aclose()
    total = len(cases)
    return {
        "hit_rate_at_k": hits / total,
        "mrr": reciprocal_rank / total,
        "recall_at_k": recalled / expected_total,
    }


async def evaluate_rag(path: Path) -> list[EvalResult]:
    cases = _load_jsonl(path, RagEvalCase)
    results: list[EvalResult] = []
    for case in cases:
        container, identities = await _ingest_fixtures(case.fixture_documents)
        try:
            response = await container.rag.answer(query=case.query, top_k=case.k)
            normalized = response.generation.content.casefold()
            missing = [term for term in case.must_contain if term.casefold() not in normalized]
            forbidden = [term for term in case.must_not_contain if term.casefold() in normalized]
            expected_citations = {identities[value] for value in case.expected_citations}
            actual_citations = {source.document_id for source in response.sources}
            citations_match = expected_citations <= actual_citations
            no_answer_matches = not case.expect_no_answer or not response.sources
            results.append(
                EvalResult(
                    case_id=case.id,
                    passed=(
                        not missing and not forbidden and citations_match and no_answer_matches
                    ),
                    missing_terms=missing,
                    forbidden_terms=forbidden,
                )
            )
        finally:
            await container.aclose()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic local AI evaluations")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if not 0 <= args.min_pass_rate <= 1:
        msg = "--min-pass-rate must be between 0 and 1"
        raise ValueError(msg)
    client = FakeChatModel()
    chat_path = args.dataset or CASES_DIRECTORY / "chat.jsonl"
    chat_results = await evaluate(path=chat_path, client=client)
    retrieval_metrics = await evaluate_retrieval(CASES_DIRECTORY / "retrieval.jsonl")
    rag_results = await evaluate_rag(CASES_DIRECTORY / "rag.jsonl")
    security_results = await evaluate_rag(CASES_DIRECTORY / "security.jsonl")
    await client.aclose()
    results = [*chat_results, *rag_results, *security_results]
    for result in results:
        print(result.model_dump_json())
    passed = sum(result.passed for result in results)
    pass_rate = passed / len(results)
    summary = {
        "passed": passed,
        "total": len(results),
        "pass_rate": pass_rate,
        **retrieval_metrics,
    }
    print(json.dumps(summary, sort_keys=True))
    retrieval_passed = retrieval_metrics["hit_rate_at_k"] >= 1.0 and retrieval_metrics["mrr"] >= 1.0
    return 0 if pass_rate >= args.min_pass_rate and retrieval_passed else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
