import argparse
import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path

import pytest

from app.adapters.llm.fake import FakeChatModel
from app.cli import download_models, ingest, model_smoke
from app.config import ModelBackend, ModelSettings, Settings
from app.domain.documents import IngestionResult, IngestionStatus


@pytest.mark.unit
def test_model_download_cli_uses_pinned_snapshots_inside_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str, Path]] = []

    def snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: Path,
        token: str | None,
    ) -> str:
        assert token is None
        calls.append((repo_id, revision, local_dir))
        return str(local_dir)

    monkeypatch.setattr(download_models, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download",
            "--generator-id",
            "org/generator",
            "--generator-revision",
            "abc123",
            "--embedding-id",
            "org/embeddings",
            "--embedding-revision",
            "def456",
            "--destination",
            str(tmp_path / "models"),
        ],
    )

    download_models.main()

    assert [call[:2] for call in calls] == [
        ("org/generator", "abc123"),
        ("org/embeddings", "def456"),
    ]
    assert "MODEL__PATH=" in capsys.readouterr().out
    with pytest.raises(ValueError, match="inside"):
        download_models.safe_destination(tmp_path / "models", "../outside")


class UploadService:
    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> IngestionResult:
        assert filename == "notes.md"
        assert content_type == "text/markdown"
        assert content == b"local notes"
        return result()


class UrlService:
    async def ingest(self, url: str) -> IngestionResult:
        assert url == "https://example.com/article"
        return result()


class Container:
    def __init__(self) -> None:
        self.ingest_upload = UploadService()
        self.ingest_url = UrlService()
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True


class Parser:
    def __init__(self, namespace: argparse.Namespace) -> None:
        self._namespace = namespace

    def parse_args(self) -> argparse.Namespace:
        return self._namespace


def result() -> IngestionResult:
    return IngestionResult(
        document_id="doc",
        document_version="v1",
        status=IngestionStatus.INDEXED,
        chunk_count=1,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_cli_reuses_file_and_url_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "notes.md"
    path.write_bytes(b"local notes")
    containers: list[Container] = []

    def build(_settings: Settings) -> Container:
        container = Container()
        containers.append(container)
        return container

    monkeypatch.setattr(ingest, "build_container", build)
    monkeypatch.setattr(ingest, "get_settings", Settings)
    monkeypatch.setattr(
        ingest,
        "build_parser",
        lambda: Parser(argparse.Namespace(command="file", path=path)),
    )
    await ingest.run()
    monkeypatch.setattr(
        ingest,
        "build_parser",
        lambda: Parser(argparse.Namespace(command="url", url="https://example.com/article")),
    )
    await ingest.run()

    assert all(container.started and container.closed for container in containers)
    assert capsys.readouterr().out.count('"document_id":"doc"') == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_smoke_requires_huggingface_and_runs_local_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(model_smoke, "get_settings", Settings)
    with pytest.raises(RuntimeError, match="MODEL__BACKEND"):
        await model_smoke.run()

    settings = Settings(
        model=ModelSettings(
            backend=ModelBackend.HUGGINGFACE,
            path=tmp_path,
            alias="smoke-model",
        )
    )
    monkeypatch.setattr(model_smoke, "get_settings", lambda: settings)
    monkeypatch.setattr(
        model_smoke,
        "build_chat_model",
        lambda _settings: FakeChatModel(model="smoke-model"),
    )
    await model_smoke.run()

    assert "Fake response: Reply with OK." in capsys.readouterr().out


@pytest.mark.unit
def test_cli_main_functions_delegate_to_asyncio(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def run(coroutine: Coroutine[object, object, object]) -> None:
        calls.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", run)
    ingest.main()
    model_smoke.main()
    assert len(calls) == 2
