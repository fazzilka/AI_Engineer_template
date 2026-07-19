import argparse
import asyncio
import mimetypes
from pathlib import Path

import anyio
import orjson

from app.bootstrap.container import build_container
from app.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a local file or safe URL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    file_parser = subparsers.add_parser("file")
    file_parser.add_argument("path", type=Path)
    url_parser = subparsers.add_parser("url")
    url_parser.add_argument("url")
    return parser


async def run() -> None:
    args = build_parser().parse_args()
    container = build_container(get_settings())
    await container.start()
    try:
        if args.command == "file":
            path: Path = args.path
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            result = await container.ingest_upload.ingest(
                filename=path.name,
                content_type=content_type,
                content=await anyio.to_thread.run_sync(path.read_bytes),
            )
        else:
            result = await container.ingest_url.ingest(args.url)
        print(
            orjson.dumps(
                {
                    "document_id": result.document_id,
                    "document_version": result.document_version,
                    "status": result.status.value,
                    "chunk_count": result.chunk_count,
                }
            ).decode()
        )
    finally:
        await container.aclose()


def main() -> None:
    asyncio.run(run())
