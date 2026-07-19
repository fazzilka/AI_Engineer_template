import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def safe_destination(base: Path, name: str) -> Path:
    resolved_base = base.resolve()
    destination = (resolved_base / name).resolve()
    if not destination.is_relative_to(resolved_base):
        msg = "Model destination must stay inside the configured models directory"
        raise ValueError(msg)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download pinned local model snapshots")
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--generator-revision", required=True)
    parser.add_argument("--embedding-id", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--destination", type=Path, default=Path("models"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = args.destination.resolve()
    base.mkdir(parents=True, exist_ok=True)
    generator = safe_destination(base, "generator")
    embeddings = safe_destination(base, "embeddings")
    token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=args.generator_id,
        revision=args.generator_revision,
        local_dir=generator,
        token=token,
    )
    snapshot_download(
        repo_id=args.embedding_id,
        revision=args.embedding_revision,
        local_dir=embeddings,
        token=token,
    )
    print("Model snapshots downloaded. Configure:")
    print("MODEL__BACKEND=huggingface")
    print("MODEL__SOURCE=filesystem")
    print(f"MODEL__PATH={generator}")
    print("EMBEDDINGS__BACKEND=huggingface")
    print("EMBEDDINGS__SOURCE=filesystem")
    print(f"EMBEDDINGS__PATH={embeddings}")
    print("MODEL__LOCAL_FILES_ONLY=true")
    print("EMBEDDINGS__LOCAL_FILES_ONLY=true")
