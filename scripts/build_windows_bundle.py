#!/usr/bin/env python3
"""Build a verifiable Windows node update bundle from fixed Git commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path


NODE_FILES = {
    "node/agent.py": "agent.py",
    "node/launch_node.py": "launch_node.py",
    "node/start-node.bat": "start-node.bat",
    "node/vlm_fallback_windows.py": "vlm_fallback_windows.py",
    "node/install_update.py": "install_update.py",
    "node/update-node.bat": "update-node.bat",
    "node/requirements-windows.txt": "requirements-windows.txt",
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", f"{ref}^{{commit}}").decode().strip()


def read_git_file(repo: Path, commit: str, name: str) -> bytes:
    return git(repo, "show", f"{commit}:{name}")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_zip(source: Path, destination: Path) -> None:
    root = source.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(root) / path.relative_to(source)
            info = zipfile.ZipInfo(str(relative).replace("\\", "/"), (2026, 9, 28, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build(
    node_repo: Path,
    node_ref: str,
    ocr_repo: Path,
    ocr_ref: str,
    version: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    node_commit = resolve_commit(node_repo, node_ref)
    ocr_commit = resolve_commit(ocr_repo, ocr_ref)
    release_name = f"PhysMeta-Windows-{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{release_name}.zip"

    with tempfile.TemporaryDirectory(prefix="physmeta-windows-bundle-") as temporary:
        bundle = Path(temporary) / release_name
        bundle.mkdir()

        payloads: dict[str, bytes] = {}
        for source, destination in NODE_FILES.items():
            payloads[destination] = read_git_file(node_repo, node_commit, source)

        expected_version = f'AGENT_VERSION = "minimal-{version}"'.encode()
        if expected_version not in payloads["agent.py"]:
            raise ValueError(f"node ref does not identify minimal-{version}")

        ocr_names = git(
            ocr_repo,
            "ls-tree",
            "-r",
            "--name-only",
            ocr_commit,
            "--",
            "src/bubble_ocr",
        ).decode().splitlines()
        for source in ocr_names:
            if source.endswith(".py"):
                destination = "bubble-src/" + source.removeprefix("src/")
                payloads[destination] = read_git_file(ocr_repo, ocr_commit, source)

        required = "bubble-src/bubble_ocr/pipeline.py"
        if required not in payloads:
            raise ValueError(f"OCR ref is missing {required}")

        readme = (
            f"PhysMeta Windows inference node {version}\r\n"
            f"Node source: {node_commit}\r\n"
            f"OCR source:  {ocr_commit}\r\n\r\n"
            "This package updates code only. It does not contain or change models, tokens, "
            "configuration, Python environments, task data, or scheduled tasks.\r\n\r\n"
            "1. Use an Administrator session on the actual windows-5060ti-01 host.\r\n"
            "2. Stop the old node console and pause its scheduled task.\r\n"
            "3. Extract this ZIP into a separate temporary directory.\r\n"
            "4. Run update-node.bat and point it at the existing node directory if needed.\r\n"
            "5. In the existing node directory run start-node.bat --check.\r\n"
            "6. Start exactly one agent only after the check succeeds.\r\n"
        ).encode("utf-8")
        payloads["README.txt"] = readme

        manifest = {
            "release": release_name,
            "agent_version": f"minimal-{version}",
            "source_commits": {
                "physmeta-VLM-inference-link": node_commit,
                "physmeta-ocr-mvp": ocr_commit,
            },
            "files": {name: sha256(payload) for name, payload in sorted(payloads.items())},
        }

        for name, payload in payloads.items():
            destination = bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (bundle / "bundle-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        write_zip(bundle, zip_path)

    digest_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    digest_path.write_text(f"{sha256(zip_path.read_bytes())}  {zip_path.name}\n", encoding="ascii")
    return zip_path, digest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--node-ref", required=True)
    parser.add_argument("--ocr-repo", type=Path, required=True)
    parser.add_argument("--ocr-ref", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    zip_path, digest_path = build(
        args.node_repo.resolve(),
        args.node_ref,
        args.ocr_repo.resolve(),
        args.ocr_ref,
        args.version,
        args.output_dir.resolve(),
    )
    print(zip_path)
    print(digest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
