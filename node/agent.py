#!/usr/bin/env python3
"""PhysMeta 最小化推理节点（Windows · 单节点 · 只做推理 · 整链路）

复用老方案的核心机制（主动出站 + 长轮询领单 + 下载校验 + 回传校验 + 失败回退），
推理单元换成 bubble-ocr-app 的完整链路：YOLO 检测 + RapidOCR 识值 + 格式规范化 + VLM 兜底 + 仲裁。

依赖：
  - Python 3.10+ 标准库（urllib / json / hashlib / zipfile）
  - bubble-ocr-app 的 src/bubble_ocr 包（YOLO=ultralytics, OCR=rapidocr_onnxruntime, VLM=transformers CUDA）

节点只主动出站，客户防火墙/NAT 无需开放入站端口。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

AGENT_VERSION = "minimal-1.0.0"
JOB_ID_PATTERN = re.compile(r"cad-\d{8}-\d{6}-[A-Za-z0-9]{6}\Z")


class AgentError(RuntimeError):
    pass


class AgentAuthenticationError(AgentError):
    pass


# ---------- 配置加载 ----------

def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"配置文件无效：{exc}") from exc
    if not isinstance(config, dict):
        raise AgentError("配置文件不是 JSON 对象")

    server = str(config.get("server") or "https://app.physmeta.cn").rstrip("/")
    parsed = urlparse(server)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise AgentError("远程推理服务必须使用 HTTPS")

    token = str(config.get("token") or os.getenv("PHYS_META_INFERENCE_TOKEN") or "")
    if len(token) < 32:
        raise AgentError("远程推理令牌缺失或过短（至少 32 字符）")

    node_id = str(config.get("node_id") or "windows-inference-01")
    root = Path(str(config.get("root") or r"C:\PhysMetaInference")).resolve()
    # bubble-ocr-app 的 src 目录（含 bubble_ocr 包）
    bubble_src = Path(str(config.get("bubble_src") or root / "bubble-ocr-app" / "src")).resolve()
    if not (bubble_src / "bubble_ocr" / "pipeline.py").is_file():
        raise AgentError(f"bubble-ocr 包不完整，缺少 pipeline.py：{bubble_src}")

    jobs_root = Path(str(config.get("jobs_root") or root / "inference-jobs")).resolve()

    return {
        **config,
        "server": server,
        "token": token,
        "node_id": node_id,
        "root": root,
        "bubble_src": bubble_src,
        "jobs_root": jobs_root,
    }


# ---------- HTTP 工具 ----------

def request_json(
    config: dict[str, Any],
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    run_token: str = "",
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config['token']}",
        "X-Inference-Node-ID": config["node_id"],
        "User-Agent": f"PhysMeta-Inference-Node/{AGENT_VERSION}",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    if run_token:
        headers["X-Inference-Run-Token"] = run_token
    request = urllib.request.Request(
        f"{config['server']}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
            return decoded if isinstance(decoded, dict) else {}
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_payload = {}
        message = str(error_payload.get("error") or f"HTTP {exc.code}")
        if exc.code in {401, 403}:
            raise AgentAuthenticationError(message) from exc
        raise AgentError(message) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AgentError(f"远程服务暂时不可用：{exc}") from exc


# ---------- 文件工具 ----------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_job_id(job_id: Any) -> str:
    value = str(job_id or "")
    if not JOB_ID_PATTERN.fullmatch(value):
        raise AgentError("服务端返回了不符合合同的任务 ID")
    return value


# ---------- 下载 / 回传 ----------

def download_input(config: dict[str, Any], job: dict[str, Any], run_token: str, target: Path) -> None:
    headers = {
        "Authorization": f"Bearer {config['token']}",
        "X-Inference-Node-ID": config["node_id"],
        "X-Inference-Run-Token": run_token,
        "User-Agent": f"PhysMeta-Inference-Node/{AGENT_VERSION}",
    }
    request = urllib.request.Request(
        f"{config['server']}/api/inference/jobs/{quote(str(job['id']))}/input",
        headers=headers,
        method="GET",
    )
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise AgentAuthenticationError("下载任务输入时身份失效") from exc
        raise AgentError(f"下载任务输入失败：HTTP {exc.code}") from exc

    if partial.stat().st_size != int(job["input_bytes"]):
        partial.unlink(missing_ok=True)
        raise AgentError("任务输入大小校验失败")
    if sha256_file(partial) != str(job["input_sha256"]):
        partial.unlink(missing_ok=True)
        raise AgentError("任务输入 SHA-256 校验失败")
    partial.replace(target)


def build_result_archive(job_root: Path, result_path: Path) -> Path:
    archive_path = job_root / "result.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_path, "result.json")
        for directory_name in ("output", "preview"):
            directory = job_root / directory_name
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, f"{directory_name}/{path.relative_to(directory).as_posix()}")
    return archive_path


def upload_result(config: dict[str, Any], job_id: str, run_token: str, archive_path: Path) -> dict[str, Any]:
    boundary = f"physmeta-{uuid.uuid4().hex}"
    digest = sha256_file(archive_path)
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="sha256"\r\n\r\n'
        f"{digest}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="artifact"; filename="result.zip"\r\n'
        "Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    body = prefix + archive_path.read_bytes() + suffix
    request = urllib.request.Request(
        f"{config['server']}/api/inference/jobs/{quote(job_id)}/result",
        data=body,
        headers={
            "Authorization": f"Bearer {config['token']}",
            "X-Inference-Node-ID": config["node_id"],
            "X-Inference-Run-Token": run_token,
            "X-Artifact-SHA256": digest,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": f"PhysMeta-Inference-Node/{AGENT_VERSION}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        raise AgentError(str(payload.get("error") or f"结果上传失败：HTTP {exc.code}")) from exc


# ---------- 推理：调 bubble-ocr 完整链路 ----------

def run_inference(config: dict[str, Any], source: Path, options: dict[str, Any]) -> dict[str, Any]:
    """调 bubble-ocr 的 pipeline 跑推理。

    支持三种任务类型（options.task_type）：
      - full          整图识别 pipeline.run（YOLO 检测全图 → 逐框读值）
      - region_single 单标注框选 pipeline.run_region（读框选区域，1 条）
      - region_detect 区域批量识别 pipeline.run_region_detect（区域内 YOLO 检测 → 逐框读值）
    region 模式额外读 options.region = [x, y, width, height]。
    """
    sys.path.insert(0, str(config["bubble_src"]))

    from bubble_ocr import config as bubble_config  # noqa
    from bubble_ocr.pipeline import run as pipeline_run  # noqa
    from bubble_ocr.pipeline import run_region, run_region_detect, _stats  # noqa

    # 权重 / 模型路径：options(任务级) 优先，其次节点配置(inference-node.json)，再环境变量，最后 bubble-ocr 默认
    weights = Path(str(options.get("weights") or config.get("weights") or os.getenv("BUBBLE_YOLO_WEIGHTS") or bubble_config.YOLO_WEIGHTS))
    model_dir = options.get("vlm_model") or config.get("vlm_model") or os.getenv("BUBBLE_VLM_MODEL") or bubble_config.VLM_MODEL_DIR
    adapter_dir = options.get("vlm_adapter") or config.get("vlm_adapter") or os.getenv("BUBBLE_VLM_ADAPTER") or bubble_config.VLM_ADAPTER_DIR
    conf = float(options.get("conf") or bubble_config.YOLO_CONF)
    low_conf = float(options.get("low_conf_threshold") or bubble_config.LOW_CONF_THRESHOLD)
    enable_vlm = bool(options.get("enable_vlm", bubble_config.ENABLE_VLM))
    use_ocr = bool(options.get("use_ocr", config.get("use_ocr", getattr(bubble_config, "USE_OCR", True))))

    model_path = Path(str(model_dir)) if model_dir else None
    adapter_path = Path(str(adapter_dir)) if adapter_dir else None
    if model_path is not None and not model_path.exists():
        model_path = None  # VLM 模型缺失则自动降级为纯 OCR 主线

    enable = enable_vlm and model_path is not None

    def _box_to_dict(r) -> dict[str, Any]:
        return {
            "box": r.box,
            "ocr_text": r.ocr_text,
            "ocr_conf": r.ocr_conf,
            "vlm_candidate": r.vlm_candidate,
            "final_text": r.final_text,
            "normalized": r.normalized,
            "source": r.source,
            "needs_review": r.needs_review,
            "matched": r.matched,
            "gold_text": r.gold_text,
            "symbols": r.symbols,
            "tools": r.tools,
            "index": r.index,
        }

    task_type = str(options.get("task_type") or "full")

    if task_type in ("region_single", "region_detect"):
        region = options.get("region")
        if not region or len(region) != 4:
            raise AgentError(f"{task_type} 任务缺少 region=[x,y,width,height]")
        region = tuple(float(v) for v in region)
        if task_type == "region_single":
            results = run_region(
                source, region,
                model_path=model_path, adapter_path=adapter_path,
                low_conf_threshold=low_conf, enable_vlm=enable, use_ocr=use_ocr,
            )
        else:
            results = run_region_detect(
                source, weights, region,
                model_path=model_path, adapter_path=adapter_path,
                conf=conf, low_conf_threshold=low_conf, enable_vlm=enable, use_ocr=use_ocr,
            )
        return {
            "image": str(source),
            "stats": _stats(results, False),
            "boxes": [_box_to_dict(r) for r in results],
        }

    result = pipeline_run(
        source,
        weights,
        model_path=model_path,
        adapter_path=adapter_path,
        conf=conf,
        low_conf_threshold=low_conf,
        enable_vlm=enable,
        use_ocr=use_ocr,
    )

    # 转成与 server.py / cli.py 一致的 payload 结构
    return {
        "image": result.image_path,
        "stats": result.stats,
        "boxes": [_box_to_dict(r) for r in result.boxes],
    }


# ---------- 单个任务执行 ----------

def run_job(config: dict[str, Any], job: dict[str, Any], run_token: str) -> None:
    job_id = validate_job_id(job.get("id"))
    job_root = config["jobs_root"] / job_id
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    source = input_dir / Path(str(job["filename"])).name
    manifest_path = job_root / "job.json"
    result_path = job_root / "result.json"

    atomic_write_json(manifest_path, job)

    # 1. 下载图纸
    if not source.is_file() or sha256_file(source) != str(job["input_sha256"]):
        download_input(config, job, run_token, source)

    # 2. 本地推理（bubble-ocr 全链路）
    options = job.get("options") or {}
    payload = run_inference(config, source, options)
    atomic_write_json(result_path, payload)

    # 3. 打包 + 回传
    archive_path = build_result_archive(job_root, result_path)
    upload_result(config, job_id, run_token, archive_path)

    # 4. 清理本地任务目录
    shutil.rmtree(job_root, ignore_errors=True)


# ---------- 主循环 ----------

def run_loop(config: dict[str, Any], once: bool) -> int:
    config["jobs_root"].mkdir(parents=True, exist_ok=True)
    connection_failures = 0

    while True:
        try:
            long_poll = 5.0 if not once else 0.0
            claim = request_json(
                config,
                "POST",
                "/api/inference/claim",
                {"wait_seconds": long_poll},
                timeout=max(10.0, long_poll + 5.0),
            )
            connection_failures = 0
        except AgentAuthenticationError:
            raise
        except AgentError as exc:
            connection_failures += 1
            delay = min(15.0, float(2 ** min(3, max(1, connection_failures))))
            print(f"控制链路中断，{delay:.0f} 秒后重试：{exc}", file=sys.stderr)
            if once:
                raise
            time.sleep(delay)
            continue

        job = claim.get("job")
        if not isinstance(job, dict):
            if once:
                return 0
            try:
                retry_delay = float(claim.get("retry_after_seconds") or 5)
            except (TypeError, ValueError):
                retry_delay = 5.0
            time.sleep(max(0.1, min(15.0, retry_delay)))
            continue

        job_id = validate_job_id(job.get("id"))
        run_token = str(claim.get("run_token") or "")
        try:
            run_job(config, job, run_token)
        except AgentAuthenticationError:
            raise
        except AgentError as exc:
            error = str(exc)[-1000:]
            print(f"任务 {job_id} 失败：{error}", file=sys.stderr)
            try:
                request_json(
                    config,
                    "POST",
                    f"/api/inference/jobs/{quote(job_id)}/fail",
                    {"reason": "remote_job_failed", "error": error},
                    run_token=run_token,
                    timeout=10.0,
                )
            except AgentError:
                pass
            if once:
                raise
            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysMeta minimal inference node (bubble-ocr)")
    parser.add_argument("--config", type=Path, default=Path(r"C:\PhysMetaInference\inference-node.json"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        return run_loop(load_config(args.config.resolve()), args.once)
    except KeyboardInterrupt:
        return 0
    except AgentError as exc:
        print(f"PhysMeta 推理节点错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
