#!/usr/bin/env python3
"""PhysMeta 最小化推理节点（Windows · 单节点 · 只做推理 · 整链路）

复用老方案的核心机制（主动出站 + 长轮询领单 + 下载校验 + 回传校验 + 失败回退），
推理单元换成 bubble-ocr-app 的完整链路：YOLO 检测 + RapidOCR 识值 + 格式规范化 + VLM 兜底 + 仲裁。

依赖：
  - Python 3.10+ 标准库（urllib / json / hashlib / zipfile）
  - bubble-ocr-app 的 src/bubble_ocr 包（YOLO=ultralytics, OCR=rapidocr_onnxruntime, VLM=transformers CUDA）

节点只主动出站，客户防火墙/NAT 无需开放入站端口。
内置本地监控面板（仅绑定 127.0.0.1，不对外暴露）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

AGENT_VERSION = "minimal-1.0.0"
JOB_ID_PATTERN = re.compile(r"cad-\d{8}-\d{6}-[A-Za-z0-9]{6}\Z")


class AgentError(RuntimeError):
    pass


class AgentAuthenticationError(AgentError):
    pass


# ---------- 本地监控面板（只绑定 127.0.0.1，不对外暴露） ----------

MONITOR_PORT = 8901
STATUS_PATH = Path(__file__).resolve().parent / "status.json"

_STATUS_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {
    "node_id": "",
    "version": AGENT_VERSION,
    "pid": os.getpid(),
    "server": "",
    "started_at": time.time(),
    "state": "starting",
    "link": {"status": "starting", "failures": 0, "last_error": ""},
    "tasks": {"claimed": 0, "succeeded": 0, "failed": 0},
    "current_job": None,
    "recent": [],
    "last_updated": time.time(),
}
_LOG_TAIL: deque[str] = deque(maxlen=200)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    with _STATUS_LOCK:
        _LOG_TAIL.append(line)


def _snapshot() -> dict[str, Any]:
    with _STATUS_LOCK:
        snapshot = json.loads(json.dumps(_STATUS))
        snapshot["logs"] = list(_LOG_TAIL)
    return snapshot


def _flush_status() -> None:
    try:
        atomic_write_json(STATUS_PATH, _snapshot())
    except OSError:
        pass


def _set_state(**kwargs: Any) -> None:
    with _STATUS_LOCK:
        _STATUS.update(kwargs)
        _STATUS["last_updated"] = time.time()
    _flush_status()


def _bump(key: str) -> None:
    with _STATUS_LOCK:
        _STATUS["tasks"][key] = _STATUS["tasks"].get(key, 0) + 1
        _STATUS["last_updated"] = time.time()
    _flush_status()


def _push_recent(entry: dict[str, Any]) -> None:
    with _STATUS_LOCK:
        _STATUS["recent"].insert(0, entry)
        del _STATUS["recent"][20:]
        _STATUS["last_updated"] = time.time()
    _flush_status()


def _nvidia_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError as exc:
        return {"error": f"nvidia-smi 调用失败：{exc}"}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"error": "nvidia-smi 不可用"}
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    keys = ["name", "mem_used", "mem_total", "util", "temp", "power", "power_limit"]
    if len(parts) < len(keys):
        return {"error": "nvidia-smi 输出格式异常"}
    return dict(zip(keys, parts))


MONITOR_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhysMeta 节点监控</title>
<style>
:root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --err:#f85149; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; padding:20px; }
h1 { font-size:18px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:12px; margin-bottom:16px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
.card h2 { font-size:12px; color:var(--muted); margin:0 0 8px; text-transform:uppercase; letter-spacing:.5px; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge.ok { background:rgba(63,185,80,.15); color:var(--ok); }
.badge.run { background:rgba(88,166,255,.15); color:var(--accent); }
.badge.down { background:rgba(248,81,73,.15); color:var(--err); }
.badge.warn { background:rgba(210,153,34,.15); color:var(--warn); }
.kv { display:flex; justify-content:space-between; gap:12px; padding:4px 0; border-bottom:1px dashed var(--border); }
.kv:last-child { border-bottom:none; }
.kv .k { color:var(--muted); }
.kv .v { font-variant-numeric:tabular-nums; font-weight:600; text-align:right; word-break:break-all; }
.big { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:500; }
td.ok { color:var(--ok); } td.fail { color:var(--err); }
#log { background:#0a0e13; border:1px solid var(--border); border-radius:8px; padding:10px 12px; font:12px/1.6 Consolas,Menlo,monospace; height:240px; overflow:auto; white-space:pre-wrap; color:#c9d1d9; }
.gpu-bar { background:#0a0e13; border-radius:4px; height:8px; overflow:hidden; margin:4px 0; }
.gpu-bar > div { height:100%; background:linear-gradient(90deg,var(--accent),#a371f7); transition:width .4s; }
.empty { color:var(--muted); font-size:13px; padding:8px 0; }
</style>
</head>
<body>
<h1>PhysMeta 推理节点监控</h1>
<div class="sub" id="sub">正在连接…</div>
<div class="grid">
  <div class="card"><h2>节点状态</h2><div id="nodeState" class="badge">…</div><div id="nodeInfo" style="margin-top:8px"></div></div>
  <div class="card"><h2>控制链路</h2><div id="link"></div></div>
  <div class="card"><h2>任务统计</h2><div id="tasks"></div></div>
  <div class="card"><h2>当前任务</h2><div id="current"></div></div>
  <div class="card"><h2>GPU</h2><div id="gpu"></div></div>
  <div class="card"><h2>运行时长</h2><div id="uptime" class="big">—</div></div>
</div>
<div class="card" style="margin-top:12px"><h2>最近任务</h2><div id="recent"></div></div>
<div class="card" style="margin-top:12px"><h2>日志</h2><pre id="log">…</pre></div>
<script>
const $ = (id) => document.getElementById(id);
function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function fmtUp(s){ if(s==null) return "—"; s=Math.floor(s); const h=Math.floor(s/3600), m=Math.floor(s%3600/60), ss=s%60; return h+"h "+m+"m "+ss+"s"; }
const ST = { idle:["待命中","ok"], starting:["启动中","warn"], claiming:["领单中","run"], running:["推理中","run"], retrying:["链路重试","down"], failed:["异常","down"] };
const badge = (cls,t) => `<span class="badge ${cls}">${esc(t)}</span>`;
const kv = (k,v) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`;
async function refresh(){
  try {
    const r = await fetch("/api/status", {cache:"no-store"});
    const s = await r.json();
    if (s.error){ $("sub").textContent = s.error; return; }
    $("sub").textContent = `节点 ${esc(s.node_id)} · 版本 ${esc(s.version)} · PID ${esc(s.pid)} · 服务 ${esc(s.server)}`;
    const [st, cls] = ST[s.state] || [s.state,"warn"];
    const ns = $("nodeState"); ns.className = "badge "+cls; ns.textContent = st;
    $("nodeInfo").innerHTML = kv("服务", s.server||"—") + kv("节点", s.node_id||"—");
    const lk = s.link||{};
    $("link").innerHTML = lk.status==="ok"
      ? badge("ok","正常") + kv("历史失败累计", lk.failures||0)
      : badge("down","中断") + kv("失败次数", lk.failures||0) + (lk.last_error?kv("最近错误", lk.last_error):"");
    const t = s.tasks||{};
    $("tasks").innerHTML = kv("累计领单", t.claimed||0) + kv("成功", t.succeeded||0) + kv("失败", t.failed||0);
    const c = s.current_job;
    $("current").innerHTML = c
      ? kv("任务", c.id||"—") + kv("类型", c.type||"—") + kv("文件", c.filename||"—") + kv("已用时", fmtUp(c.started_at ? Date.now()/1000 - c.started_at : null))
      : '<div class="empty">当前无任务</div>';
    const g = s.gpu||{};
    if (g.error) $("gpu").innerHTML = '<div class="empty">'+esc(g.error)+'</div>';
    else {
      const pct = g.mem_total ? Math.min(100,Math.round(g.mem_used/g.mem_total*100)) : 0;
      $("gpu").innerHTML = kv("型号", g.name||"—")
        + kv("显存", (g.mem_used||"—")+" / "+(g.mem_total||"—")+" MB")
        + '<div class="gpu-bar"><div style="width:'+pct+'%"></div></div>'
        + kv("利用率", (g.util||"—")+"%") + kv("温度", (g.temp||"—")+"°C")
        + kv("功耗", (g.power||"—")+" / "+(g.power_limit||"—")+" W");
    }
    $("uptime").textContent = fmtUp(s.uptime);
    const rec = s.recent||[];
    if (!rec.length) $("recent").innerHTML = '<div class="empty">暂无完成记录</div>';
    else {
      let rows = "<table><tr><th>时间</th><th>任务</th><th>类型</th><th>耗时</th><th>结果</th></tr>";
      for (const j of rec) rows += `<tr><td>${esc(j.at)}</td><td>${esc(j.id)}</td><td>${esc(j.type)}</td><td>${esc(j.elapsed)}s</td><td class="${j.status==='ok'?'ok':'fail'}">${j.status==='ok'?'成功':'失败'}</td></tr>`;
      rows += "</table>";
      $("recent").innerHTML = rows;
    }
    const el = $("log");
    el.textContent = (s.logs||[]).join("\n");
    el.scrollTop = el.scrollHeight;
  } catch(e) {
    $("sub").textContent = "监控服务连接失败：" + e.message;
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class _MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # 静音访问日志
        return

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            payload = _snapshot()
            payload["gpu"] = _nvidia_smi()
            payload["uptime"] = round(time.time() - payload.get("started_at", time.time()), 1)
            self._send(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        else:
            self._send(MONITOR_HTML.encode("utf-8"), "text/html; charset=utf-8")


def _start_monitor(port: int, *, open_browser: bool) -> None:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _MonitorHandler)
    except OSError as exc:
        _log(f"[节点] 监控面板启动失败（端口 {port} 被占用）：{exc}")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    _log(f"[节点] 监控面板已启动：{url}")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()


# ---------- 配置加载 ----------

def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"配置文件无效：{exc}") from exc
    if not isinstance(config, dict):
        raise AgentError("配置文件不是 JSON 对象")

    server = str(config.get("server") or "https://yolobubble.physmeta.cn").rstrip("/")
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

# 直连 opener（绕过系统代理）。节点访问的 yolobubble.physmeta.cn 是国内服务器，
# 直连稳定快速；而本机系统代理（127.0.0.1:xxxx 科学上网/抓包工具）会对 HTTPS
# 长轮询连接间歇性 reset，表现为 SSL: UNEXPECTED_EOF_WHILE_READING（周期性断链）。
# 故默认直连；仅当显式设 BUBBLE_USE_PROXY=1 时才走系统代理。
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_USE_PROXY = os.getenv("BUBBLE_USE_PROXY", "0").lower() in ("1", "true", "yes")


def _urlopen(request: urllib.request.Request, timeout: float):
    """统一出站：默认直连（绕过系统代理），避免代理抖动导致 SSL EOF。"""
    if _USE_PROXY:
        return urllib.request.urlopen(request, timeout=timeout)
    return _DIRECT_OPENER.open(request, timeout=timeout)


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
        with _urlopen(request, timeout) as response:
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


def _to_bool(value: Any, default: bool = False) -> bool:
    """安全解析布尔值：正确处理字符串 'false'/'0'/'no'，避免 bool('false')==True 的坑。

    表单/JSON 透传的 options 里 use_ocr/enable_vlm 可能是字符串，直接 bool() 会把
    非空字符串都当成 True。这里按常见语义解析：
      - bool 原样返回
      - 字符串 'false'/'0'/'no'/'off'/'none'/''（不区分大小写）→ False
      - 其余非空字符串 → True
      - None → default
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "false", "0", "no", "off", "none", "null"):
            return False
        return True
    return bool(value)


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
        with _urlopen(request, 180) as response, partial.open("wb") as output:
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
        with _urlopen(request, 300) as response:
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
    enable_vlm = _to_bool(options.get("enable_vlm", bubble_config.ENABLE_VLM))
    use_ocr = _to_bool(options.get("use_ocr", config.get("use_ocr", getattr(bubble_config, "USE_OCR", True))))

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
    _set_state(
        node_id=config["node_id"],
        server=config["server"],
        state="idle",
        link={"status": "ok", "failures": 0, "last_error": ""},
    )
    _log(f"[节点] 已启动：{config['node_id']}（{AGENT_VERSION}），服务 {config['server']}")

    while True:
        try:
            _set_state(state="claiming")
            long_poll = 5.0 if not once else 0.0
            claim = request_json(
                config,
                "POST",
                "/api/inference/claim",
                {"wait_seconds": long_poll},
                timeout=max(10.0, long_poll + 5.0),
            )
            if connection_failures:
                _log("[节点] 控制链路已恢复")
            connection_failures = 0
            _set_state(state="idle", link={"status": "ok", "failures": 0, "last_error": ""})
        except AgentAuthenticationError:
            raise
        except AgentError as exc:
            connection_failures += 1
            delay = min(8.0, float(2 ** min(3, max(1, connection_failures))))
            _set_state(
                state="retrying",
                link={"status": "down", "failures": connection_failures, "last_error": str(exc)[-200:]},
            )
            # 日志降噪：间歇性 SSL/网络抖动会自动恢复，避免每条都刷屏
            if connection_failures == 1:
                _log(f"[节点] 控制链路中断：{exc}")
                _log("[节点] 自动重试中（网络/代理间歇性中断通常很快恢复）...")
            elif connection_failures % 5 == 0:
                _log(f"[节点] 控制链路持续中断（已重试 {connection_failures} 次，最近错误：{exc}）")
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
        task_type = str((job.get("options") or {}).get("task_type") or "full")
        started = time.time()
        _set_state(
            state="running",
            current_job={
                "id": job_id,
                "type": task_type,
                "filename": str(job.get("filename") or ""),
                "started_at": started,
            },
        )
        _bump("claimed")
        _log(f"[节点] 开始任务 {job_id}（{task_type}）")
        try:
            run_job(config, job, run_token)
            elapsed = time.time() - started
            _push_recent({
                "id": job_id,
                "type": task_type,
                "elapsed": round(elapsed, 1),
                "status": "ok",
                "at": time.strftime("%H:%M:%S"),
            })
            _bump("succeeded")
            _set_state(state="idle", current_job=None)
            _log(f"[节点] 任务 {job_id} 完成（{elapsed:.1f}s）")
        except AgentAuthenticationError:
            # 身份失效（token 被吊销/改错）：必须停止，交给人工排查，不能静默吞掉
            raise
        except Exception as exc:  # noqa: BLE001 —— 兜底捕获一切异常（ImportError/模型加载失败/OOM/ValueError 等）
            import traceback
            elapsed = time.time() - started
            error = str(exc)[-1000:] or exc.__class__.__name__
            tb = traceback.format_exc()
            _push_recent({
                "id": job_id,
                "type": task_type,
                "elapsed": round(elapsed, 1),
                "status": "fail",
                "at": time.strftime("%H:%M:%S"),
            })
            _bump("failed")
            _set_state(state="idle", current_job=None)
            _log(f"[节点] 任务 {job_id} 失败：{error}")
            _log(f"[节点] 任务 {job_id} 异常堆栈（末尾）：\n{tb[-2000:]}")
            try:
                request_json(
                    config,
                    "POST",
                    f"/api/inference/jobs/{quote(job_id)}/fail",
                    {"reason": "remote_job_failed", "error": error},
                    run_token=run_token,
                    timeout=10.0,
                )
            except Exception:  # noqa: BLE001 —— 回传失败也不能让节点崩溃
                _log(f"[节点] 任务 {job_id} 回传 /fail 失败（服务端可能已回收该任务）")
            if once:
                raise
            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysMeta minimal inference node (bubble-ocr)")
    parser.add_argument("--config", type=Path, default=Path(r"C:\PhysMetaInference\inference-node.json"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-monitor", action="store_true", help="禁用本地监控面板")
    parser.add_argument("--no-browser", action="store_true", help="启动监控面板但不自动打开浏览器")
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
        if not args.no_monitor:
            port = int(config.get("monitor_port") or MONITOR_PORT)
            open_browser = (not args.no_browser) and bool(config.get("monitor_open_browser", True))
            _start_monitor(port, open_browser=open_browser)
        return run_loop(config, args.once)
    except KeyboardInterrupt:
        return 0
    except AgentError as exc:
        _log(f"PhysMeta 推理节点错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
