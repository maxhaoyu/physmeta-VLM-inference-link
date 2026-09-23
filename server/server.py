"""PhysMeta 最小化推理链路 · 服务器端（标准库实现，对齐平台惯例）

跟 identity-service / dxf-bubble-app / bubble-ocr-app 一致：
原生 http.server + ThreadingHTTPServer + sqlite3，零第三方 Web 框架。

职责：接收上传的图纸 → 入队 → 供 Windows 节点长轮询领取 → 收结果。

接口：
  POST /api/inference/upload               客户上传图纸，创建 pending 任务
  POST /api/inference/claim                节点领任务（长轮询）
  GET  /api/inference/jobs/{id}/input      节点下载图纸
  POST /api/inference/jobs/{id}/progress   节点回传进度
  POST /api/inference/jobs/{id}/result     节点回传结果 zip
  POST /api/inference/jobs/{id}/fail       节点失败，回退云端
  GET  /api/inference/jobs/{id}            前端查询任务状态

认证：
  节点用 Bearer token（NODE_TOKEN 环境变量，constant-time 比较）
  任务级用 X-Inference-Run-Token
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

# ---------- 配置 ----------
ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("INFERENCE_DB", str(ROOT / "inference.db")))
STORAGE_DIR = Path(os.environ.get("INFERENCE_STORAGE", str(ROOT / "storage")))
NODE_TOKEN = os.environ.get("INFERENCE_NODE_TOKEN", "")
UPLOAD_TOKEN = os.environ.get("INFERENCE_UPLOAD_TOKEN", "")  # 空则不校验上传
HOST = os.environ.get("INFERENCE_HOST", "0.0.0.0")
PORT = int(os.environ.get("INFERENCE_PORT", "8090"))

(STORAGE_DIR / "input").mkdir(parents=True, exist_ok=True)
(STORAGE_DIR / "result").mkdir(parents=True, exist_ok=True)

JOB_ID_RE = re.compile(r"cad-\d{8}-\d{6}-[A-Za-z0-9]{6}\Z")
DB_LOCK = threading.Lock()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_job_id() -> str:
    import datetime
    return "cad-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


# ---------- DB ----------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inference_jobs (
            id           TEXT PRIMARY KEY,
            user_id      TEXT,
            input_path   TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            input_bytes  INTEGER NOT NULL,
            filename     TEXT,
            options      TEXT,
            status       TEXT NOT NULL DEFAULT 'pending',
            node_id      TEXT,
            run_token    TEXT,
            result_path  TEXT,
            error        TEXT,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON inference_jobs (status, created_at);
    """)
    conn.commit()
    conn.close()


def job_row(job_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM inference_jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return row


def next_pending_job():
    """FIFO 取队首 pending 任务。"""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM inference_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def update_status(job_id: str, status: str, **fields):
    sets = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    values = [status]
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        values.append(v)
    values.append(job_id)
    conn = get_db()
    conn.execute(f"UPDATE inference_jobs SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()
    conn.close()


# ---------- 常量时间 token 比较 ----------
def token_ok(provided: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(provided or "", expected)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bearer(self) -> str:
        auth = self.headers.get("Authorization", "")
        return auth.removeprefix("Bearer ").strip()

    def _require_node(self) -> bool:
        if not token_ok(self._bearer(), NODE_TOKEN):
            self._json(401, {"error": "节点认证失败"})
            return False
        return True

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ---------------- 路由 ----------------
    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path

        if p == "/healthz":
            self._json(200, {"status": "ok"})
            return

        m = re.fullmatch(r"/api/inference/jobs/([^/]+)/input", p)
        if m:
            self._handle_input(m.group(1))
            return

        m = re.fullmatch(r"/api/inference/jobs/([^/]+)/result", p)
        if m:
            self._handle_result_get(m.group(1))
            return

        m = re.fullmatch(r"/api/inference/jobs/([^/]+)", p)
        if m:
            self._handle_status(m.group(1))
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        p = parsed.path

        if p == "/api/inference/upload":
            self._handle_upload()
            return
        if p == "/api/inference/claim":
            self._handle_claim()
            return

        m = re.fullmatch(r"/api/inference/jobs/([^/]+)/progress", p)
        if m:
            self._handle_progress(m.group(1))
            return
        m = re.fullmatch(r"/api/inference/jobs/([^/]+)/result", p)
        if m:
            self._handle_result(m.group(1))
            return
        m = re.fullmatch(r"/api/inference/jobs/([^/]+)/fail", p)
        if m:
            self._handle_fail(m.group(1))
            return

        self._json(404, {"error": "not found"})

    # ---------------- 上传 ----------------
    def _handle_upload(self):
        if UPLOAD_TOKEN and not token_ok(self._bearer(), UPLOAD_TOKEN):
            self._json(401, {"error": "上传认证失败"})
            return
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._json(400, {"error": "需要 multipart/form-data"})
            return
        body = self._read_body()
        boundary = re.search(r"boundary=([^;]+)", ctype)
        if not boundary:
            self._json(400, {"error": "缺少 boundary"})
            return
        boundary = boundary.group(1).strip().strip('"').encode()

        # 取 image 字段（及可选 task_type / region / options）
        image_bytes = None
        filename = "upload.png"
        task_type = "full"
        region = None
        extra_options: dict = {}

        for part in body.split(b"--" + boundary):
            if b"Content-Disposition" not in part or b'name="' not in part:
                continue
            header, _, data = part.partition(b"\r\n\r\n")
            name_m = re.search(rb'name="([^"]*)"', header)
            if not name_m:
                continue
            name = name_m.group(1).decode("utf-8", "ignore")
            data = data.rsplit(b"\r\n", 1)[0] if data.endswith(b"\r\n") else data
            if name == "image":
                m = re.search(rb'filename="([^"]*)"', header)
                if m:
                    filename = m.group(1).decode("utf-8", "ignore") or "upload.png"
                image_bytes = data
            elif name == "task_type":
                task_type = data.decode("utf-8", "ignore").strip() or "full"
            elif name == "region":
                try:
                    region = json.loads(data.decode("utf-8", "ignore"))
                except (ValueError, UnicodeDecodeError):
                    region = None
            elif name == "options":
                try:
                    extra_options = json.loads(data.decode("utf-8", "ignore")) or {}
                except (ValueError, UnicodeDecodeError):
                    extra_options = {}

        if not image_bytes:
            self._json(400, {"error": "未找到 image 字段"})
            return

        options = dict(extra_options)
        if task_type and task_type != "full":
            options["task_type"] = task_type
        if region:
            options["region"] = region

        job_id = new_job_id()
        input_path = STORAGE_DIR / "input" / job_id
        input_path.write_bytes(image_bytes)
        digest = sha256(image_bytes)

        conn = get_db()
        conn.execute(
            "INSERT INTO inference_jobs (id, input_path, input_sha256, input_bytes, filename, options, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            (job_id, str(input_path), digest, len(image_bytes), filename,
             json.dumps(options, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()

        self._json(200, {"job_id": job_id, "status": "pending"})

    # ---------------- 领任务（长轮询） ----------------
    def _handle_claim(self):
        if not self._require_node():
            return
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            payload = {}
        wait_seconds = max(0.0, min(30.0, float(payload.get("wait_seconds") or 5)))

        deadline = time.time() + wait_seconds
        while True:
            row = next_pending_job()
            if row is not None:
                run_token = secrets.token_hex(32)
                node_id = self.headers.get("X-Inference-Node-ID", "")
                update_status(row["id"], "claimed", node_id=node_id, run_token=run_token)
                job = dict(row)
                self._json(200, {
                    "job": {
                        "id": job["id"],
                        "filename": job["filename"],
                        "input_bytes": job["input_bytes"],
                        "input_sha256": job["input_sha256"],
                        "options": json.loads(job["options"]) if job["options"] else {},
                    },
                    "run_token": run_token,
                })
                return
            if time.time() >= deadline:
                self._json(200, {"retry_after_seconds": 5})
                return
            time.sleep(0.5)

    # ---------------- 下载图纸 ----------------
    def _handle_input(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        if not token_ok(self.headers.get("X-Inference-Run-Token", ""), row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return
        data = Path(row["input_path"]).read_bytes()
        self._send(200, data, "application/octet-stream")

    # ---------------- 回传进度 ----------------
    def _handle_progress(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        if not token_ok(self.headers.get("X-Inference-Run-Token", ""), row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return
        update_status(job_id, "processing")
        self._json(200, {"ok": True})

    # ---------------- 回传结果 ----------------
    def _handle_result(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        if not token_ok(self.headers.get("X-Inference-Run-Token", ""), row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return

        # multipart：读 artifact 字段
        ctype = self.headers.get("Content-Type", "")
        body = self._read_body()
        if ctype.startswith("multipart/form-data"):
            boundary = re.search(r"boundary=([^;]+)", ctype)
            if boundary:
                boundary = boundary.group(1).strip().strip('"').encode()
                for part in body.split(b"--" + boundary):
                    if b'name="artifact"' not in part:
                        continue
                    _, _, data = part.partition(b"\r\n\r\n")
                    data = data.rsplit(b"\r\n", 1)[0] if data.endswith(b"\r\n") else data
                    body = data
                    break

        digest = sha256(body)
        expected = self.headers.get("X-Artifact-SHA256", "")
        if expected and digest != expected:
            self._json(400, {"error": "结果 SHA-256 校验失败"})
            return

        result_path = STORAGE_DIR / "result" / f"{job_id}.zip"
        result_path.write_bytes(body)
        update_status(job_id, "done", result_path=str(result_path))
        self._json(200, {"ok": True, "status": "done"})

    # ---------------- 失败回退 ----------------
    def _handle_fail(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        if not token_ok(self.headers.get("X-Inference-Run-Token", ""), row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            payload = {}
        update_status(job_id, "failed", error=str(payload.get("error") or "")[-1000:])
        # TODO: 在这里接入云端回退 Worker
        self._json(200, {"ok": True, "status": "failed"})

    # ---------------- 查询状态 ----------------
    def _handle_status(self, job_id: str):
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        self._json(200, {
            "id": row["id"],
            "status": row["status"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # ---------------- 取回结果（前端/上传服务轮询用） ----------------
    def _handle_result_get(self, job_id: str):
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        if row["status"] == "failed":
            self._json(200, {"status": "failed", "error": row["error"]})
            return
        if row["status"] != "done" or not row["result_path"]:
            self._json(200, {"status": row["status"]})
            return
        result_path = Path(row["result_path"])
        if not result_path.is_file():
            self._json(200, {"status": row["status"]})
            return
        try:
            import zipfile
            with zipfile.ZipFile(result_path) as zf:
                data = zf.read("result.json")
            payload = json.loads(data.decode("utf-8"))
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError, OSError):
            self._json(500, {"error": "结果包损坏"})
            return
        self._json(200, {"status": "done", "result": payload})

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


def main():
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"PhysMeta inference server listening on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
