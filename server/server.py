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
UPLOAD_TOKEN = os.environ.get("INFERENCE_UPLOAD_TOKEN", "")  # 生产环境必须非空（见 main 的启动校验）
HOST = os.environ.get("INFERENCE_HOST", "0.0.0.0")
PORT = int(os.environ.get("INFERENCE_PORT", "8090"))
# 上传大小上限（对齐 bubble-ocr 的 50MB），防止公网经 Caddy 访问 upload 时被大文件打爆存储
MAX_UPLOAD_BYTES = int(os.environ.get("INFERENCE_MAX_UPLOAD", str(50 * 1024 * 1024)))
# 结果回传大小上限（结果 zip 正常 1~10MB，设上限防持 token 者撑爆磁盘）
MAX_RESULT_BYTES = int(os.environ.get("INFERENCE_MAX_RESULT", str(50 * 1024 * 1024)))
# claimed 任务超时回收：节点崩溃后，超过该时长的 claimed 任务重置为 pending 重新派发
CLAIM_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_CLAIM_TIMEOUT", "300"))
# processing 任务超时回收：节点回传进度后崩溃，超过该时长未完成则重置为 pending（需要租约心跳续期）
PROCESSING_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_PROCESSING_TIMEOUT", "3600"))
# 输入/结果文件保留期（秒）：超过该时长的文件与任务记录被清理，避免磁盘无限增长
RETENTION_SECONDS = int(os.environ.get("INFERENCE_RETENTION", str(7 * 24 * 3600)))
# 结果 zip 内允许的最大文件数与单文件最大解压后大小（防 zip 炸弹）
MAX_ARCHIVE_FILES = int(os.environ.get("INFERENCE_MAX_ARCHIVE_FILES", "200"))
MAX_ARCHIVE_UNCOMPRESSED = int(os.environ.get("INFERENCE_MAX_ARCHIVE_UNCOMPRESSED", str(200 * 1024 * 1024)))

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
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            heartbeat_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON inference_jobs (status, created_at);
    """)
    # SQLite 的 CREATE TABLE IF NOT EXISTS 不会给已有部署补列；显式迁移旧库。
    columns = {row[1] for row in conn.execute("PRAGMA table_info(inference_jobs)")}
    if "heartbeat_at" not in columns:
        conn.execute("ALTER TABLE inference_jobs ADD COLUMN heartbeat_at TIMESTAMP")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_heartbeat "
        "ON inference_jobs (status, heartbeat_at)"
    )
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


def _reap_stale_claimed(conn: sqlite3.Connection) -> None:
    """把超时未完成的 claimed / processing 任务重置为 pending（节点崩溃兜底，避免任务永久卡死）。

    - claimed：认领后超过 CLAIM_TIMEOUT_SECONDS 未开始（未回传进度）→ 重置。
    - processing：回传进度后超过 PROCESSING_TIMEOUT_SECONDS 未完成 → 重置（依赖心跳续期）。
    """
    conn.execute(
        "UPDATE inference_jobs SET status='pending', node_id=NULL, run_token=NULL, heartbeat_at=NULL "
        "WHERE status='claimed' AND updated_at < datetime('now', ?)",
        (f"-{CLAIM_TIMEOUT_SECONDS} seconds",),
    )
    conn.execute(
        "UPDATE inference_jobs SET status='pending', node_id=NULL, run_token=NULL, heartbeat_at=NULL "
        "WHERE status='processing' AND COALESCE(heartbeat_at, updated_at) < datetime('now', ?)",
        (f"-{PROCESSING_TIMEOUT_SECONDS} seconds",),
    )


# 合法状态转换表：确保 progress/result/fail 只能从合法前置状态转移，避免旧节点/重放覆盖终态
_VALID_TRANSITIONS = {
    "progress": ("claimed", "processing"),
    "result": ("claimed", "processing"),
    "fail": ("claimed", "processing"),
}


def claim_next_pending(node_id: str, run_token: str):
    """原子领取队首 pending 任务，返回任务行或 None。

    用 BEGIN IMMEDIATE 加写锁，保证「查 pending + 标记 claimed」原子，
    避免多节点并发把同一任务重复领取；顺带回收超时未完成的 claimed/processing 任务。
    """
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _reap_stale_claimed(conn)
        row = conn.execute(
            "SELECT * FROM inference_jobs WHERE status='pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        conn.execute(
            "UPDATE inference_jobs SET status='claimed', node_id=?, run_token=?, "
            "updated_at=CURRENT_TIMESTAMP, heartbeat_at=CURRENT_TIMESTAMP WHERE id=? AND status='pending'",
            (node_id, run_token, row["id"]),
        )
        conn.commit()
        return dict(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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


def transition(job_id: str, run_token: str, action: str, status: str, **fields) -> bool:
    """带状态条件的原子状态转换，返回是否成功。

    action ∈ {"progress","result","fail"}，对应合法前置状态见 _VALID_TRANSITIONS。
    job_id 与 run_token 必须同时匹配，避免任务回收重派后旧节点更新新租约。
    """
    allowed = _VALID_TRANSITIONS.get(action)
    if not allowed:
        raise ValueError(f"未知状态转换动作：{action}")
    sets = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    values = [status]
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        values.append(v)
    values.extend([job_id, run_token, *allowed])
    placeholders = ", ".join("?" for _ in allowed)
    conn = get_db()
    try:
        cur = conn.execute(
            f"UPDATE inference_jobs SET {', '.join(sets)} "
            f"WHERE id = ? AND run_token = ? AND status IN ({placeholders})",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def heartbeat(job_id: str, run_token: str) -> bool:
    """刷新任务心跳时间戳（processing 阶段续期，防误回收）。"""
    conn = get_db()
    cur = conn.execute(
        "UPDATE inference_jobs SET heartbeat_at=CURRENT_TIMESTAMP "
        "WHERE id = ? AND run_token = ? AND status IN ('claimed','processing')",
        (job_id, run_token),
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0


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
        raw = self.headers.get("Content-Length", "0")
        try:
            length = int(raw)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return b""
        # 循环读满 length 字节：TCP 分片下 rfile.read(n) 不保证一次读满
        buf = b""
        while len(buf) < length:
            chunk = self.rfile.read(length - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

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
        # 大小上限：读 body 前拦截，避免超大文件耗尽存储/内存
        try:
            upload_len = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._json(400, {"error": "Content-Length 无效"})
            return
        if upload_len > MAX_UPLOAD_BYTES:
            self._json(413, {"error": f"文件超过大小上限 {MAX_UPLOAD_BYTES // (1024*1024)}MB"})
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
        raw_wait = payload.get("wait_seconds", 5)
        try:
            wait_seconds = max(0.0, min(30.0, float(raw_wait)))
        except (TypeError, ValueError):
            wait_seconds = 5.0

        deadline = time.time() + wait_seconds
        node_id = self.headers.get("X-Inference-Node-ID", "")
        while True:
            run_token = secrets.token_hex(32)
            row = claim_next_pending(node_id, run_token)
            if row is not None:
                self._json(200, {
                    "job": {
                        "id": row["id"],
                        "filename": row["filename"],
                        "input_bytes": row["input_bytes"],
                        "input_sha256": row["input_sha256"],
                        "options": json.loads(row["options"]) if row["options"] else {},
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
        run_token = self.headers.get("X-Inference-Run-Token", "")
        if not token_ok(run_token, row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return
        data = Path(row["input_path"]).read_bytes()
        heartbeat(job_id, run_token)
        self._send(200, data, "application/octet-stream")

    # ---------------- 回传进度 ----------------
    def _handle_progress(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        run_token = self.headers.get("X-Inference-Run-Token", "")
        if not token_ok(run_token, row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return
        # 进度回传即心跳续期；状态转换：claimed→processing 或 processing→processing
        if not transition(job_id, run_token, "progress", "processing"):
            # 已处于 done/failed 终态，旧节点重放，忽略而非报错
            self._json(200, {"ok": True, "status": row["status"], "stale": True})
            return
        heartbeat(job_id, run_token)
        self._json(200, {"ok": True})

    # ---------------- 回传结果 ----------------
    def _handle_result(self, job_id: str):
        if not self._require_node():
            return
        row = job_row(job_id)
        if row is None:
            self._json(404, {"error": "任务不存在"})
            return
        run_token = self.headers.get("X-Inference-Run-Token", "")
        if not token_ok(run_token, row["run_token"] or ""):
            self._json(403, {"error": "run_token 不匹配"})
            return

        # 结果大小上限：读 body 前拦截（与上传一致，防撑爆存储/内存）
        raw_len = self.headers.get("Content-Length", "0")
        try:
            body_len = int(raw_len)
        except (TypeError, ValueError):
            body_len = 0
        if body_len > MAX_RESULT_BYTES:
            self._json(413, {"error": f"结果超过大小上限 {MAX_RESULT_BYTES // (1024*1024)}MB"})
            return

        # multipart：读 artifact 字段（流式读入，避免一次性读完整 body 进内存时放大）
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

        # 强制 SHA-256 校验：生产环境必须携带且匹配，缺失或错误一律拒绝
        digest = sha256(body)
        expected = self.headers.get("X-Artifact-SHA256", "")
        if not expected:
            self._json(400, {"error": "缺少 X-Artifact-SHA256 头"})
            return
        if digest != expected:
            self._json(400, {"error": "结果 SHA-256 校验失败"})
            return

        # zip 炸弹防护：限制文件数与解压后总大小，且只接受合法 zip
        try:
            import zipfile
            with zipfile.ZipFile(__import__("io").BytesIO(body)) as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ARCHIVE_FILES:
                    self._json(400, {"error": f"结果包文件数超限（>{MAX_ARCHIVE_FILES}）"})
                    return
                total = sum(i.file_size for i in infos)
                if total > MAX_ARCHIVE_UNCOMPRESSED:
                    self._json(400, {"error": "结果包解压后体积超限"})
                    return
        except zipfile.BadZipFile:
            self._json(400, {"error": "结果包不是合法 zip"})
            return

        # 先写入带 run_token 的唯一文件，成功后再以同一 token 提交 done 状态。
        # 这样旧节点不会覆盖新租约的结果，也不会留下 done 但无文件的记录。
        result_path = STORAGE_DIR / "result" / f"{job_id}.{run_token}.{uuid.uuid4().hex}.zip"
        tmp_path = result_path.with_name(f".{result_path.name}.part")
        try:
            tmp_path.write_bytes(body)
            tmp_path.replace(result_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        if not transition(job_id, run_token, "result", "done", result_path=str(result_path)):
            result_path.unlink(missing_ok=True)
            self._json(200, {"ok": True, "status": row["status"], "stale": True})
            return
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
        # 状态转换（claimed/processing → failed）：终态后旧节点重放被忽略
        run_token = self.headers.get("X-Inference-Run-Token", "")
        if not transition(job_id, run_token, "fail", "failed", error=str(payload.get("error") or "")[-1000:]):
            self._json(200, {"ok": True, "status": row["status"], "stale": True})
            return
        # TODO: 在这里接入云端回退 Worker
        self._json(200, {"ok": True, "status": "failed"})

    # ---------------- 查询状态 ----------------
    def _handle_status(self, job_id: str):
        # 状态接口经 Caddy 暴露公网，必须校验上传令牌（与 _handle_result_get 一致），
        # 否则任何人可凭 job_id 枚举任务状态/错误/时间戳。
        if UPLOAD_TOKEN and not token_ok(self._bearer(), UPLOAD_TOKEN):
            self._json(401, {"error": "上传认证失败"})
            return
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
        # 该接口会经 Caddy 暴露公网，必须校验上传令牌（与 _handle_upload 一致），
        # 否则任何人可凭 job_id 读取识别结果，破坏「图纸不出内网」红线。
        if UPLOAD_TOKEN and not token_ok(self._bearer(), UPLOAD_TOKEN):
            self._json(401, {"error": "上传认证失败"})
            return
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


def _cleanup_expired() -> int:
    """清理超过保留期的任务记录与输入/结果文件，返回清理条数。

    只清理终态（done/failed）且超过 RETENTION_SECONDS 的任务；
    pending/claimed/processing 永不清理（仍在流转中）。
    """
    cutoff = time.time() - RETENTION_SECONDS
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, input_path, result_path FROM inference_jobs "
            "WHERE status IN ('done','failed') AND updated_at < datetime('now', ?)",
            (f"-{RETENTION_SECONDS} seconds",),
        ).fetchall()
    finally:
        conn.close()

    removed = 0
    for row in rows:
        for col in ("input_path", "result_path"):
            p = row[col]
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
    if rows:
        ids = [r["id"] for r in rows]
        conn = get_db()
        try:
            conn.executemany("DELETE FROM inference_jobs WHERE id = ?", [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()
        removed = len(ids)
    return removed


def _cleanup_loop() -> None:
    """周期性磁盘清理线程（每小时一次）。"""
    while True:
        time.sleep(3600)
        try:
            n = _cleanup_expired()
            if n:
                print(f"[cleanup] 清理了 {n} 条过期任务记录", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cleanup] 清理失败：{exc}", flush=True)


def main():
    # 生产模式强制认证：两个 token 必须是强随机值，否则拒绝启动（防静默失去保护）
    if not NODE_TOKEN or len(NODE_TOKEN) < 32:
        raise SystemExit("启动失败：INFERENCE_NODE_TOKEN 缺失或过短（至少 32 字符）")
    if not UPLOAD_TOKEN or len(UPLOAD_TOKEN) < 32:
        raise SystemExit("启动失败：INFERENCE_UPLOAD_TOKEN 缺失或过短（至少 32 字符）")

    init_db()
    # 启动时清理一次历史过期数据，之后由后台线程周期清理
    _cleanup_expired()
    threading.Thread(target=_cleanup_loop, daemon=True).start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    # 请求线程随主进程退出，避免关闭时残留后台线程；公网并发限制由反向代理负责。
    server.daemon_threads = True
    print(f"PhysMeta inference server listening on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
