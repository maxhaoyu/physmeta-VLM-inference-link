-- PhysMeta 最小化推理链路 · 服务器端建表（单节点 FIFO 队列）
-- 兼容 SQLite / PostgreSQL。MySQL 请把 TEXT 改 VARCHAR，TIMESTAMP 加 DEFAULT。

CREATE TABLE IF NOT EXISTS inference_jobs (
    id           TEXT PRIMARY KEY,             -- 形如 cad-20260923-132500-abc123
    user_id      TEXT,                         -- 上传客户（可空）
    input_path   TEXT NOT NULL,                -- 服务器上的图纸文件路径
    input_sha256 TEXT NOT NULL,
    input_bytes  INTEGER NOT NULL,
    filename     TEXT,                         -- 原始文件名
    options      TEXT,                         -- 透传给 run_pipeline 的 JSON 字符串
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending/claimed/processing/done/failed/dead_letter
    node_id      TEXT,                         -- 认领节点
    run_token    TEXT,                         -- 每次任务独立 token
    result_path  TEXT,                         -- 回传结果 zip 的存储路径
    error        TEXT,                         -- 失败原因
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    heartbeat_at TIMESTAMP,                   -- processing 阶段租约心跳
    lease_managed INTEGER NOT NULL DEFAULT 0,  -- 1=可按心跳超时回收
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    agent_version TEXT,
    protocol_version INTEGER,
    last_phase TEXT,
    last_progress REAL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON inference_jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_heartbeat ON inference_jobs (status, heartbeat_at);

CREATE TABLE IF NOT EXISTS inference_nodes (
    node_id TEXT PRIMARY KEY,
    agent_version TEXT,
    protocol_version INTEGER,
    capabilities TEXT,
    current_job TEXT,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    node_id TEXT,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events (job_id, id);
