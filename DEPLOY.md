# 部署 Prompt（给 codeX）：yolobubble 框选识别 → 客户 Windows GPU 推理

> 这是一份**自包含**的部署指令。代码已全部改好并 push 到 GitHub（见第 2 节），
> 你的任务只是**在服务器上完成部署**。按下面步骤执行即可，遇到和本文件不一致的
> 环境细节（路径/端口）时，以服务器实际现状为准并回传说明。

---

## 0. 目标（一句话）

让线上 `https://yolobubble.physmeta.cn` 的「▭ 单标注框选」和「◫ 区域批量识别」两个功能，
把推理任务路由到**客户 Windows 本机 NVIDIA GPU** 执行（YOLO 检测 + VLM 逐框读取），而不是服务器本地跑。

## 1. 架构与数据流

```
客户浏览器（yolobubble 网页）
  │ ① 上传图纸 → 点「单标注框选」或「区域批量识别」
  ▼
bubble-ocr 服务（服务器，SSO 鉴权，已在跑）
  │ ② POST /region 时转发（内网 INFERENCE_QUEUE_URL + Bearer UPLOAD_TOKEN）
  ▼
inference-queue 服务（新部署，任务队列，http.server + sqlite3）
  │ ③ Windows 节点长轮询 POST /claim 领单（公网 HTTPS + Bearer NODE_TOKEN）
  ▼
Windows 节点（客户本机 GPU：YOLO 定位 → VLM 逐框读取）
  │ ④ 回传 result.zip
  ▼
inference-queue → bubble-ocr 阻塞轮询取回 → 前端渲染
```

关键红线：图纸数据只在「服务器 ↔ Windows 节点」之间流动；节点**主动出站**领单/回传，
服务器永不主动连进客户内网，客户防火墙/NAT 无需开入站端口。

## 2. 已完成的代码（无需再写代码，只需部署）

### 仓库 A：`maxhaoyu/bubble-ocr-mvp`（私有，服务器 `/opt/bubble-ocr` 已在跑）
本次已 push 的改动（`git pull` 即可获得）：
- `src/bubble_ocr/server.py`：`_handle_region` 在配置了 `INFERENCE_QUEUE_URL` 时，
  把「图纸 + 框选区域」转发给推理队列并**阻塞轮询**结果（前端零改动，保持同步交互）。
  新增 `_forward_inference` / `_wait_inference` 两个方法。
- `src/bubble_ocr/config.py`：新增 `INFERENCE_QUEUE_URL` / `INFERENCE_UPLOAD_TOKEN` / `INFERENCE_WAIT_TIMEOUT`。
- `src/bubble_ocr/pipeline.py`：`run_region` / `run_region_detect` 加 `use_ocr` 参数（VLM-only）。
- `src/bubble_ocr/vlm_fallback_windows.py`：Windows 加速版 VLM 后端（bf16 + SDPA + read_all）。

### 仓库 B：`maxhaoyu/physmeta-VLM-inference-link`（公开，本仓库）
- `server/server.py`：队列服务（标准库 http.server + sqlite3，零第三方依赖）。
  接口：`POST /api/inference/upload`、`POST /api/inference/claim`、
  `GET /api/inference/jobs/{id}/input`、`POST /api/inference/jobs/{id}/progress|result|fail`、
  `GET /api/inference/jobs/{id}`（状态）、`GET /api/inference/jobs/{id}/result`（取回结果）。
  已支持 `task_type`/`region`/`options` 表单字段透传（full / region_single / region_detect）。
- `server/Dockerfile`：可直接 build。
- `node/agent.py`：Windows 节点（长轮询领单 + 三种任务分发）。

## 3. 服务器部署步骤（在 ECS `47.243.175.149` 上执行）

### 3.1 拉取两个仓库最新代码

```bash
# 仓库 A：更新 bubble-ocr（拿到 /region 转发逻辑）
cd /opt/bubble-ocr && git pull --ff-only

# 仓库 B：克隆推理链路（拿 inference-queue 的 server/）
git clone https://github.com/maxhaoyu/physmeta-VLM-inference-link.git /opt/inference-link
# 若服务器无 HTTPS 权限，改用 SSH：git@github.com:maxhaoyu/physmeta-VLM-inference-link.git
```

### 3.2 生成两个 token（各 ≥32 字符，随机）

```bash
python3 -c "import secrets; print('NODE_TOKEN  =', secrets.token_hex(32))"
python3 -c "import secrets; print('UPLOAD_TOKEN=', secrets.token_hex(32))"
```

- `NODE_TOKEN`：**Windows 节点的身份令牌**，部署完成后下发给客户节点。
- `UPLOAD_TOKEN`：bubble-ocr → 队列的上传鉴权令牌（服务器内部使用，不下发）。
- 两个 token 必须**不同**，且保存好（写入服务器 `.env` / secrets）。

### 3.3 把 inference-queue 挂进 compose

编辑 `/opt/bubble-ocr/docker-compose.yml`，新增一个服务（`build.context` 按 3.1 实际路径改）：

```yaml
services:
  inference-queue:
    build:
      context: /opt/inference-link/server
    image: inference-queue:1.0.0
    container_name: inference-queue
    restart: unless-stopped
    cap_drop: [ALL]
    security_opt: [no-new-privileges:true]
    environment:
      INFERENCE_NODE_TOKEN: "<3.2 生成的 NODE_TOKEN>"
      INFERENCE_UPLOAD_TOKEN: "<3.2 生成的 UPLOAD_TOKEN>"
      INFERENCE_DB: /data/inference.db
      INFERENCE_STORAGE: /data/storage
      INFERENCE_HOST: 0.0.0.0
      INFERENCE_PORT: "8090"
    volumes:
      - inference-data:/data
    ports:
      - "127.0.0.1:18090:8090"     # 仅回环，供 Caddy 反代，不直接暴露公网
    networks:
      default: {}
      edge:
        aliases: [inference-queue]   # 供 bubble-ocr 通过 http://inference-queue:8090 内网访问
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3).status==200 else 1)"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
    logging:
      driver: json-file
      options: { max-size: "10m", max-file: "3" }

# 顶层新增一个命名卷
volumes:
  inference-data:
```

要点：
- `inference-queue` 必须和 `bubble-ocr` 在**同一个 network**（现有 `edge` = `physmeta_edge` external），
  这样 bubble-ocr 才能用服务名 `inference-queue:8090` 访问它。
- 回环端口 `18090` 供 Caddy 反代（下一节），不要暴露到公网。

### 3.4 bubble-ocr 服务加两个环境变量

在同一个 compose 的 `bubble-ocr` 服务的 `environment:` 里**追加**：

```yaml
      INFERENCE_QUEUE_URL: "http://inference-queue:8090"
      INFERENCE_UPLOAD_TOKEN: "<3.2 生成的 UPLOAD_TOKEN，与 inference-queue 一致>"
```

`INFERENCE_QUEUE_URL` 设了之后，bubble-ocr 的 `/region` 才会走转发；不设则仍是服务器本地推理（旧行为），
因此这两个变量是关键开关。

### 3.5 Caddy 路由（关键，容易漏）

Windows 节点通过公网域名 `https://yolobubble.physmeta.cn/api/inference/*` 访问队列。
所以 Caddy 必须把 `/api/inference/*` 前缀路由到 inference-queue，其余仍走 bubble-ocr。

在 `yolobubble.physmeta.cn` 站点里，**在现有的 `reverse_proxy` 行之前**加一段 `handle` 路由；
现有的 `reverse_proxy` 行**保持原样不动**（它的目标端口以服务器现状为准，仓库里 `deploy/caddy-yolobubble.Caddyfile` 是 `127.0.0.1:8000`）：

```
yolobubble.physmeta.cn {
    # 推理队列接口（节点 claim/input/result/fail 走这里）
    handle /api/inference/* {
        reverse_proxy 127.0.0.1:18090
    }
    # 其余请求仍走 bubble-ocr（这一行是服务器现状，别改，可能是 8000）
    handle {
        reverse_proxy 127.0.0.1:8000
    }
    # ... 保留原有 request_body 上限 50MB、安全头、日志等 ...
}
```

然后 `caddy reload --config /etc/caddy/Caddyfile`（或你的 reload 方式）。

> 说明：bubble-ocr 自身没有 `/api/inference/*` 路由（它的路由是 `/`、`/region`、`/upload`、`/result` 等），
> 所以前缀路由不会冲突。节点侧接口与上传侧接口都汇聚到 inference-queue，只是
> 上传侧走内网 `http://inference-queue:8090`，节点侧走公网 Caddy。

### 3.6 重启并验证

```bash
cd /opt/bubble-ocr
docker compose up -d --build

# 1) 队列健康检查
curl -f http://127.0.0.1:18090/healthz          # 期望 {"status":"ok"}

# 2) 队列 claim 鉴权生效（无 token 应 401）
curl -i -X POST http://127.0.0.1:18090/api/inference/claim -d '{}'   # 期望 401 节点认证失败

# 3) 队列 claim 带正确 NODE_TOKEN 应 200（无任务时返回 retry_after_seconds）
curl -i -X POST http://127.0.0.1:18090/api/inference/claim \
  -H "Authorization: Bearer <NODE_TOKEN>" \
  -H "Content-Type: application/json" -d '{"wait_seconds":1}'        # 期望 200

# 4) 公网链路（经 Caddy）也应可达
curl -f https://yolobubble.physmeta.cn/api/inference/healthz 2>/dev/null || \
  curl -f https://yolobubble.physmeta.cn/api/inference/jobs/nonexist 2>/dev/null
```

## 4. 验收标准（做到什么算部署完成）

1. `http://127.0.0.1:18090/healthz` 返回 `{"status":"ok"}`。
2. 无 token 调 `POST /api/inference/claim` 返回 401；带正确 `NODE_TOKEN` 返回 200（空队列时 `retry_after_seconds`）。
3. 公网 `https://yolobubble.physmeta.cn/api/inference/*` 经 Caddy 可达队列。
4. bubble-ocr 服务 `environment` 已含 `INFERENCE_QUEUE_URL` + `INFERENCE_UPLOAD_TOKEN`。

## 5. 你要回传给用户的东西

**只回传一个值**：`INFERENCE_NODE_TOKEN`（3.2 生成的 NODE_TOKEN），
用户会把它填进 Windows 节点的 `inference-node.json` 启动节点。

> 不要回传 `INFERENCE_UPLOAD_TOKEN`（服务器内部用）。

## 6. Windows 节点端（客户本地，不在你的职责内，但附上以便你了解全貌）

- 节点目录已就绪：`C:\PhysMetaInference-ocr\`（`agent.py` + `inference-node.json` + `vlm_fallback_windows.py`）。
- `inference-node.json` 里 `server=https://yolobubble.physmeta.cn`、`node_id=windows-5060ti-01`、
  `use_ocr=false`，`token` 待填（就是上面下发的 NODE_TOKEN）。
- **VLM 模型版本**：基座 = `models/vlm/MiniCPM-V-4.6-hf-ra600`（已 merge checkpoint-600），
  adapter = `models/adapters/checkpoint-150-best`。**不能挂原始 `MiniCPM-V-4.6-hf`**，详见 `docs/MODEL_VERSION.md`。
- 用户填完 token 后，在 Windows 上运行一个常驻进程：
  ```
  C:\Users\maxzh\WorkBuddy\2026-09-23-16-10-09\bubble-ocr-app\.venv\Scripts\python.exe ^
    C:\PhysMetaInference-ocr\agent.py --config C:\PhysMetaInference-ocr\inference-node.json
  ```
  该进程长轮询 `POST /api/inference/claim` 领单 → 下载图纸 → 本地 GPU 跑 YOLO+VLM → 回传 result.zip。

## 7. 红线（务必遵守）

- 节点只主动出站，服务器**不要**主动连客户内网；不要给节点开放入站端口。
- inference-queue 不要暴露公网端口（只用 `127.0.0.1:18090` 回环 + Caddy 反代）。
- token 走环境变量 / secrets，不要硬编码进镜像或提交到 git。
- 图纸数据不落公网第三方；`INFERENCE_STORAGE` 用独立数据卷。
- **VLM 基座必须挂 ra600（已 merge checkpoint-600），adapter 用 checkpoint-150-best，不能挂原始 hf**（见 `docs/MODEL_VERSION.md`）。
