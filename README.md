# PhysMeta 最小化推理链路（单节点 · Windows 整链路推理 · 含 VLM）

> 基于老方案（`06_BAT工具/01_PDF_Inference_Node`）的代码级复用 + 大幅精简。
> 目标：客户在 **bubble-ocr 上传入口**上传图纸 → 服务器下发 → 客户 Windows 本地跑完整识值链路（YOLO + OCR + VLM）→ 回传。
> 核心诉求：**VLM 兜底在客户 Windows NVIDIA GPU 上本地跑**，守住「图纸不出内网」红线（`mlx-vlm` 只支持 macOS，Windows 需换 transformers CUDA 后端）。

---

## 一、核心机制（复用老方案的精华，6 件事）

老方案最值钱的是「**Windows 主动出站**」——节点主动 HTTPS 连服务器，天然穿透客户防火墙/NAT，服务器永不主动连进客户内网。

精简版只保留 6 件事：

1. **主动出站 + 长轮询领单**：节点循环 `POST /claim`，有任务就发，没任务挂起等待。
2. **Bearer token 认证**：节点身份 token，每次请求带上。
3. **下载校验**：`.part` 临时文件 + 字节数 + SHA-256 双重校验。
4. **本地整链路推理**：调 bubble-ocr 的 `pipeline.run`（YOLO 检测 + OCR + VLM + 仲裁）。
5. **回传校验**：结果打包 zip + `X-Artifact-SHA256` 回执校验。
6. **失败回退**：节点失败 → `POST /fail`，服务器回退云端 Worker。

## 二、砍掉了什么（对单节点推理都是负担）

| 砍掉 | 原用途 | 为何不需要 |
|---|---|---|
| Resident 常驻进程 | 模型常驻内存、多任务复用 | 单节点，agent 本身常驻即可复用模型 |
| GPU 门禁（nvidia-smi） | 防止和训练抢 GPU | Windows 只做推理，无训练 |
| 训练暂停文件（TRAINING_ACTIVE） | 训练排空时暂停 | 无训练 |
| YOLO 双模型 + 自适应路由 + 分块 | 复杂图纸增强 | 单模型 + 固定参数够用 |
| 十几个 OCR 引擎/候选路由 | 多引擎投票、稀疏路由 | 单引擎 OCR 够用 |
| PDF 矢量快速路由 | 特殊 PDF 优化 | 一般图纸用不上 |
| zip 解压路径穿越防御（200行） | 防恶意数据集 | 客户图纸是自己人传的 |
| 训练节点整套（SSH 配对、配对码） | 局域网训模型 | 客户场景用不上 |

## 三、与老方案的关键差异：推理单元变了

| | 老方案 | 本方案 |
|---|---|---|
| 推理入口 | `bubble_inspection.run_pipeline`（老算法） | **`bubble_ocr.pipeline.run`**（bubble-ocr-app，YOLO cv88 + RapidOCR + VLM） |
| VLM 后端 | mlx-vlm（macOS 专属） | **transformers + CUDA**（Windows 可跑，见 `node/vlm_fallback_windows.py`） |
| 上传入口 | 老 SaaS 上传 | **bubble-ocr 的 `POST /upload`**（已有 SSO 鉴权） |

## 四、接口契约（服务器端 5 个接口 + 1 张表）

### 节点端 → 服务器端

| 接口 | 方法 | 作用 |
|---|---|---|
| `/api/inference/claim` | POST | 领任务（长轮询） |
| `/api/inference/jobs/{id}/input` | GET | 下载图纸（`X-Inference-Run-Token`） |
| `/api/inference/jobs/{id}/progress` | POST | 回传进度（可选） |
| `/api/inference/jobs/{id}/result` | POST | 回传结果 zip（`X-Artifact-SHA256`） |
| `/api/inference/jobs/{id}/fail` | POST | 失败回退云端 |

### 上传入口（客户浏览器）

**直接复用 bubble-ocr-app 已有的 `POST /upload`**（`src/bubble_ocr/server.py` 里已实现，含 SSO 鉴权 + 图片安全校验）。改动点只有一处：上传后不再本地同步跑 pipeline，而是**把图纸交给服务器任务队列**（调用 `POST /api/inference/upload` 或直接写 `inference_jobs` 表），由节点认领。

### 认证头

```
Authorization: Bearer <node_token>
X-Inference-Node-ID: <node_id>
X-Inference-Run-Token: <run_token>
```

### 数据库一张表

```sql
inference_jobs(
  id           TEXT PRIMARY KEY,   -- cad-20260923-132500-abc123
  user_id      TEXT,
  input_path   TEXT,
  input_sha256 TEXT,
  input_bytes  INTEGER,
  filename     TEXT,
  options      TEXT,               -- 透传 pipeline 参数（JSON）
  status       TEXT,               -- pending/claimed/processing/done/failed
  node_id      TEXT,
  run_token    TEXT,
  result_path  TEXT,
  error        TEXT,
  created_at   TIMESTAMP,
  updated_at   TIMESTAMP
)
```

## 五、目录结构

```
PhysMeta-最小化推理链路/
├── README.md                        ← 本文件
├── node/
│   ├── agent.py                     ← 精简版节点端（调 bubble_ocr.pipeline.run）
│   ├── inference-node.example.json  ← 节点配置
│   ├── vlm_fallback_windows.py      ← Windows 版 VLM 后端（transformers 替换 mlx）
│   └── requirements-windows.txt     ← Windows 依赖（CUDA 版 torch + transformers）
├── server/
│   ├── schema.sql                   ← 建表 SQL（文档性质，server.py 内联建表）
│   ├── server.py                    ← 标准库实现（http.server + sqlite3，零第三方依赖）
│   └── requirements.txt             ← 服务端依赖（为空，纯标准库）
└── docs/
    └── 节点安装说明.md               ← Windows 节点安装 + VLM 后端切换 + 开机自启
```

## 六、VLM 微调（已完成决策）

macOS 上的 train3 LoRA 是 **mlx 格式**，Windows 的 transformers 读不了（mlx 的 `linear_attn.*` 层 peft 无法映射）。因此：

- **方案已定**：在 Windows 上用 **ms-swift 重训** peft 格式 LoRA（基座 `openbmb/MiniCPM-V-4.6`）。
- `node/vlm_fallback_windows.py` 已支持通过 `adapter_path` 挂载 peft LoRA，挂载失败自动降级为裸模型。
- 训练数据 6000 条（4500 真实裁剪 + 1500 合成符号图，448×448）已随 `bubble-ocr-app` 仓库推送，Windows 端 `git pull` 即可获取。

详见 `docs/节点安装说明.md` 第四节。

## 七、技术栈已确认（与你平台一致）

查清了 physmeta.cn 的实际技术栈，**整个平台统一是「原生标准库 + http.server + sqlite3」，无 Web 框架**：

| 模块 | 语言 | 框架 |
|---|---|---|
| portal | Node.js | 原生 `http` |
| identity-service | Python | 原生 `http.server` |
| dxf-bubble-app | Python | 原生 `http.server` |
| bubble-ocr-app | Python | 原生 `http.server` |

因此 `server/server.py` 已改为**同款标准库实现**（`http.server` + `sqlite3`，零第三方依赖），配 Dockerfile 可直接挂进 compose。

## 八、上传入口改造（推荐做法 A）

上传入口**复用 bubble-ocr 现有的 `POST /upload`**（含 SSO + 图片校验），只在存图后**转发**给独立的推理队列服务（`inference-queue`），由 Windows 节点认领。详见 `docs/上传入口改造方案.md`。

一句话结论：新增一个独立 `inference-queue` 服务（就是本方案 `server/`），bubble-ocr 上传后转发给它，前端改轮询 job_id。这是最贴合你模块化架构的做法。
