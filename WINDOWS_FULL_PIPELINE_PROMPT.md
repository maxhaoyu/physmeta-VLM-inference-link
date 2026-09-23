# Windows 全流程 Prompt：从 VLM 训练一路跑到线上 yolobubble 用 Windows 推理

> 用法：在 Windows 机器 WorkBuddy 里新建会话，把下面「任务正文」整段粘贴，按顺序执行。
> 目标：让线上 physmeta.cn 的 yolobubble（气泡识值）用这台 Windows 的 GPU 做整链路推理（YOLO + OCR + VLM）。

---

## 任务正文

你是负责把「气泡识值（yolobubble）VLM」从训练一路推到「线上用 Windows GPU 推理」的工程师。当前你已经在本机跑 VLM 重训练（MiniCPM-V-4.6 的 peft LoRA）。本任务从你当前的训练进度继续，一直做到推理节点上线可领任务为止。

### 关键决策（先明确，别绕弯）

**YOLO + OCR + VLM 三层全部放在 Windows 本地跑。** 理由：
- YOLO（ultralytics）和 OCR（rapidocr_onnxruntime）本来就是跨平台 pip 包，Windows 直接装就能跑，零改造。
- 如果把 VLM 单独拆出去（粒度②），反而要拆分 pipeline、加中间的图/框传输，更慢更复杂。
- 整链路本地跑最干净，也最守「图纸不出内网」红线。

所以你最终交付的是一个**在 Windows 上跑完整识值链路、主动向服务器领任务的推理节点**。

### 你要完成的四件事（按顺序）

#### 第 1 步：先把 VLM 重训练跑完

如果 VLM 训练还没跑完，先按 `bubble-ocr-app/docs/Windows重训练说明.md` 把训练跑完：
```
python scripts\train_swift_windows.py --quantization_bit 4
```
训练完成后验证：
```
python scripts\eval_swift_lora_windows.py --adapter output\swift-train3\vx-xxx\checkpoint-xxx
```
记录盲测集数值等价准确率（对比裸模型基线 65.4%，目标逼近 88.5%）。
产出是 `checkpoint-xxx` 目录（`adapter_config.json` + `adapter_model.safetensors`）。

#### 第 2 步：拉最新代码（含我这次修的 Windows 支持）

你的两个代码源：
- 识值主代码仓库：`git@github.com:maxhaoyu/bubble-ocr-mvp.git`（或 HTTPS `https://github.com/maxhaoyu/bubble-ocr-mvp.git`）
- 推理节点仓库（新增）：`git@github.com:maxhaoyu/physmeta-inference-link.git`

在项目目录执行（二选一，已 clone 过就 pull）：
```
git pull origin main
```
或首次：
```
git clone git@github.com:maxhaoyu/bubble-ocr-mvp.git
git clone git@github.com:maxhaoyu/physmeta-inference-link.git
```

> ⚠️ 若 SSH 报权限错，改 HTTPS：`git remote set-url origin https://github.com/maxhaoyu/<repo>.git`；
> 若 HTTPS 需要凭据，向用户索要 PAT token。GitHub 被墙时用 codeload tarball：
> `https://codeload.github.com/maxhaoyu/<repo>/tar.gz/refs/heads/main`。

**重要**：我这次已经改了 `bubble-ocr-mvp` 的 `src/bubble_ocr/pipeline.py`，让它按平台自动选 VLM 后端（Windows 用 `vlm_fallback_windows.py`，macOS 用 mlx 版）。所以你**不再需要手动覆盖 vlm_fallback.py**——拉最新代码后，Windows 上 `pipeline.run` 会自动走 transformers 版 VLM。

#### 第 3 步：搭推理节点（Windows 本地）

按 `physmeta-inference-link/docs/节点安装说明.md` 操作，核心是：

1. 目录准备 `C:\PhysMetaInference\`，放入：
   - `agent.py`、`vlm_fallback_windows.py`、`requirements-windows.txt`（来自 `physmeta-inference-link/node/`）
   - `bubble-ocr-app\`（整个识值代码，含 `models\yolo\cv88-baseline-epoch11.pt`、`models\vlm\` 的 MiniCPM-V-4.6、`models\adapters\` 的第 1 步 LoRA）
2. 装依赖：
   ```powershell
   python -m venv C:\PhysMetaInference\.venv
   C:\PhysMetaInference\.venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r C:\PhysMetaInference\requirements-windows.txt
   ```
3. 配 `inference-node.json`（复制 example，填 `server`、`token`、`bubble_src`、`vlm_adapter` 指向第 1 步的 checkpoint）。
4. 先 `--once` 手动验证一次，再配开机自启（任务计划）。

#### 第 4 步：打通服务器端 + 上传入口（需要和 Mac 端配合）

这一步分「代码」和「部署」两半：

- **代码（Mac 端已完成方案，你可能要协助落地）**：
  - 推理队列服务 `physmeta-inference-link/server/server.py` 要部署上线（标准库，Dockerfile 已备好）。
  - bubble-ocr 的 `/upload` 要加「转发给队列」那一步（见 `physmeta-inference-link/docs/上传入口改造方案.md` 做法 A）。
  - 前端改成轮询 `job_id`。
- **部署**：把 `inference-queue` 挂进 compose，配 `INFERENCE_NODE_TOKEN`（这个 token 会下发给 Windows 节点填进 `inference-node.json`）。

如果你在 Windows 端拿不到服务器部署权限，就把「节点端已就绪、等服务器 token」这个状态回传，让 Mac 端完成服务器部署 + 下发 token。

### 验收标准（做到什么算完成）

1. Windows 节点 `agent.py --once` 能成功领到一个任务、跑完 YOLO+OCR+VLM、回传 result.zip。
2. 线上 yolobubble 上传一张图纸 → 状态从 pending 走到 done → 前端能渲染识别结果。
3. VLM 走的是 Windows 重训的 LoRA（不是裸模型），盲测准确率 ≥ 80%。

### 红线（必须遵守）

- VLM 只读「已定位框内文本」，不做检测；VLM 输出只是候选，与 OCR 仲裁，不直接写质检表。
- 图纸数据全程不出客户内网（只主动出站领任务/回传结果，不开放入站端口）。

### 回传内容

1. VLM 训练结果：盲测准确率 + checkpoint 路径。
2. 节点验证：`agent.py --once` 的完整日志 + 是否成功回传。
3. 卡住的环节：具体报错原文 + 你已尝试的解决方式。

请从你当前的训练进度开始，先报告第 1 步（训练）的状态，然后继续往下推。
