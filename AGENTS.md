# AGENTS.md — physmeta-VLM-inference-link（模块级）

## 全局速览（自举必读）

PhysMeta 是面向工程图纸（钣金 / DXF 气泡）的 AI 识别与标注 SaaS 平台，生产地址 `https://physmeta.cn`。10 个独立 git 仓库分三层：

| 层 | 仓库 | 职责 |
|----|------|------|
| 平台层 | platform / portal / identity | 编排、官网登录、账号授权 |
| 业务层 | annotationsheet / dxf / 2dcomparison | 钣金标注、DXF 气泡、图纸比对 |
| 算法层 | ocr-mvp / vlm-training / yolo-enhancements / VLM-inference-link | YOLO 检测+OCR、VLM 训练、YOLO 优化交接、本地推理链路 |

核心业务链：**上传图纸 → YOLO 检测气泡 → OCR/VLM 识值 → 结构化结果 / 质检 Excel**。

**三条铁律**：① 图纸不出内网（推理在客户本地 GPU 跑，服务器永不主动连内网）；② VLM 不当检测器（定位靠 YOLO，VLM 只候选留痕，不一致标 `needs_review` 转人工）；③ 密钥/客户图纸/模型权重不入 git（私密配置放仓库外，大权重放 Release）。

**权威链路**：全局唯一总导航在 `physmeta-platform/AGENTS.md`；本仓库职责/红线/验收见下文；线上真相读 `physmeta-platform/deploy-cn/STATUS.md`。任何 agent 不得凭猜测或旧对话推断，以这三个文件最新内容为准。

---


## 一、职责

Windows 本地推理链路（队列 + 节点）。线上路径 `yolobubble.physmeta.cn /api/inference/*`。

负责推理的调度和通信：推理队列 `inference-queue` 任务中转，Windows 节点（客户 GPU）
主动出站领单。**不产框**——真正「画框」的 YOLO 模型在节点上跑，框选算法逻辑归属 `physmeta-ocr-mvp`。

## 二、边界红线（不可违反）

- **图纸不出内网**：服务器永不主动连客户内网，节点主动出站领单。
- **Windows 节点只跑 node/agent.py 做推理，不写代码、不改代码**。
- **密钥/客户图纸/模型权重不入 git**（铁律第三条）。
- 本模块管「链路/调度」，不管「框选算法」——框选逻辑在 ocr-mvp，勿越界。

## 三、验收命令

> TODO：clone 完成后按本仓库真实脚本补全。

## 四、关联模块

- 框选算法：`physmeta-ocr-mvp`（YOLO 检测）
- 识值训练：`physmeta-ocr-vlm-training`
- 编排：`physmeta-platform`

## 五、指回总导航

全局权威导航在 `physmeta-platform/AGENTS.md`；线上状态在 `physmeta-platform/deploy-cn/STATUS.md`。
