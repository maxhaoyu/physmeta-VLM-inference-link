# VLM 模型版本与部署红线

> ⚠️ **重要**：部署推理节点时，VLM 基座**必须**挂 `MiniCPM-V-4.6-hf-ra600`，
> **不能**挂原始 `MiniCPM-V-4.6-hf`。否则 `checkpoint-150-best` 的 LoRA 无法对齐，识别结果错误或加载失败。

## 版本链

```
原始基座 openbmb/MiniCPM-V-4.6 (1.3B)
   │  训练 + checkpoint-600（LoRA 增量）
   │  merge 进基座
   ▼
models/vlm/MiniCPM-V-4.6-hf-ra600   ← 部署基座（已含 checkpoint-600 增量，约 2.6GB）
   │  继续训练 + checkpoint-150-best（新 LoRA）
   ▼
models/adapters/checkpoint-150-best  ← 部署 adapter（20MB，4 文件）
   = 最终推理模型（ra600 基座 + checkpoint-150-best adapter）
```

## 部署配置（节点）

```json
{
  "vlm_model": "C:\\PhysMetaInference\\bubble-ocr-app\\models\\vlm\\MiniCPM-V-4.6-hf-ra600",
  "vlm_adapter": "C:\\PhysMetaInference\\bubble-ocr-app\\models\\adapters\\checkpoint-150-best",
  "use_ocr": false
}
```

## 红线

1. **基座必须是 ra600**。`checkpoint-150-best` 的 `adapter_config.json` 里
   `base_model_name_or_path` 明确指向 `MiniCPM-V-4.6-hf-ra600`，挂原始 hf 会错位。
2. adapter 目录需含 `adapter_config.json` + `adapter_model.safetensors`（peft 挂载必需）。
   `additional_config.json` / `README.md` 是 ms-swift 附加物，可保留。

## 模型文件分发（不进 git）

- 模型大文件不入 git（`.gitignore` 排除 `*.safetensors`、`models/vlm/`、`models/adapters/`）。
- `MiniCPM-V-4.6-hf-ra600`（约 2.6GB）与 `checkpoint-150-best`（20MB）走内网盘 / 服务器独立分发。
- Git 只同步**配置与文档**：本文件、`node/inference-node.example.json`、`DEPLOY.md`。
