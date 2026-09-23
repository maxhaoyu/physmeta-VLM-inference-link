# -*- coding: utf-8 -*-
"""Windows 版 VLM 兜底后端（替换 mlx-vlm，跑在 Windows NVIDIA CUDA 上）。

为什么需要这个文件：
  - 原 vlm_fallback.py 用 mlx-vlm（Apple Silicon 专属），Windows 装不了。
  - 本文件用 HuggingFace transformers + CUDA，让 VLM 能在客户 Windows 8G 显卡上本地跑，
    守住「图纸不出内网」红线。

接口对齐 vlm_fallback.py：
  - read_box(image, box, model_path, adapter_path, prompt, max_tokens) -> str
  - fallback(image, boxes, model_path, adapter_path, threshold) -> list[dict]

用法：在 pipeline 里按平台选择 import（Windows 用本文件，macOS 用 vlm_fallback.py）。
两者接口一致，其余代码无需改动。

依赖：pip install transformers>=5.7.0 accelerate torch（CUDA 版）peft av
模型：基座用 openbmb/MiniCPM-V-4.6（1.3B，FP16 约 2.8GB，8G 显存轻松跑）。
  adapter_path 指向重训出的 peft LoRA（adapter_config.json + adapter_model.safetensors）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda"
_LOAD_LOCK = None


def _load(model_path: Path, adapter_path: Path | None = None):
    """加载 VLM 基座到 CUDA，可选挂载 peft LoRA（adapter_path 存在才挂）。"""
    global _MODEL, _PROCESSOR, _LOAD_LOCK
    if _LOAD_LOCK is None:
        import threading
        _LOAD_LOCK = threading.Lock()
    if _MODEL is None:
        with _LOAD_LOCK:
            if _MODEL is None:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

                # trust_remote_code=True：MiniCPM-V 需要自定义建模代码
                model = AutoModelForImageTextToText.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto" if _DEVICE == "cuda" else None,
                ).eval()

                # 挂载 peft LoRA（Windows 重训产物）。挂不上则告警降级为裸模型。
                if adapter_path is not None and Path(adapter_path).is_dir():
                    try:
                        from peft import PeftModel

                        model = PeftModel.from_pretrained(model, str(adapter_path))
                        model = model.merge_and_unload().eval()
                        print(f"[vlm] 已挂载 LoRA adapter: {adapter_path}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[vlm] 挂载 LoRA 失败，降级为裸模型：{exc}", flush=True)

                _MODEL = model
                _PROCESSOR = AutoProcessor.from_pretrained(
                    str(model_path), trust_remote_code=True
                )
    return _MODEL, _PROCESSOR


def _crop_box(image: Image.Image, box: dict[str, Any], pad: int = 12) -> Image.Image:
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["width"])
    h = int(box["height"])
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad)
    crop = image.crop((left, top, right, bottom))
    return crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)


def read_box(
    image: Image.Image,
    box: dict[str, Any],
    model_path: Path,
    adapter_path: Path | None = None,
    prompt: str = "读出图中标注的文本内容。",
    max_tokens: int = 64,
) -> str:
    """用本地 VLM 读单个框内文本。"""
    import torch

    model, processor = _load(model_path, adapter_path)
    crop = _crop_box(image, box)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=crop, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # 只解码「新生成的 token」，不解码输入 prompt，避免 prompt 前缀污染答案
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()
    # 去掉思考链包装与残余特殊 token，只留实际答案
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL)
    decoded = re.sub(r"<[^>]+>", "", decoded).strip()
    return decoded


def fallback(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    model_path: Path,
    adapter_path: Path | None = None,
    threshold: float = 0.82,
) -> list[dict[str, Any]]:
    """对低置信度框跑 VLM 兜底，接口与 vlm_fallback.fallback 一致。"""
    out: list[dict[str, Any]] = []
    for b in boxes:
        item = dict(b)
        if item.get("ocr_conf", 0.0) < threshold:
            item["vlm_candidate"] = read_box(image, item, model_path, adapter_path)
        else:
            item["vlm_candidate"] = None
        out.append(item)
    return out
