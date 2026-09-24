# -*- coding: utf-8 -*-
"""Windows 版 VLM 推理后端（加速版，替换 mlx-vlm）。

与 vlm_fallback.py 接口对齐（read_box / fallback），另新增 read_all 支持
「VLM-only」全框读取（评测已证明 OCR 层冗余，见 scripts/benchmark_ocr_vlm_accel.py）。

加速手段（实测数据见 output/ocr_vlm_accel_report.json）：
  - bf16（5060 Ti 原生支持，比 fp16 更快更稳；CPU 兜底回退 fp32）
  - SDPA 注意力（attn_implementation="sdpa"，比 eager 快 ~12%）
  - merge_and_unload 挂载 LoRA（消除 adapter 层逐层开销）
  - torch.inference_mode（比 no_grad 快，且禁用 autograd）
  - max_tokens=32（答案很短，64 是浪费；实测 32 无精度损失）
  - 模型常驻缓存（模块级单例 + 双重检查锁，进程内只 load 一次）

依赖：pip install transformers>=5.7.0 accelerate peft torch(CUDA) av
模型：基座用 ra600（models/vlm/MiniCPM-V-4.6-hf-ra600，已 merge checkpoint-600 的完整模型，
  约 2.6GB，8G 显存轻松跑）。adapter_path 指向重训出的 peft LoRA
  （checkpoint-150-best：adapter_config.json + adapter_model.safetensors）。
  推理时先 crop 2×，再做 up_median 保边去噪（先 4× 放大 → 中值滤波 → 缩回 2×），
  盲测 84.6% → 92.3%。
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda"
_LOAD_LOCK = threading.Lock()
# 记录当前已加载的 model_path，避免换模型时复用旧模型
_LOADED_FOR: tuple[str, str] | None = None

DEFAULT_PROMPT = "读出图中标注的文本内容。"
DEFAULT_MAX_TOKENS = 32
# MiniCPM-V 的 scale_resolution：crop 放大后最长边不得超过该值，否则会触发 slice 切分，
# 而 transformers 5.17.0 的 MiniCPM-V 4.6 在部分尺寸下 source/slice patch 数不一致，
# 导致 vit_merger reshape 崩溃（RuntimeError: shape mismatch）。
_MAX_EDGE = 448

# 切边补偿（与 recognize.py 口径一致）：底部额外补偿，下标公差多落在框底。
_CROP_PAD = int(os.getenv("BUBBLE_CROP_PAD", "8"))
_CROP_PAD_BOTTOM = int(os.getenv("BUBBLE_CROP_PAD_BOTTOM", "4"))


def _load(model_path: Path, adapter_path: Path | None = None):
    """加载 VLM 基座到 CUDA（bf16 + SDPA），可选挂载 peft LoRA 并 merge。

    进程内只加载一次；model_path/adapter_path 变化时重新加载。
    """
    global _MODEL, _PROCESSOR, _DEVICE, _LOADED_FOR

    key = (str(model_path), str(adapter_path) if adapter_path else "")
    if _MODEL is not None and _LOADED_FOR == key:
        return _MODEL, _PROCESSOR

    with _LOAD_LOCK:
        if _MODEL is not None and _LOADED_FOR == key:
            return _MODEL, _PROCESSOR

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
        print(f"[vlm] 加载基座 {model_path}（device={_DEVICE}, dtype={dtype}, attn=sdpa）...", flush=True)

        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

        # SDPA 加速；若该模型不支持则回退 eager，保证可用性
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                device_map="auto" if _DEVICE == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="sdpa",
            ).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm] SDPA 初始化失败，回退 eager：{exc}", flush=True)
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                device_map="auto" if _DEVICE == "cuda" else None,
                trust_remote_code=True,
            ).eval()

        # 挂载 peft LoRA 并 merge（消除推理期 adapter 开销）；失败降级裸模型
        if adapter_path is not None and Path(adapter_path).is_dir():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter_path))
                model = model.merge_and_unload().eval()
                print(f"[vlm] 已挂载并合并 LoRA adapter: {adapter_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[vlm] 挂载 LoRA 失败，降级为裸模型：{exc}", flush=True)

        _MODEL = model
        _PROCESSOR = processor
        _LOADED_FOR = key
    return _MODEL, _PROCESSOR


def _crop_box(
    image: Image.Image,
    box: dict[str, Any],
    pad: int | None = None,
    preprocess: str = "up_median",
) -> Image.Image:
    if pad is None:
        pad = _CROP_PAD
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["width"])
    h = int(box["height"])
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad + _CROP_PAD_BOTTOM)
    crop = image.crop((left, top, right, bottom))
    # 放大 2x 提升小字识别；但限制最长边 ≤ _MAX_EDGE（448），避免触发 slice 切分
    # （slice 切分在部分尺寸下 patch 数不一致会崩溃，见 _MAX_EDGE 注释）。
    scale = min(2.0, _MAX_EDGE / max(crop.width, crop.height))
    new_w = max(1, round(crop.width * scale))
    new_h = max(1, round(crop.height * scale))
    crop = crop.resize((new_w, new_h), Image.LANCZOS)
    # up_median 保边去噪：先 4x 放大 → 中值滤波 → 缩回 2x。
    # 真实扫描图的噪点/锯齿是数字与竖排 Ra 误读主因，此步去噪同时保住小数点/细笔画，
    # 盲测 84.6% → 92.3%（24/26）。不加此步只有 84.6%。
    if preprocess == "up_median":
        crop = crop.resize((new_w * 2, new_h * 2), Image.LANCZOS)
        crop = crop.filter(ImageFilter.MedianFilter(3))
        crop = crop.resize((new_w, new_h), Image.LANCZOS)
    return crop


def read_box(
    image: Image.Image,
    box: dict[str, Any],
    model_path: Path,
    adapter_path: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    preprocess: str = "up_median",
) -> str:
    """用本地 VLM 读单个框内文本（加速路径）。"""
    import torch

    model, processor = _load(model_path, adapter_path)
    crop = _crop_box(image, box, preprocess=preprocess)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=crop, return_tensors="pt").to(_DEVICE)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # 只解码「新生成的 token」，避免 prompt 前缀污染答案
    input_len = inputs["input_ids"].shape[1]
    decoded = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
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


def read_all(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    model_path: Path,
    adapter_path: Path | None = None,
) -> list[dict[str, Any]]:
    """VLM-only：对全部框用 VLM 读取（不做 OCR），每个框写入 vlm_candidate。

    评测结论：OCR（30.8%）被 VLM-LoRA（80.8%）严格支配，组合反而降到 50%，
    因此识值链路应改为「YOLO 定位 → VLM 逐框读取」，跳过 OCR 层。
    """
    out: list[dict[str, Any]] = []
    for b in boxes:
        item = dict(b)
        item["vlm_candidate"] = read_box(image, item, model_path, adapter_path)
        item["ocr_text"] = ""
        item["ocr_conf"] = 0.0
        out.append(item)
    return out
