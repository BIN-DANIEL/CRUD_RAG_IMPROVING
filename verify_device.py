"""
验证脚本：确认当前环境是否 fallback 到 CPU
不依赖正在运行的建库进程，独立执行。

目的：
1. 打印 torch 的版本与 GPU 能力（cuda / musa）
2. 实际加载 bge 嵌入模型，看它的参数落在哪个设备
3. 推断 device_map="auto" 会解析到什么
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线，避免联网
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

print("=" * 60)
print("[1] torch 基本信息")
print("=" * 60)
print(f"torch 版本      : {torch.__version__}")
print(f"cuda.is_available: {torch.cuda.is_available()}")
print(f"cuda.device_count: {torch.cuda.device_count()}")
print(f"hasattr torch.musa: {hasattr(torch, 'musa')}")

# 尝试 import torch_musa
try:
    import torch_musa
    print(f"torch_musa 已安装: {torch_musa.__version__}")
    print(f"torch.musa.is_available(): {torch.musa.is_available()}")
except ImportError:
    print("torch_musa 未安装（无法使用 MUSA GPU）")

print()
print("=" * 60)
print("[2] transformers 的 device_map 解析")
print("=" * 60)
try:
    from accelerate.utils import get_balanced_memory, infer_auto_device_map
    # 用一个已缓存的模型做推断（只做 device_map 解析，不真正加载权重）
    try:
        device_map = infer_auto_device_map(None, max_memory=None)
        print("infer_auto_device_map(None) 结果:", device_map)
    except Exception as e:
        print("infer_auto_device_map 异常:", repr(e))
except ImportError as e:
    print("accelerate 导入失败:", e)

print()
print("=" * 60)
print("[3] 实际加载 bge 嵌入模型，看设备")
print("=" * 60)
from transformers import AutoModel, AutoConfig

model_name = "BAAI/bge-base-zh-v1.5"
try:
    config = AutoConfig.from_pretrained(model_name)
    # 用 device_map="auto" 加载，看它落到哪
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    # 打印第一个参数的设备
    first_param = next(model.parameters())
    print(f"模型参数设备: {first_param.device}")
    print(f"模型 device: {model.device if hasattr(model, 'device') else 'N/A'}")
    if first_param.device.type == "cpu":
        print(">>> 结论：device_map='auto' 落到了 CPU")
    elif first_param.device.type == "cuda":
        print(">>> 结论：落到了 CUDA GPU")
    elif first_param.device.type == "musa":
        print(">>> 结论：落到了 MUSA GPU")
    else:
        print(f">>> 结论：落到了未知设备 {first_param.device.type}")
except Exception as e:
    print("加载失败:", repr(e))
