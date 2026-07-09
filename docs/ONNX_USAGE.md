# ONNX 导出与推理说明

本目录介绍如何将 OV-DEIM 目标检测模型导出为 ONNX 格式，以及如何使用导出后的 ONNX 模型进行推理。

## 目录

- [1. 概述](#1-概述)
- [2. 环境准备](#2-环境准备)
- [3. 导出 ONNX 模型](#3-导出-onnx-模型)
- [4. ONNX 模型输入输出说明](#4-onnx-模型输入输出说明)
- [5. 使用 ONNX 模型推理](#5-使用-onnx-模型推理)
  - [5.1 图像预处理](#51-图像预处理)
  - [5.2 文本特征提取](#52-文本特征提取)
  - [5.3 运行 ONNX 推理](#53-运行-onnx-推理)
  - [5.4 检测框后处理](#54-检测框后处理)
  - [5.5 完整示例](#55-完整示例)
- [6. 性能优化建议](#6-性能优化建议)
- [7. 常见问题](#7-常见问题)

---

## 1. 概述

OV-DEIM 是一个实时的开放词汇目标检测模型。导出为 ONNX 后，可以在以下运行时中使用：

- [ONNX Runtime](https://onnxruntime.ai/)（CPU / CUDA）
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [OpenVINO](https://docs.openvino.ai/)
- 其他支持 ONNX 标准的推理框架

导出脚本 `export_onnx.py` 将原始 PyTorch 模型（backbone + encoder + decoder）以及不依赖逐图像几何信息的后处理（sigmoid + top-k + gather）封装为一个 ONNX 计算图。检测框坐标还原（去除 padding / 还原缩放到原图尺寸）需要由调用方完成，具体步骤见 [第 5.4 节](#54-检测框后处理)。

## 2. 环境准备

在项目根目录的 conda 环境基础上，额外安装 ONNX 相关依赖：

```bash
pip install onnx onnxruntime onnxsim
# 如果需要使用 GPU 推理：
pip install onnxruntime-gpu
```

同时请确保：

1. 已按 `README.md` 中的 [Installation](../README.md#installation) 步骤安装了项目依赖（torch、hydra-core、albumentations 等）。
2. 已下载 DINOv3 backbone 预训练权重（如 `weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`），并且 `dinov3/` 子目录存在（backbone 通过 `torch.hub.load('./dinov3', ...)` 加载）。
3. 已下载 OV-DEIM 模型权重（如 `weights/ovdeim_l.pth`）。
4. 已准备好文本特征文件（通过 MobileCLIP-B(LT) 提取，如 `data/lvis_text_embeddings.pth`）。

## 3. 导出 ONNX 模型

### 基本用法

```bash
cd /path/to/OV-DEIM

# 导出 LVIS 开放词汇版本 (base_l 配置)
python export_onnx.py \
    --config base_l \
    --checkpoint weights/ovdeim_l.pth \
    --output weights/ovdeim_l.onnx

# 导出 COCO 版本 (coco_l 配置)
python export_onnx.py \
    --config coco_l \
    --checkpoint weights/ovdeim_coco_l.pth \
    --output weights/ovdeim_coco_l.onnx
```

### 完整参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | str | `base_l` | Hydra 配置名，可选 `base_l/m/s` 或 `coco_l/m/s` |
| `--checkpoint` | str | （必填） | OV-DEIM 权重路径（`.pth`），支持 EMA / 普通 / 裸 state_dict 格式 |
| `--output` | str | `ovdeim.onnx` | 输出 ONNX 文件路径 |
| `--opset` | int | `17` | ONNX opset 版本 |
| `--num-top-queries` | int | `300` | 每张图保留的 top-k 检测框数量 |
| `--num-texts` | int | 配置中的 `num_training_classes` | 每个样本的文本特征数量（词表大小） |
| `--batch-size` | int | `1` | 静态 batch size（`--no-dynamic-batch` 时生效） |
| `--dynamic-batch` / `--no-dynamic-batch` | flag | 启用 | 是否允许动态 batch 维度 |
| `--simplify` / `--no-simplify` | flag | 启用 | 是否运行 onnx-simplifier |
| `--verify` / `--no-verify` | flag | 启用 | 是否对比 ONNX 与 PyTorch 输出一致性 |

### 为 TensorRT 导出固定 batch 的模型

TensorRT 对动态维度支持有限，建议导出固定 batch 的模型：

```bash
python export_onnx.py \
    --config base_l \
    --checkpoint weights/ovdeim_l.pth \
    --output weights/ovdeim_l_static.onnx \
    --batch-size 1 \
    --no-dynamic-batch \
    --no-simplify
```

### 导出流程说明

导出脚本会自动完成以下步骤：

1. 根据 `--config` 构建 OVDEIM 模型（backbone + encoder + decoder）；
2. 加载 `--checkpoint` 权重（自动识别 EMA / 普通 / 裸 state_dict 格式，并跳过推理时不用的 `denoising_class_embed` 权重）；
3. 将 `SyncBatchNorm` 替换为 `BatchNorm2d`（保留 running stats），并切换到 `eval` 模式；
4. 设置 `decoder.num_enc_queries = 0`，使 decoder 在推理时返回单一 `out` 字典；
5. 用 `torch.onnx.export` 导出计算图；
6. （可选）用 onnx-simplifier 简化模型；
7. （可选）用 onnxruntime 对比 ONNX 与 PyTorch 输出，验证精度。

## 4. ONNX 模型输入输出说明

### 输入

| 名称 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `image` | float32 | `[B, 3, 640, 640]` | 预处理后的图像张量（见 [5.1](#51-图像预处理)） |
| `text_feats` | float32 | `[B, num_texts, text_dim]` | 文本特征，由 MobileCLIP-B(LT) 提取（见 [5.2](#52-文本特征提取)） |

> 当启用动态 batch 时，`B` 为动态维度。

### 输出

| 名称 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `scores` | float32 | `[B, num_top_queries]` | 置信度分数（已做 sigmoid，按降序排列） |
| `labels` | int64 | `[B, num_top_queries]` | 预测类别索引（对应 `text_feats` 中的第几条文本） |
| `boxes` | float32 | `[B, num_top_queries, 4]` | 检测框，`cxcywh` 格式，归一化到 `[0, 1]`（相对于 640×640 的 padded 输入） |

> **注意**：`boxes` 是相对于预处理后的 640×640 图像（含 padding）的归一化坐标，需要经过后处理才能还原到原图尺寸（见 [5.4](#54-检测框后处理)）。

## 5. 使用 ONNX 模型推理

### 5.1 图像预处理

预处理流程与训练/评估时一致，包含三步：

1. **KeepRatioResize**：保持长宽比缩放，使长边不超过 640；
2. **LetterResize**：将缩放后的图像用固定值（114）填充到 640×640；
3. **归一化**：先除以 255 归一化到 `[0, 1]`，再按 ImageNet 均值方差归一化。

同时需要记录缩放因子 `scale_factor = (scale_x, scale_y)` 和填充参数 `pad_param = [top, bottom, left, right]`，用于后处理还原坐标。

```python
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = (640, 640)  # (height, width)
PAD_VALUE = 114

def preprocess(image: np.ndarray):
    """Letterbox 预处理，返回 (input_tensor, scale_factor, pad_param, ori_h, ori_w)。

    Args:
        image: 原始图像，HWC, BGR (cv2 读取) 或 RGB 均可，uint8。
    Returns:
        input_tensor: float32, shape (1, 3, 640, 640)
        scale_factor: (scale_x, scale_y)
        pad_param: [top, bottom, left, right]
    """
    ori_h, ori_w = image.shape[:2]
    target_h, target_w = TARGET_SIZE

    # 1. KeepRatioResize + LetterResize 等价的 letterbox 操作
    ratio = min(target_h / ori_h, target_w / ori_w)
    new_h = int(round(ratio * ori_h))
    new_w = int(round(ratio * ori_w))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2. 计算 padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = int(round(pad_h / 2))
    left = int(round(pad_w / 2))
    bottom = pad_h - top
    right = pad_w - left
    pad_param = [top, bottom, left, right]

    # 用 PAD_VALUE 填充
    canvas = np.full((target_h, target_w, image.shape[2]), PAD_VALUE, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    # 3. 归一化: HWC uint8 -> CHW float32
    img = canvas[:, :, ::-1] if image.shape[2] == 3 else canvas  # BGR -> RGB（如果原始是 BGR）
    img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    scale_factor = (new_w / ori_w, new_h / ori_h)
    return input_tensor, scale_factor, pad_param, ori_h, ori_w
```

### 5.2 文本特征提取

OV-DEIM 使用 [MobileCLIP-B(LT)](https://github.com/apple/ml-mobileclip) 提取文本特征。每条文本对应词表中的一个类别。

#### 方式一：使用项目预提取的缓存文件

项目提供了预提取好的文本特征文件（如 `data/lvis_text_embeddings.pth`），格式为 `[num_texts, text_dim]` 的张量（`text_dim=512`）：

```python
import torch

# 加载预提取的文本特征
text_embeddings = torch.load("data/lvis_text_embeddings.pth", map_location="cpu")
# text_embeddings shape: [num_texts, 512]

# 每次推理时传入一个 batch 的文本特征
text_feats = text_embeddings.unsqueeze(0)  # [1, num_texts, 512]
```

#### 方式二：使用 MobileCLIP 自行提取

```python
import mobileclip

# 加载 MobileCLIP 模型
model, _, _ = mobileclip.create_model_and_transforms("mobileclip_blt", pretrained="checkpoints/mobileclip_blt.pt")
tokenizer = mobileclip.get_tokenizer("mobileclip_blt")
model.eval()

# 对类别名列表进行编码
texts = ["a photo of a dog", "a photo of a cat", ...]  # 你的词表
text_tokens = tokenizer(texts)
with torch.no_grad():
    text_feats = model.encode_text(text_tokens)  # [num_texts, 512]
text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # 归一化
```

> **注意**：导出 ONNX 时 `--num-texts` 应与推理时实际传入的文本数量一致。`base_l` 配置默认 `num_training_classes=150`，`coco_l` 配置默认为 COCO 类别数。

### 5.3 运行 ONNX 推理

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # 按需选择
session = ort.InferenceSession("weights/ovdeim_l.onnx", providers=providers)

# 运行推理
outputs = session.run(
    None,
    {
        "image": input_tensor.numpy(),       # [1, 3, 640, 640] float32
        "text_feats": text_feats.numpy(),    # [1, num_texts, 512] float32
    },
)
scores, labels, boxes = outputs  # 分别对应三个输出
# scores: [1, 300] float32
# labels: [1, 300] int64
# boxes:  [1, 300, 4] float32  (cxcywh, 归一化)
```

### 5.4 检测框后处理

ONNX 模型输出的 `boxes` 是 `cxcywh` 格式、归一化到 `[0, 1]`（相对于 640×640 padded 图像）。需要经过以下步骤还原到原图坐标：

1. 乘以 640 转换为像素坐标；
2. 从 `cxcywh` 转换为 `xyxy`；
3. 减去 padding（`left_pad` 对应 x，`top_pad` 对应 y）；
4. 除以缩放因子（`scale_x` 对应 x，`scale_y` 对应 y）；
5. 裁剪到原图范围。

```python
def postprocess(boxes, scores, labels, scale_factor, pad_param, ori_h, ori_w,
                conf_threshold=0.3):
    """将 ONNX 输出的归一化框还原到原图坐标。

    Args:
        boxes: np.ndarray, shape [num_top_queries, 4], cxcywh 归一化坐标
        scores: np.ndarray, shape [num_top_queries]
        labels: np.ndarray, shape [num_top_queries]
        scale_factor: (scale_x, scale_y)
        pad_param: [top, bottom, left, right]
        ori_h, ori_w: 原图高宽
        conf_threshold: 置信度阈值
    Returns:
        list of [x, y, w, h, score, label]  (xywh, 原图像素坐标)
    """
    scale_x, scale_y = scale_factor
    top, _, left, _ = pad_param
    img_size = 640

    # 过滤低置信度
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 1. 归一化坐标 -> 像素坐标
    boxes = boxes * img_size  # cxcywh, 640 空间

    # 2. cxcywh -> xyxy
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 3. 去除 padding
    x1 -= left
    y1 -= top
    x2 -= left
    y2 -= top

    # 4. 还原缩放
    x1 /= scale_x
    y1 /= scale_y
    x2 /= scale_x
    y2 /= scale_y

    # 5. 裁剪到原图范围
    x1 = np.clip(x1, 0, ori_w)
    y1 = np.clip(y1, 0, ori_h)
    x2 = np.clip(x2, 0, ori_w)
    y2 = np.clip(y2, 0, ori_h)

    # 转回 xywh
    w = x2 - x1
    h = y2 - y1

    results = []
    for i in range(len(scores)):
        if w[i] > 0 and h[i] > 0:
            results.append([x1[i], y1[i], w[i], h[i], scores[i], labels[i]])
    return results
```

### 5.5 完整示例

下面是一个完整的推理脚本示例，将上述步骤串联起来：

```python
import cv2
import numpy as np
import onnxruntime as ort
import torch

# ---- 配置 ----
ONNX_PATH = "weights/ovdeim_l.onnx"
IMAGE_PATH = "test.jpg"
TEXT_EMBEDDINGS_PATH = "data/lvis_text_embeddings.pth"  # 或自行用 MobileCLIP 提取
CONF_THRESHOLD = 0.3

# ---- 图像预处理 ----
image = cv2.imread(IMAGE_PATH)  # BGR, HWC
ori_h, ori_w = image.shape[:2]

input_tensor, scale_factor, pad_param, ori_h, ori_w = preprocess(image)

# ---- 文本特征 ----
text_embeddings = torch.load(TEXT_EMBEDDINGS_PATH, map_location="cpu")
if text_embeddings.dim() == 2:
    text_feats = text_embeddings.unsqueeze(0)  # [1, num_texts, text_dim]
else:
    text_feats = text_embeddings

# ---- ONNX 推理 ----
session = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
scores, labels, boxes = session.run(None, {
    "image": input_tensor.numpy(),
    "text_feats": text_feats.numpy(),
})
scores = scores[0]   # [num_top_queries]
labels = labels[0]    # [num_top_queries]
boxes = boxes[0]     # [num_top_queries, 4]

# ---- 后处理 ----
results = postprocess(boxes, scores, labels, scale_factor, pad_param, ori_h, ori_w,
                      conf_threshold=CONF_THRESHOLD)

# ---- 可视化 ----
for x, y, w, h, score, label in results:
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label}: {score:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("result.jpg", image)
print(f"检测到 {len(results)} 个目标，结果已保存到 result.jpg")
```

## 6. 性能优化建议

### 使用 TensorRT

对于 NVIDIA GPU 部署，建议将 ONNX 转换为 TensorRT 引擎以获得最佳性能：

```bash
# 固定 batch 导出（推荐用于 TensorRT）
python export_onnx.py --config base_l --checkpoint weights/ovdeim_l.pth \
    --output weights/ovdeim_l_trt.onnx --batch-size 1 --no-dynamic-batch --no-simplify

# 转换为 TensorRT 引擎
trtexec --onnx=weights/ovdeim_l_trt.onnx --saveEngine=weights/ovdeim_l.engine --fp16
```

### 使用 onnxruntime-gpu

```python
session = ort.InferenceSession(
    "weights/ovdeim_l.onnx",
    providers=[
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
        }),
        "CPUExecutionProvider",
    ],
)
```

### 批量推理

如果启用了动态 batch，可以一次处理多张图片：

```python
# 假设已预处理得到 batch_images: [B, 3, 640, 640], batch_text_feats: [B, num_texts, 512]
scores, labels, boxes = session.run(None, {
    "image": batch_images.numpy(),
    "text_feats": batch_text_feats.numpy(),
})
```

## 7. 常见问题

### Q1: 导出时报错 `RuntimeError: Exporting the operator ... to ONNX opset ...`

部分算子需要较高的 opset 版本。可以尝试提升 opset 版本：

```bash
python export_onnx.py --config base_l --checkpoint weights/ovdeim_l.pth --opset 18 ...
```

### Q2: onnx-simplifier 简化失败

简化失败不会影响模型的正确性，导出脚本会保留未简化的原模型。如果简化失败，可以：

- 检查 ONNX 模型是否包含不支持的算子；
- 使用 `--no-simplify` 跳过简化步骤；
- 升级 `onnx` 和 `onnxsim` 版本。

### Q3: 验证时 scores/labels 一致但 boxes 差异较大

确保 PyTorch 和 ONNX 都在 CPU 上运行对比（`export_onnx.py` 的验证默认使用 CPU）。如果使用了不同的 ONNX Runtime provider，浮点精度可能有细微差异，通常在 `1e-3` 以内属于正常范围。

### Q4: 如何使用不同的词表（类别集）

OV-DEIM 是开放词汇检测器，词表通过 `text_feats` 输入动态指定。只需将你需要的类别名通过 MobileCLIP 编码为文本特征，作为 `text_feats` 传入即可。**不需要重新导出 ONNX 模型**，只要 `text_feats` 的 `text_dim` 维度（512）与导出时一致。

### Q5: `num_enc_queries`（Fixed AP）功能是否支持

当前导出脚本设置 `num_enc_queries = 0`（标准 AP 推理模式）。Fixed AP 模式（`num_enc_queries > 0`）会让 decoder 额外返回 encoder 预测的检测框，需要额外的拼接逻辑，目前未包含在 ONNX 导出中。如需 Fixed AP，建议直接使用原始 PyTorch 推理脚本 `test_lvis.py` 中的 `eval_fixed_ap` 函数。

### Q6: SyncBatchNorm 转换后结果是否正确

导出脚本将 `SyncBatchNorm` 替换为 `BatchNorm2d`，并完整复制了 `weight`、`bias`、`running_mean`、`running_var` 等参数。在 `eval` 模式下，两者前向计算完全一致（都使用 running stats），不影响推理精度。导出脚本会在最后自动对比 ONNX 与 PyTorch 的输出以验证正确性。
