# Models

本目录只保留模型说明，不存放模型权重。程序不会从 `models/` 读取正式资源；
实际资源由项目脚本导入到顶层 `resource-library/`，OCR 原始归档则从
`ocr-model-archives/` 读取。这些目录都不应提交到 Git。

## 本地资源类型

| 类型 | 正式位置 | 说明 |
| --- | --- | --- |
| Caption Tagger | `resource-library/tagging-models/` | E621/Danbooru 图片标签模型 |
| Classification Index | `resource-library/classification-indexes/` | 分类、count 与词表索引 |
| Replacement Index | `resource-library/replacement-indexes/` | E621 替换规则 |
| Policy Model | `resource-library/dropout-models/` | 可选质量评分模型 |
| Tokenizer | `resource-library/tokenizers/` | Token Budget 精确计数 |
| OCR Model | `resource-library/ocr-models/` | 用户提供并验证后的 PaddleOCR 资源 |

资源包必须带有项目要求的 manifest、文件哈希和兼容性元数据。不要只复制单个
权重文件，也不要绕过导入脚本的预览和校验。

## OCR 手动归档

OCR is disabled by default。`Install-WebUI.bat` 只安装基础 WebUI，不安装 OCR runtime
或模型。请按项目根 [OCR_MODEL_DOWNLOAD.md](../OCR_MODEL_DOWNLOAD.md) 中固定的三个官方
URL、文件名、大小和 SHA-256，将未解压归档放入 `ocr-model-archives`，再使用
`Install-Ocr.bat` 安装并验证 CPU runtime；需要 CUDA 时，在 CPU fallback 完整后使用
`Install-OcrGpu.bat`。

可选入口会验证归档、在项目内 staging 运行 offline CPU OCR probe，并只在验证通过后发布
`resource-library/ocr-models/`。没有通过该验证时，基础 WebUI 仍可用，only OCR-enabled
jobs are blocked。`Install-WebUI.bat` 没有 `-OcrMode` 参数；OCR 归档、正式 OCR 资源和
runtime 都不应提交到 Git。

## 上游来源

- CL Tagger v2：<https://huggingface.co/cella110n/cl_tagger_v2/tree/main/v2_00>
- WD EVA02-Large v3：<https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3>
- Qwen3 0.6B tokenizer：<https://huggingface.co/Qwen/Qwen3-0.6B>
- Qwen3-VL 4B tokenizer：<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- PaddleOCR 模型列表：<https://www.paddleocr.ai/latest/en/version3.x/model_list.html>

模型与 tokenizer 受各自许可证、访问限制和使用条款约束。本项目不授予其权重的
再分发权限；下载、安装和使用前应自行核对上游条款。
