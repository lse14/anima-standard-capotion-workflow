# Third-Party Notices

Anima 源码依赖第三方开源软件，并可连接或加载第三方模型资源。各组件仍受其
各自许可证、版权声明和使用条款约束。本文件用于定位上游声明，不替代上游完整
许可证文本，也不改变 [项目许可证](../LICENSE)。

本源码仓库不包含 Python/Node 运行时、浏览器二进制、模型权重、tokenizer
文件、OCR 模型或数据集。`resource-library/classification-indexes/` 和
`resource-library/replacement-indexes/` 中明确列出的 E621 索引是下述例外。

## 前端与开发工具

精确版本记录在 `frontend/package-lock.json`，其中包含 npm 包的 license 元数据。
主要直接依赖包括：

| 组件 | 上游许可证/项目 |
| --- | --- |
| React / React DOM | <https://github.com/facebook/react/blob/main/LICENSE> |
| Vite | <https://github.com/vitejs/vite/blob/main/LICENSE> |
| TypeScript | <https://github.com/microsoft/TypeScript/blob/main/LICENSE.txt> |
| Playwright | <https://github.com/microsoft/playwright/blob/main/LICENSE> |

## Python 直接依赖

精确版本记录在 `packaging/requirements/*.in` 和对应 lock 文件。主要组件的上游
许可证入口如下：

| 组件 | 上游许可证/项目 |
| --- | --- |
| FastAPI | <https://github.com/fastapi/fastapi/blob/master/LICENSE> |
| Uvicorn | <https://github.com/encode/uvicorn/blob/master/LICENSE.md> |
| Pillow | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| NumPy | <https://github.com/numpy/numpy/blob/main/LICENSE.txt> |
| ONNX Runtime | <https://github.com/microsoft/onnxruntime/blob/main/LICENSE> |
| HTTPX | <https://github.com/encode/httpx/blob/master/LICENSE.md> |
| PyTorch | <https://github.com/pytorch/pytorch/blob/main/LICENSE> |
| torchvision | <https://github.com/pytorch/vision/blob/main/LICENSE> |
| OpenCLIP | <https://github.com/mlfoundations/open_clip/blob/main/LICENSE> |
| safetensors | <https://github.com/huggingface/safetensors/blob/main/LICENSE> |
| tokenizers | <https://github.com/huggingface/tokenizers/blob/main/LICENSE> |
| PaddlePaddle | <https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE> |
| PaddleOCR | <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE> |
| PaddleX | <https://github.com/PaddlePaddle/PaddleX> |

传递依赖也适用各自许可证。生成或分发组装包时，应同时审查所选 lock 文件、wheel
元数据及随包许可证文件。

## 随源码提交的 E621 索引

本仓库随源码提交
`resource-library/classification-indexes/e621-classify-20260724-v1` 的三个文件，
用于本地 E621 标签分类与 Count 规则；还提交
`resource-library/replacement-indexes/e621-replace-20260726-v2` 的 CSV、资源清单和
本地构建说明书，用于本地 E621 标签规范化。它们不是模型权重或 tokenizer。分类版本的
`resource.json` 固定如下文件身份：

| 文件 | 大小（bytes） | SHA-256 |
| --- | ---: | --- |
| `e621_tag_dictionary.json` | 37,569,404 | `87c42e0021ea637bc93195c6d37ac4f8b967dd989a8bd5de4b7ebb7546264e59` |
| `e621_count_wiki.sqlite3` | 73,728 | `6aa0f944f07de490413aa49bf59d6ead555b6eeaad2e00022dedd6109d0abff9` |

替换版本的 `resource.json` 固定其 CSV 文件身份：

| 文件 | 大小（bytes） | SHA-256 |
| --- | ---: | --- |
| `e621_tag_replacement_index.csv` | 3,902,020 | `24ad8388580a6c3628dec44bd813897c278e4f1b04fccd810f22acaf97c1cbe7` |

字典元数据标记其来源为 E621，创建时间为 `2026-07-23T15:57:10+00:00`，包含
120,978 个标签/别名条目。Count 数据库标记为
`e621-wiki-count-20260724-v1`，包含 21 条 Wiki 投影。来源和适用条款入口为：

- <https://e621.net/help/api>
- <https://e621.net/terms_of_use>

2026-08-12 核对时，E621 Terms of Use 第 4 节将网站内容置于其权利人保护下，
并限制复制、分发、修改和再发布。2026-08-13，项目负责人通过
`user-confirmed-project-owner` 决定允许项目随源码/GitHub 分发当前
`e621-classify-20260724-v1` 和 `e621-replace-20260726-v2` 的六个账本绑定文件；
该决定、Terms URL、Terms 响应 SHA-256 与全部文件大小/SHA-256 记录在
`packaging/installer/license-ledger.json`。这是项目分发决定，不是 E621 上游
授予的法律许可，也不替代对上述条款或适用法律的独立核对。

替换 CSV 的随附说明书记录其生成输入为 E621 tags、aliases、implications 和 Wiki
数据导出，官方导出 URL 模板为
`https://static1.e621.net/data/db_export/{tags,tag_aliases,tag_implications,wiki_pages}.csv.gz`；
Danbooru 索引仅作为本地辅助碰撞检测证据，未作为该 CSV 的上游再分发内容。上述项目
负责人决定只覆盖账本列出的当前替换 CSV、其 `resource.json` 和说明书字节身份；它不构成
对替换 CSV 或说明书的 E621 上游独立再分发许可，任何下游分发仍须按上述 E621 Terms 和
适用法律完成核对。

## 安装清单模型与 tokenizer

这些资源只从清单固定的上游身份下载，当前源码不重新分发权重或 tokenizer 文件。
下表记录实现使用的不可变身份和核对入口；2026-08-13 的 Hugging Face API 响应分类和
响应 SHA-256 仅记录在 `packaging/installer/license-ledger.json` 作为取证事实，不能把
它们当作完整法律审批或对模型权重的镜像许可。

| 资源 | 固定上游身份 | 来源/许可证入口 | 状态 |
| --- | --- | --- | --- |
| E621 EVA02 Tagger | `nzs234/eva02_large_E621_FULL_V1@04a88fab40711ea5fdad1a8d051d25bdcb77a4e3` | <https://huggingface.co/nzs234/eva02_large_E621_FULL_V1> | API evidence collected; direct upstream only |
| Qwen3 0.6B tokenizer | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` | <https://huggingface.co/Qwen/Qwen3-0.6B> | API evidence collected; direct upstream only |
| LSE14 fusion head | `lse14/lse14-scorer@655377cb813d35291a2010031f724e778b7d80dd` | <https://huggingface.co/lse14/lse14-scorer> | API evidence collected; direct upstream only |
| JTP-3 | `RedRocket/Hydra@d82e15954de3d99b94217fe015d5005d30e24332` | <https://huggingface.co/RedRocket/Hydra> | API evidence collected; direct upstream only |
| Waifu scorer | `Eugeoter/waifu-scorer-v3@c2a747fd61d310a90e9cbbf8fc590c522f234424` | <https://huggingface.co/Eugeoter/waifu-scorer-v3> | API evidence collected; direct upstream only |
| CLIP `ViT-L-14.pt` | OpenAI CDN SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` | <https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt> | Direct upstream only; evidence remains incomplete |
| PaddleOCR PP-OCRv5 Server | 固定模型归档由发布清单记录 | <https://www.paddleocr.ai/latest/en/version3.x/model_list.html> | User local-only archives; not mirrored |

## PaddleOCR 手动模型归档

PaddleOCR 模型权重不包含在源码、缓存或项目 Release 中，也不会由
`Install-WebUI.bat` 自动下载或镜像。OCR 默认关闭；只有 OCR-enabled job 在本地模型
缺失或 SHA-256 不符时被预检阻止。用户必须使用
[OCR_MODEL_DOWNLOAD.md](../OCR_MODEL_DOWNLOAD.md) 中不变的三个官方 URL、文件名、大小和
SHA-256，将原始归档放入项目根 `ocr-model-archives`，再使用 `Install-Ocr.bat`；需要 GPU
时在 CPU fallback 完整后使用 `Install-OcrGpu.bat`。这些可选入口只在项目内离线探测成功
后发布 OCR 资源，且 `Install-WebUI.bat` 没有 `-OcrMode` 参数。

在逐项核对许可证、版本、来源和适用限制前，不得把上述模型镜像到项目 Release；
`Validate-SourceBootstrapRelease.ps1` 会拒绝包含未解决许可证状态的公开门禁。
