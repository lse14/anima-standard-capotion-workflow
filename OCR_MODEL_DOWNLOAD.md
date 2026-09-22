# OCR Model Download

`Install-WebUI.bat` installs the base WebUI only. It does not install the optional
OCR CPU/GPU runtimes and it does not download or redistribute OCR model weights.
OCR is disabled by default. The WebUI and non-OCR jobs can run without these
files; only a job where OCR is explicitly enabled is blocked until both the
selected local runtime and the OCR resource pass verification. `Install-WebUI.bat`
has no `-OcrMode` parameter.

Download the following three files from their official Paddle HTTPS URLs. Do not
rename, unpack, or edit them. Create `ocr-model-archives` in the project root and
place all three files directly in that directory:

| File | Official URL | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `PP-OCRv5_server_det_infer.tar` | <https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar> | 88340480 | `22a33e0ba6a21425ea4192da03bf4395c9a0c67902bd924b7328fc859073045d` |
| `PP-OCRv5_server_rec_infer.tar` | <https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar> | 84869120 | `d99be2ffd348943ab52876179168be4fb5b14f5f0812f2ae4c76d89ec2ea750a` |
| `PP-LCNet_x1_0_textline_ori_infer.tar` | <https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar> | 6871040 | `6171f69605215a85624d650e9079fa45f7c3eaf944296181bcc5395bf3ddc7f6` |

For CPU OCR, after the files are present, double-click `Install-Ocr.bat`. The
release must contain the matching `optional-components\\ocr-cpu` payload. This
optional installer safely stages the model resource, verifies the packaged CPU
runtime, runs an offline CPU OCR probe, and publishes both only after the probe
succeeds. Incomplete, renamed, corrupt, or hash-mismatched archives do not publish
a partial OCR resource, and a release without the optional payload fails clearly
instead of rebuilding a runtime implicitly.

For GPU OCR, complete the CPU OCR installation first, then use
`Install-OcrGpu.bat`. The release must contain the matching
`optional-components\\ocr-gpu` payload. The GPU installation keeps the CPU runtime
as its fallback and publishes only after its CUDA probe succeeds.
`Import-OcrResource.bat -Apply` can be used to re-import the local model archives
when the CPU runtime is already available.

The same guide is shown when an OCR-enabled job finds a missing or hash-mismatched
resource. Replace the affected archive in `ocr-model-archives` and run the
corresponding optional OCR installer; jobs with OCR disabled remain available.

These models remain local-only. Their upstream terms and license status are not a
permission to mirror them in a project Release; see
`docs/THIRD_PARTY_NOTICES.md` and the Paddle model list:
<https://www.paddleocr.ai/latest/en/version3.x/model_list.html>.
