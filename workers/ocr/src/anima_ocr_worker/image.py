from __future__ import annotations

import hashlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .protocol import OcrWorkItem


MAX_PIXELS = 40_000_000
MAX_SIDE = 16_384
MAX_HEADER_BYTES = 16 * 1024 * 1024
FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".bmp": "bmp",
}


class OcrImageError(RuntimeError):
    pass


class OcrImageDecodeError(OcrImageError):
    def __init__(self, message: str, *, width: int | None = None, height: int | None = None) -> None:
        super().__init__(message)
        self.width = width
        self.height = height


class OcrImageTooLargeError(OcrImageError):
    def __init__(self, width: int, height: int) -> None:
        super().__init__("OCR image dimensions exceed the first-release safety limit.")
        self.width = width
        self.height = height


class OcrSourceFingerprintError(OcrImageError):
    pass


@dataclass(frozen=True)
class DecodedImage:
    image: object
    width: int
    height: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ImageHeader:
    image_format: str
    encoded_width: int
    encoded_height: int
    visible_width: int
    visible_height: int


class _BoundedHeaderReader:
    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        remaining = MAX_HEADER_BYTES - self._read_bytes
        if remaining < 0:
            raise OcrImageDecodeError("OCR image header exceeds the safety limit")
        requested = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self._stream.read(requested)
        self._read_bytes += len(data)
        if self._read_bytes > MAX_HEADER_BYTES:
            raise OcrImageDecodeError("OCR image header exceeds the safety limit")
        return data

    def readinto(self, buffer: object) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()

    def close(self) -> None:
        self._stream.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


def image_exceeds_limits(width: int, height: int) -> bool:
    return width < 1 or height < 1 or width * height > MAX_PIXELS or width > MAX_SIDE or height > MAX_SIDE


def scaled_dimensions(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if not image_exceeds_limits(width, height):
        return width, height
    scale = min((MAX_PIXELS / (width * height)) ** 0.5, MAX_SIDE / max(width, height))
    scaled_width = max(1, math.floor(width * scale))
    scaled_height = max(1, math.floor(height * scale))
    while image_exceeds_limits(scaled_width, scaled_height):
        if scaled_width >= scaled_height:
            scaled_width -= 1
        else:
            scaled_height -= 1
    return scaled_width, scaled_height


def _is_reparse(path: Path) -> bool:
    try:
        information = os.lstat(path)
    except OSError as exc:
        raise OcrSourceFingerprintError("OCR source cannot be inspected") from exc
    attributes = getattr(information, "st_file_attributes", 0)
    return stat.S_ISLNK(information.st_mode) or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _source_path(item: OcrWorkItem) -> Path:
    path = Path(item.image_path)
    if not path.is_absolute() or _is_reparse(path):
        raise OcrSourceFingerprintError("OCR source path is invalid")
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise OcrSourceFingerprintError("OCR source cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise OcrSourceFingerprintError("OCR source is not a regular file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OcrSourceFingerprintError("OCR source cannot be read") from exc
    return digest.hexdigest()


def verify_source_fingerprint(item: OcrWorkItem) -> Path:
    path = _source_path(item)
    try:
        before = path.stat()
    except OSError as exc:
        raise OcrSourceFingerprintError("OCR source cannot be inspected") from exc
    if before.st_size != item.image_size or _sha256(path) != item.image_sha256:
        raise OcrSourceFingerprintError("OCR source size or SHA-256 changed after preflight")
    try:
        after = path.stat()
    except OSError as exc:
        raise OcrSourceFingerprintError("OCR source cannot be inspected") from exc
    if after.st_size != item.image_size:
        raise OcrSourceFingerprintError("OCR source size changed during verification")
    return path


def _visible_dimensions(source: object) -> tuple[int, int]:
    width = getattr(source, "width", None)
    height = getattr(source, "height", None)
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise OcrImageDecodeError("OCR image dimensions are invalid")
    try:
        orientation = source.getexif().get(274, 1)
    except Exception as exc:
        raise OcrImageDecodeError("OCR image EXIF metadata is invalid", width=width, height=height) from exc
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def _inspect_header(source: object, expected_format: str) -> ImageHeader:
    visible_width, visible_height = _visible_dimensions(source)
    image_format = str(source.format or "").lower()
    if image_format != expected_format:
        raise OcrImageDecodeError(
            "OCR image bytes do not match their extension",
            width=visible_width,
            height=visible_height,
        )
    if getattr(source, "n_frames", 1) != 1:
        raise OcrImageDecodeError(
            "OCR does not support multi-frame images",
            width=visible_width,
            height=visible_height,
        )
    try:
        source.seek(1)
    except EOFError:
        source.seek(0)
    else:
        raise OcrImageDecodeError(
            "OCR does not support multi-frame images",
            width=visible_width,
            height=visible_height,
        )
    return ImageHeader(
        image_format=image_format,
        encoded_width=source.width,
        encoded_height=source.height,
        visible_width=visible_width,
        visible_height=visible_height,
    )


def _read_header(path: Path, Image: object, expected_format: str) -> ImageHeader:
    with path.open("rb") as stream:
        bounded = _BoundedHeaderReader(stream)
        with Image.open(bounded) as source:
            return _inspect_header(source, expected_format)


def _jpeg_decoder_target(width: int, height: int, *, visible_width: int, visible_height: int) -> tuple[int, int]:
    for divisor in (2, 4, 8):
        decoded_width = math.ceil(width / divisor)
        decoded_height = math.ceil(height / divisor)
        if not image_exceeds_limits(decoded_width, decoded_height):
            return max(1, width // divisor), max(1, height // divisor)
    raise OcrImageTooLargeError(visible_width, visible_height)


def decode_and_verify(item: OcrWorkItem) -> DecodedImage:
    path = verify_source_fingerprint(item)
    suffix = path.suffix.lower()
    expected_format = FORMATS.get(suffix)
    if expected_format is None:
        raise OcrImageDecodeError("OCR image extension is unsupported")
    try:
        from PIL import Image, ImageFile, ImageOps
    except ImportError as exc:
        raise OcrImageDecodeError("Pillow is unavailable in the OCR runtime") from exc
    try:
        header = _read_header(path, Image, expected_format)
        decoder_target: tuple[int, int] | None = None
        if image_exceeds_limits(header.encoded_width, header.encoded_height):
            if header.image_format != "jpeg":
                raise OcrImageTooLargeError(header.visible_width, header.visible_height)
            decoder_target = _jpeg_decoder_target(
                header.encoded_width,
                header.encoded_height,
                visible_width=header.visible_width,
                visible_height=header.visible_height,
            )
        with path.open("rb") as stream:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(stream) as source:
                decoded_header = _inspect_header(source, expected_format)
                if decoded_header != header:
                    raise OcrImageDecodeError(
                        "OCR image header changed during decoding",
                        width=decoded_header.visible_width,
                        height=decoded_header.visible_height,
                    )
                if decoder_target is not None:
                    source.draft("RGB", decoder_target)
                    if image_exceeds_limits(source.width, source.height):
                        raise OcrImageTooLargeError(header.visible_width, header.visible_height)
                source.load()
                transposed = ImageOps.exif_transpose(source)
                transposed.load()
                target_width, target_height = scaled_dimensions(transposed.width, transposed.height)
                if (target_width, target_height) != (transposed.width, transposed.height):
                    transposed = transposed.resize((target_width, target_height), Image.Resampling.LANCZOS)
                rgba = transposed.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(white, rgba).convert("RGB")
        with path.open("rb") as verification_stream:
            with Image.open(verification_stream) as verification:
                verification.verify()
    except (OcrImageDecodeError, OcrImageTooLargeError):
        raise
    except Exception as exc:
        raise OcrImageDecodeError("OCR image cannot be decoded under the frozen policy") from exc
    verify_source_fingerprint(item)
    if image_exceeds_limits(rgb.width, rgb.height) or not math.isfinite(float(rgb.width * rgb.height)):
        raise OcrImageDecodeError("OCR image dimensions are invalid")
    return DecodedImage(rgb, rgb.width, rgb.height, item.image_size, item.image_sha256)
