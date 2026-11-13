"""Decode untrusted uploads once, preserving EXIF orientation."""
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
from app.core.config import MAX_BYTES, MAX_PIXELS

# Keep Pillow's decoder: some Ultralytics versions patch Image.open to install
# optional HEIF support on a decoding failure. This API accepts JPEG/PNG only.
_PIL_OPEN = Image.open


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Image must be nonempty and at most 5 MB.')
    if not (data.startswith(b'\x89PNG\r\n\x1a\n') or data.startswith(b'\xff\xd8\xff')):
        raise ValueError('Only JPEG and PNG images are supported.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with _PIL_OPEN(io.BytesIO(data)) as image:
                if image.format not in {'JPEG', 'PNG'}:
                    raise ValueError('Only JPEG and PNG images are supported.')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Image exceeds the 16 megapixel limit.')
                image.load()
                return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('The image is damaged or cannot be decoded safely.') from exc
