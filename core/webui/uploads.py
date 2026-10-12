"""Validate and normalize uploads without importing any model libraries."""
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 24_000_000
MAX_EDGE = 2048


def read_image(stream):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Please upload a PNG, JPEG, or WebP image.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Images must be at most 24 megapixels.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this image. Use a PNG, JPEG, or WebP under 24 megapixels.") from exc


def prepare_upload(directory: Path, image_stream, mask_stream=None):
    image = read_image(image_stream)
    mask = None
    mode = "automatic"
    if mask_stream is not None:
        uploaded_mask = read_image(mask_stream)
        if uploaded_mask.size != image.size:
            raise ValueError("The mask and image must have the same dimensions after orientation.")
        # Transparent masks use alpha; opaque masks use white foreground on black.
        alpha = uploaded_mask.getchannel("A")
        mask = alpha if alpha.getextrema()[0] < 255 else uploaded_mask.convert("L")
        mode = "uploaded mask"
    elif image.getchannel("A").getextrema()[0] < 255:
        mask = image.getchannel("A")
        mode = "transparency"
    original_size = image.size
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    if mask is not None:
        mask = mask.resize(image.size, Image.Resampling.NEAREST).point(lambda x: 1 if x >= 128 else 0)
        if not mask.getbbox():
            raise ValueError("The foreground mask is empty. Mark the object in white or use opaque alpha.")
    scene = directory / "input" / "scene"
    scene.mkdir(parents=True)
    rgb = Image.new("RGB", image.size, "white")
    rgb.paste(image, mask=image.getchannel("A"))
    rgb.save(scene / "0000.png")
    if mask is not None:
        mask.save(scene / "0000_mask.png")
    rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
    rgb.save(directory / "input.jpg", quality=85)
    return {"mask": mode, "original_size": original_size, "size": image.size}
