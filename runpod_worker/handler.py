"""RunPod Serverless worker for the Akio Studio video stage.

Deployed as the endpoint's container image; :mod:`akio_studio.runpod_transport`
is the client that drives it. The contract between them is narrow on purpose:

Input (``job["input"]``, produced by ``VideoRenderRequest.to_payload``)::

    {"model", "prompt", "negative_prompt", "seed", "num_frames",
     "width", "height", "denoise", ...passthrough extras}

Output::

    {"video_base64": "...", "sha256": "...", "frames": N, "fps": F,
     "duration_s": D, "model": "..."}

or, when ``AKIO_S3_BUCKET`` is configured::

    {"video_url": "https://...", "sha256": "...", ...}

The ``sha256`` is what lets the client detect a truncated transfer instead of
writing a corrupt shot into the production tree, so it is always returned.

Cold starts dominate serverless GPU cost, so the pipeline is built once at
import and reused across invocations on the same worker.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import runpod

logger = logging.getLogger("akio.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = os.environ.get("AKIO_VIDEO_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
#: Weights live on a network volume so the image stays small and cold starts
#: do not re-download tens of GiB per worker.
MODEL_CACHE = os.environ.get("AKIO_MODEL_CACHE", "/runpod-volume/models")
MASTER_FPS = 24000 / 1001  # 23.976, matching the studio master timeline

_PIPELINE: Any = None


def _load_pipeline() -> Any:
    """Build the diffusion pipeline once per worker process."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    logger.info("loading %s (cache=%s)", MODEL_ID, MODEL_CACHE)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=MODEL_CACHE
    )
    pipeline = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE
    )
    pipeline.to("cuda")
    # Frees VRAM between shots without paying a full reload per job.
    pipeline.enable_model_cpu_offload()
    _PIPELINE = pipeline
    logger.info("pipeline ready")
    return _PIPELINE


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    """Clamp and type-check inputs before spending GPU time on them."""
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("input.prompt is required and must be non-empty")

    num_frames = int(payload.get("num_frames", 81))
    if not 1 <= num_frames <= 241:
        raise ValueError(f"num_frames {num_frames} outside supported range 1-241")

    width = int(payload.get("width", 1280))
    height = int(payload.get("height", 720))
    for name, value in (("width", width), ("height", height)):
        if value % 16 or not 256 <= value <= 1920:
            raise ValueError(f"{name} must be a multiple of 16 within 256-1920")

    denoise = float(payload.get("denoise", 0.30))
    if not 0.0 <= denoise <= 1.0:
        raise ValueError(f"denoise {denoise} outside 0.0-1.0")

    return {
        "prompt": prompt,
        "negative_prompt": str(payload.get("negative_prompt", "")),
        "seed": int(payload.get("seed", 0)),
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "denoise": denoise,
        "steps": max(1, min(int(payload.get("steps", 30)), 80)),
        "guidance_scale": float(payload.get("guidance_scale", 5.0)),
    }


def _encode_video(frames: list[Any], fps: float, out_dir: Path) -> Path:
    """Write frames to an H.264 MP4 the studio's AE pipeline can ingest."""
    from diffusers.utils import export_to_video

    path = out_dir / "render.mp4"
    export_to_video(frames, str(path), fps=fps)
    return path


def _maybe_upload(path: Path, digest: str) -> str | None:
    """Upload to S3 when configured; return the object URL, else ``None``.

    Inline base64 is fine for short shots but roughly doubles the payload and
    is capped by RunPod's response size, so anything long should go to object
    storage.
    """
    bucket = os.environ.get("AKIO_S3_BUCKET")
    if not bucket:
        return None
    import boto3

    key = f"akio/{digest[:16]}.mp4"
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=os.environ.get("AKIO_S3_ENDPOINT") or None,
        region_name=os.environ.get("AKIO_S3_REGION") or None,
    )
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(os.environ.get("AKIO_S3_URL_TTL", "3600")),
    )
    logger.info("uploaded %s to s3://%s/%s", path.name, bucket, key)
    return str(url)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod entry point: one render per invocation.

    Errors are returned as ``{"error": ...}`` rather than raised, so the
    client sees a clean FAILED status with a usable message instead of a
    traceback buried in worker logs.
    """
    try:
        params = _validate(job.get("input") or {})
    except (ValueError, TypeError) as exc:
        logger.warning("rejected job input: %s", exc)
        return {"error": f"invalid input: {exc}"}

    try:
        import torch

        pipeline = _load_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(params["seed"])
        logger.info(
            "rendering seed=%s frames=%s %sx%s",
            params["seed"],
            params["num_frames"],
            params["width"],
            params["height"],
        )
        result = pipeline(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"] or None,
            height=params["height"],
            width=params["width"],
            num_frames=params["num_frames"],
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            generator=generator,
        )
        frames = result.frames[0]

        with tempfile.TemporaryDirectory() as tmp:
            video_path = _encode_video(frames, MASTER_FPS, Path(tmp))
            payload = video_path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            url = _maybe_upload(video_path, digest)

        response: dict[str, Any] = {
            "sha256": digest,
            "frames": params["num_frames"],
            "fps": MASTER_FPS,
            "duration_s": params["num_frames"] / MASTER_FPS,
            "model": MODEL_ID,
            "seed": params["seed"],
        }
        if url:
            response["video_url"] = url
        else:
            response["video_base64"] = base64.b64encode(payload).decode("ascii")
        logger.info("render complete: %d bytes, sha256=%s", len(payload), digest[:16])
        return response
    except Exception as exc:  # surfaced to the client as FAILED
        logger.exception("render failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if os.environ.get("AKIO_PRELOAD_ON_BOOT", "1") == "1":
        try:
            _load_pipeline()  # pay the load once, at worker boot
        except Exception:
            logger.exception("preload failed; will retry on first job")
    runpod.serverless.start({"handler": handler})
