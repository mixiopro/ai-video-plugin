import json
import os
from typing import Any, Dict, Optional

import fal_client


_DEFAULT_IMAGE_MODEL = "fal-ai/flux-pro/kontext"
_DEFAULT_VIDEO_MODEL = "fal-ai/veo3.1/fast"
_DEFAULT_AUDIO_MODEL = ""


def _load_models_config() -> Dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.json")
    if not os.path.exists(cfg_path):
        return {
            "image_model_id": _DEFAULT_IMAGE_MODEL,
            "video_model_id": _DEFAULT_VIDEO_MODEL,
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = {}
    return {
        "image_model_id": data.get("image_model_id", _DEFAULT_IMAGE_MODEL),
        "video_model_id": data.get("video_model_id", _DEFAULT_VIDEO_MODEL),
        "audio_model_id": data.get("audio_model_id", _DEFAULT_AUDIO_MODEL),
    }


def _resolve_model_id(task: str, override: Optional[str]) -> str:
    if override:
        return override
    cfg = _load_models_config()
    if task == "image":
        return cfg.get("image_model_id", _DEFAULT_IMAGE_MODEL)
    if task == "video":
        return cfg.get("video_model_id", _DEFAULT_VIDEO_MODEL)
    if task == "audio":
        return cfg.get("audio_model_id", _DEFAULT_AUDIO_MODEL)
    return _DEFAULT_IMAGE_MODEL


def generate_image(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = _resolve_model_id("image", model_id)
    args = {"prompt": prompt}
    if arguments:
        args.update(arguments)
    return fal_client.run(model, arguments=args)


async def generate_image_async(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    with_logs: bool = False,
) -> Dict[str, Any]:
    model = _resolve_model_id("image", model_id)
    args = {"prompt": prompt}
    if arguments:
        args.update(arguments)
    submission = await fal_client.submit_async(model, arguments=args)
    async for _ in submission.iter_events(with_logs=with_logs):
        pass
    return await submission.get()


def generate_audio(
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generic audio generation/transcription via Fal.

    Note: You must set config/models.json audio_model_id or pass model_id.
    """
    model = _resolve_model_id("audio", model_id)
    args = {}
    if arguments:
        args.update(arguments)
    if not model:
        # No model configured; return empty structure
        return {"audio": {"url": ""}}
    return fal_client.run(model, arguments=args)


def generate_video(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = _resolve_model_id("video", model_id)
    args = {"prompt": prompt}
    if arguments:
        args.update(arguments)
    return fal_client.run(model, arguments=args)


async def generate_video_async(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    with_logs: bool = False,
) -> Dict[str, Any]:
    model = _resolve_model_id("video", model_id)
    args = {"prompt": prompt}
    if arguments:
        args.update(arguments)
    submission = await fal_client.submit_async(model, arguments=args)
    async for _ in submission.iter_events(with_logs=with_logs):
        pass
    return await submission.get()
