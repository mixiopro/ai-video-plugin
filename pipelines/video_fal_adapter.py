from typing import Any, Dict, Optional

from ..providers import media_fal


def generate_video(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Thin adapter for Fal AI video generation.

    - prompt: text prompt for generation
    - model_id: optional override (defaults from config/models.json)
    - arguments: additional Fal model arguments (e.g., duration, width, height, seed)
    Returns the Fal response dict (e.g., video URL(s)).
    """
    return media_fal.generate_video(prompt, model_id=model_id, arguments=arguments)


class VideoFalAdapter:
    def __init__(self, *, model_id: Optional[str] = None):
        self.model_id = model_id

    def __call__(
        self,
        prompt: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return generate_video(prompt, model_id=self.model_id, arguments=arguments)
