from typing import Any, Dict, Optional

from ..providers import media_fal


def generate_image(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Thin adapter replacing local Flux diffusers pipeline with Fal AI image generation.

    - prompt: text prompt for generation
    - model_id: optional override (defaults from config/models.json)
    - arguments: additional Fal model arguments (e.g., width, height, guidance_scale, seed)
    Returns the Fal response dict (e.g., images URLs).
    """
    return media_fal.generate_image(prompt, model_id=model_id, arguments=arguments)


class FluxFalAdapter:
    def __init__(self, *, model_id: Optional[str] = None):
        self.model_id = model_id

    def __call__(
        self,
        prompt: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return generate_image(prompt, model_id=self.model_id, arguments=arguments)
