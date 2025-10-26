# Migration Plan: Move from Local Models to Fal AI (media) and OpenAI (LLM)

## Summary
- Replace local PyTorch/Diffusers-based image/video/audio generation with Fal AI hosted models via `fal-client`.
- Replace any prompt/LLM logic with OpenAI API (`openai` Python SDK).
- Minimize local GPU/accelerate/deepspeed dependencies; standardize env with API keys and lightweight client libs.

## Findings (current codebase)
- **Local model + diffusers usage**:
  - `pipelines/pipeline_flux_de_distill.py`: heavy `torch`, `transformers`, `diffusers` usage (Flux pipeline).
  - `pipelines/flex_pipeline.py`: inherits from Flux pipelines, `torch`, `diffusers`, control/inpaint latents.
  - `free_lunch_utils.py`: torch utilities patching UNet upblocks; assumes local diffusers UNet modules.
  - `__init__.py`: very large file; grep indicates extensive local model logic and utilities (assume similar patterns).
- **Provider usage**:
  - Minimal evidence of current cloud LLM providers. A single mention of `openai` in `pipeline_flux_de_distill.py` import text; appears not actively used for LLM.
- **Other assets**:
  - `deepspeed/`, `OpenPose_by_BlazzzX4/`, `Voices/`, `sounds/` directories may support local processing. Evaluate for removal or adaptation.

## Target architecture
- **LLM prompts, orchestration, small text tasks**: OpenAI Chat Completions.
- **Image generation/editing**: Fal AI image models (e.g., SDXL/FLUX variants on Fal).
- **Audio (TTS/STT)**: Fal AI for STT (e.g., `fal-ai/whisper`) and optionally TTS models; keep OpenAI Whisper only if needed.
- **Video generation/editing**: Fal AI video models (choose based on current features required; e.g., text-to-video, image-to-video, control). Map features from `flex_pipeline.py` to available Fal pipelines.

## Dependencies (to add/remove)
- **Add**:
  - `fal-client` (Python): `pip install fal-client`
  - `openai` (Python): `pip install openai`
- **Remove/De-scope (when migration done)**:
  - `torch`, `torchvision`, `xformers`, `accelerate`, `diffusers`, `transformers`, `deepspeed`, related CUDA deps
  - Custom utils tightly coupled to diffusers UNet (`free_lunch_utils.py`)

## Environment variables
- `FAL_KEY`: Fal AI API key
- `OPENAI_API_KEY`: OpenAI API key

Export examples:
```bash
export FAL_KEY=your-fal-api-key
export OPENAI_API_KEY=your-openai-api-key
```

## Usage patterns (for replacement)
- **Fal AI (sync/asynchronous runs)**
```python
# pip install fal-client
import fal_client

# Synchronous image generation (example model id placeholder)
resp = fal_client.run("fal-ai/fast-sdxl", arguments={
    "prompt": "a cinematic orange cat, ultra-detailed",
})
first_image_url = resp["images"][0]["url"]

# Async submit with event streaming
import asyncio

async def generate():
    submission = await fal_client.submit_async(
        "fal-ai/fast-sdxl",
        arguments={"prompt": "a cozy cabin in the woods, photorealistic"},
    )
    async for event in submission.iter_events(with_logs=True):
        # handle fal_client.Queued / InProgress / Completed
        pass
    result = await submission.get()
    return result

asyncio.run(generate())

# File input helpers
audio_url = fal_client.upload_file("path/to/audio.wav")
# or
audio_data_url = fal_client.encode_file("path/to/audio.wav")
resp = fal_client.run("fal-ai/whisper", arguments={"audio_url": audio_url})
```

- **OpenAI (chat/LLM)**
```python
# pip install openai
from openai import OpenAI

client = OpenAI()
chat = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Draft 3 prompts for a cute cat image."},
    ],
)
text = chat.choices[0].message.content
```

- Optional: streaming and structured parsing are available via the OpenAI Python SDK if needed.

## File-by-file migration tasks
- **__init__.py**
  - Identify all entrypoints that call local diffusers/torch pipelines.
  - Extract high-level workflows (text-to-image, image-to-image, inpaint, control, video gen, audio transcribe/tts).
  - Refactor each workflow:
    - Replace local pipeline assembly with Fal AI `fal_client.run`/`submit_async` calls.
    - Map arguments: `prompt`, negative prompt, steps/guidance to closest Fal model params.
    - Replace local file IO with Fal `upload_file` or `encode_file` for inputs (images, audio).
    - Handle outputs as URLs; download if local saving is needed.
  - Remove GPU/precision/device management code.
  - Introduce thin provider layer (see “Abstractions” below).

- **pipelines/pipeline_flux_de_distill.py**
  - Remove/stop importing: `torch`, `transformers`, `diffusers`, Flux models.
  - Replace entire pipeline class with a Fal-backed adapter, e.g., `FluxFalAdapter`:
    - `generate_image(prompt, width, height, guidance_scale, steps, seed, ...) -> images` via `fal_client.run`.
    - Note: Parameter parity may not be exact. Document supported subset.
  - If the file defines public API, keep function/class names but internally delegate to Fal.

- **pipelines/flex_pipeline.py**
  - Current logic packs custom control/inpaint latents and concatenates into channels, then runs transformer steps.
  - Replace with Fal model(s) that support inpaint/control if required, or split features:
    - Inpaint: choose a Fal inpainting model; send `image`, `mask`, `prompt` via `upload_file` + args.
    - Control: pick a Fal ControlNet-like model if available; otherwise scope out and note as a limitation or alternate flow.
  - Provide a compatibility wrapper maintaining the same call signature where possible.

- **free_lunch_utils.py**
  - Remove or stub out. This file modifies UNet internals (not applicable with hosted models).
  - If APIs previously depended on these patches, document that behavior is no longer available or find a Fal parameter equivalent if exists.

- **deepspeed/**
  - Remove references and any runtime hooks. Not used once generation is remote.

- **OpenPose_by_BlazzzX4/**
  - If used for control signals, decide whether to:
    - Keep local preprocessing (e.g., to produce a pose image) and then `upload_file` to Fal; or
    - Replace with Fal models that take raw images and detect pose internally. Document chosen path.

- **Assets (Voices/, sounds/)**
  - Keep as static assets. For TTS/STT, prefer Fal models; upload inputs and download outputs.

## New abstraction layer (recommended)
- Create `providers/` with two modules:
  - `providers/media_fal.py`
    - Functions: `generate_image(args)`, `transcribe_audio(args)`, `generate_video(args)` …
    - Wrap `fal_client` calls; enforce consistent argument names and response shapes.
  - `providers/llm_openai.py`
    - Functions: `chat(messages, model="gpt-4o", **opts)` …
  - This isolates codebase from provider SDKs and eases future swaps.

## Configuration
- Add a config file or env-based settings to select models per task, e.g.:
  - `FAL_IMAGE_MODEL=fal-ai/fast-sdxl`
  - `FAL_STT_MODEL=fal-ai/whisper`
  - `FAL_VIDEO_MODEL=<choose-appropriate-model>`
- Centralize timeout/retry/backoff for network calls.

## Testing and validation
- Unit tests for provider wrappers using mocked responses.
- Golden tests comparing a small set of previous outputs (if available) to new outputs qualitatively.
- Integration tests that:
  - Generate an image from a prompt
  - Inpaint with a mask
  - Transcribe an audio sample
  - (Optional) Generate a short video

## Rollout plan
- Phase 1: Introduce provider wrappers; add env/config; keep old paths behind a flag.
- Phase 2: Migrate primary entrypoints in `__init__.py` to wrappers; deprecate local code paths.
- Phase 3: Remove heavy local dependencies and modules; update README with new setup.

## README updates
- Add installation: `pip install fal-client openai`
- Document required env vars (`FAL_KEY`, `OPENAI_API_KEY`).
- Provide minimal code examples for common tasks using the new provider layer.

## Open items / Decisions needed
- Confirm exact Fal model IDs for:
  - Text-to-image (SDXL/FLUX variant)
  - Inpainting
  - Control/Image-to-Image (if required)
  - Video generation
  - TTS (if required)
- Confirm whether to keep any local preprocessing (pose, masks) vs. rely on Fal model capabilities.
- Accept potential behavior differences vs. custom UNet patches.
