from typing import Any, Dict, List, Optional

from openai import OpenAI


_DEFAULT_LLM_MODEL = "gpt-4o"


def chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    client = OpenAI()
    m = model or _DEFAULT_LLM_MODEL
    kwargs: Dict[str, Any] = {}
    if options:
        kwargs.update(options)
    completion = client.chat.completions.create(model=m, messages=messages, **kwargs)
    return completion.choices[0].message.content or ""
