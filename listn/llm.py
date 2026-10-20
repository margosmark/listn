"""Local text generation.

The default model is ``mlx-community/Qwen2.5-7B-Instruct-4bit`` (about 4.3 GB).
Qwen2.5-7B follows JSON instructions well enough to pull decisions and
action items out of a transcript, and 4-bit keeps it small next to
Whisper large-v3-turbo (about 1.6 GB) on a 48 GB Mac. A 3B model is
faster but drops commitments; a 14B model still fits, and can be selected
in Settings, but it is slower than this meeting needs.

The weights download once from Hugging Face. After that, loads set
``HF_HUB_OFFLINE``. Machines without MLX (and the test suite) use the
extractive fallback in ``listn.summary`` instead of calling this module.
"""

from __future__ import annotations

import os

from listn.hub import load_local_or_download

DEFAULT_LLM = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def mlx_lm_available() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except Exception:
        return False


def generate(prompt: str, model_id: str = DEFAULT_LLM, max_tokens: int = 900) -> str:
    """Run one prompt on a local MLX model. Raises if MLX is not installed."""
    import mlx_lm
    from huggingface_hub import snapshot_download

    repo = (model_id or DEFAULT_LLM).strip()

    def cached():
        return snapshot_download(repo, local_files_only=True)

    def fetched():
        return snapshot_download(repo)

    path = load_local_or_download(cached, fetched)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model, tokenizer = mlx_lm.load(path)
    chat = prompt
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None:
        try:
            chat = apply([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            chat = prompt
    return mlx_lm.generate(model, tokenizer, prompt=chat, max_tokens=max_tokens, verbose=False)
