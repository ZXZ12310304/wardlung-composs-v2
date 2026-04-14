from __future__ import annotations

import gc
import os
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.json_utils import safe_json_loads
from src.utils.prompts import SYSTEM_PROMPT
from src.utils.runtime_secrets import load_runtime_secrets

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_runtime_secrets(base_dir=_REPO_ROOT)
DEFAULT_HF_CACHE_DIR = os.path.join(_REPO_ROOT, "models")
DEFAULT_HF_ENDPOINT = os.getenv("MED_HF_ENDPOINT", "https://hf-mirror.com")

os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", DEFAULT_HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_CACHE_DIR)


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y")


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _from_pretrained_compat(fn, *, token: Optional[str], **kwargs):
    try:
        if token:
            return fn(**kwargs, token=token)
        return fn(**kwargs)
    except TypeError:
        if token:
            return fn(**kwargs, use_auth_token=token)
        return fn(**kwargs)


def _looks_like_repo_id(model_name: str) -> bool:
    parts = str(model_name or "").split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def _looks_like_local_path(model_name: str) -> bool:
    model_name = str(model_name or "").strip()
    if os.path.isabs(model_name):
        return True
    if _looks_like_repo_id(model_name):
        return False
    if os.path.altsep and os.path.altsep in model_name:
        return True
    if os.path.sep in model_name:
        return True
    if model_name.startswith(".") or model_name.startswith("models"):
        return True
    if ":" in model_name:
        return True
    return False


def _has_model_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    config_ok = os.path.isfile(os.path.join(path, "config.json"))
    if not config_ok:
        return False
    tokenizer_ok = (
        os.path.isfile(os.path.join(path, "tokenizer.json"))
        or os.path.isfile(os.path.join(path, "tokenizer.model"))
        or os.path.isfile(os.path.join(path, "tokenizer_config.json"))
    )
    return tokenizer_ok


def _resolve_snapshot_dir(path: str, max_depth: int = 4) -> Optional[str]:
    root = os.path.normpath(path)
    if _has_model_files(root):
        return root
    if not os.path.isdir(root):
        return None

    candidates: list[tuple[float, str]] = []
    for current, dirs, files in os.walk(root):
        rel = os.path.relpath(current, root)
        depth = 0 if rel == "." else rel.count(os.path.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        if "config.json" not in files:
            continue
        if not (
            "tokenizer.json" in files
            or "tokenizer.model" in files
            or "tokenizer_config.json" in files
        ):
            continue
        cfg_path = os.path.join(current, "config.json")
        try:
            score = os.path.getmtime(cfg_path)
        except OSError:
            score = 0.0
        candidates.append((score, current))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _snapshot_download_compat(*, repo_id: str, local_dir: str, token: Optional[str]) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. Install it with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "local_dir_use_symlinks": False,
    }
    try:
        if token:
            return snapshot_download(**kwargs, token=token)
        return snapshot_download(**kwargs)
    except TypeError:
        if token:
            return snapshot_download(**kwargs, use_auth_token=token)
        return snapshot_download(**kwargs)


def _resolve_model_source(model_id: str, token: Optional[str]) -> str:
    model_id = str(model_id or "").strip()
    if not _looks_like_local_path(model_id):
        return model_id

    local_dir = os.path.normpath(model_id)
    resolved = _resolve_snapshot_dir(local_dir)
    if resolved:
        return resolved

    auto_download = _bool_env("CHAT_QWEN_AUTO_DOWNLOAD", True)
    if not auto_download:
        raise RuntimeError(
            f"CHAT_QWEN_MODEL points to local path but model files were not found: {local_dir}. "
            "Set CHAT_QWEN_AUTO_DOWNLOAD=1 or provide a valid local snapshot directory."
        )

    repo_id = str(
        os.getenv("CHAT_QWEN_AUTO_DOWNLOAD_REPO")
        or os.getenv("CHAT_QWEN_MODEL_REPO")
        or "Qwen/Qwen2.5-0.5B-Instruct"
    ).strip()
    print(f"[QwenChat] Local model missing at {local_dir}; downloading {repo_id} ...")
    os.makedirs(local_dir, exist_ok=True)
    _snapshot_download_compat(repo_id=repo_id, local_dir=local_dir, token=token)

    resolved = _resolve_snapshot_dir(local_dir)
    if resolved:
        return resolved
    raise RuntimeError(
        f"Auto-download finished but usable model files were not found in {local_dir}. "
        "Expected config.json + tokenizer files."
    )


class QwenChatClient:
    def __init__(self, model_id: Optional[str] = None) -> None:
        requested_model = str(
            model_id
            or os.getenv("CHAT_QWEN_MODEL")
            or os.getenv("CHAT_TRANSLATE_MODEL")
            or "Qwen/Qwen2.5-0.5B-Instruct"
        ).strip()
        self.default_max_new_tokens = _int_env("CHAT_QWEN_MAX_NEW_TOKENS", 320, 96, 1024)
        self.retry_max_new_tokens = _int_env("CHAT_QWEN_RETRY_MAX_NEW_TOKENS", 448, 128, 1536)
        self.max_input_tokens = _int_env("CHAT_QWEN_MAX_INPUT_TOKENS", 3072, 512, 8192)
        token = _hf_token()
        self.model_id = _resolve_model_source(requested_model, token)
        self.requested_model_id = requested_model

        force_cuda = _bool_env("FORCE_CUDA", False)
        requested_device = str(os.getenv("CHAT_QWEN_DEVICE", "auto")).strip().lower()
        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError("FORCE_CUDA is enabled but CUDA is not available.")
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CHAT_QWEN_DEVICE=cuda but CUDA is not available.")
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = (
            (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if self.device == "cuda"
            else torch.float32
        )
        self.tokenizer = _from_pretrained_compat(
            AutoTokenizer.from_pretrained,
            token=token,
            pretrained_model_name_or_path=self.model_id,
            cache_dir=DEFAULT_HF_CACHE_DIR,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = _from_pretrained_compat(
            AutoModelForCausalLM.from_pretrained,
            token=token,
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch_dtype,
            cache_dir=DEFAULT_HF_CACHE_DIR,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.requested_model_id != self.model_id:
            print(f"[QwenChat] Requested model={self.requested_model_id}, resolved local path={self.model_id}")
        print(f"[QwenChat] Loaded {self.model_id} on {self.device}")

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    @staticmethod
    def _cleanup_cuda() -> None:
        if not torch.cuda.is_available():
            return
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_prompt(self, prompt: str, image: Optional[Image.Image]) -> str:
        user_prompt = str(prompt or "").strip()
        if image is not None:
            user_prompt = (
                f"{user_prompt}\n\n"
                "[Note] User uploaded an image. This text-only chat backend cannot inspect pixels; "
                "if image interpretation is required, advise nurse/doctor review."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_prompt}\n[ASSISTANT]\n"

    def _generate_text(self, prompt: str, *, max_new_tokens: int, image: Optional[Image.Image]) -> str:
        rendered_prompt = self._build_prompt(prompt, image)
        inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_device = self.model.device if hasattr(self.model, "device") else torch.device(self.device)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_json(self, prompt: str, *, max_new_tokens: int, image: Optional[Image.Image]) -> Dict[str, Any]:
        text = self._generate_text(prompt, max_new_tokens=max_new_tokens, image=image)
        return safe_json_loads(text)

    def run(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        target_tokens = int(max_new_tokens or self.default_max_new_tokens)
        try:
            return self._generate_json(prompt, max_new_tokens=target_tokens, image=image)
        except ValueError as exc:
            print(f"[QwenChat] JSON parse failed, retrying with strict prompt: {exc}")
            retry_tokens = max(target_tokens, self.retry_max_new_tokens)
            strict_prompt = (
                "You MUST output ONE valid JSON object only.\n"
                "No markdown, no preface, no suffix, no explanations.\n"
                "Follow the exact schema requested in the original task.\n\n"
                "Original task:\n"
                f"{prompt}"
            )
            try:
                out = self._generate_json(strict_prompt, max_new_tokens=retry_tokens, image=image)
                if isinstance(out, dict):
                    out.setdefault("degraded_mode", "json_format_retry")
                return out
            except Exception as retry_exc:
                print(f"[QwenChat] JSON retry failed: {retry_exc}")
                return {
                    "error": f"json_parse_failed: {retry_exc}",
                    "gentle_summary": "Model output format invalid.",
                }
        except RuntimeError as exc:
            if self.device == "cuda" and self._is_oom_error(exc):
                self._cleanup_cuda()
                retry_tokens = min(target_tokens, max(128, self.default_max_new_tokens // 2))
                try:
                    return self._generate_json(prompt, max_new_tokens=retry_tokens, image=image)
                except Exception as retry_exc:
                    print(f"[QwenChat] OOM retry failed: {retry_exc}")
                    return {"error": str(retry_exc), "gentle_summary": "GPU memory is insufficient."}
            print(f"[QwenChat] Inference error: {exc}")
            return {"error": str(exc), "gentle_summary": "Error in processing."}
        except Exception as exc:
            print(f"[QwenChat] Inference error: {exc}")
            return {"error": str(exc), "gentle_summary": "Error in processing."}
