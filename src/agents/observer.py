# src/agents/observer.py
import os
import re
import gc
from typing import Any, Dict, Optional, List, Tuple

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

import torch
from PIL import Image
from transformers import (
    AutoModelForImageClassification,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from src.utils.json_utils import safe_json_loads
from src.utils.prompts import SYSTEM_PROMPT

FORCE_CUDA_ENV = "FORCE_CUDA"
MEDSIGLIP_DEVICE_ENV = "MEDSIGLIP_DEVICE"
MEDGEMMA_MODEL_ID_ENV = "MEDGEMMA_MODEL_ID"


def _force_cuda_enabled() -> bool:
    return os.getenv(FORCE_CUDA_ENV, "").strip().lower() in ("1", "true", "yes", "y")


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _resolve_runtime_device(force_cuda: bool, device_env_name: str, default_mode: str = "auto") -> str:
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FORCE_CUDA is set but CUDA is not available. Install a CUDA-enabled "
                "PyTorch build or unset FORCE_CUDA."
            )
        return "cuda"
    mode = os.getenv(device_env_name, default_mode).strip().lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device_env_name}=cuda but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _from_pretrained_compat(fn, *, token: Optional[str], **kwargs):
   
    try:
        if token:
            return fn(**kwargs, token=token)
        return fn(**kwargs)
    except TypeError:
        if token:
            return fn(**kwargs, use_auth_token=token)
        return fn(**kwargs)


def _is_label_interpretable(label: str) -> bool:
    if not label:
        return False
    
    if re.match(r"^LABEL_\d+$", label.strip()):
        return False
    return True


def _evidence_strength(interpretable: bool, confidence: float) -> str:
    if not interpretable:
        return "low"
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


class MedGemmaClient:
    def __init__(self, model_id: Optional[str] = None) -> None:
        resolved_model_id = str(model_id or os.getenv(MEDGEMMA_MODEL_ID_ENV, "google/medgemma-1.5-4b-it")).strip()
        self.model_id = resolved_model_id or "google/medgemma-1.5-4b-it"
        token = _hf_token()
        self.default_max_new_tokens = _int_env("MEDGEMMA_MAX_NEW_TOKENS", 384, 64, 1024)
        self.retry_max_new_tokens = _int_env("MEDGEMMA_RETRY_MAX_NEW_TOKENS", 192, 32, 512)
        self.max_input_tokens = _int_env("MEDGEMMA_MAX_INPUT_TOKENS", 3072, 512, 8192)

        force_cuda = _force_cuda_enabled()
        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "FORCE_CUDA is set but CUDA is not available. Install a CUDA-enabled "
                "PyTorch build or unset FORCE_CUDA."
            )

        if force_cuda:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "cuda"
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto"

        self.processor = _from_pretrained_compat(
            AutoProcessor.from_pretrained,
            token=token,
            pretrained_model_name_or_path=self.model_id,
            cache_dir=DEFAULT_HF_CACHE_DIR,
        )
        model_kwargs = {
            "token": token,
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "cache_dir": DEFAULT_HF_CACHE_DIR,
        }
        try:
            self.model = _from_pretrained_compat(
                AutoModelForImageTextToText.from_pretrained,
                **model_kwargs,
            )
        except ImportError as exc:
            if "requires Accelerate" in str(exc):
                print("[MedGemma] accelerate not found; retrying load without device_map.")
                model_kwargs.pop("device_map", None)
                model_kwargs["low_cpu_mem_usage"] = False
                self.model = _from_pretrained_compat(
                    AutoModelForImageTextToText.from_pretrained,
                    **model_kwargs,
                )
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                if target_device == "cuda":
                    self.model = self.model.to("cuda")
            else:
                raise

        self.model.eval()
        device = self.model.device if hasattr(self.model, "device") else "unknown"
        print(f"[MedGemma] Loaded on device: {device} (force_cuda={force_cuda})")

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

    def _build_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_device = self.model.device
        model_dtype = self.model.dtype if hasattr(self.model, "dtype") else (
            torch.bfloat16 if model_device.type == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if model_device.type == "cuda" else torch.float32
            )
        )
        inputs: Dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if torch.is_tensor(value):
                tensor = value.to(model_device)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=model_dtype)
                inputs[key] = tensor
        return inputs

    def _generate_text(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        inputs = self._build_inputs(messages)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return generated_text

    def _generate_json(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        generated_text = self._generate_text(messages, max_new_tokens=max_new_tokens)
        return safe_json_loads(generated_text)

    def run(
        self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        target_tokens = int(max_new_tokens or self.default_max_new_tokens)
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image", "image": image})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

        try:
            return self._generate_json(messages, max_new_tokens=target_tokens)
        except ValueError as exc:
            # Model returned non-JSON text; do one short retry with explicit format-only constraints.
            print(f"[MedGemma] JSON parse failed, retrying with strict JSON formatter prompt: {exc}")
            # If JSON was truncated before first "{", increasing output budget is usually required.
            retry_tokens = max(target_tokens, 320)
            retry_tokens = min(retry_tokens, 768)
            strict_prompt = (
                "You MUST output ONE valid JSON object only.\n"
                "No markdown, no preface, no suffix, no explanations.\n"
                "Follow the exact schema requested in the original task.\n\n"
                "Original task:\n"
                f"{prompt}"
            )
            strict_content = [{"type": "text", "text": strict_prompt}]
            if image is not None:
                strict_content.append({"type": "image", "image": image})
            strict_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": strict_content},
            ]
            try:
                out = self._generate_json(strict_messages, max_new_tokens=retry_tokens)
                if isinstance(out, dict):
                    out.setdefault("degraded_mode", "json_format_retry")
                return out
            except Exception as retry_exc:
                print(f"[MedGemma] JSON retry failed: {retry_exc}")
                return {
                    "error": f"json_parse_failed: {retry_exc}",
                    "gentle_summary": "模型输出格式无效。",
                }
        except RuntimeError as exc:
            if self.model.device.type == "cuda" and self._is_oom_error(exc):
                print(f"[MedGemma] OOM: retry with lower tokens (from {target_tokens})")
                self._cleanup_cuda()
                retry_tokens = min(target_tokens, self.retry_max_new_tokens)
                try:
                    return self._generate_json(messages, max_new_tokens=retry_tokens)
                except Exception as retry_exc:
                    print(f"[MedGemma] Retry failed: {retry_exc}")
                    if image is not None:
                        # Last resort: drop image to reduce KV/cache and vision token overhead.
                        text_only_messages = [
                            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {"role": "user", "content": [{"type": "text", "text": prompt}]},
                        ]
                        self._cleanup_cuda()
                        try:
                            out = self._generate_json(text_only_messages, max_new_tokens=retry_tokens)
                            if isinstance(out, dict):
                                out.setdefault("degraded_mode", "text_only_after_oom")
                            return out
                        except Exception as retry2_exc:
                            print(f"[MedGemma] Text-only retry failed: {retry2_exc}")
                    return {"error": str(retry_exc), "gentle_summary": "GPU 显存不足。"}
            print(f"[MedGemma] Inference error: {exc}")
            return {"error": str(exc), "gentle_summary": "处理过程中发生错误。"}

        except Exception as exc:
            print(f"[MedGemma] Inference error: {exc}")
            return {"error": str(exc), "gentle_summary": "处理过程中发生错误。"}


class MedSigLIPAnalyzer:
    

    
    DEFAULT_CANDIDATE_LABELS: List[str] = [
        "normal chest x-ray",
        "no pneumothorax",
        "pneumonia",
        "atypical pneumonia",
        "aspiration pneumonia",
        "right lower lobe consolidation",
        "left lower lobe consolidation",
        "interstitial opacities",
        "pleural effusion",
    ]

    def __init__(self, model_id: str = "google/medsiglip-448") -> None:
        token = _hf_token()
        force_cuda = _force_cuda_enabled()
        self.device = _resolve_runtime_device(force_cuda, MEDSIGLIP_DEVICE_ENV, default_mode="auto")
        self.model_id = model_id

       
        self.processor = _from_pretrained_compat(
            AutoProcessor.from_pretrained,
            token=token,
            pretrained_model_name_or_path=model_id,
            cache_dir=DEFAULT_HF_CACHE_DIR,
        )

      
        self.zero_shot = False
        self.zs_cls = None
        try:
            from transformers import AutoModelForZeroShotImageClassification  # type: ignore

            self.zs_cls = AutoModelForZeroShotImageClassification
            self.zero_shot = True
        except Exception:
            self.zero_shot = False

        if self.zero_shot and self.zs_cls is not None:
            self.model = _from_pretrained_compat(
                self.zs_cls.from_pretrained,
                token=token,
                pretrained_model_name_or_path=model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
            ).to(self.device)
        else:
            # fallback
            self.model = _from_pretrained_compat(
                AutoModelForImageClassification.from_pretrained,
                token=token,
                pretrained_model_name_or_path=model_id,
                cache_dir=DEFAULT_HF_CACHE_DIR,
            ).to(self.device)

        self.model.eval()
        print(f"[MedSigLIP] Loaded on device: {self.device} zero_shot={self.zero_shot} (force_cuda={force_cuda})")

    def analyze(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        labels = candidate_labels or self.DEFAULT_CANDIDATE_LABELS
        issues: List[str] = []
        top_candidates: List[Dict[str, Any]] = []

        # --- zero-shot path ---
        if self.zero_shot:
            try:
                inputs = self.processor(
                    images=image,
                    text=labels,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    out = self.model(**inputs)
                    
                    logits = getattr(out, "logits_per_image", None)
                    if logits is None:
                        logits = getattr(out, "logits", None)
                    if logits is None:
                        raise RuntimeError("zero-shot model output has no logits")

                    probs = torch.softmax(logits[0], dim=-1)  # (num_labels,)
                    k = min(int(top_k), probs.shape[-1])
                    vals, idxs = torch.topk(probs, k=k)

                for p, i in zip(vals.tolist(), idxs.tolist()):
                    top_candidates.append({"label": labels[i], "prob": round(float(p), 4)})

                primary_label = top_candidates[0]["label"] if top_candidates else "Unknown"
                confidence = float(top_candidates[0]["prob"]) if top_candidates else 0.0

                interpretable = _is_label_interpretable(primary_label)
                if not interpretable:
                    issues.append("vision_label_not_interpretable")

                suggests_pneumonia = "pneumonia" in primary_label.lower() or "consolidation" in primary_label.lower()

                return {
                    "model": "MedSigLIP",
                    "mode": "zero_shot",
                    "primary_finding": primary_label,
                    "confidence": round(confidence, 4),
                    "top_candidates": top_candidates,
                    "interpretable": interpretable,
                    "suggests_pneumonia": bool(suggests_pneumonia),
                    "evidence_strength": _evidence_strength(interpretable, confidence),
                    "issues": issues,
                }
            except Exception as exc:
                issues.append(f"zero_shot_failed: {exc}")
                # fallback to classification below

        # --- fallback classification path ---
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                k = min(int(top_k), probs.shape[-1])
                vals, idxs = torch.topk(probs, k=k)

            for p, i in zip(vals.tolist(), idxs.tolist()):
                label = self.model.config.id2label.get(int(i), f"LABEL_{int(i)}")
                top_candidates.append({"label": label, "prob": round(float(p), 4)})

            idx = int(idxs[0].item()) if idxs.numel() > 0 else int(probs.argmax().item())
            label = self.model.config.id2label.get(idx, f"LABEL_{idx}")
            confidence = float(probs[idx])

            interpretable = _is_label_interpretable(label)
            if not interpretable:
                issues.append("vision_label_not_interpretable")
            suggests_pneumonia = interpretable and ("pneumonia" in label.lower())

            return {
                "model": "MedSigLIP",
                "mode": "classification_fallback",
                "primary_finding": label,
                "confidence": round(float(confidence), 4),
                "top_candidates": top_candidates,
                "interpretable": interpretable,
                "suggests_pneumonia": bool(suggests_pneumonia),
                "evidence_strength": _evidence_strength(interpretable, float(confidence)),
                "issues": issues,
            }

        except Exception as exc:
            return {
                "model": "MedSigLIP",
                "mode": "failed",
                "primary_finding": "Unknown",
                "confidence": 0.0,
                "top_candidates": [],
                "interpretable": False,
                "suggests_pneumonia": False,
                "evidence_strength": "low",
                "issues": issues + [f"vision_failed: {exc}"],
            }
