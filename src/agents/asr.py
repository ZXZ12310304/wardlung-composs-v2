# src/agents/asr.py
import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

from src.utils.runtime_secrets import load_runtime_secrets

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_runtime_secrets(base_dir=_REPO_ROOT)
DEFAULT_MODEL_CACHE_DIR = os.path.join(_REPO_ROOT, "models")
DEFAULT_MODELSCOPE_CACHE_DIR = os.path.join(DEFAULT_MODEL_CACHE_DIR, "modelscope")
DEFAULT_HF_ENDPOINT = os.getenv("MED_HF_ENDPOINT", "https://hf-mirror.com")

os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("MODELSCOPE_CACHE", DEFAULT_MODELSCOPE_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", DEFAULT_MODEL_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", DEFAULT_MODEL_CACHE_DIR)

import torch

# ====== Hard constraint: ONLY Fun-ASR-Nano-2512 ======
FUN_ASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"

FORCE_CUDA_ENV = "FORCE_CUDA"
FUN_ASR_MODEL_ID_ENV = "FUN_ASR_MODEL_ID"
FUN_ASR_DEVICE_ENV = "FUN_ASR_DEVICE"
FUN_ASR_DEBUG_ENV = "FUN_ASR_DEBUG"
FUN_ASR_LANGUAGE_ENV = "FUN_ASR_LANGUAGE"
FUN_ASR_ITN_ENV = "FUN_ASR_ITN"

# Backward-compatible env aliases (legacy MedASR keys)
MED_ASR_MODEL_ID_ENV = "MED_ASR_MODEL_ID"
MED_ASR_DEVICE_ENV = "MED_ASR_DEVICE"
MED_ASR_DEBUG_ENV = "MED_ASR_DEBUG"
MED_ASR_USE_FP16_ENV = "MED_ASR_USE_FP16"

def _truthy(value: str) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "y")


def _force_cuda_enabled() -> bool:
    return _truthy(os.getenv(FORCE_CUDA_ENV, ""))


def _debug_enabled() -> bool:
    return _truthy(os.getenv(FUN_ASR_DEBUG_ENV, "") or os.getenv(MED_ASR_DEBUG_ENV, ""))


def _resolve_device(force_cuda: bool) -> str:
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FORCE_CUDA is set but CUDA is not available. "
                "Install CUDA-enabled PyTorch or unset FORCE_CUDA."
            )
        return "cuda:0"

    mode = (
        os.getenv(FUN_ASR_DEVICE_ENV, "").strip().lower()
        or os.getenv(MED_ASR_DEVICE_ENV, "").strip().lower()
        or "auto"
    )
    if mode == "cpu":
        return "cpu"
    if mode in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("FUN_ASR_DEVICE=cuda but CUDA is not available.")
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _ensure_only_funasr(model_id: str) -> str:
    mid = (model_id or "").strip() or FUN_ASR_MODEL_ID
    if mid != FUN_ASR_MODEL_ID:
        raise ValueError(
            f"Only Fun-ASR-Nano-2512 is allowed in this project. "
            f"Expected model_id='{FUN_ASR_MODEL_ID}', got '{mid}'."
        )
    return mid


def _ensure_ffmpeg_and_pydub() -> None:
    try:
        from pydub import AudioSegment  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'pydub'. Install it with: pip install pydub\n"
            "And ensure ffmpeg exists: apt-get update && apt-get install -y ffmpeg"
        ) from exc


def _resolve_funasr_remote_code() -> Optional[str]:
    try:
        import funasr  # type: ignore
    except Exception:
        return None

    pkg_dir = os.path.dirname(getattr(funasr, "__file__", "") or "")
    if not pkg_dir:
        return None

    remote_code = os.path.join(pkg_dir, "models", "fun_asr_nano", "model.py")
    if not os.path.isfile(remote_code):
        return None
    # funasr's dynamic importer splits on "/" when deriving the module name.
    # Normalize Windows paths so it resolves "model" instead of the whole path.
    return remote_code.replace("\\", "/")


def _normalize_audio_to_wav16k_mono(audio_path: str, force_resample_wav: bool = True) -> Tuple[str, bool]:
    if not audio_path:
        raise ValueError("audio_path is empty")

    _ensure_ffmpeg_and_pydub()
    from pydub import AudioSegment

    lower = audio_path.lower()
    if lower.endswith(".wav") and not force_resample_wav:
        return audio_path, False

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name, True


def _post_clean(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("</s>", "").replace("<s>", "").strip()

    rep = {
        "{period}": ".",
        "{comma}": ",",
        "{colon}": ":",
        "{semicolon}": ";",
        "{question}": "?",
        "{exclamation}": "!",
        "{new paragraph}": "\n",
        "{newline}": "\n",
    }
    for k, v in rep.items():
        t = t.replace(k, v)

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([.,:;!?])", r"\1", t)
    return t.strip()


def _extract_text_from_funasr_result(res: Any) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for key in ("text", "sentence", "transcript"):
            value = res.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(res, (list, tuple)):
        texts = []
        for item in res:
            text = _extract_text_from_funasr_result(item)
            if text:
                texts.append(text)
        return " ".join(texts).strip()
    return ""


class FunASRTranscriber:
    def __init__(
        self,
        model_id: Optional[str] = None,
        force_resample_wav: bool = True,
    ) -> None:
        env_mid = os.getenv(FUN_ASR_MODEL_ID_ENV, "").strip() or os.getenv(MED_ASR_MODEL_ID_ENV, "").strip()
        self.model_id = _ensure_only_funasr(model_id or env_mid or FUN_ASR_MODEL_ID)
        self.device = _resolve_device(_force_cuda_enabled())
        self.force_resample_wav = bool(force_resample_wav)
        self.language = os.getenv(FUN_ASR_LANGUAGE_ENV, "").strip()
        self.itn = _truthy(os.getenv(FUN_ASR_ITN_ENV, "1"))

        # Keep cache locations inside the repo so the project can be moved without machine-specific paths.
        os.environ.setdefault("HF_HOME", DEFAULT_MODEL_CACHE_DIR)

        try:
            from funasr import AutoModel
        except Exception as exc:
            missing_dep = getattr(exc, "name", "") if isinstance(exc, ModuleNotFoundError) else ""
            if missing_dep == "torchaudio":
                raise RuntimeError(
                    "Missing dependency 'torchaudio'. Install it with: "
                    "pip install torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124"
                ) from exc
            if missing_dep == "funasr":
                raise RuntimeError(
                    "Missing dependency 'funasr'. Install it with: pip install -U funasr"
                ) from exc
            raise RuntimeError(
                f"Failed to initialize FunASR dependencies: {exc}"
            ) from exc

        auto_model_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "device": self.device,
        }
        remote_code = _resolve_funasr_remote_code()
        if remote_code:
            auto_model_kwargs["trust_remote_code"] = True
            auto_model_kwargs["remote_code"] = remote_code

        self.model = AutoModel(**auto_model_kwargs)
        print(f"[FunASR] Loaded model={self.model_id} device={self.device}")

    def transcribe(self, audio_path: str) -> str:
        wav_path, is_temp = _normalize_audio_to_wav16k_mono(audio_path, force_resample_wav=self.force_resample_wav)
        try:
            if _debug_enabled():
                try:
                    import torchaudio

                    w, sr = torchaudio.load(wav_path)
                    dur = w.shape[-1] / float(sr)
                    peak = float(w.abs().max().item())
                    print(f"[FunASR][debug] sr={sr} dur={dur:.2f}s peak={peak:.4f} path={wav_path}")
                except Exception as exc:
                    print(f"[FunASR][debug] torchaudio inspect failed: {exc}")

            generate_kwargs: Dict[str, Any] = {
                "input": wav_path,
                "cache": {},
                "batch_size": 1,
                "itn": self.itn,
            }
            if self.language:
                generate_kwargs["language"] = self.language

            res = self.model.generate(**generate_kwargs)
            text = _post_clean(_extract_text_from_funasr_result(res))
            return text if text else "[empty transcript]"
        finally:
            if is_temp:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


# Backward-compatible alias for existing imports.
MedASRTranscriber = FunASRTranscriber
