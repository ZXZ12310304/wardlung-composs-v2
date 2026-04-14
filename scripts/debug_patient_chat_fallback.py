from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.runtime_secrets import load_runtime_secrets

load_runtime_secrets(base_dir=REPO_ROOT)

from src.agents.chat_agent import ChatAgent


DEFAULT_MESSAGE = "I still have cough and chest tightness. What should I do now?"


def _capture_env() -> Dict[str, Any]:
    keys = [
        "CHAT_BACKEND_PROVIDER",
        "CHAT_QWEN_MODEL",
        "CHAT_QWEN_DEVICE",
        "CHAT_QWEN_MAX_NEW_TOKENS",
        "CHAT_QWEN_RETRY_MAX_NEW_TOKENS",
        "CHAT_TRANSLATION_ENABLED",
        "MED_HF_ENDPOINT",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "FORCE_CUDA",
    ]
    out = {k: str(os.getenv(k) or "") for k in keys}
    provider_raw = out.get("CHAT_BACKEND_PROVIDER", "").strip().lower() or "qwen"
    out["CHAT_BACKEND_PROVIDER_EFFECTIVE"] = (
        "qwen" if provider_raw in ("medgemma", "med_gemma") else provider_raw
    )
    out["HF_TOKEN_PRESENT"] = bool(out.get("HF_TOKEN") or out.get("HUGGINGFACE_HUB_TOKEN"))
    out.pop("HF_TOKEN", None)
    out.pop("HUGGINGFACE_HUB_TOKEN", None)
    return out


def _import_patient_app() -> Tuple[Optional[Any], str]:
    try:
        from src.ui import patient_app as mod

        return mod, ""
    except Exception as exc:
        return None, repr(exc)


def _probe_model_client(max_new_tokens: int, patient_app_mod: Optional[Any]) -> Tuple[Dict[str, Any], Optional[Any]]:
    result: Dict[str, Any] = {
        "step": "model_client_probe",
        "ok": False,
    }
    client = None
    if patient_app_mod is not None:
        start = time.perf_counter()
        client = patient_app_mod._get_chat_model_client()
        result["init_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
        result["client_type"] = type(client).__name__ if client is not None else None
        result["chat_model_error"] = patient_app_mod._BACKEND_CACHE.get("chat_model_error")
        result["client_source"] = "patient_app"
    else:
        try:
            from src.agents.qwen_chat_client import QwenChatClient

            start = time.perf_counter()
            client = QwenChatClient()
            result["init_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
            result["client_type"] = type(client).__name__
            result["client_source"] = "direct_qwen"
            result["chat_model_error"] = "patient_app_unavailable"
        except Exception as exc:
            result["detail"] = f"direct qwen init failed: {exc!r}"
            result["traceback"] = traceback.format_exc()[-1600:]
            return result, None
    if client is None:
        result["detail"] = "chat model init failed"
        return result, None

    prompt = (
        "Return exactly one JSON object with keys: "
        "answer,suggested_actions,need_escalation,escalation_reason,safety_flags,"
        "citations,new_gaps,topic_tag,assistant_summary_for_memory. "
        "User: mild cough for 2 days."
    )
    try:
        raw = client.run(prompt, max_new_tokens=max_new_tokens)
        result["raw_preview"] = json.dumps(raw, ensure_ascii=False)[:1200]
        result["raw_has_error"] = bool(isinstance(raw, dict) and raw.get("error"))
        result["ok"] = isinstance(raw, dict) and not raw.get("error")
        result["detail"] = "model run ok" if result["ok"] else "model returned error payload"
    except Exception as exc:
        result["detail"] = f"model run raised: {exc!r}"
        result["traceback"] = traceback.format_exc()[-1600:]
    return result, client


def _probe_chat_agent(
    *,
    message: str,
    patient_id: str,
    lang: str,
    client: Optional[Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "step": "chat_agent_probe",
        "ok": False,
    }
    if client is None:
        result["detail"] = "chat model unavailable"
        return result

    try:
        agent = ChatAgent(client, rag_engine=None, lang=lang)
        out = agent.answer(
            role="patient",
            patient_id=patient_id,
            user_message=message,
            timeline={},
            memory_summaries=[],
            recent_turns=[],
            lang=lang,
            asr_quality=None,
            image=None,
        )
        safety_flags = [str(x) for x in (out.get("safety_flags") or [])]
        model_error_fallback = "model_error" in safety_flags
        result["ok"] = True
        result["model_error_fallback"] = model_error_fallback
        result["answer_preview"] = str(out.get("answer") or "")[:300]
        result["safety_flags"] = safety_flags
        result["topic_tag"] = str(out.get("topic_tag") or "")
        result["detail"] = "chat agent answered"
    except Exception as exc:
        result["detail"] = f"chat agent raised: {exc!r}"
        result["traceback"] = traceback.format_exc()[-1600:]
    return result


def _probe_patient_chat_pipeline(
    *,
    message: str,
    patient_id: str,
    ui_lang: str,
    timeout_sec: float,
    patient_app_mod: Optional[Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "step": "patient_chat_pipeline_probe",
        "ok": False,
    }
    if patient_app_mod is None:
        result["detail"] = "skipped: patient_app import failed (likely missing gradio)"
        return result

    state: Dict[str, Any] = {
        "authed": True,
        "role": "patient",
        "patient_id": patient_id,
        "settings_lang": ui_lang,
        "chat_history": [],
        "chat_pending": False,
    }

    # Avoid UI rendering dependencies (icons/page config) from masking backend errors.
    original_render = getattr(patient_app_mod, "render_patient_view", None)
    if callable(original_render):
        patient_app_mod.render_patient_view = lambda _state: None

    try:
        try:
            payload = json.dumps({"message": message}, ensure_ascii=False)
            state, _ = patient_app_mod.chat_send(payload, None, state)
        except Exception as exc:
            result["detail"] = f"chat_send raised: {exc!r}"
            result["traceback"] = traceback.format_exc()[-1600:]
            return result

        deadline = time.time() + max(1.0, float(timeout_sec))
        try:
            while state.get("chat_pending") and time.time() < deadline:
                time.sleep(0.2)
                state, _ = patient_app_mod.poll_chat_updates(state)
        except Exception as exc:
            result["detail"] = f"poll_chat_updates raised: {exc!r}"
            result["traceback"] = traceback.format_exc()[-1600:]
            return result

        history = state.get("chat_history") or []
        assistant_text = ""
        if history and isinstance(history[-1], dict):
            assistant_text = str(history[-1].get("text") or "")

        result["ok"] = True
        result["chat_pending_after_poll"] = bool(state.get("chat_pending"))
        result["chat_model_warned"] = bool(state.get("_chat_model_warned"))
        result["toast"] = str(state.get("toast") or "")
        result["assistant_text_preview"] = assistant_text[:300]
        result["history_len"] = len(history)
        result["detail"] = "pipeline completed"
        return result
    finally:
        if callable(original_render):
            patient_app_mod.render_patient_view = original_render


def _guess_root_cause(model_probe: Dict[str, Any], agent_probe: Dict[str, Any], pipeline_probe: Dict[str, Any]) -> str:
    if not model_probe.get("ok"):
        if not model_probe.get("client_type"):
            return "chat_model_init_failed"
        return "chat_model_run_failed"
    if bool(agent_probe.get("model_error_fallback")):
        return "chat_agent_internal_fallback(model_error)"
    if bool(pipeline_probe.get("chat_model_warned")):
        return "patient_pipeline_exception_fallback"
    if bool(pipeline_probe.get("chat_pending_after_poll")):
        return "pipeline_timeout"
    return "no_fallback_detected_in_probe"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose why patient chat is falling back.")
    parser.add_argument("--patient-id", default="", help="Patient ID (default: first patient in DB).")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="User message to probe.")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="Model reply language in ChatAgent probe.")
    parser.add_argument(
        "--ui-lang",
        default="English",
        choices=["English", "Chinese"],
        help="UI language used in full patient_app pipeline probe.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220, help="max_new_tokens for model client probe.")
    parser.add_argument("--timeout-sec", type=float, default=45.0, help="Timeout for threaded chat pipeline probe.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when fallback is detected.")
    args = parser.parse_args()

    patient_app_mod, patient_app_import_error = _import_patient_app()
    if patient_app_mod is not None:
        patient_id = str(args.patient_id or patient_app_mod._get_any_patient_id()).strip() or "demo_patient_001"
    else:
        patient_id = str(args.patient_id or "demo_patient_001").strip() or "demo_patient_001"

    report: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "repo_root": REPO_ROOT,
        "patient_id": patient_id,
        "env": _capture_env(),
        "patient_app_import_error": patient_app_import_error,
        "checks": [],
    }

    model_probe, model_client = _probe_model_client(
        max_new_tokens=max(64, int(args.max_new_tokens)),
        patient_app_mod=patient_app_mod,
    )
    agent_probe = _probe_chat_agent(
        message=args.message,
        patient_id=patient_id,
        lang=args.lang,
        client=model_client,
    )
    pipeline_probe = _probe_patient_chat_pipeline(
        message=args.message,
        patient_id=patient_id,
        ui_lang=args.ui_lang,
        timeout_sec=max(5.0, float(args.timeout_sec)),
        patient_app_mod=patient_app_mod,
    )

    report["checks"].append(model_probe)
    report["checks"].append(agent_probe)
    report["checks"].append(pipeline_probe)
    report["root_cause_guess"] = _guess_root_cause(model_probe, agent_probe, pipeline_probe)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.strict:
        return 0
    if report["root_cause_guess"] == "no_fallback_detected_in_probe":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
