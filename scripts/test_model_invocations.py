from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.runtime_secrets import load_runtime_secrets

load_runtime_secrets(base_dir=REPO_ROOT)


def _row(step: str, ok: bool, detail: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "step": str(step),
        "ok": bool(ok),
        "detail": str(detail),
    }
    if extra:
        out["extra"] = extra
    return out


def _run_step(rows: List[Dict[str, Any]], step: str, fn: Callable[[], Any]) -> Any:
    try:
        value = fn()
        rows.append(_row(step, True, "ok"))
        return value
    except Exception as exc:
        rows.append(_row(step, False, repr(exc), {"traceback": traceback.format_exc()[-1600:]}))
        return None


def _smoke_qwen_chat(client: Any) -> Dict[str, Any]:
    prompt = (
        "Return JSON with keys answer,suggested_actions,need_escalation,safety_flags,topic_tag,"
        "assistant_summary_for_memory. User: mild cough for 2 days."
    )
    return client.run(prompt, max_new_tokens=128)


def _smoke_medgemma(client: Any) -> Dict[str, Any]:
    prompt = (
        "Return JSON with keys answer,suggested_actions,need_escalation,safety_flags,topic_tag,"
        "assistant_summary_for_memory. User: fever and cough."
    )
    return client.run(prompt, max_new_tokens=128)


def _smoke_translator(translator: Any) -> Dict[str, str]:
    zh_in = "我咳嗽两天，今晚有点气短。"
    en_mid = translator.zh_query_to_en(zh_in)
    zh_out = translator.en_answer_to_zh("Please rest, drink water, and contact nurse if breathing worsens.")
    return {"zh_query_to_en": str(en_mid or ""), "en_answer_to_zh": str(zh_out or "")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test model init and invocation paths.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any test step fails.")
    parser.add_argument("--skip-medgemma", action="store_true", help="Skip MedGemma init/inference tests.")
    parser.add_argument("--skip-qwen-chat", action="store_true", help="Skip Qwen chat init/inference tests.")
    parser.add_argument("--skip-translator", action="store_true", help="Skip Qwen translator init/inference tests.")
    parser.add_argument("--init-only", action="store_true", help="Only initialize models, skip inference calls.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    rows.append(_row("python", True, sys.version.replace("\n", " ")))
    rows.append(_row("repo_root", True, REPO_ROOT))
    rows.append(
        _row(
            "env",
            True,
            "captured",
            {
                "CHAT_BACKEND_PROVIDER": str(os.getenv("CHAT_BACKEND_PROVIDER")),
                "CHAT_QWEN_MODEL": str(os.getenv("CHAT_QWEN_MODEL")),
                "CHAT_TRANSLATE_MODEL": str(os.getenv("CHAT_TRANSLATE_MODEL")),
                "MEDGEMMA_MODEL_ID": str(os.getenv("MEDGEMMA_MODEL_ID")),
                "CHAT_TRANSLATE_DEVICE": str(os.getenv("CHAT_TRANSLATE_DEVICE")),
                "CHAT_QWEN_DEVICE": str(os.getenv("CHAT_QWEN_DEVICE")),
                "FORCE_CUDA": str(os.getenv("FORCE_CUDA")),
            },
        )
    )

    medgemma_client = None
    qwen_chat_client = None
    translator = None

    if not args.skip_medgemma:
        observer_mod = _run_step(rows, "import_src.agents.observer", lambda: __import__("src.agents.observer", fromlist=["*"]))
        if observer_mod is not None:
            medgemma_client = _run_step(rows, "init_medgemma_client", lambda: getattr(observer_mod, "MedGemmaClient")())
            if medgemma_client is not None and not args.init_only:
                result = _run_step(rows, "run_medgemma_smoke", lambda: _smoke_medgemma(medgemma_client))
                if isinstance(result, dict):
                    rows.append(
                        _row(
                            "run_medgemma_output",
                            True,
                            json.dumps(result, ensure_ascii=False)[:800],
                        )
                    )

    if not args.skip_qwen_chat:
        qwen_mod = _run_step(rows, "import_src.agents.qwen_chat_client", lambda: __import__("src.agents.qwen_chat_client", fromlist=["*"]))
        if qwen_mod is not None:
            qwen_chat_client = _run_step(rows, "init_qwen_chat_client", lambda: getattr(qwen_mod, "QwenChatClient")())
            if qwen_chat_client is not None and not args.init_only:
                result = _run_step(rows, "run_qwen_chat_smoke", lambda: _smoke_qwen_chat(qwen_chat_client))
                if isinstance(result, dict):
                    rows.append(
                        _row(
                            "run_qwen_chat_output",
                            True,
                            json.dumps(result, ensure_ascii=False)[:800],
                        )
                    )

    if not args.skip_translator:
        trans_mod = _run_step(rows, "import_src.tools.qwen_translator", lambda: __import__("src.tools.qwen_translator", fromlist=["*"]))
        if trans_mod is not None:
            translator = _run_step(rows, "init_qwen_translator", lambda: getattr(trans_mod, "QwenTranslator")())
            if translator is not None and not args.init_only:
                result = _run_step(rows, "run_qwen_translator_smoke", lambda: _smoke_translator(translator))
                if isinstance(result, dict):
                    rows.append(_row("run_qwen_translator_output", True, json.dumps(result, ensure_ascii=False)[:800]))

    failed = [r for r in rows if not r.get("ok")]
    rows.append(_row("summary", len(failed) == 0, f"failed_steps={len(failed)}"))
    print(json.dumps(rows, ensure_ascii=False, indent=2))

    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

