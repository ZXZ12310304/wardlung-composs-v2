from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.runtime_secrets import load_runtime_secrets

load_runtime_secrets(base_dir=REPO_ROOT)


def _result(step: str, ok: bool, detail: str) -> dict:
    return {"step": step, "ok": bool(ok), "detail": str(detail)}


def main() -> int:
    out: list[dict] = []
    out.append(_result("python", True, sys.version.replace("\n", " ")))
    out.append(_result("cwd", True, os.getcwd()))
    out.append(_result("env_MEDGEMMA_MAX_NEW_TOKENS", True, str(os.getenv("MEDGEMMA_MAX_NEW_TOKENS"))))
    out.append(_result("env_MEDGEMMA_RETRY_MAX_NEW_TOKENS", True, str(os.getenv("MEDGEMMA_RETRY_MAX_NEW_TOKENS"))))

    try:
        importlib.import_module("llama_index")
        out.append(_result("import_llama_index", True, "ok"))
    except Exception as exc:
        out.append(_result("import_llama_index", False, repr(exc)))

    try:
        observer_mod = importlib.import_module("src.agents.observer")
        out.append(_result("import_src.agents.observer", True, "ok"))
    except Exception as exc:
        out.append(_result("import_src.agents.observer", False, repr(exc)))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    try:
        MedGemmaClient = getattr(observer_mod, "MedGemmaClient")
        client = MedGemmaClient()
        out.append(_result("init_medgemma", True, "ok"))
    except Exception as exc:
        out.append(_result("init_medgemma", False, repr(exc)))
        out.append(_result("trace", False, traceback.format_exc()[-1200:]))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 2

    try:
        from src.utils.chat_prompts import build_chat_prompt

        prompt = build_chat_prompt(
            role="patient",
            lang="en",
            user_message="I feel worse since yesterday. What should I do now?",
            timeline={"patient_profile": {"age": 65}},
            memory_summaries=["User reported worsening symptoms."],
            recent_turns=[
                {"role": "user", "text": "hello"},
                {"role": "assistant", "text": "What is your main symptom?"},
                {"role": "user", "text": "I feel worse"},
            ],
        )
        reply = client.run(prompt, max_new_tokens=192)
        called = isinstance(reply, dict)
        out.append(_result("run_medgemma_called", called, json.dumps(reply, ensure_ascii=False)[:600]))
        ok = isinstance(reply, dict) and not reply.get("error")
        out.append(_result("run_medgemma_valid_json", ok, json.dumps(reply, ensure_ascii=False)[:600]))
    except Exception as exc:
        out.append(_result("run_medgemma", False, repr(exc)))
        out.append(_result("trace", False, traceback.format_exc()[-1200:]))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 3

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
