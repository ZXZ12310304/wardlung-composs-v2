"""Agents package with lazy exports to avoid heavy import side effects."""

from __future__ import annotations

from typing import Any

__all__ = [
    "WardAgent",
    "MedGemmaClient",
    "QwenChatClient",
    "ChatAgent",
    "CareCardAgent",
    "FunASRTranscriber",
]


def __getattr__(name: str) -> Any:
    if name == "WardAgent":
        from .ward_agent import WardAgent

        return WardAgent
    if name == "MedGemmaClient":
        from .observer import MedGemmaClient

        return MedGemmaClient
    if name == "QwenChatClient":
        from .qwen_chat_client import QwenChatClient

        return QwenChatClient
    if name == "ChatAgent":
        from .chat_agent import ChatAgent

        return ChatAgent
    if name == "CareCardAgent":
        from .care_card_agent import CareCardAgent

        return CareCardAgent
    if name == "FunASRTranscriber":
        from .asr import FunASRTranscriber

        return FunASRTranscriber
    raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
