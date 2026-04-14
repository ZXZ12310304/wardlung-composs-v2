from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple


def _get_env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in ("1", "true", "True", "yes", "YES")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        lines = [line.strip(" -\t") for line in value.splitlines()]
        return [line for line in lines if line]
    return [str(value)]


def _pick_value(data: Dict[str, Any], keys: List[str]) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data and data.get(key) not in (None, "", "-", "--"):
            return data.get(key)
    return None


def _fmt_optional(value: Any, suffix: str = "", lang: str = "en") -> str:
    if value in (None, "", "-", "--"):
        return "未记录" if str(lang or "").lower().startswith("zh") else "not recorded"
    text = str(value).strip()
    if not text:
        return "未记录" if str(lang or "").lower().startswith("zh") else "not recorded"
    return f"{text}{suffix}" if suffix else text


def _format_vitals_text(vitals: Any, lang: str = "en") -> str:
    zh = str(lang or "").lower().startswith("zh")
    if not isinstance(vitals, dict) or not vitals:
        return "未记录" if zh else "not recorded"
    temp = _pick_value(vitals, ["temperature_c", "temperature"])
    hr = _pick_value(vitals, ["heart_rate", "hr"])
    rr = _pick_value(vitals, ["resp_rate", "respiratory_rate", "rr"])
    bp = _pick_value(vitals, ["bp", "blood_pressure"])
    spo2 = _pick_value(vitals, ["spo2", "spo2_pct"])
    pain = _pick_value(vitals, ["pain", "pain_score"])
    parts: List[str] = []
    if zh:
        if temp is not None:
            parts.append(f"体温 {_fmt_optional(temp, lang='zh')}℃")
        if hr is not None:
            parts.append(f"心率 {_fmt_optional(hr, lang='zh')} 次/分")
        if rr is not None:
            parts.append(f"呼吸 {_fmt_optional(rr, lang='zh')} 次/分")
        if bp is not None:
            parts.append(f"血压 {_fmt_optional(bp, lang='zh')}")
        if spo2 is not None:
            parts.append(f"血氧 {_fmt_optional(spo2, lang='zh')}%")
        if pain is not None:
            parts.append(f"疼痛 {_fmt_optional(pain, lang='zh')}/10")
        return "；".join(parts) if parts else "未记录"
    if temp is not None:
        parts.append(f"Temp {_fmt_optional(temp)} C")
    if hr is not None:
        parts.append(f"HR {_fmt_optional(hr)} bpm")
    if rr is not None:
        parts.append(f"RR {_fmt_optional(rr)}/min")
    if bp is not None:
        parts.append(f"BP {_fmt_optional(bp)}")
    if spo2 is not None:
        parts.append(f"SpO2 {_fmt_optional(spo2)}%")
    if pain is not None:
        parts.append(f"Pain {_fmt_optional(pain)}/10")
    return ", ".join(parts) if parts else "not recorded"


def _risk_level_zh(value: Any) -> str:
    low = str(value or "").strip().lower()
    if low in ("red", "high"):
        return "高"
    if low in ("yellow", "medium", "attention"):
        return "中"
    if low in ("green", "low", "stable"):
        return "低"
    return "未知"


def _sex_zh(value: Any) -> str:
    low = str(value or "").strip().lower()
    if low in ("male", "m", "男"):
        return "男"
    if low in ("female", "f", "女"):
        return "女"
    if low in ("other", "其他"):
        return "其他"
    return str(value or "未记录").strip() or "未记录"


def _zhify_text(text: Any) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    replacements = [
        ("Community Acquired Pneumonia", "社区获得性肺炎"),
        ("Pneumonia", "肺炎"),
        ("COPD", "慢阻肺"),
        ("Asthma", "哮喘"),
        ("persistent cough", "持续咳嗽"),
        ("reduced appetite", "食欲下降"),
        ("fever", "发热"),
        ("respiratory rate increased", "呼吸频率增快"),
        ("missing vital signs data", "缺少生命体征数据"),
        ("Missing vital signs data", "缺少生命体征数据"),
        ("Measure vital signs", "监测生命体征"),
        ("Continue monitoring", "持续监测"),
        ("Notify doctor if worsening", "若病情加重及时通知医生"),
        ("Male", "男"),
        ("Female", "女"),
        ("Nausea", "恶心"),
        ("Medium", "中"),
        ("High", "高"),
        ("Low", "低"),
    ]
    for src, dst in replacements:
        out = out.replace(src, dst)
    out = re.sub(r"\bRisk light\s*=\s*([A-Z]+)", r"风险等级=\1", out, flags=re.I)
    out = out.replace("风险等级=YELLOW", "风险等级=中")
    out = out.replace("风险等级=RED", "风险等级=高")
    out = out.replace("风险等级=GREEN", "风险等级=低")
    return out


def build_sbar_skeleton(
    timeline: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    lang: str = "en",
) -> Tuple[str, List[str]]:
    lang = (lang or "en").lower()
    zh = lang.startswith("zh")
    patient = (timeline or {}).get("patient_profile") or {}
    latest_log = (timeline or {}).get("latest_daily_log") or {}
    latest_admin = (timeline or {}).get("latest_nurse_admin") or {}
    latest_assessment = (timeline or {}).get("latest_assessment_summary") or {}

    risk_data = risk_snapshot or {}
    risk_level = risk_data.get("risk_level", "green")
    flags = _as_list([f.get("message") for f in (risk_data.get("flags") or []) if isinstance(f, dict)])
    actions = _as_list(risk_data.get("next_actions") or [])

    vitals = latest_admin.get("vitals_json") or {}
    key_points: List[str] = []
    if zh:
        risk_text = _risk_level_zh(risk_level)
        flag_text = _zhify_text(flags[0]) if flags else "暂无紧急风险提示。"
        diet_text = _zhify_text(_fmt_optional(latest_log.get("diet"), lang="zh"))
        water_text = _fmt_optional(latest_log.get("water_ml"), "ml", lang="zh")
        sleep_text = _fmt_optional(latest_log.get("sleep_hours"), "小时", lang="zh")
        vitals_text = _format_vitals_text(vitals, lang="zh")
        dx_text = _zhify_text(_fmt_optional(latest_assessment.get("primary_diagnosis"), lang="zh"))
        assess_risk = _risk_level_zh(latest_assessment.get("risk_level"))
        gaps_text = _fmt_optional(latest_assessment.get("gaps_count"), lang="zh")
        rec_items = [_zhify_text(x) for x in actions[:3]] if actions else ["持续监测生命体征", "若病情加重及时通知医生"]
        rec_text = "；".join([str(x).strip("。 ") for x in rec_items if str(x).strip()]) or "持续监测"
        sbar_lines = [
            f"S（现状）：风险等级={risk_text}。{flag_text}",
            (
                f"B（背景）：床位{patient.get('bed_id') or '未记录'}，年龄{patient.get('age') or '未记录'}，"
                f"性别{_sex_zh(patient.get('sex'))}。今日饮食{diet_text}，饮水{water_text}，睡眠{sleep_text}。"
            ),
            f"A（评估）：最新体征：{vitals_text}。评估结论：{dx_text}，风险{assess_risk}，缺口数{gaps_text}。",
            f"R（建议）：{rec_text}。",
        ]
        key_points.extend([_zhify_text(x) for x in flags[:3]])
        key_points.extend([_zhify_text(x) for x in actions[:3]])
        return "\n".join(sbar_lines), [x for x in key_points if str(x).strip()][:6]

    diet_text = _fmt_optional(latest_log.get("diet"))
    water_text = _fmt_optional(latest_log.get("water_ml"), " ml")
    sleep_text = _fmt_optional(latest_log.get("sleep_hours"), " hrs")
    vitals_text = _format_vitals_text(vitals)
    dx_text = _fmt_optional(latest_assessment.get("primary_diagnosis"))
    risk_text = _fmt_optional(latest_assessment.get("risk_level"))
    gaps_text = _fmt_optional(latest_assessment.get("gaps_count"))
    rec_items = actions[:3] if actions else ["Continue monitoring", "Notify doctor if worsening"]
    rec_text = "; ".join([str(x).strip().rstrip(".") for x in rec_items if str(x).strip()]) or "Continue monitoring"
    sbar_lines = [
        f"**S (Situation)**: Risk light={str(risk_level).upper()}. {flags[0] if flags else 'No urgent red flags.'}",
        (
            f"**B (Background)**: Bed {patient.get('bed_id') or '-'}, age {patient.get('age') or '-'}, "
            f"sex {patient.get('sex') or '-'}. Diet {diet_text}, water {water_text}, sleep {sleep_text}."
        ),
        f"**A (Assessment)**: Latest vitals {vitals_text}; assessment {dx_text}, risk {risk_text}, gaps {gaps_text}.",
        f"**R (Recommendation)**: {rec_text}.",
    ]
    key_points.extend(flags[:3])
    key_points.extend(actions[:3])
    return "\n".join(sbar_lines), key_points[:6]


class HandoverAgent:
    def __init__(self, medgemma_client=None) -> None:
        self.medgemma_client = medgemma_client

    def generate(self, timeline: Dict[str, Any], risk_snapshot: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        sbar_md, key_points = build_sbar_skeleton(timeline, risk_snapshot, lang=lang)

        if _get_env_flag("HANDOVER_USE_LLM", "0") and self.medgemma_client is not None:
            if str(lang or "").lower().startswith("zh"):
                prompt = (
                    "请润色以下 SBAR，保持 S/B/A/R 结构，不新增事实，仅返回中文 SBAR 文本。\n\n"
                    + sbar_md
                )
            else:
                prompt = (
                    "Polish the following SBAR for clarity. Keep structure and do not add new facts. "
                    "Return only the polished SBAR text.\n\n"
                    + sbar_md
                )
            try:
                res = self.medgemma_client.run(prompt)
                if isinstance(res, dict) and res.get("answer"):
                    sbar_md = str(res.get("answer"))
            except Exception:
                pass

        if str(lang or "").lower().startswith("zh"):
            sbar_md = _zhify_text(sbar_md)
            key_points = [_zhify_text(x) for x in key_points]
        return {"sbar_md": sbar_md, "key_points": key_points}
