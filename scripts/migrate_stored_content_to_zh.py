from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB = os.path.join(REPO_ROOT, "data", "ward_demo.db")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            return [str(v) for v in loaded if str(v).strip()]
        lines = [ln.strip(" -\t") for ln in value.splitlines()]
        return [ln for ln in lines if ln]
    return [str(value)]


PHRASE_MAP: dict[str, str] = {
    "Today's Care Card": "今日护理卡",
    "Breathing & Chest Comfort": "呼吸与胸部舒缓",
    "Cough Relief & Airway Care": "咳嗽缓解与气道护理",
    "Nutrition & Hydration": "营养与补液",
    "Rest & Recovery": "休息与恢复",
    "Medication & Routine": "用药与日常管理",
    "Recovery Basics": "基础康复",
    "Follow your care team's guidance.": "按医护团队建议执行护理计划。",
    "Rest between activities and pace yourself.": "活动与休息交替进行，注意节奏。",
    "Keep your call bell within reach.": "将呼叫铃放在随手可及的位置。",
    "Don't push through worsening symptoms.": "症状加重时不要硬扛。",
    "Don't stop prescribed medicines on your own.": "不要自行停用处方药。",
    "Breathing suddenly gets worse.": "呼吸突然明显变差。",
    "Chest pain is new or getting worse.": "出现新的胸痛或胸痛加重。",
    "You cannot speak a full sentence.": "无法完整说一句话。",
    "You feel confused, very drowsy, or faint.": "出现意识模糊、明显嗜睡或将要晕倒。",
    "You cough up blood.": "咳血。",
    "Sit upright or prop yourself with pillows.": "尽量保持坐位或半卧位，可用枕头垫高上身。",
    "Pace activities and rest between tasks.": "活动分段进行，中间充分休息。",
    "Use prescribed oxygen or inhalers as directed.": "按医嘱使用氧气或吸入药物。",
    "Practice slow, gentle breathing to reduce anxiety.": "做缓慢、平稳的呼吸，减轻紧张。",
    "Don't lie flat if it makes breathing harder.": "若平躺会加重气短，不要平躺。",
    "Don't push through breathlessness.": "气短时不要勉强活动。",
    "Don't delay asking for help if symptoms worsen.": "症状加重时不要拖延求助。",
    "Breathing is much worse than usual.": "呼吸比平时明显更差。",
    "You feel faint or very drowsy.": "出现将要晕倒或明显嗜睡。",
    "Sip warm fluids if allowed.": "如允许饮水，可少量多次喝温水。",
    "Cough gently to clear mucus.": "轻柔咳嗽，帮助排出痰液。",
    "Use prescribed medicines as directed.": "按医嘱规律用药。",
    "Rest your voice and avoid long talking.": "减少长时间说话，让喉咙休息。",
    "Don't smoke or stay around smoke.": "不要吸烟，也避免二手烟环境。",
    "Don't suppress a cough that brings up mucus.": "有痰咳嗽时不要强行憋咳。",
    "Coughing up blood.": "咳血。",
    "Cough with severe chest pain.": "咳嗽伴明显胸痛。",
    "Breathing becomes much worse.": "呼吸明显变差。",
    "Have small, frequent meals if tolerated.": "若能耐受，少量多餐。",
    "Choose easy-to-digest foods.": "优先选择易消化食物。",
    "Sip fluids regularly if allowed.": "如允许饮水，规律少量补液。",
    "Ask the nurse if nausea limits eating.": "恶心影响进食时及时告知护士。",
    "Don't force large meals.": "不要勉强一次吃太多。",
    "Don't skip fluids if you can tolerate small sips.": "若可少量饮水，不要长时间不补液。",
    "You cannot keep fluids down.": "喝水后持续无法保留液体。",
    "Repeated vomiting or severe nausea.": "反复呕吐或明显恶心。",
    "Severe abdominal pain or weakness.": "剧烈腹痛或明显乏力。",
    "Plan short rest periods through the day.": "白天安排多次短时休息。",
    "Keep the room quiet and lights dim for sleep.": "睡前保持环境安静、光线柔和。",
    "Ask for help with repositioning if needed.": "需要翻身或调整体位时及时求助。",
    "Use relaxation breathing before sleep.": "睡前做放松呼吸。",
    "Don't overexert yourself when tired.": "疲劳时不要勉强活动。",
    "Don't stay in one position too long.": "不要长时间保持同一体位。",
    "Severe dizziness or fainting.": "明显头晕或晕厥。",
    "Breathing worsens at rest.": "静息时呼吸也在变差。",
    "New confusion or extreme sleepiness.": "新发意识混乱或异常嗜睡。",
    "Focus today: take medicines safely and on schedule.": "今日重点：安全、按时用药。",
    "Take prescribed medicines as directed.": "按医嘱服用处方药。",
    "Ask the nurse if you missed a dose.": "漏服后及时咨询护士，不要自行加量。",
    "Report side effects promptly.": "出现不良反应请尽快上报。",
    "Don't double up doses to catch up.": "不要通过双倍剂量“补服”。",
    "Don't stop medicines on your own.": "不要自行停药。",
    "Severe dizziness or fainting after medicines.": "用药后出现明显头晕或晕厥。",
    "Rash, swelling, or trouble breathing.": "出现皮疹、肿胀或呼吸困难。",
    "New severe nausea or vomiting.": "新发严重恶心或呕吐。",
    "Focus today: rest, hydrate, and listen to your body.": "今日重点：休息、补液，并留意身体变化。",
    "Don't overexert yourself.": "不要过度劳累。",
    "Don't ignore worsening symptoms.": "不要忽视症状加重信号。",
    "You feel faint, confused, or very drowsy.": "出现将要晕倒、意识混乱或明显嗜睡。",
    "Next steps:": "下一步：",
    "DO": "建议做",
    "DON'T": "避免做",
    "GET HELP NOW": "请立即求助",
    "Nurse station": "护士站",
    "Post-discharge care steps": "出院后护理步骤",
    "Dr. Chen": "陈医生",
    "Reviewing your daily check": "已查看你的每日打卡",
    "Weekly summary available": "每周总结已生成",
    "System": "系统",
    "Hello,\nPlease follow your daily care card instructions.\nNext steps:\n- Complete daily check\n- Hydration reminders\n": "你好，\n请按今日护理卡执行。\n下一步：\n- 完成每日打卡\n- 按医嘱补液与休息\n",
    "We reviewed your daily check. Please continue resting and monitor symptoms.": "我们已查看你的每日打卡。请继续休息并观察症状变化。",
    "Your weekly summary is now available in Care Cards.": "你的每周总结已生成，可在护理卡中查看。",
}


def _translate_text(text: str) -> str:
    raw = str(text or "")
    out = raw.replace("\r\n", "\n")
    for src in sorted(PHRASE_MAP.keys(), key=len, reverse=True):
        out = out.replace(src, PHRASE_MAP[src])

    low = out.lower().strip()
    if low.startswith("focus today:"):
        if "worse" in low or "worsening" in low:
            return "今日重点：症状在加重，请放慢节奏并优先保证安全。"
        if "improv" in low:
            return "今日重点：状态在好转，请继续循序渐进恢复。"
        if "fluctuat" in low or "up and down" in low:
            return "今日重点：症状有波动，请减少负担并留意变化。"
        return "今日重点：请注意休息，按计划恢复。"
    return out


def _update_care_cards(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "care_cards"):
        return 0
    from src.utils.care_card_render import render_care_card

    rows = conn.execute(
        "SELECT card_id, title, one_liner, bullets_json, red_flags_json, followup_json, text_md, language FROM care_cards"
    ).fetchall()
    changed = 0
    for row in rows:
        card_id = str(row[0])
        title = _translate_text(str(row[1] or ""))
        one_liner = _translate_text(str(row[2] or ""))
        bullets = [_translate_text(x) for x in _ensure_list(row[3])]
        red_flags = [_translate_text(x) for x in _ensure_list(row[4])]
        follow_up = [_translate_text(x) for x in _ensure_list(row[5])]
        old_text_md = str(row[6] or "")
        old_lang = str(row[7] or "").strip().lower()
        card_json = {
            "title": title or "今日护理卡",
            "one_liner": one_liner,
            "bullets": bullets,
            "red_flags": red_flags,
            "follow_up": follow_up,
        }
        new_text_md = render_care_card(card_json, lang="zh", show_footer=True)
        if (
            title != str(row[1] or "")
            or one_liner != str(row[2] or "")
            or json.dumps(bullets, ensure_ascii=False) != str(row[3] or "")
            or json.dumps(red_flags, ensure_ascii=False) != str(row[4] or "")
            or json.dumps(follow_up, ensure_ascii=False) != str(row[5] or "")
            or new_text_md != old_text_md
            or old_lang != "zh"
        ):
            conn.execute(
                """
                UPDATE care_cards
                SET title = ?, one_liner = ?, bullets_json = ?, red_flags_json = ?, followup_json = ?, text_md = ?, language = 'zh'
                WHERE card_id = ?
                """,
                (
                    title or "今日护理卡",
                    one_liner,
                    json.dumps(bullets, ensure_ascii=False),
                    json.dumps(red_flags, ensure_ascii=False),
                    json.dumps(follow_up, ensure_ascii=False),
                    new_text_md,
                    card_id,
                ),
            )
            changed += 1
    return changed


def _update_inbox(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "inbox_messages"):
        return 0
    rows = conn.execute("SELECT message_id, sender_name, subject, body FROM inbox_messages").fetchall()
    changed = 0
    for row in rows:
        mid = str(row[0])
        sender_name = _translate_text(str(row[1] or ""))
        subject = _translate_text(str(row[2] or ""))
        body = _translate_text(str(row[3] or ""))
        if sender_name != str(row[1] or "") or subject != str(row[2] or "") or body != str(row[3] or ""):
            conn.execute(
                "UPDATE inbox_messages SET sender_name = ?, subject = ?, body = ? WHERE message_id = ?",
                (sender_name, subject, body, mid),
            )
            changed += 1
    return changed


def _update_patient_cards(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "patient_cards"):
        return 0
    rows = conn.execute("SELECT card_id, content_md FROM patient_cards").fetchall()
    changed = 0
    for row in rows:
        card_id = str(row[0])
        content = str(row[1] or "")
        new_content = _translate_text(content)
        if new_content != content:
            conn.execute("UPDATE patient_cards SET content_md = ? WHERE card_id = ?", (new_content, card_id))
            changed += 1
    return changed


def _update_doctor_orders_preview(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "doctor_orders_plan"):
        return 0
    rows = conn.execute("SELECT patient_id, patient_preview_text FROM doctor_orders_plan").fetchall()
    changed = 0
    for row in rows:
        pid = str(row[0])
        preview = str(row[1] or "")
        new_preview = _translate_text(preview)
        if new_preview != preview:
            conn.execute(
                "UPDATE doctor_orders_plan SET patient_preview_text = ? WHERE patient_id = ?",
                (new_preview, pid),
            )
            changed += 1
    return changed


def _backup_db(db_path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{db_path}.bak_{ts}"
    shutil.copy2(db_path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate stored patient-facing content to Chinese.")
    parser.add_argument("--db-path", default=DEFAULT_DB, help="SQLite db path")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup copy before update")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"db not found: {db_path}")

    backup_path = ""
    if not args.no_backup:
        backup_path = _backup_db(db_path)

    summary: dict[str, Any] = {
        "db_path": db_path,
        "backup_path": backup_path,
        "updated": {},
    }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        summary["updated"]["care_cards"] = _update_care_cards(conn)
        summary["updated"]["inbox_messages"] = _update_inbox(conn)
        summary["updated"]["patient_cards"] = _update_patient_cards(conn)
        summary["updated"]["doctor_orders_plan.patient_preview_text"] = _update_doctor_orders_preview(conn)
        conn.commit()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
