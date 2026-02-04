#!/usr/bin/env python3
"""
Конвертация экспорта Instagram (JSON) в один Markdown-файл для NotebookLM.

- Вход: папка экспорта Instagram (в src/ или переданный путь)
- Выход: один .md с разделами: Профиль, Социальные связи, Переписки, Активность, Повторяющиеся темы
- Медиа не включаются; голосовые сообщения опционально транскрибируются через Whisper
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(SCRIPT_DIR, "src")
DEFAULT_DIST = os.path.join(SCRIPT_DIR, "dist")

# Формат даты в логе сообщений
DATE_FMT = "%Y-%m-%d %H:%M"
# Как часто выводить прогресс внутри одного чата (по сообщениям)
CHAT_PROGRESS_INTERVAL = 100

_WS_RE = re.compile(r"\s+")
_whisper_model = None


def _normalize_text(s: str) -> str:
    """Нормализует текст: убирает лишние пробелы."""
    if not s:
        return ""
    return _WS_RE.sub(" ", s).strip()


def load_json_safe(root: str, *path_parts: str) -> dict | list | None:
    """Читает JSON по относительному пути от root. При ошибке возвращает None."""
    path = os.path.join(root, *path_parts)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_instagram_export_root(path: str) -> str | None:
    """
    Ищет корень экспорта Instagram: папка должна содержать
    personal_information/personal_information/personal_information.json
    и your_instagram_activity/messages/inbox/
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    profile = os.path.join(path, "personal_information", "personal_information", "personal_information.json")
    inbox = os.path.join(path, "your_instagram_activity", "messages", "inbox")
    if os.path.isfile(profile) and os.path.isdir(inbox):
        return path
    # Поиск в подпапках (например src/instagram-.../)
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and not name.startswith("."):
            found = find_instagram_export_root(sub)
            if found:
                return found
    return None


def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Транскрибирует аудио/видео через Whisper. Возвращает текст или пустую строку при ошибке.
    Требует: pip install openai-whisper и ffmpeg в системе.
    """
    global _whisper_model
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        import whisper
    except ImportError:
        return ""
    try:
        if _whisper_model is None:
            _whisper_model = whisper.load_model(model_size)
        result = _whisper_model.transcribe(file_path, fp16=False, language=None)
        text = (result.get("text") or "").strip()
        return _normalize_text(text)
    except Exception:
        return ""


# --- Профиль ---


def load_profile(root: str) -> list[str]:
    """Собирает блок Markdown для раздела «Профиль»."""
    lines = []
    data = load_json_safe(root, "personal_information", "personal_information", "personal_information.json")
    if not data or "profile_user" not in data:
        return lines
    for block in data.get("profile_user", [])[:1]:
        sm = block.get("string_map_data") or {}
        fields = [
            ("Name", "Имя"),
            ("Username", "Username"),
            ("Email", "Email"),
            ("Phone Number", "Телефон"),
            ("Date of birth", "Дата рождения"),
            ("Gender", "Пол"),
            ("Private Account", "Приватный аккаунт"),
        ]
        for key, label in fields:
            if key in sm and sm[key].get("value"):
                lines.append(f"- {label}: {sm[key]['value']}")
    return lines


# --- Социальные связи ---


def _load_relationship_list(root: str, rel_path: str, key: str, title_key: str = "title") -> list[tuple[str, int]]:
    """Загружает список из relationships_* или string_list_data. Возвращает [(username или value, timestamp), ...]."""
    data = load_json_safe(root, rel_path)
    if not data:
        return []
    items = data.get(key, data.get("string_list_data", []))
    if not items:
        return []
    result = []
    for item in items:
        if isinstance(item, dict):
            title = item.get(title_key) or (item.get("string_list_data") or [{}])[0].get("value", "")
            ts = 0
            if "string_list_data" in item and item["string_list_data"]:
                ts = item["string_list_data"][0].get("timestamp", 0)
            elif "string_map_data" in item:
                for v in item["string_map_data"].values():
                    if isinstance(v, dict) and "timestamp" in v:
                        ts = v["timestamp"]
                        break
            if title:
                result.append((str(title).strip(), ts))
        elif isinstance(item, str):
            result.append((item, 0))
    return result


def load_connections(root: str) -> dict[str, list[tuple[str, int]]]:
    """Собирает данные для раздела «Социальные связи»."""
    base = "connections", "followers_and_following"
    out = {}
    # following
    data = load_json_safe(root, *base, "following.json")
    if data and "relationships_following" in data:
        out["following"] = [(item.get("title", ""), (item.get("string_list_data") or [{}])[0].get("timestamp", 0)) for item in data["relationships_following"] if item.get("title")]
    else:
        out["following"] = []
    # followers (followers_1.json, followers_2.json ...) — может быть массив в корне или объект с ключом
    followers = []
    for i in range(1, 20):
        fn = f"followers_{i}.json"
        data = load_json_safe(root, *base, fn)
        if not data:
            continue
        arr = data if isinstance(data, list) else data.get("relationships_followers", data.get("relationships_following", []))
        for item in arr or []:
            sl = (item.get("string_list_data") or [{}])[0]
            title = (item.get("title") or sl.get("value", "")).strip()
            ts = sl.get("timestamp", 0)
            if not title and isinstance(sl.get("value"), str):
                title = sl["value"]
            if title:
                followers.append((title, ts))
    out["followers"] = followers
    # close_friends
    data = load_json_safe(root, *base, "close_friends.json")
    if data and "relationships_close_friends" in data:
        out["close_friends"] = []
        for item in data["relationships_close_friends"]:
            for sl in (item.get("string_list_data") or []):
                val = sl.get("value", "").strip()
                if val:
                    out["close_friends"].append((val, sl.get("timestamp", 0)))
    else:
        out["close_friends"] = []
    return out


# --- Переписки ---


def _message_text(msg: dict, export_root: str, transcribe_fn) -> str:
    """
    Извлекает текст одного сообщения: content, share (link + share_text), или транскрипция аудио.
    transcribe_fn(path) -> str
    """
    parts = []
    content = (msg.get("content") or "").strip()
    share = msg.get("share")
    audio_files = msg.get("audio_files") or []

    # Голосовое сообщение
    if audio_files:
        uri = audio_files[0].get("uri") if isinstance(audio_files[0], dict) else None
        if uri:
            full_path = os.path.join(export_root, uri)
            text = transcribe_fn(full_path) if transcribe_fn else ""
            if text:
                parts.append(text)
            else:
                parts.append("[Голосовое сообщение: не удалось распознать]")
        else:
            parts.append("[Голосовое сообщение]")

    # Текст (если не служебная подпись к шарингу)
    if content and content not in ("You sent an attachment.", "Liked a message"):
        if "sent an attachment" not in content and "Reacted" not in content:
            parts.append(content)
        elif not parts:
            parts.append(content)

    # Шаринг: ссылка и подпись
    if share:
        link = share.get("link", "")
        share_text = (share.get("share_text") or "").strip()
        owner = share.get("original_content_owner", "")
        if link:
            parts.append(f"Поделился ссылкой: {link}")
        if share_text:
            parts.append(share_text)
        if owner and owner not in str(parts):
            parts.append(f"(автор: {owner})")

    # Если только "You sent an attachment." / "Liked a message" / "Reacted ..."
    if not parts and content:
        parts.append(content)

    if not parts:
        return "[Сообщение без текста]"
    return " ".join(parts).strip()


def _format_reactions(reactions: list) -> str:
    if not reactions:
        return ""
    return " " + " ".join(f"({r.get('actor', '')}: {r.get('reaction', '')})" for r in reactions)


def _log_noop(_msg: str, _pct: int | None = None) -> None:
    pass


def load_inbox_conversations(root: str, transcribe_fn, log=None) -> list[tuple[str, list[str]]]:
    """
    Загружает все чаты из inbox. Возвращает список (имя_чата, [строки Markdown]).
    transcribe_fn(file_path) -> str или None (тогда плейсхолдер для аудио).
    log(msg, pct=None) — опциональный вывод прогресса; pct — общий прогресс 0–100.
    """
    if log is None:
        log = _log_noop
    inbox_dir = os.path.join(root, "your_instagram_activity", "messages", "inbox")
    if not os.path.isdir(inbox_dir):
        return []

    folders = sorted([f for f in os.listdir(inbox_dir) if os.path.isdir(os.path.join(inbox_dir, f))])
    total_folders = len(folders)
    log(f"  Переписки: найдено {total_folders} чатов.")

    conversations = []
    for idx, folder in enumerate(folders):
        conv_path = os.path.join(inbox_dir, folder)
        # Все message_*.json в этой папке
        pattern = os.path.join(conv_path, "message_*.json")
        files = sorted(glob.glob(pattern), key=lambda p: (len(p), p))
        all_messages = []
        participants_names = []
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if participants_names == [] and data.get("participants"):
                participants_names = [p.get("name", "") for p in data["participants"] if p.get("name")]
            all_messages.extend(data.get("messages", []))

        if not all_messages:
            continue
        all_messages.sort(key=lambda m: m.get("timestamp_ms", 0))

        # Имя чата: участники (без владельца можно оставить всех)
        chat_name = ", ".join(participants_names) if participants_names else folder
        short_name = (chat_name[:50] + "…") if len(chat_name) > 50 else chat_name
        # Переписки занимают 10–80% общего прогресса
        pct = 10 + int(70 * (idx + 1) / total_folders) if total_folders else 80
        log(f"    Чат {idx + 1}/{total_folders}: {short_name} ({len(all_messages)} сообщ.)", pct)

        lines = []
        total_msgs = len(all_messages)
        for msg_idx, msg in enumerate(all_messages):
            ts = msg.get("timestamp_ms", 0)
            try:
                dt = datetime.utcfromtimestamp(ts / 1000.0).strftime(DATE_FMT)
            except (OSError, ValueError):
                dt = str(ts)
            sender = (msg.get("sender_name") or "").strip()
            text = _message_text(msg, root, transcribe_fn)
            reactions = _format_reactions(msg.get("reactions") or [])
            line = f"- {dt} | {sender} | {text}{reactions}"
            lines.append(line)
            # Прогресс внутри чата: каждые CHAT_PROGRESS_INTERVAL сообщений или в конце
            if total_msgs > CHAT_PROGRESS_INTERVAL and (
                (msg_idx + 1) % CHAT_PROGRESS_INTERVAL == 0 or (msg_idx + 1) == total_msgs
            ):
                log(f"      обработано {msg_idx + 1}/{total_msgs} сообщ.")
        conversations.append((chat_name, lines))
    return conversations


# --- Комментарии ---


def load_comments(root: str) -> list[tuple[int, str, str]]:
    """Возвращает [(timestamp, media_owner, comment_text), ...] из post_comments и hype."""
    out = []
    base = "your_instagram_activity", "comments"
    for filename in ("post_comments_1.json", "hype.json"):
        data = load_json_safe(root, *base, filename)
        if not data:
            continue
        items = data if isinstance(data, list) else data.get("comments_story_comments", [])
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sm = item.get("string_map_data") or {}
            if not isinstance(sm, dict):
                continue
            comment = (sm.get("Comment") or {}).get("value", "")
            owner = (sm.get("Media Owner") or {}).get("value", "")
            ts = (sm.get("Time") or {}).get("timestamp", 0)
            if comment or owner:
                out.append((ts, owner, comment))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# --- Активность (лайки, сохранённое, посты Threads) ---


def load_activity(root: str) -> dict[str, list]:
    """Собирает данные для раздела «Активность»."""
    base_act = "your_instagram_activity"
    out = {"liked_posts": [], "liked_comments": [], "saved_posts": [], "threads_posts": []}

    # liked_posts
    data = load_json_safe(root, base_act, "likes", "liked_posts.json")
    if data and "likes_media_likes" in data:
        for item in data["likes_media_likes"]:
            title = item.get("title", "")
            for sl in (item.get("string_list_data") or []):
                href = sl.get("href", "")
                ts = sl.get("timestamp", 0)
                out["liked_posts"].append((title, href, ts))

    # liked_comments
    data = load_json_safe(root, base_act, "likes", "liked_comments.json")
    if data and "likes_comment_likes" in data:
        for item in data["likes_comment_likes"]:
            title = item.get("title", "")
            for sl in (item.get("string_list_data") or []):
                href = sl.get("href", "")
                ts = sl.get("timestamp", 0)
                out["liked_comments"].append((title, href, ts))

    # saved_posts
    data = load_json_safe(root, base_act, "saved", "saved_posts.json")
    if data and "saved_saved_media" in data:
        for item in data["saved_saved_media"]:
            title = item.get("title", "")
            sm = item.get("string_map_data") or {}
            saved = sm.get("Saved on", {})
            href = saved.get("href", "") if isinstance(saved, dict) else ""
            ts = saved.get("timestamp", 0) if isinstance(saved, dict) else 0
            out["saved_posts"].append((title, href, ts))

    # threads_and_replies — посты Threads (заголовок + дата)
    data = load_json_safe(root, base_act, "threads", "threads_and_replies.json")
    if data and "text_post_app_text_posts" in data:
        for item in data["text_post_app_text_posts"]:
            for media in item.get("media", [])[:1]:
                title = (media.get("title") or "").strip()
                ts = media.get("creation_timestamp", 0)
                uri = media.get("uri", "")
                out["threads_posts"].append((title, uri, ts))

    return out


# --- Повторяющиеся темы ---


def load_topics(root: str) -> list[str]:
    """Загружает список тем из recommended_topics."""
    data = load_json_safe(root, "preferences", "your_topics", "recommended_topics.json")
    if not data or "topics_your_topics" not in data:
        return []
    topics = []
    for item in data.get("topics_your_topics", []):
        sm = item.get("string_map_data") or {}
        name = (sm.get("Name") or {}).get("value", "")
        if name:
            topics.append(name.strip())
    return topics


# --- Сборка Markdown ---


def build_markdown(
    root: str,
    transcribe_fn,
    username_fallback: str = "Instagram",
    log=None,
) -> str:
    """Собирает один Markdown-документ из всех разделов. log(msg, pct=None) — опциональный вывод прогресса; pct 0–100."""
    if log is None:
        log = _log_noop
    sections = []

    log("Сборка документа...", 0)
    # Заголовок
    profile_lines = load_profile(root)
    log("  Профиль загружен.", 5)
    username = username_fallback
    for line in profile_lines:
        if line.startswith("- Username:"):
            username = line.replace("- Username:", "").strip()
            break
    sections.append(f"# Instagram Export — {username}\n")

    # Профиль
    sections.append("## Профиль\n")
    sections.extend(profile_lines if profile_lines else ["- Нет данных"])
    sections.append("")

    # Социальные связи
    sections.append("## Социальные связи\n")
    conn = load_connections(root)
    log("  Социальные связи загружены.", 10)
    if conn.get("following"):
        sections.append("### Подписки (following)\n")
        for name, _ts in conn["following"][:500]:
            sections.append(f"- {name}")
        sections.append("")
    if conn.get("followers"):
        sections.append("### Подписчики (followers)\n")
        for name, _ts in conn["followers"][:500]:
            sections.append(f"- {name}")
        sections.append("")
    if conn.get("close_friends"):
        sections.append("### Близкие друзья\n")
        for name, _ts in conn["close_friends"]:
            sections.append(f"- {name}")
        sections.append("")

    # Переписки
    sections.append("## Переписки\n")
    conversations = load_inbox_conversations(root, transcribe_fn, log=log)
    log(f"  Переписки обработаны: {len(conversations)} чатов.", 80)
    for chat_name, lines in conversations:
        safe_title = chat_name.replace("\n", " ").strip() or "Без имени"
        sections.append(f"### Чат: {safe_title}\n")
        sections.extend(lines)
        sections.append("")

    # Активность
    sections.append("## Активность\n")
    act = load_activity(root)
    log("  Активность загружена.", 85)
    if act["liked_posts"]:
        sections.append("### Лайки постов\n")
        for title, href, ts in act["liked_posts"][:300]:
            try:
                dt = datetime.utcfromtimestamp(ts).strftime(DATE_FMT) if ts else ""
            except (OSError, ValueError):
                dt = ""
            sections.append(f"- {dt} | {title} | {href}")
        sections.append("")
    if act["liked_comments"]:
        sections.append("### Лайки комментариев\n")
        for title, href, ts in act["liked_comments"][:200]:
            try:
                dt = datetime.utcfromtimestamp(ts).strftime(DATE_FMT) if ts else ""
            except (OSError, ValueError):
                dt = ""
            sections.append(f"- {dt} | {title} | {href}")
        sections.append("")
    if act["saved_posts"]:
        sections.append("### Сохранённые посты\n")
        for title, href, ts in act["saved_posts"][:300]:
            try:
                dt = datetime.utcfromtimestamp(ts).strftime(DATE_FMT) if ts else ""
            except (OSError, ValueError):
                dt = ""
            sections.append(f"- {dt} | {title} | {href}")
        sections.append("")
    if act["threads_posts"]:
        sections.append("### Посты Threads\n")
        for title, uri, ts in act["threads_posts"][:100]:
            try:
                dt = datetime.utcfromtimestamp(ts).strftime(DATE_FMT) if ts else ""
            except (OSError, ValueError):
                dt = ""
            sections.append(f"- {dt} | {title} | {uri}")
        sections.append("")

    # Комментарии (мои)
    sections.append("### Мои комментарии\n")
    comments = load_comments(root)
    log("  Комментарии загружены.", 90)
    for ts, owner, text in comments[:300]:
        try:
            dt = datetime.utcfromtimestamp(ts).strftime(DATE_FMT) if ts else ""
        except (OSError, ValueError):
            dt = ""
        sections.append(f"- {dt} | {owner} | {text[:200]}")
    sections.append("")

    # Повторяющиеся темы
    sections.append("## Повторяющиеся темы\n")
    topics = load_topics(root)
    log("  Темы загружены.", 95)
    for t in topics:
        sections.append(f"- {t}")
    if not topics:
        sections.append("- Нет данных")
    sections.append("")

    log("Сборка завершена.", 95)
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Конвертация экспорта Instagram (JSON) в Markdown для NotebookLM."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_SRC,
        help="Папка с экспортом (по умолчанию: src/)",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_DIST,
        help="Папка для выходного .md (по умолчанию: dist/)",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Не транскрибировать голосовые сообщения (быстрее, в логе будет плейсхолдер)",
    )
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(SCRIPT_DIR, args.input)
    export_root = find_instagram_export_root(input_path)
    if not export_root:
        print(
            f"Ошибка: не найдена папка экспорта Instagram в {input_path}. "
            "Ожидается наличие personal_information/personal_information/personal_information.json и your_instagram_activity/messages/inbox/.",
            file=sys.stderr,
        )
        return 1

    last_pct: list[int] = [-1]

    def progress(msg: str, pct: int | None = None) -> None:
        if pct is not None and pct != last_pct[0]:
            print(f"Общий прогресс: {pct}%", flush=True)
            last_pct[0] = pct
        print(msg, flush=True)

    print(f"Экспорт Instagram: {export_root}", flush=True)
    transcribe_fn = None if args.no_transcribe else transcribe_audio
    if args.no_transcribe:
        print("Транскрипция голосовых: отключена.", flush=True)
    else:
        try:
            import whisper  # noqa: F401
            print("Транскрипция голосовых: включена (Whisper).", flush=True)
        except ImportError:
            print("Предупреждение: Whisper не установлен. Голосовые будут помечены плейсхолдером. Установите: pip install openai-whisper (нужен ffmpeg).", flush=True)
            transcribe_fn = None

    md = build_markdown(export_root, transcribe_fn, log=progress)
    export_folder_name = os.path.basename(os.path.normpath(export_root))
    out_dir = os.path.join(args.output, export_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    out_name = export_folder_name + "_notebooklm.md"
    out_path = os.path.join(out_dir, out_name)
    progress("Запись в файл...", 98)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Общий прогресс: 100%", flush=True)
    print(f"Готово: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
