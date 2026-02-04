#!/usr/bin/env python3
"""
Конвертация папки экспорта Telegram (result.json + медиа) в Markdown для ИИ/NotebookLM.

Схема экспорта: https://core.telegram.org/import-export
- Вход: папка с result.json (или messages.html) и подпапками медиа
- Выход: dist/*.md — сообщения по времени, сервисные события, подписи к медиа
- Без пустых строк и лишних переносов: один перевод строки между блоками, внутри блока — без пустых строк
- Прогресс и возобновление с места остановки (state и log в dist/)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(SCRIPT_DIR, "src")
DEFAULT_DIST = os.path.join(SCRIPT_DIR, "dist")

def _format_elapsed(secs: float) -> str:
    """Форматирует секунды в MM:SS."""
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m}:{s:02d}"


# State и лог хранятся в output_dir (dist)
def _state_file(output_dir: str) -> str:
    return os.path.join(output_dir, ".export_state.json")


def _log_file(output_dir: str) -> str:
    return os.path.join(output_dir, "export.log")

# Лимиты для разбиения (NotebookLM ~500k слов на источник)
DEFAULT_MAX_WORDS_PER_FILE = 450_000
CHECKPOINT_EVERY = 200  # сохранять чекпоинт каждые N сообщений

# Папки медиа в экспорте Telegram (schema)
MEDIA_VOICE_DIRS = ("voice_messages",)
MEDIA_VIDEO_DIRS = ("round_video_messages", "video_files")

_WS_RE = re.compile(r"\s+")
_whisper_model = None


def _extract_text(obj) -> str:
    """Извлекает текст из поля text (строка, объект или массив с type/text)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return str(obj.get("text", ""))
    if isinstance(obj, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in obj
        )
    return str(obj)


def _normalize_text(s: str) -> str:
    """Нормализует текст: одна строка, без лишних пробелов."""
    if not s:
        return ""
    return _WS_RE.sub(" ", s).strip()


def _is_media_path(s: str) -> bool:
    """Путь к медиа (а не предупреждение). В экспорте пути могут быть заменены на warning (schema)."""
    if not s or not isinstance(s, str):
        return False
    return "/" in s or "\\" in s


def _media_label(media_type: str | None, path: str) -> str:
    """Подпись для медиа по схеме: sticker, video_message, voice_message, animation, video_file, audio_file."""
    if not path:
        return ""
    name = path if not _is_media_path(path) else os.path.basename(path)
    # Когда путь — предупреждение (File not included и т.п.), всё равно указываем тип по media_type
    if not _is_media_path(path):
        if media_type == "voice_message":
            return f"[Голосовое сообщение: {path}]"
        if media_type in ("video_message", "video_file"):
            return f"[Видео: {path}]"
        if media_type == "audio_file":
            return f"[Аудио: {path}]"
        if media_type == "animation":
            return f"[GIF/анимация: {path}]"
        if media_type == "sticker":
            return f"[Стикер: {path}]"
        return f"[Медиа: {path}]"
    if media_type == "voice_message":
        return f"[Голосовое сообщение: {name}]"
    if media_type in ("video_message", "video_file"):
        return f"[Видео: {name}]"
    if media_type == "audio_file":
        return f"[Аудио: {name}]"
    if media_type == "animation":
        return f"[GIF/анимация: {name}]"
    if media_type == "sticker":
        return f"[Стикер: {name}]"
    if path.lower().endswith((".jpg", ".jpeg", ".png")) or "photos" in path:
        return f"[Фото: {name}]"
    return f"[Файл: {name}]"


def scan_media_folders(export_root: str) -> dict[str, list[str]]:
    """
    Сканирует voice_messages/, round_video_messages/, video_files/ в папке экспорта.
    Возвращает {"voice": [полные_пути], "video": [полные_пути]}, отсортированные по имени.
    """
    out = {"voice": [], "video": []}
    for sub in MEDIA_VOICE_DIRS:
        d = os.path.join(export_root, sub)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith((".ogg", ".opus", ".m4a", ".mp3", ".wav")):
                    out["voice"].append(os.path.join(d, name))
    for sub in MEDIA_VIDEO_DIRS:
        d = os.path.join(export_root, sub)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith((".mp4", ".webm", ".mov")) and "_thumb" not in name.lower():
                    out["video"].append(os.path.join(d, name))
    return out


def _resolve_media_path(msg: dict, export_root: str, media_type: str, path_from_list: str | None) -> str | None:
    """Возвращает полный путь к файлу: из msg['file'] если путь валидный, иначе path_from_list."""
    file_val = msg.get("file")
    if file_val and _is_media_path(file_val):
        full = os.path.join(export_root, file_val)
        if os.path.isfile(full):
            return full
    return path_from_list if path_from_list and os.path.isfile(path_from_list) else None


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


# Действия и типы сообщений, которые не добавляем в dist
SKIP_ACTIONS = ("phone_call", "group_call", "invite_to_group_call", "send_payment")
SKIP_VIA_BOT = ("@wallet",)
SKIP_TEXT_PATTERNS = ("Crypto transfer", "✅ Crypto transfer")


def _should_skip_message(msg: dict) -> bool:
    """
    Пропускаем: денежные транзакции, звонки, пропущенные звонки, сообщения-только-вложения (файл/фото без текста).
    Голосовые и видео оставляем (будут с транскрипцией).
    """
    msg_type = msg.get("type", "message")
    action = msg.get("action", "")

    # Звонки и пропущенные (service + phone_call/group_call)
    if msg_type == "service" and action in SKIP_ACTIONS:
        return True

    # Денежные транзакции: via_bot @wallet или send_payment
    if msg.get("via_bot") in SKIP_VIA_BOT:
        return True
    if action == "send_payment":
        return True

    # Текст про крипто-перевод
    text = msg.get("text") or ""
    if not text and msg.get("text_entities"):
        text = " ".join(_extract_text(e) for e in msg["text_entities"])
    else:
        text = _extract_text(text)
    text = _normalize_text(str(text))
    for pat in SKIP_TEXT_PATTERNS:
        if pat in text:
            return True

    # Прикреплённые файлы: только файл/фото/стикер без текста (голос и видео оставляем)
    if not text:
        mt = msg.get("media_type")
        if mt in ("voice_message", "video_message", "video_file"):
            return False  # голос и видео — оставляем (транскрипция)
        if msg.get("file") or msg.get("photo"):
            return True  # файл/фото без текста — пропуск

    return False


def message_to_md(
    msg: dict,
    export_root: str,
    voice_text: str | None = None,
    video_text: str | None = None,
) -> str:
    """
    Один объект Message → блок Markdown (schema: Message, type message|service).
    Учитывает: text/text_entities, photo, file, media_type, action, information_text,
    reply_to_message_id, edited, forwarded_from, saved_from, poll, contact_information.
    """
    date = msg.get("date", "")
    if isinstance(date, str) and "T" in date:
        date = date.replace("T", " ")[:16]
    author = msg.get("from") or msg.get("actor") or ""
    msg_type = msg.get("type", "message")
    lines = [f"### {date} | {author}"]

    # Сервисное сообщение (action): используем information_text или action для контекста
    if msg_type == "service":
        action = msg.get("action", "")
        info = msg.get("information_text") or ""
        if info:
            lines.append(info)
        elif action:
            lines.append(f"[Событие: {action}]")

    # Текст сообщения (строка или text_entities)
    text = msg.get("text") or ""
    if not text and msg.get("text_entities"):
        text = " ".join(_extract_text(e) for e in msg["text_entities"])
    else:
        text = _extract_text(text)
    text = _normalize_text(text)
    if text:
        if text.startswith("#"):
            text = "\\" + text
        lines.append(text)

    # Ответ на сообщение (reply_to_message_id)
    reply_id = msg.get("reply_to_message_id")
    if reply_id is not None:
        lines.append(f"[Ответ на сообщение #{reply_id}]")

    # Редактирование (edited)
    edited = msg.get("edited")
    if edited:
        if isinstance(edited, str) and "T" in edited:
            edited = edited.replace("T", " ")[:16]
        lines.append(f"(ред. {edited})")

    # Пересланное / сохранённое
    if msg.get("forwarded_from"):
        lines.append(f"[Переслано из: {msg['forwarded_from']}]")
    if msg.get("saved_from"):
        lines.append(f"[Сохранено из: {msg['saved_from']}]")

    # Опрос (poll)
    poll = msg.get("poll")
    if isinstance(poll, dict) and poll.get("question"):
        lines.append(f"[Опрос: {poll['question']}]")

    # Контакт (contact_information)
    contact = msg.get("contact_information")
    if isinstance(contact, dict):
        name = contact.get("first_name", "") or contact.get("last_name", "") or "Контакт"
        lines.append(f"[Контакт: {name}]")

    # Медиа: file, photo. Если есть транскрипт голоса/видео — подпись "[Медиа: ...]" не выводим.
    media_type = msg.get("media_type")
    file_path = msg.get("file")
    photo_path = msg.get("photo")
    if file_path:
        if media_type == "voice_message":
            if voice_text:
                lines.append(f"({_normalize_text(voice_text)})")
            else:
                lines.append(_media_label(media_type, file_path))
        elif media_type in ("video_message", "video_file"):
            if video_text:
                lines.append(f"({_normalize_text(video_text)})")
            else:
                lines.append(_media_label(media_type, file_path))
        else:
            lines.append(_media_label(media_type, file_path))
    if photo_path and photo_path != file_path:
        lines.append(_media_label("photo", photo_path))

    # Убираем пустые строки и лишние переносы
    lines = [x for x in lines if x.strip()]
    return re.sub(r"\n+", "\n", "\n".join(lines)).strip()


def count_messages(export_root: str) -> int:
    """Подсчёт общего числа сообщений в result.json (для прогресса)."""
    result_path = os.path.join(export_root, "result.json")
    if not os.path.isfile(result_path):
        return 0
    total = 0
    with open(result_path, "rb") as f:
        # Один чат: result = Chat с полем messages
        try:
            import ijson
            try:
                for _ in ijson.items(f, "messages.item"):
                    total += 1
                return total
            except ijson.JSONError:
                pass
        except ImportError:
            pass
        f.seek(0)
    # Fallback: полная загрузка (для полного экспорта с chats.list)
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "chats" in data and "list" in data["chats"]:
        for chat in data["chats"]["list"]:
            total += len(chat.get("messages", []))
        return total
    if "messages" in data:
        return len(data["messages"])
    return 0


def load_state(export_root: str, output_dir: str) -> dict:
    """Загружает чекпоинт, если он есть и совпадает с текущим экспортом."""
    path = _state_file(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("export_path") != os.path.abspath(export_root):
            return {}
        return state
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(
    export_root: str, output_dir: str, last_index: int, total: int, part_index: int
) -> None:
    """Сохраняет чекпоинт для возобновления."""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "export_path": os.path.abspath(export_root),
        "last_message_index": last_index,
        "total": total,
        "part_index": part_index,
    }
    path = _state_file(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


class _Logger:
    """Пишет в stdout и в dist/export.log."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path = _log_file(output_dir)
        self._file = None

    def _ensure_open(self) -> None:
        if self._file is None and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self._file = open(self.log_path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(msg, flush=True)
        self._ensure_open()
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def progress_line(self, msg: str) -> None:
        """Печатает строку прогресса в stdout без переноса (обновление на месте). В лог не пишем."""
        print(f"\r{msg}  ", end="", flush=True)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def _html_date_to_iso(title: str) -> str:
    """Из title '30.10.2025 11:06:02 UTC+02:00' → '2025-10-30 11:06' для заголовка MD."""
    if not title or not isinstance(title, str):
        return ""
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})", title.strip())
    if m:
        d, mo, y, h, mi = m.groups()
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)} {h.zfill(2)}:{mi}"
    return title[:16] if len(title) >= 16 else (title or "")


def _parse_html_message(div_el) -> dict | None:
    """
    Парсит один div.message из messages.html в dict, совместимый с message_to_md.
    Структура: .message.service → .body.details; .message.default → .body .date.details[title], .from_name, .text, .media_wrap.
    """
    classes = div_el.get("class") or []
    if "message" not in classes:
        return None
    msg = {"type": "message", "date": "", "from": "", "text": ""}
    body = div_el.find("div", class_="body")
    if not body:
        return None
    if "service" in classes:
        msg["type"] = "service"
        details = body.find("div", class_="details")
        if details:
            raw = details.get_text(strip=True)
            msg["information_text"] = raw
            m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
            if m:
                d, mon, y = m.groups()
                months = "January February March April May June July August September October November December".split()
                try:
                    mo = str(months.index(mon) + 1).zfill(2)
                    msg["date"] = f"{y}-{mo}-{d.zfill(2)} 00:00"
                except ValueError:
                    pass
        return msg
    date_el = body.find("div", class_="date", attrs={"title": True})
    if date_el and date_el.get("title"):
        msg["date"] = _html_date_to_iso(date_el["title"])
    from_el = body.find("div", class_="from_name")
    if from_el:
        msg["from"] = from_el.get_text(strip=True)
    text_el = body.find("div", class_="text")
    if text_el:
        msg["text"] = text_el.get_text(separator=" ", strip=True)
    forwarded = body.find("div", class_="forwarded")
    if forwarded:
        fwd_from = forwarded.find("div", class_="from_name")
        if fwd_from:
            msg["forwarded_from"] = fwd_from.get_text(strip=True)
        fwd_text = forwarded.find("div", class_="text")
        if fwd_text and not msg["text"]:
            msg["text"] = fwd_text.get_text(separator=" ", strip=True)
    media_wrap = body.find("div", class_="media_wrap")
    if media_wrap:
        voice = media_wrap.find("a", class_=lambda c: c and "media_voice_message" in c)
        video = media_wrap.find("a", class_=lambda c: c and "media_video" in c)
        photo = media_wrap.find("div", class_=lambda c: c and "media_photo" in (c or []))
        if voice and voice.get("href"):
            msg["file"] = voice["href"]
            msg["media_type"] = "voice_message"
        if video and video.get("href"):
            msg["file"] = msg.get("file") or video["href"]
            if not msg.get("media_type"):
                msg["media_type"] = "video_message"
        if photo:
            title_el = photo.find("div", class_="title")
            desc_el = photo.find("div", class_="description")
            if title_el:
                title_text = title_el.get_text(strip=True)
                if "Sticker" in title_text:
                    msg["media_type"] = msg.get("media_type") or "sticker"
                if not msg.get("file") and not msg.get("photo"):
                    msg["photo"] = desc_el.get_text(strip=True) if desc_el else title_text
    return msg


def count_messages_html(export_root: str) -> int:
    """Подсчёт сообщений в messages.html."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Для работы с messages.html установите beautifulsoup4: pip install beautifulsoup4"
        ) from None
    html_path = os.path.join(export_root, "messages.html")
    if not os.path.isfile(html_path):
        return 0
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return len(soup.find_all("div", class_=lambda c: c and "message" in (c or [])))


def iter_messages_from_html(export_root: str):
    """Итератор по сообщениям из messages.html (один чат)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Для работы с messages.html установите beautifulsoup4: pip install beautifulsoup4"
        ) from None
    html_path = os.path.join(export_root, "messages.html")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for div in soup.find_all("div", class_=lambda c: c and "message" in (c or [])):
        parsed = _parse_html_message(div)
        if parsed:
            yield parsed


def iter_messages_single_chat(export_root: str):
    """Итератор по сообщениям одного чата (result.json = один Chat)."""
    result_path = os.path.join(export_root, "result.json")
    with open(result_path, "rb") as f:
        try:
            import ijson
            for msg in ijson.items(f, "messages.item"):
                yield msg
        except ImportError:
            data = json.load(f)
            for msg in data.get("messages", []):
                yield msg


def iter_messages_full_export(data: dict):
    """Итератор по всем сообщениям полного экспорта (chats.list), по чатам."""
    chats = data.get("chats", {}).get("list", [])
    for chat in chats:
        name = chat.get("name", "unknown")
        for msg in chat.get("messages", []):
            msg["_chat_name"] = name
            yield msg


def run_export(
    export_root: str,
    output_dir: str,
    max_words_per_file: int,
    resume: bool,
    progress_interval: int = 100,
    transcribe: bool = True,
) -> list[str]:
    """
    Конвертирует экспорт в MD с прогрессом и чекпоинтами.
    Поддерживает result.json или messages.html (fallback).
    Возвращает список созданных .md файлов.
    """
    result_path = os.path.join(export_root, "result.json")
    html_path = os.path.join(export_root, "messages.html")

    if os.path.isfile(result_path):
        # JSON: один чат или полный экспорт
        with open(result_path, "r", encoding="utf-8") as f:
            peek = json.load(f)
        if "chats" in peek and "list" in peek["chats"]:
            total = sum(len(c.get("messages", [])) for c in peek["chats"]["list"])
            message_iter = iter_messages_full_export(peek)
            single_chat = False
        else:
            total = count_messages(export_root)
            message_iter = iter_messages_single_chat(export_root)
            single_chat = True
    elif os.path.isfile(html_path):
        # HTML fallback: один чат из messages.html
        total = count_messages_html(export_root)
        message_iter = iter_messages_from_html(export_root)
        single_chat = True
    else:
        raise FileNotFoundError(
            f"В папке нет result.json ни messages.html: {export_root}. "
            "При экспорте включите JSON или используйте папку с messages.html."
        )

    if total == 0:
        print("Сообщений не найдено.", flush=True)
        return []

    log = _Logger(output_dir)
    try:
        state = load_state(export_root, output_dir) if resume else {}
        start_index = state.get("last_message_index", 0)
        current_part = state.get("part_index", 1)
        # Имя файла в dist = имя папки экспорта (например ChatExport_test → ChatExport_test_1.md)
        out_prefix = os.path.basename(os.path.normpath(export_root)) or ("chat" if single_chat else "export")

        if transcribe:
            try:
                import whisper  # noqa: F401
            except ImportError:
                log.log(
                    "Предупреждение: Whisper не установлен. Транскрипция пропущена. Установите: pip install openai-whisper (нужен ffmpeg)."
                )
                transcribe = False
        media_folders = scan_media_folders(export_root) if transcribe else {"voice": [], "video": []}
        voice_idx = 0
        video_idx = 0

        os.makedirs(output_dir, exist_ok=True)
        created = []
        current_blocks = []
        current_words = 0
        sep = "\n"  # без пустых строк между сообщениями
        part_index = current_part
        last_checkpoint = start_index
        idx = -1

        if transcribe:
            log.log("Транскрипция голосовых и видео: включена (Whisper).")
        log.log(f"Всего сообщений: {total:,}. Начало с индекса: {start_index:,}.")
        if transcribe:
            log.log("При большом числе голосовых/видео между обновлениями возможны паузы (транскрипция).")
        start_time = time.time()

        for msg in message_iter:
            idx += 1
            if idx < start_index:
                continue

            if _should_skip_message(msg):
                if transcribe:
                    mt = msg.get("media_type")
                    if mt == "voice_message":
                        voice_idx += 1
                    if mt in ("video_message", "video_file"):
                        video_idx += 1
                continue

            voice_path = None
            video_path = None
            if transcribe:
                mt = msg.get("media_type")
                if mt == "voice_message":
                    voice_path = _resolve_media_path(
                        msg,
                        export_root,
                        "voice",
                        media_folders["voice"][voice_idx] if voice_idx < len(media_folders["voice"]) else None,
                    )
                    voice_idx += 1
                if mt in ("video_message", "video_file"):
                    video_path = _resolve_media_path(
                        msg,
                        export_root,
                        "video",
                        media_folders["video"][video_idx] if video_idx < len(media_folders["video"]) else None,
                    )
                    video_idx += 1
            if voice_path or video_path:
                elapsed = _format_elapsed(time.time() - start_time)
                log.progress_line(f"  Транскрипция голос/видео ({idx + 1:,}/{total:,}) — {elapsed}")
            voice_text = transcribe_audio(voice_path) if voice_path else ""
            video_text = transcribe_audio(video_path) if video_path else ""
            block = message_to_md(
                msg,
                export_root,
                voice_text=voice_text or None,
                video_text=video_text or None,
            )
            block_words = len(block.split())

            if current_blocks and (current_words + block_words > max_words_per_file):
                out_name = os.path.join(output_dir, f"{out_prefix}_{part_index}.md")
                with open(out_name, "w", encoding="utf-8") as f:
                    f.write(sep.join(current_blocks))
                created.append(os.path.abspath(out_name))
                log.log(f"  Записана часть {part_index}: {out_name}")
                current_blocks = []
                current_words = 0
                part_index += 1

            current_blocks.append(block)
            current_words += block_words

            if (idx - last_checkpoint) >= CHECKPOINT_EVERY:
                save_state(export_root, output_dir, idx + 1, total, part_index)
                last_checkpoint = idx
            pct = (idx + 1) / total * 100.0
            elapsed = _format_elapsed(time.time() - start_time)
            if (idx + 1) % 10 == 0:
                log.progress_line(f"  Прогресс: {pct:.1f}% ({idx + 1:,} / {total:,}) — {elapsed}")
            if progress_interval and (idx + 1) % progress_interval == 0:
                log.log(f"  Прогресс: {pct:.1f}% ({idx + 1:,} / {total:,}) — {elapsed}")

        if current_blocks:
            out_name = os.path.join(output_dir, f"{out_prefix}_{part_index}.md")
            with open(out_name, "w", encoding="utf-8") as f:
                f.write(sep.join(current_blocks))
            created.append(os.path.abspath(out_name))
            log.log(f"  Записана часть {part_index}: {out_name}")

        save_state(export_root, output_dir, total, total, part_index + 1)
        log.log(f"Готово: {len(created)} файл(ов). Прогресс: 100%")
        # Удаляем state при успешном завершении
        state_path = _state_file(output_dir)
        if os.path.isfile(state_path):
            try:
                os.remove(state_path)
            except OSError:
                pass
        return created
    finally:
        log.close()


def main() -> None:
    # Подкоманда clean: очистить папку dist
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        parser_clean = argparse.ArgumentParser(description="Очистить папку dist")
        parser_clean.add_argument(
            "--output-dir",
            "-o",
            default=DEFAULT_DIST,
            help=f"Папка для очистки (по умолчанию: {DEFAULT_DIST})",
        )
        args_clean = parser_clean.parse_args(sys.argv[2:])
        folder = args_clean.output_dir if os.path.isabs(args_clean.output_dir) else os.path.join(SCRIPT_DIR, args_clean.output_dir)
        if not os.path.isdir(folder):
            print(f"Папка не найдена: {folder}", flush=True)
            sys.exit(1)
        removed = 0
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        print(f"Очищено {folder}: удалено файлов {removed}", flush=True)
        return

    parser = argparse.ArgumentParser(
        description="Конвертация папки экспорта Telegram (result.json + медиа) в Markdown с прогрессом и возобновлением."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_SRC,
        help=f"Папка экспорта (по умолчанию: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_DIST,
        help=f"Папка для .md (по умолчанию: {DEFAULT_DIST})",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS_PER_FILE,
        help=f"Макс. слов в одном .md (по умолчанию: {DEFAULT_MAX_WORDS_PER_FILE})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Игнорировать чекпоинт и начать с начала",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        metavar="N",
        help="Писать в лог прогресс каждые N сообщений (0 = только в конце)",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Не транскрибировать голосовые и видео (по умолчанию транскрипция включена)",
    )
    args = parser.parse_args()

    export_root = args.input if os.path.isabs(args.input) else os.path.join(SCRIPT_DIR, args.input)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(SCRIPT_DIR, args.output_dir)

    if not os.path.isdir(export_root):
        print(f"Ошибка: папка не найдена: {export_root}", file=sys.stderr)
        sys.exit(1)

    # Если в указанной папке нет result.json ни messages.html — ищем в подпапках
    has_json = os.path.isfile(os.path.join(export_root, "result.json"))
    has_html = os.path.isfile(os.path.join(export_root, "messages.html"))
    if not has_json and not has_html:
        subdirs = [
            os.path.join(export_root, d)
            for d in sorted(os.listdir(export_root))
            if os.path.isdir(os.path.join(export_root, d)) and not d.startswith(".")
        ]
        for sub in subdirs:
            if os.path.isfile(os.path.join(sub, "result.json")):
                export_root = sub
                print(f"Найден экспорт (JSON): {export_root}", flush=True)
                break
            if os.path.isfile(os.path.join(sub, "messages.html")):
                export_root = sub
                print(f"Найден экспорт (HTML): {export_root}", flush=True)
                break
        else:
            print(
                "Ошибка: в папке нет result.json ни messages.html. "
                "Положите сюда папку экспорта Telegram (с result.json или messages.html).",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        files = run_export(
            export_root=export_root,
            output_dir=output_dir,
            max_words_per_file=args.max_words,
            resume=not args.no_resume,
            progress_interval=args.progress_interval,
            transcribe=not args.no_transcribe,
        )
        for p in files:
            print(f"  {p}")
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
