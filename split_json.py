#!/usr/bin/env python3
"""
Потоковое разбиение/конвертация большого JSON (массив объектов) в части .md или .json (ijson).

Оптимизации JSON → MD:
- Потоковое чтение (ijson): файл не загружается целиком в память.
- Один проход конвертации: каждый объект конвертируется в MD один раз, при записи — только join.
- Разбиение по лимитам: части пишутся по мере накопления.
"""

from __future__ import annotations

import argparse
import ijson
import json
import os
import re
import sys
from datetime import datetime

# Путь к config.json рядом со скриптом (работает при любом текущем каталоге)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_CONFIG = {
    "max_file_size_mb": 150.0,
    "max_objects_per_file": 400_000,
    "max_words_per_file": 450_000,  # NotebookLM ~500k limit, margin for safety
    "array_path": "messages",
    "output_format": "md",
    "author_at_top": True,
    "skip_empty_messages": True,
}


def load_config() -> dict:
    """Читает config.json из папки скрипта, при отсутствии — значения по умолчанию."""
    if not os.path.isfile(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = DEFAULT_CONFIG.copy()
        for key in DEFAULT_CONFIG:
            if key in data:
                out[key] = data[key]
        return out
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    """Сохраняет настройки в config.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _extract_text(obj) -> str:
    """
    Извлекает обычный текст из поля text (строка, объект или массив с type/text).
    Используется для экспорта Telegram.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return str(obj.get("text", ""))
    if isinstance(obj, list):
        parts = []
        for item in obj:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(str(p) for p in parts)
    return str(obj)


_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    """
    Нормализует текст для NotebookLM:
    - убирает переносы строк и любые повторяющиеся пробельные символы
    - удаляет пустые строки (как частный случай)
    """
    if not s:
        return ""
    return _WS_RE.sub(" ", s).strip()


def obj_to_md(obj: dict) -> str:
    """
    Конвертирует один объект (сообщение Telegram и т.п.) в компактный блок Markdown:
    - без пустых строк
    - без переносов внутри текста (в одну строку)
    """
    date = obj.get("date", "")
    if date and "T" in str(date):
        date = str(date).replace("T", " ")[:16]
    author = obj.get("from") or obj.get("actor") or ""
    text = obj.get("text") or ""
    if not text and obj.get("text_entities"):
        text = " ".join(_extract_text(e) for e in obj["text_entities"])
    else:
        text = _extract_text(text)
    text = _normalize_text(text)
    lines = [f"### {date} | {author}"]
    if text:
        if text.startswith("#"):
            text = "\\" + text
        lines.append(text)
    return "\n".join(lines)


def json_to_txt(
    input_path: str,
    output_path: str | None = None,
    output_dir: str = "dist",
    array_path: str = "",
    format_: str = "chat",
    progress_interval: int = 50_000,
) -> str:
    """
    Потоково конвертирует JSON (массив объектов) в TXT.

    Args:
        input_path: путь к входному JSON
        output_path: полный путь к выходному .txt (если None — dist/имя_файла.txt)
        output_dir: папка для выхода (если output_path не задан)
        array_path: путь к массиву (например messages для Telegram)
        format_: "chat" — дата | автор: текст (для Telegram); "jsonl" — по одному JSON-объекту на строку
        progress_interval: интервал вывода прогресса (0 = без вывода)

    Returns:
        Путь к созданному .txt файлу.
    """
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base}.txt")

    ijson_prefix = f"{array_path}.item" if array_path else "item"
    total_read = 0

    if progress_interval > 0:
        print(f"  Конвертация в TXT: {os.path.basename(input_path)} → {os.path.basename(output_path)}", flush=True)

    with open(input_path, "rb") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            parser = ijson.items(fin, ijson_prefix)

            for obj in parser:
                total_read += 1

                if format_ == "chat":
                    # Формат для Telegram: 2023-12-27 10:28 | Автор: текст
                    date = obj.get("date", "")
                    if "T" in str(date):
                        date = str(date).replace("T", " ")[:16]  # YYYY-MM-DD HH:MM
                    author = obj.get("from") or obj.get("actor") or ""
                    text = obj.get("text") or ""
                    if not text and obj.get("text_entities"):
                        text = " ".join(_extract_text(e) for e in obj["text_entities"])
                    else:
                        text = _extract_text(text)
                    text = _normalize_text(text)
                    line = f"{date} | {author}: {text}"
                    fout.write(line.rstrip() + "\n")
                else:
                    # jsonl: один JSON-объект на строку
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                if progress_interval > 0 and total_read % progress_interval == 0:
                    print(f"  Обработано: {total_read:,}", flush=True)

    if progress_interval > 0:
        print(f"  Готово: {total_read:,} записей → {output_path}", flush=True)

    return os.path.abspath(output_path)


# Чекпоинты для возобновления
CHECKPOINT_EVERY = 10_000
STATE_FILE = ".split_json_state.json"
LOG_FILE = "split_json.log"


def _state_path(output_dir: str) -> str:
    return os.path.join(output_dir, STATE_FILE)


def _log_path(output_dir: str) -> str:
    return os.path.join(output_dir, LOG_FILE)


def _load_split_state(input_path: str, output_dir: str) -> dict:
    """Загружает чекпоинт, если он есть и совпадает с входным файлом."""
    path = _state_path(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("input_path") != os.path.abspath(input_path):
            return {}
        return state
    except (json.JSONDecodeError, OSError):
        return {}


def _save_split_state(
    input_path: str,
    output_dir: str,
    output_prefix: str,
    last_object_index: int,
    part_index: int,
    current_md_blocks: list[str],
    current_objects: list,
    current_size: int,
    current_words: int,
    output_format: str,
) -> None:
    """Сохраняет чекпоинт для возобновления."""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "input_path": os.path.abspath(input_path),
        "output_prefix": output_prefix,
        "last_object_index": last_object_index,
        "part_index": part_index,
        "current_size": current_size,
        "current_words": current_words,
        "output_format": output_format,
    }
    if output_format == "md":
        state["current_md_blocks"] = current_md_blocks
    else:
        # Для JSON не сохраняем объекты (могут быть огромными), только количество
        state["current_object_count"] = len(current_objects)
    with open(_state_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


class _SplitLogger:
    """Пишет в stdout и в dist/split_json.log."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path = _log_path(output_dir)
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

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def split_json(
    input_path: str,
    output_dir: str = "dist",
    output_prefix: str = "part",
    max_file_size_mb: float | None = None,
    max_objects_per_file: int | None = None,
    max_words_per_file: int | None = None,
    array_path: str = "",
    progress_interval: int = 50_000,
    output_format: str = "md",
    author_at_top: bool = True,
    skip_empty_messages: bool = True,
    resume: bool = True,
) -> list[str]:
    """
    Читает JSON-массив потоково и сохраняет части в output_dir (part_1.json или part_1.md).
    Поддерживает возобновление после прерывания (state и log в output_dir).

    Args:
        output_format: "md" — Markdown (дата | автор: текст); "json" — компактный JSON.
        author_at_top: для MD — вынести автора один раз в начало, в блоках только ### дата и текст.
        skip_empty_messages: для MD — не писать блоки без текста.
        max_words_per_file: для MD — макс. слов в одном файле (NotebookLM ~500k, с запасом 450k).
        resume: возобновить с чекпоинта при прерывании.
    """
    os.makedirs(output_dir, exist_ok=True)
    has_limit = (
        max_file_size_mb is not None
        or max_objects_per_file is not None
        or (output_format == "md" and max_words_per_file is not None)
    )
    if not has_limit:
        raise ValueError(
            "Укажите хотя бы одно ограничение: max_file_size_mb, max_objects_per_file или max_words_per_file (для md)"
        )

    ext = ".md" if output_format == "md" else ".json"
    max_size_bytes = int(max_file_size_mb * 1024 * 1024) if max_file_size_mb else None
    created_files: list[str] = []
    log = _SplitLogger(output_dir)

    # MD: разделитель между блоками (короче чем \n---\n)
    sep = "\n\n"
    sep_len = len(sep.encode("utf-8"))

    # Загрузка чекпоинта для возобновления
    state = _load_split_state(input_path, output_dir) if resume else {}
    start_index = 0
    current_md_blocks: list[str] = []
    current_objects: list = []
    current_size = 0
    current_words = 0
    part_index = 1

    if state:
        last_idx = state.get("last_object_index", 0)
        if output_format == "md":
            current_md_blocks = state.get("current_md_blocks", [])
            start_index = last_idx - len(current_md_blocks)
        else:
            obj_count = state.get("current_object_count", 0)
            start_index = last_idx - obj_count
        start_index = max(0, start_index)
        part_index = state.get("part_index", 1)
        current_size = state.get("current_size", 0)
        current_words = state.get("current_words", 0)
        log.log(f"Возобновление с объекта {start_index:,}, часть {part_index}")

    # MD: накапливаем готовые строки. JSON: накапливаем объекты.

    total_read = start_index
    last_checkpoint = start_index

    def write_part() -> None:
        nonlocal current_objects, current_size, part_index, current_md_blocks, current_words
        n = len(current_md_blocks) if output_format == "md" else len(current_objects)
        if n == 0:
            return
        out_name = os.path.join(output_dir, f"{output_prefix}_{part_index}{ext}")
        if progress_interval > 0:
            log.log(f"  Запись части {part_index}: {os.path.basename(out_name)} ({n:,} объектов)...")
        with open(out_name, "w", encoding="utf-8") as f:
            if output_format == "md":
                if author_at_top and current_md_blocks:
                    first_block = current_md_blocks[0]
                    first_line = first_block.split("\n", 1)[0]
                    author = first_line.split(" | ", 1)[1] if " | " in first_line else ""
                    f.write(f"Source: {author}\n\n")
                    transformed = []
                    for block in current_md_blocks:
                        first_ln = block.split("\n", 1)[0]
                        date_only = first_ln.split(" | ", 1)[0]  # "### 2025-01-01 01:34"
                        rest = block.split("\n", 1)[1] if "\n" in block else ""
                        transformed.append(date_only + ("\n" + rest if rest else ""))
                    f.write(sep.join(transformed))
                else:
                    f.write(sep.join(current_md_blocks))
            else:
                json.dump(current_objects, f, ensure_ascii=False, separators=(",", ":"))
        created_files.append(os.path.abspath(out_name))
        if progress_interval > 0:
            size_mb = current_size / (1024 * 1024)
            log.log(f"  Готово: {os.path.basename(out_name)} ({size_mb:.1f} МБ)")
        part_index += 1
        current_objects = []
        current_md_blocks = []
        current_size = 0
        current_words = 0

    # Путь к элементам массива для ijson: "item" для корня, "messages.item" для obj["messages"]
    ijson_prefix = f"{array_path}.item" if array_path else "item"

    if progress_interval > 0:
        log.log(f"  Чтение и разбиение: {os.path.basename(input_path)}")

    try:
        with open(input_path, "rb") as f:
            # Потоковый парсинг массива: объекты приходят по одному
            parser = ijson.items(f, ijson_prefix)

            for obj in parser:
                if total_read < start_index:
                    total_read += 1
                    if progress_interval > 0 and total_read % progress_interval == 0:
                        log.log(f"  Пропуск (resume): {total_read:,}")
                    continue

                if output_format == "md":
                    obj_content = obj_to_md(obj)
                    if skip_empty_messages and "\n" not in obj_content.strip():
                        total_read += 1
                        if progress_interval > 0 and total_read % progress_interval == 0:
                            log.log(f"  Обработано объектов: {total_read:,}")
                        continue
                    obj_size = len(obj_content.encode("utf-8")) + (sep_len if current_md_blocks else 0)
                    obj_words = len(obj_content.split())
                else:
                    obj_json = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    obj_size = len(obj_json.encode("utf-8"))

                # Сначала сбросить часть, если уже набрали лимит по размеру (жёсткий лимит)
                if max_size_bytes and current_size >= max_size_bytes:
                    write_part()
                # Перед добавлением следующего объекта — сбросить, если он не влезет
                n_current = len(current_md_blocks) if output_format == "md" else len(current_objects)
                if max_size_bytes and n_current and (current_size + obj_size > max_size_bytes):
                    write_part()
                # Лимит по количеству объектов
                n_current = len(current_md_blocks) if output_format == "md" else len(current_objects)
                if max_objects_per_file and n_current >= max_objects_per_file:
                    write_part()
                # Лимит по словам (для MD, под NotebookLM ~500k)
                if (
                    output_format == "md"
                    and max_words_per_file
                    and current_md_blocks
                    and (current_words + obj_words > max_words_per_file)
                ):
                    write_part()

                if output_format == "md":
                    current_md_blocks.append(obj_content)
                    current_words += obj_words
                else:
                    current_objects.append(obj)
                current_size += obj_size
                total_read += 1

                if progress_interval > 0 and total_read % progress_interval == 0:
                    log.log(f"  Обработано объектов: {total_read:,}")

                # Чекпоинт для возобновления
                if (total_read - last_checkpoint) >= CHECKPOINT_EVERY:
                    _save_split_state(
                        input_path,
                        output_dir,
                        output_prefix,
                        total_read,
                        part_index,
                        current_md_blocks,
                        current_objects,
                        current_size,
                        current_words,
                        output_format,
                    )
                    last_checkpoint = total_read

        if current_objects or current_md_blocks:
            write_part()

        # В первый файл добавляем краткий отчёт с общим количеством сообщений.
        if output_format == "md" and created_files:
            first_path = created_files[0]
            tmp_path = first_path + ".tmp"
            report_line = f"### Отчёт | Всего сообщений: {total_read:,}"
            with open(first_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
                fout.write(report_line + "\n\n")
                fout.write(fin.read())
            os.replace(tmp_path, first_path)

        if progress_interval > 0:
            log.log(f"  Всего обработано: {total_read:,} объектов, частей: {len(created_files)}")

        # Удаляем state при успешном завершении
        state_path = _state_path(output_dir)
        if os.path.isfile(state_path):
            try:
                os.remove(state_path)
            except OSError:
                pass

        return created_files
    finally:
        log.close()


def main() -> None:
    config = load_config()

    # Подкоманда config: обновить config.json через консоль
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        parser_c = argparse.ArgumentParser(description="Обновить настройки в config.json")
        parser_c.add_argument(
            "--max-size-mb",
            type=float,
            default=None,
            metavar="MB",
            help="Максимальный размер одного выходного файла в МБ",
        )
        parser_c.add_argument(
            "--max-objects",
            type=int,
            default=None,
            metavar="N",
            help="Максимальное количество объектов в одном файле",
        )
        parser_c.add_argument(
            "--max-words",
            type=int,
            default=None,
            metavar="N",
            help="Макс. слов в одном .md файле (NotebookLM ~500k, по умолчанию 450k с запасом)",
        )
        parser_c.add_argument(
            "--array-path",
            type=str,
            default=None,
            metavar="KEY",
            help="Путь к массиву в JSON (например: messages)",
        )
        parser_c.add_argument(
            "--output-format",
            "--format",
            type=str,
            choices=["md", "json"],
            default=None,
            dest="output_format",
            help="Выходной формат: md или json",
        )
        parser_c.add_argument(
            "--author-at-top",
            action="store_true",
            default=None,
            dest="author_at_top",
            help="В MD вынести автора один раз в начало (по умолчанию: True)",
        )
        parser_c.add_argument(
            "--no-author-at-top",
            action="store_false",
            dest="author_at_top",
            help="В MD не выносить автора в начало",
        )
        parser_c.add_argument(
            "--skip-empty-messages",
            action="store_true",
            default=None,
            dest="skip_empty_messages",
            help="Не писать в MD блоки без текста (по умолчанию: True)",
        )
        parser_c.add_argument(
            "--no-skip-empty-messages",
            action="store_false",
            dest="skip_empty_messages",
            help="Писать в MD и пустые блоки",
        )
        parser_c.add_argument(
            "--show",
            action="store_true",
            help="Показать текущий конфиг и выйти",
        )
        args_c = parser_c.parse_args(sys.argv[2:])
        has_any = (
            args_c.max_size_mb is not None,
            args_c.max_objects is not None,
            args_c.max_words is not None,
            args_c.array_path is not None,
            args_c.output_format is not None,
            args_c.author_at_top is not None,
            args_c.skip_empty_messages is not None,
        )
        if args_c.show and not any(has_any):
            print("Текущий config.json:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            return
        if args_c.max_size_mb is not None:
            config["max_file_size_mb"] = args_c.max_size_mb
        if args_c.max_objects is not None:
            config["max_objects_per_file"] = args_c.max_objects
        if args_c.max_words is not None:
            config["max_words_per_file"] = args_c.max_words
        if args_c.array_path is not None:
            config["array_path"] = args_c.array_path
        if args_c.output_format is not None:
            config["output_format"] = args_c.output_format
        if args_c.author_at_top is not None:
            config["author_at_top"] = args_c.author_at_top
        if args_c.skip_empty_messages is not None:
            config["skip_empty_messages"] = args_c.skip_empty_messages
        save_config(config)
        print("config.json обновлён:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        return

    # Подкоманда clean: очистить dist и/или src
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        parser_clean = argparse.ArgumentParser(description="Очистить папки dist и/или src")
        parser_clean.add_argument(
            "folders",
            nargs="+",
            choices=["dist", "src"],
            metavar="dist|src",
            help="dist и/или src",
        )
        args_clean = parser_clean.parse_args(sys.argv[2:])
        for name in args_clean.folders:
            folder = os.path.join(SCRIPT_DIR, name)
            if not os.path.isdir(folder):
                print(f"Папка не найдена: {name}/", flush=True)
                continue
            removed = 0
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            print(f"Очищено {name}/: удалено файлов {removed}", flush=True)
        return

    # Подкоманда to-txt: конвертация JSON в TXT
    if len(sys.argv) > 1 and sys.argv[1] == "to-txt":
        parser_txt = argparse.ArgumentParser(description="Конвертировать JSON (массив объектов) в TXT")
        parser_txt.add_argument(
            "input",
            nargs="?",
            default=None,
            help="Входной JSON (например src/data.json). По умолчанию — все .json в src/",
        )
        parser_txt.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            metavar="FILE",
            help="Выходной .txt файл (по умолчанию: dist/имя_входа.txt)",
        )
        parser_txt.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Папка для выхода (по умолчанию: dist)",
        )
        parser_txt.add_argument(
            "--array-path",
            type=str,
            default=None,
            metavar="KEY",
            help=f"Путь к массиву (конфиг: {config.get('array_path', '') or '(корень)'})",
        )
        parser_txt.add_argument(
            "--format",
            "-f",
            choices=["chat", "jsonl"],
            default="chat",
            help="chat — дата | автор: текст (Telegram); jsonl — по объекту на строку (по умолчанию: chat)",
        )
        args_txt = parser_txt.parse_args(sys.argv[2:])

        array_path = args_txt.array_path if args_txt.array_path is not None else (config.get("array_path") or "")
        output_dir = args_txt.output_dir or os.path.join(SCRIPT_DIR, "dist")

        if args_txt.input is not None:
            input_path = args_txt.input if os.path.isabs(args_txt.input) else os.path.join(SCRIPT_DIR, args_txt.input)
            if not os.path.isfile(input_path):
                print(f"Ошибка: файл не найден: {input_path}", file=sys.stderr)
                sys.exit(1)
            input_files = [input_path]
        else:
            src_dir = os.path.join(SCRIPT_DIR, "src")
            if not os.path.isdir(src_dir):
                print(f"Ошибка: папка не найдена: {src_dir}", file=sys.stderr)
                sys.exit(1)
            input_files = sorted(
                os.path.join(src_dir, f)
                for f in os.listdir(src_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(src_dir, f))
            )
            if not input_files:
                print("В папке src нет .json файлов", file=sys.stderr)
                sys.exit(1)

        created = []
        for inp in input_files:
            out = args_txt.output
            if out and len(input_files) > 1:
                out = None  # при нескольких входах — игнорируем общий --output
            try:
                path = json_to_txt(
                    input_path=inp,
                    output_path=out,
                    output_dir=output_dir,
                    array_path=array_path,
                    format_=args_txt.format,
                )
                created.append(path)
            except Exception as e:
                print(f"Ошибка при конвертации {inp}: {e}", file=sys.stderr)
                sys.exit(1)

        print(f"Создано TXT: {len(created)}")
        for p in created:
            print(f"  {p}")
        return

    parser = argparse.ArgumentParser(
        description="Потоковое разбиение большого JSON (массив объектов) на части (ijson)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Входной JSON-файл из src (например: src/data.json). По умолчанию — все .json в src/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Папка для выходных файлов (по умолчанию: dist рядом со скриптом)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        metavar="MB",
        help=f"Максимальный размер одного выходного файла в МБ (конфиг: {config['max_file_size_mb']})",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        metavar="N",
        help=f"Максимальное количество объектов в одном файле (конфиг: {config['max_objects_per_file']})",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        metavar="N",
        help=f"Макс. слов в одном .md (конфиг: {config.get('max_words_per_file', 450_000)})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Префикс имён выходных файлов (по умолчанию: имя входного файла без расширения)",
    )
    parser.add_argument(
        "--array-path",
        type=str,
        default=None,
        metavar="KEY",
        help=f"Путь к массиву в JSON (конфиг: {config['array_path'] or '(корень)'})",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["md", "json"],
        default=None,
        help="Выходной формат: md (Markdown) или json (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Игнорировать чекпоинт и начать с начала",
    )
    args = parser.parse_args()

    # Подставляем значения из config, если параметр не задан в консоли
    if args.format is None:
        args.format = config.get("output_format", "md")
    if args.max_size_mb is None:
        args.max_size_mb = config["max_file_size_mb"]
    if args.max_objects is None:
        args.max_objects = config["max_objects_per_file"]
    if args.max_words is None and args.format == "md":
        args.max_words = config.get("max_words_per_file")
    if args.array_path is None:
        args.array_path = config.get("array_path", "") or ""

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "dist")

    # Показываем применяемые лимиты (из конфига или консоли)
    limits = f"max_size={args.max_size_mb} МБ, max_objects={args.max_objects}"
    if args.format == "md" and args.max_words is not None:
        limits += f", max_words={args.max_words:,}"
    print(f"Лимиты: {limits}, array_path={repr(args.array_path) or '(корень)'}")
    sys.stdout.flush()

    # Список файлов для обработки (src — рядом со скриптом)
    if args.input is not None:
        input_path = args.input if os.path.isabs(args.input) else os.path.join(SCRIPT_DIR, args.input)
        if not os.path.isfile(input_path):
            print(f"Ошибка: файл не найден: {input_path}", file=sys.stderr)
            sys.exit(1)
        input_files = [input_path]
    else:
        src_dir = os.path.join(SCRIPT_DIR, "src")
        if not os.path.isdir(src_dir):
            print(f"Ошибка: папка не найдена: {src_dir}", file=sys.stderr)
            sys.exit(1)
        input_files = sorted(
            os.path.join(src_dir, f)
            for f in os.listdir(src_dir)
            if f.endswith(".json") and os.path.isfile(os.path.join(src_dir, f))
        )
        if not input_files:
            print("В папке src нет .json файлов", file=sys.stderr)
            sys.exit(1)

    if args.max_size_mb is not None and args.max_size_mb <= 0:
        print("Ошибка: --max-size-mb должно быть положительным", file=sys.stderr)
        sys.exit(1)

    if args.max_objects is not None and args.max_objects <= 0:
        print("Ошибка: --max-objects должно быть положительным", file=sys.stderr)
        sys.exit(1)
    if args.max_words is not None and args.max_words <= 0:
        print("Ошибка: --max-words должно быть положительным", file=sys.stderr)
        sys.exit(1)

    all_created: list[str] = []
    for input_path in input_files:
        # Префикс = имя файла без расширения (src/foo.json → foo → dist/foo_1.json, foo_2.json)
        prefix = args.prefix if args.prefix is not None else os.path.splitext(os.path.basename(input_path))[0]
        try:
            files = split_json(
                input_path=input_path,
                output_dir=args.output_dir,
                output_prefix=prefix,
                max_file_size_mb=args.max_size_mb,
                max_objects_per_file=args.max_objects,
                max_words_per_file=args.max_words,
                array_path=args.array_path,
                output_format=args.format,
                author_at_top=config.get("author_at_top", True),
                skip_empty_messages=config.get("skip_empty_messages", True),
                resume=not getattr(args, "no_resume", False),
            )
            ext = ".md" if args.format == "md" else ".json"
            print(f"{input_path} → {len(files)} файл(ов) {prefix}_*{ext}")
            all_created.extend(files)
        except Exception as e:
            print(f"Ошибка при обработке {input_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Всего создано файлов: {len(all_created)}")
    for p in all_created:
        print(f"  {p}")


if __name__ == "__main__":
    main()
