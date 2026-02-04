# Telegram & Instagram Export → NotebookLM

Turn Telegram or Instagram data exports into Markdown for NotebookLM. Put the export folder in **src/** (or pass its path). Output goes to **dist/**.

## Instagram Export (instagram_export_to_md.py)

1. **Instagram** — скачайте данные через Настройки → Аккаунт → Загрузка данных (JSON).
2. **instagram_export_to_md.py** находит папку экспорта в **src/** и собирает один `.md` с разделами: **Профиль**, **Социальные связи**, **Переписки** (по пользователям), **Активность** (лайки, комментарии, сохранённое, посты Threads), **Повторяющиеся темы**. Медиа не включаются; голосовые сообщения опционально транскрибируются через Whisper.
3. Результат: `dist/<имя_папки_экспорта>/<имя_папки_экспорта>_notebooklm.md`, готовый к загрузке в NotebookLM.

| Команда | Описание |
|--------|----------|
| `python3 instagram_export_to_md.py` | Конвертация из **src/** → **dist/** |
| `python3 instagram_export_to_md.py src/MyInstagramExport` | Указать папку экспорта |
| `python3 instagram_export_to_md.py -o /path/to/out` | Папка для выходного .md |
| `python3 instagram_export_to_md.py --no-transcribe` | Не транскрибировать голосовые (быстрее) |

## Telegram Export (export_to_md.py) — How it works

1. **Telegram Desktop** exports a folder with `result.json` (enable “Machine-readable JSON”) and media subfolders (`voice_messages/`, `photos/`, etc.).
2. **export_to_md.py** reads `result.json`, writes messages in order as Markdown (`### date | author` + text). Voice/video are transcribed with Whisper and inserted as text; other media become labels like `[Photo: …]`. Progress is saved in **dist/** so you can resume after interruption.
3. **split_json.py** is for a single large JSON file (no export folder): streams the array and splits it into `.md` or `.json` parts by size/word/object limits. Optional **config.json** sets limits and format.

NotebookLM limits: ~500k words per source; default split is 450k words per file.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# For voice/video transcription:
pip install openai-whisper   # and install ffmpeg
```

## Commands

### instagram_export_to_md.py — Instagram JSON → один .md

См. таблицу выше в разделе «Instagram Export».

### export_to_md.py — Telegram export folder → .md

| Command | Description |
|--------|-------------|
| `python3 export_to_md.py` | Convert folder (default: **src/**) → **dist/** |
| `python3 export_to_md.py src/MyExport` | Convert specific folder |
| `python3 export_to_md.py -o /path/to/out` | Custom output directory |
| `python3 export_to_md.py --max-words 450000` | Max words per .md file |
| `python3 export_to_md.py --no-resume` | Ignore checkpoint, start from beginning |
| `python3 export_to_md.py --no-transcribe` | Skip voice/video transcription (faster) |
| `python3 export_to_md.py --progress-interval 500` | Log progress every N messages (0 = only at end) |
| `python3 export_to_md.py clean` | Delete contents of **dist/** (default) or `-o` folder |

### split_json.py — raw JSON → split .md / .json

| Command | Description |
|--------|-------------|
| `python3 split_json.py` | All `.json` in **src/** → **dist/** |
| `python3 split_json.py src/file.json` | Split one file |
| `python3 split_json.py -o /path/to/out` | Custom output directory |
| `python3 split_json.py --no-resume` | Start from beginning (ignore checkpoint) |
| `python3 split_json.py config --show` | Show current **config.json** |
| `python3 split_json.py config --max-words 450000 --array-path messages` | Set limits / array path / format |
| `python3 split_json.py clean dist` | Clean **dist/** and/or **src/** (`clean dist`, `clean src`, or `clean dist src`) |
| `python3 split_json.py to-txt [input]` | Convert JSON array to one .txt (e.g. `date \| author: text`) |

Config options in **config.json**: `max_file_size_mb`, `max_objects_per_file`, `max_words_per_file`, `array_path`, `output_format` (md/json).
