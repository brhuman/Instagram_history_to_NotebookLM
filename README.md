# Instagram Export → NotebookLM

This project processes **Instagram data export only** (logs, DMs, profile, activity) and converts it into a single Markdown file for [NotebookLM](https://notebooklm.google/).

## How it works

1. **Instagram** — download your data via Settings → Account → Download your information (JSON format).
2. Put the export folder in **src/** (or pass its path as an argument).
3. Run **instagram_export_to_md.py** — the script builds one `.md` with sections:
   - **Profile** (name, username, email, date of birth, etc.)
   - **Connections** (following, followers, close friends)
   - **Conversations** (all inbox chats, messages in chronological order)
   - **Activity** (liked posts and comments, saved posts, Threads posts, your comments)
   - **Suggested topics**
4. Output: `dist/<export_folder_name>/<export_folder_name>_notebooklm.md` — ready to upload to NotebookLM.

Media (photos/videos) are not included; voice messages can optionally be transcribed via Whisper.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Optional, for voice transcription:
pip install openai-whisper   # and install ffmpeg on your system
```

## Commands

| Command | Description |
|--------|-------------|
| `python3 instagram_export_to_md.py` | Convert: export folder from **src/** → **dist/** |
| `python3 instagram_export_to_md.py src/MyInstagramExport` | Specify Instagram export folder |
| `python3 instagram_export_to_md.py -o /path/to/out` | Output directory for .md (default **dist/**) |
| `python3 instagram_export_to_md.py --no-transcribe` | Skip voice transcription (faster) |

## Instagram export requirements

The export folder must contain at its root:

- `personal_information/personal_information/personal_information.json`
- `your_instagram_activity/messages/inbox/`

The script can find this folder in nested subfolders inside **src/** as well.

---

This repo also contains **export_to_md.py** (Telegram export) and **split_json.py** (splitting large JSON files) — they are not used for Instagram and are kept for related use cases.
