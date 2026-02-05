# Instagram Export → NotebookLM

Takes an **Instagram data export** (JSON) and turns it into Markdown for [NotebookLM](https://notebooklm.google/): profile, connections, DMs, activity, and suggested topics. Voice messages can be transcribed with Whisper. **Default output:** one `.md` per chat (named by the other participant) plus one `00_profile_and_activity.md` in `dist/<export_name>_export/`. Use `--single-file` to get one big `.md` (or split by word limit) instead.

---

## How it works

1. Request your data from Instagram (Settings → Account → Download your information, JSON).
2. Put the unzipped export folder in **src/** or pass its path.
3. Run **instagram_export_to_md.py**. It reads the export, normalizes text and emoji, optionally transcribes voice messages (Whisper), and writes Markdown to **dist/**.
4. **Default output:** folder `dist/<export_folder_name>_export/` with one `.md` per conversation (filename = other participant) and `00_profile_and_activity.md`. Use **`--single-file`** to get one big `.md` (or several by word limit) in `dist/<export_folder_name>/` instead.

Photos and videos are not included; only text, links, and transcribed audio.

---

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Optional, for voice transcription:
pip install openai-whisper   # requires ffmpeg
```

---

## Commands

**Basic run** (export from `src/` → `dist/`):

```bash
python3 instagram_export_to_md.py
```

**Custom input folder:**

```bash
python3 instagram_export_to_md.py /path/to/MyInstagramExport
```

**Custom output folder:**

```bash
python3 instagram_export_to_md.py -o /path/to/output
```

**Skip voice transcription** (faster):

```bash
python3 instagram_export_to_md.py --no-transcribe
```

**Limit words per file** (for NotebookLM; default 450000; use `0` for a single file):

```bash
python3 instagram_export_to_md.py --max-words 450000
```

**One big .md file** (or several by word limit) instead of one file per chat:

```bash
python3 instagram_export_to_md.py --single-file
```

**Collapse consecutive “Liked a message” / “Reacted …”** into one line:

```bash
python3 instagram_export_to_md.py --collapse-actions
```

**Whisper model** (default `medium`; `tiny` = faster, `large` = best quality):

```bash
python3 instagram_export_to_md.py --whisper-model small
```

**Example:** no transcription, custom output (default: one file per chat):

```bash
python3 instagram_export_to_md.py src/ -o dist/ --no-transcribe
```

---

## Whisper models (`--whisper-model`)

Default is **medium**. Comparison:

| Model   | Parameters | Speed     | Quality   | RAM   | Disk  |
|---------|------------|-----------|-----------|-------|-------|
| **tiny**  | ~39M   | Faster    | Lower     | ~1 GB | ~75 MB  |
| **base**  | ~74M   | Faster    | Medium    | ~1 GB | ~150 MB |
| **small** | ~244M  | Faster    | Good      | ~2 GB | ~500 MB |
| **medium** | ~769M | **(default)** | High   | ~5 GB | ~1.5 GB |
| **large** | ~1.5B  | Slower    | Highest   | ~10 GB| ~3 GB   |

- **Faster, lower quality:** `--whisper-model tiny`
- **Better quality, slower:** `--whisper-model small` or `medium`
- **Best quality:** `--whisper-model large`

---

## Export folder requirements

The Instagram export root must contain:

- `personal_information/personal_information/personal_information.json`
- `your_instagram_activity/messages/inbox/`

The script also looks for this structure in nested folders under the path you pass.

---

This repo also includes **export_to_md.py** (Telegram) and **split_json.py**; they are separate from the Instagram flow.
