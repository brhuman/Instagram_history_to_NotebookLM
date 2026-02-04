# Instagram Export → NotebookLM

Этот проект обрабатывает **только экспорт данных Instagram** (логи, переписки, профиль, активность) и конвертирует их в один Markdown-файл для загрузки в [NotebookLM](https://notebooklm.google/).

## Как это работает

1. **Instagram** — скачайте свои данные через Настройки → Аккаунт → Загрузка данных (формат JSON).
2. Положите папку экспорта в **src/** (или укажите путь к ней аргументом).
3. Запустите **instagram_export_to_md.py** — скрипт соберёт один `.md` с разделами:
   - **Профиль** (имя, username, email, дата рождения и т.д.)
   - **Социальные связи** (подписки, подписчики, близкие друзья)
   - **Переписки** (все чаты из inbox, сообщения по времени)
   - **Активность** (лайки постов и комментариев, сохранённое, посты Threads, мои комментарии)
   - **Повторяющиеся темы**
4. Результат: `dist/<имя_папки_экспорта>/<имя_папки_экспорта>_notebooklm.md` — готов к загрузке в NotebookLM.

Медиа (фото/видео) в файл не включаются; голосовые сообщения можно опционально транскрибировать через Whisper.

## Установка

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Опционально, для транскрипции голосовых:
pip install openai-whisper   # и установите ffmpeg в системе
```

## Команды

| Команда | Описание |
|--------|----------|
| `python3 instagram_export_to_md.py` | Конвертация: папка экспорта из **src/** → **dist/** |
| `python3 instagram_export_to_md.py src/MyInstagramExport` | Указать папку с экспортом Instagram |
| `python3 instagram_export_to_md.py -o /path/to/out` | Папка для выходного .md (по умолчанию **dist/**) |
| `python3 instagram_export_to_md.py --no-transcribe` | Не транскрибировать голосовые (быстрее) |

## Требования к экспорту Instagram

В корне папки экспорта должны быть:

- `personal_information/personal_information/personal_information.json`
- `your_instagram_activity/messages/inbox/`

Скрипт может найти такую папку и во вложенных подпапках внутри **src/**.

---

В репозитории также лежат скрипты **export_to_md.py** (экспорт Telegram) и **split_json.py** (разбиение больших JSON) — они не относятся к обработке Instagram и сохранены для смежных задач.
