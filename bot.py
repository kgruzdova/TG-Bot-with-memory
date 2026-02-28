"""
Telegram-бот с двойной памятью: короткой и долгой.

Короткая память — последние MAX_HISTORY_SIZE сообщений диалога (in-memory).
Долгая память  — документы → эмбеддинги, хранятся в JSON-файлах на диске.

При ответе на вопрос бот:
  1. Ищет релевантные фрагменты в документах (долгая память).
  2. Подставляет найденный контекст + историю диалога (короткая память).
  3. Генерирует ответ через OpenAI ChatCompletion.

Переменные окружения:
    BOT_TOKEN      — токен Telegram-бота
    OPENAI_API_KEY — ключ API OpenAI
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, ContentType
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    sys.exit("Ошибка: переменная окружения BOT_TOKEN не задана.")
if not OPENAI_API_KEY:
    sys.exit("Ошибка: переменная окружения OPENAI_API_KEY не задана.")

MAX_HISTORY_SIZE = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
MEMORY_DIR = Path("./memory")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = (
    "Ты — умный и дружелюбный ассистент в Telegram с двумя видами памяти.\n"
    "1) Ты помнишь недавний диалог с пользователем (короткая память).\n"
    "2) Если пользователь загрузил документы, ты можешь искать в них информацию (долгая память).\n\n"
    "Правила:\n"
    "- Если предоставлен контекст из документов — опирайся на него и не выдумывай.\n"
    "- Если контекста нет — отвечай как обычный ассистент, используя диалог.\n"
    "- Отвечай кратко и по делу."
)

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
dp = Dispatcher()
router = Router()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

chat_history: dict[int, list[dict[str, str]]] = defaultdict(list)

# ---------------------------------------------------------------------------
# Файловое векторное хранилище (замена ChromaDB, совместимо с Python 3.14)
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Файловое хранилище эмбеддингов с косинусным поиском.
    Для каждого user_id — отдельный JSON-файл в storage_dir.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: int) -> Path:
        return self.storage_dir / f"user_{user_id}.json"

    def _load(self, user_id: int) -> list[dict]:
        path = self._path(user_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, user_id: int, data: list[dict]) -> None:
        with open(self._path(user_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def count(self, user_id: int) -> int:
        return len(self._load(user_id))

    def add(self, user_id: int, chunks: list[str], embeddings: list[list[float]], source: str) -> int:
        data = self._load(user_id)
        for text, emb in zip(chunks, embeddings):
            data.append({"text": text, "embedding": emb, "source": source})
        self._save(user_id, data)
        return len(chunks)

    def search(self, user_id: int, query_embedding: list[float], top_k: int = TOP_K) -> list[str]:
        """Косинусный поиск TOP-K ближайших фрагментов."""
        data = self._load(user_id)
        if not data:
            return []

        query_vec = np.array(query_embedding)
        scores = []
        for item in data:
            doc_vec = np.array(item["embedding"])
            cos_sim = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10))
            scores.append((cos_sim, item["text"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:top_k]]

    def clear(self, user_id: int) -> bool:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
            return True
        return False


vector_store = VectorStore(MEMORY_DIR)


# ---------------------------------------------------------------------------
# Короткая память
# ---------------------------------------------------------------------------


def trim_history(user_id: int) -> None:
    buf = chat_history[user_id]
    if len(buf) > MAX_HISTORY_SIZE:
        chat_history[user_id] = buf[-MAX_HISTORY_SIZE:]


# ---------------------------------------------------------------------------
# Долгая память — работа с документами
# ---------------------------------------------------------------------------


def load_document(file_path: str) -> str:
    """Читает PDF / TXT / DOCX и возвращает текст."""
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    if ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Долгая память — эмбеддинги и поиск
# ---------------------------------------------------------------------------


async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Получает эмбеддинги через OpenAI Embeddings API."""
    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    return [item.embedding for item in response.data]


async def store_chunks(user_id: int, chunks: list[str], doc_name: str) -> int:
    embeddings = await embed_chunks(chunks)
    return vector_store.add(user_id, chunks, embeddings, source=doc_name)


async def retrieve_context(user_id: int, query: str, top_k: int = TOP_K) -> list[str]:
    if vector_store.count(user_id) == 0:
        return []
    query_embedding = (await embed_chunks([query]))[0]
    return vector_store.search(user_id, query_embedding, top_k)


# ---------------------------------------------------------------------------
# Генерация ответа — объединение обеих памятей
# ---------------------------------------------------------------------------


async def generate_answer(user_id: int, user_text: str) -> str:
    doc_fragments = await retrieve_context(user_id, user_text)

    system_content = SYSTEM_PROMPT
    if doc_fragments:
        context_block = "\n\n---\n\n".join(doc_fragments)
        system_content += (
            "\n\n--- КОНТЕКСТ ИЗ ДОКУМЕНТОВ ---\n\n"
            f"{context_block}\n\n"
            "--- КОНЕЦ КОНТЕКСТА ---"
        )

    messages = [
        {"role": "system", "content": system_content},
        *chat_history[user_id],
        {"role": "user", "content": user_text},
    ]

    response = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
       )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Обработчики Telegram
# ---------------------------------------------------------------------------


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    user_id = message.from_user.id
    chat_history[user_id].clear()

    doc_count = vector_store.count(user_id)
    docs_status = f"В долгой памяти: *{doc_count}* фрагментов." if doc_count else "Долгая память пуста — загрузите документ."

    await message.answer(
        "Привет! Я бот с *двойной памятью*:\n\n"
        "🧠 *Короткая память* — помню последние сообщения диалога.\n"
        "📚 *Долгая память* — ищу ответы в загруженных документах.\n\n"
        f"{docs_status}\n\n"
        "*Команды:*\n"
        "/start — сбросить диалог\n"
        "/clear — очистить базу документов\n"
        "/status — показать состояние памяти"
    )


@router.message(Command("clear"))
async def handle_clear(message: Message) -> None:
    user_id = message.from_user.id
    if vector_store.clear(user_id):
        await message.answer("📚 База документов очищена.")
    else:
        await message.answer("📚 База документов уже пуста.")


@router.message(Command("status"))
async def handle_status(message: Message) -> None:
    user_id = message.from_user.id
    history_len = len(chat_history[user_id])
    doc_count = vector_store.count(user_id)

    await message.answer(
        "*Состояние памяти:*\n\n"
        f"🧠 Короткая память: *{history_len}* / {MAX_HISTORY_SIZE} сообщений\n"
        f"📚 Долгая память: *{doc_count}* фрагментов в базе"
    )


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_document(message: Message) -> None:
    user_id = message.from_user.id
    doc = message.document
    file_name = doc.file_name or "unknown"
    ext = Path(file_name).suffix.lower()

    if ext not in (".txt", ".pdf", ".docx"):
        await message.answer(f"Формат `{ext}` не поддерживается. Отправьте PDF, TXT или DOCX.")
        return

    status_msg = await message.answer(f"⏳ Обрабатываю `{file_name}`...")

    try:
        file = await bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            await bot.download_file(file.file_path, destination=tmp)

        text = load_document(tmp_path)
        os.unlink(tmp_path)

        if not text.strip():
            await status_msg.edit_text("Документ пустой или не удалось извлечь текст.")
            return

        chunks = split_into_chunks(text)
        saved = await store_chunks(user_id, chunks, doc_name=file_name)

        await status_msg.edit_text(
            f"📚 Документ `{file_name}` добавлен в долгую память.\n"
            f"Символов: {len(text)} → фрагментов: {saved}\n\n"
            "Теперь задавайте вопросы — я буду искать ответы в документах "
            "и учитывать контекст нашего диалога."
        )
    except ValueError as e:
        await status_msg.edit_text(str(e))
    except Exception as e:
        logger.error("Ошибка обработки документа для user %s: %s", user_id, e)
        await status_msg.edit_text("Произошла ошибка при обработке документа.")


@router.message(F.text)
async def handle_message(message: Message) -> None:
    user_id = message.from_user.id
    user_text = message.text

    try:
        assistant_text = await generate_answer(user_id, user_text)
    except Exception as e:
        logger.error("Ошибка генерации ответа для user %s: %s", user_id, e)
        assistant_text = "Произошла ошибка при обращении к модели. Попробуйте позже."

    chat_history[user_id].append({"role": "user", "content": user_text})
    chat_history[user_id].append({"role": "assistant", "content": assistant_text})
    trim_history(user_id)

    try:
        await message.answer(assistant_text)
    except TelegramBadRequest:
        await message.answer(assistant_text, parse_mode=None)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------


async def main() -> None:
    dp.include_router(router)
    logger.info("Бот с двойной памятью запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
