# update_index.py
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Заменили OpenAI
from langchain_community.vectorstores import FAISS
from datetime import datetime

# Загружаем .env файл (ключ OpenAI больше не нужен)
load_dotenv()

# Настройка логирования
logging.basicConfig(
    filename="update.log",
    level=logging.INFO,
    encoding='utf-8',  # Добавляем кодировку для русского текста
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_documents():
    """Загрузка документов из папки docs"""
    loader = DirectoryLoader("./docs", glob="**/*.txt")
    return loader.load()


def process_documents(docs):
    """Разбиение документов на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks


def update_vector_store(chunks):
    """Создание/обновление векторного индекса"""
    # Используем локальную модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Поддерживает русский
    )
    vectorstore = FAISS.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    vectorstore.save_local("./vector_store")
    return vectorstore


def main():
    start_time = datetime.now()
    logging.info("Запуск обновления индекса")

    try:
        # Загрузка документов из папки docs
        docs = load_documents()
        logging.info(f"Загружено {len(docs)} документов из папки docs")

        if not docs:
            logging.warning("Нет документов для обработки")
            return

        # Обработка документов
        chunks = process_documents(docs)
        logging.info(f"Создано {len(chunks)} чанков")

        # Обновление векторного индекса
        vectorstore = update_vector_store(chunks)
        logging.info("Индекс успешно обновлен и сохранен в ./vector_store")

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        return

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Обновление завершено за {duration:.2f} секунд")


if __name__ == "__main__":
    main()
    