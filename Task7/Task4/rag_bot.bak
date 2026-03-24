#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import logging
import sys
from enum import Enum

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


class RAGBot:
    DB_DIR_POSTFIX = "faiss_db\\"

    class Prompts(Enum):
        BASE = "base"
        FEW_SHOT = "few-shot"
        COT = "cot"

    FILE_BASE_PROMPT = "base_prompt.txt"
    FILE_FEW_SHOT_PROMPT = "few_shot_prompt.txt"
    FILE_COT_PROMPT = "cot_prompt.txt"

    DEFAULT_MAX_DOCUMENTS = 42
    DEFAULT_VERBOSE = False

    def __init__(
        self,
        setup_dir: str = "",
        rag_max_results=DEFAULT_MAX_DOCUMENTS,
        temperature=0.1,
        faiss_storage: FAISS | None = None,
        verbose=DEFAULT_VERBOSE,
    ):
        # Определяем базовую директорию проекта
        #self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #self.faiss_db_dir = os.path.join(self.project_root, "Task3", "faiss_db")

        self.current_dir = setup_dir or os.path.dirname(os.path.abspath(__file__))
        self.faiss_db_dir = "e:\\1\\faiss_db\\" #os.path.join(self.current_dir, self.DB_DIR_POSTFIX)
        
        # **ВАЖНО: Сначала создаем логгер!**
        self._logger = self._setup_logger(verbose=verbose)
        self._logger.info("Инициализация RAG бота...")
        
        # Теперь можно использовать логгер во всех методах валидации
        try:
            self._validate_rag_max_results(rag_max_results)
            self.rag_max_results = rag_max_results
            
            self._validate_temperature(temperature)
            
            self._validate_faiss_directory()
            
            self.vector_db = faiss_storage or self._load_db()
            self.ollama = self._connect_ollama(temperature)
            self.prompts = self._create_prompts()
            
            self._logger.info("RAG бот успешно инициализирован")
            
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            raise

    def _validate_rag_max_results(self, value):
        """Валидация количества документов для RAG"""
        if not isinstance(value, int):
            raise TypeError(f"rag_max_results должен быть int, получен {type(value)}")
        if value <= 0:
            raise ValueError(f"rag_max_results должен быть > 0, получен {value}")
        if value > 100:
            self._logger.warning(f"Большое значение rag_max_results ({value}) может замедлить работу")

    def _validate_temperature(self, value):
        """Валидация температуры для LLM"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"temperature должна быть числом, получен {type(value)}")
        if value < 0 or value > 1:
            raise ValueError(f"temperature должна быть между 0 и 1, получена {value}")

    def _validate_faiss_directory(self):
        """Проверка существования директории с FAISS БД"""
        if not os.path.exists(self.faiss_db_dir):
            raise FileNotFoundError(
                f"Директория с FAISS БД не найдена: {self.faiss_db_dir}\n"
                f"Убедитесь, что БД создана и находится в правильном месте."
            )
        if not os.path.isdir(self.faiss_db_dir):
            raise NotADirectoryError(f"{self.faiss_db_dir} не является директорией")
        
        self._logger.info(f"FAISS БД найдена в: {self.faiss_db_dir}")

    def search_documents(self, query: str) -> list[Document]:
        """Поиск документов с валидацией запроса"""
        # Валидация запроса
        if not query or not isinstance(query, str):
            self._logger.error(f"Некорректный запрос: {query}")
            return []
        
        query = query.strip()
        if not query:
            self._logger.warning("Пустой запрос после очистки")
            return []
        
        try:
            results = self.vector_db.similarity_search(
                query, k=self.rag_max_results, fetch_k=25000
            )
            self._logger.info(f"Найдено {len(results)} документов по запросу '{query}'.")
            
            # Логируем источники для отладки
            for doc in results:
                source = doc.metadata.get('source', 'Неизвестный источник')
                self._logger.debug(f"Используется документ: {source}")

            return results
        except Exception as error:
            self._logger.error(f"Ошибка при поиске документов: {error}", exc_info=True)
            return []

    def format_context(self, documents: list[Document]) -> str:
        """Форматирование контекста с проверкой входных данных"""
        if not documents:
            self._logger.warning("Нет документов для форматирования контекста")
            return "Нет информации в RAG базе данных."

        # Проверка, что все элементы - Document
        valid_docs = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                self._logger.error(f"Элемент {i} не является Document: {type(doc)}")
            else:
                valid_docs.append(doc)
        
        if not valid_docs:
            self._logger.error("Нет валидных документов для форматирования")
            return "Ошибка: все документы некорректны"

        context_parts = []
        for i, doc in enumerate(valid_docs, 1):
            source = doc.metadata.get("source", "Неизвестный источник")
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            context_parts.append(
                f"""
--- Документ {i} ---
Источник: {source}
Чанк: {chunk_id}
Содержание:
{doc.page_content}
"""
            )

        self._logger.debug(
            f"Подготовлено {len(context_parts)} частей контекста для {len(valid_docs)} документов."
        )
        return "\n".join(context_parts)

    def ask(self, question: str, prompt: Prompts = Prompts.BASE):
        """Основной метод с комплексной обработкой ошибок"""
        try:
            # Валидация входных параметров
            if not question or not isinstance(question, str):
                raise ValueError("Вопрос должен быть непустой строкой")
            
            question = question.strip()
            if not question:
                raise ValueError("Вопрос не может быть пустым")
            
            if not isinstance(prompt, self.Prompts):
                try:
                    prompt = self.Prompts(prompt)
                except (ValueError, TypeError):
                    raise ValueError(f"Некорректный тип промпта: {prompt}")
            
            self._logger.info(f"Получен промпт {prompt.name} с вопросом: '{question}'.")
            
            # Поиск документов
            documents = self.search_documents(question)
            if not documents:
                self._logger.warning(f"Не найдено документов по запросу: '{question}'")
            
            # Форматирование контекста
            context = self.format_context(documents)
            
            # Проверка наличия промпта
            if prompt not in self.prompts:
                raise KeyError(f"Промпт {prompt} не найден в загруженных промптах")
            
            # Создание и выполнение цепочки
            chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | self.prompts[prompt]
                | self.ollama
                | StrOutputParser()
            )

            response = chain.invoke({"context": context, "question": question})
            self._logger.info(f"Ответ получен (первые 100 символов): '{response[:100]}...'.")

            # Формирование результата
            result = {
                "query": question,
                "timestamp": int(time.time()),
                "prompt-type": prompt.value,
                "response": response,
                "context": context,
                "prompt": self.prompts[prompt].format(context=context, question=question),
                "sources": [
                    {
                        "source": doc.metadata.get("source", "Неизвестный источник"),
                        "category": doc.metadata.get("category", "Неизвестная категория"),
                        "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                        "content_preview": f"{doc.page_content[:200]}...",
                    }
                    for doc in documents if isinstance(doc, Document)
                ],
                "num_sources": len(documents),
                "success": True
            }
            
            return result
            
        except FileNotFoundError as e:
            self._logger.error(f"Файл не найден: {e}")
            return self._error_response(question, f"Ошибка файловой системы: {e}")
            
        except ValueError as e:
            self._logger.error(f"Ошибка валидации: {e}")
            return self._error_response(question, f"Некорректные входные данные: {e}")
            
        except KeyError as e:
            self._logger.error(f"Ошибка ключа: {e}")
            return self._error_response(question, f"Внутренняя ошибка конфигурации: {e}")
            
        except Exception as e:
            self._logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            return self._error_response(question, f"Произошла внутренняя ошибка: {e}")

    def _error_response(self, question: str, error_message: str):
        """Создание ответа с ошибкой"""
        return {
            "query": question,
            "timestamp": int(time.time()),
            "prompt-type": "error",
            "response": f"Извините, произошла ошибка: {error_message}",
            "context": "",
            "prompt": "",
            "sources": [],
            "num_sources": 0,
            "success": False,
            "error": error_message
        }

    def _load_db(self, location: str | None = None):
        """Загрузка векторной БД с обработкой ошибок"""
        try:
            self._logger.info("Загрузка FAISS базы данных...")
            
            # Кэширование эмбеддингов
            if not hasattr(self, '_embeddings'):
                self._logger.debug("Создание эмбеддингов all-mpnet-base-v2")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="all-mpnet-base-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            
            db_path = location or self.faiss_db_dir
            
            # Проверка наличия файлов FAISS
            required_files = ['index.faiss', 'index.pkl']
            for file in required_files:
                file_path = os.path.join(db_path, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Файл {file} не найден в {db_path}. "
                        f"Убедитесь, что БД создана правильно."
                    )
                self._logger.debug(f"Найден файл: {file_path}")
            
            self._logger.info(f"Загрузка FAISS из {db_path}")
            vector_db = FAISS.load_local(
                folder_path=db_path,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
            
            self._logger.info("FAISS база данных успешно загружена")
            return vector_db
            
        except Exception as e:
            self._logger.error(f"Ошибка загрузки FAISS БД: {e}", exc_info=True)
            raise

    def _connect_ollama(self, temperature):
        """Подключение к Ollama с проверкой доступности"""
        try:
            self._logger.info("Подключение к Ollama...")
            
            # Проверка, запущена ли Ollama
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode != 0:
                self._logger.warning("Ollama не отвечает, но продолжаем попытку подключения")
            else:
                self._logger.info("Ollama доступна")
            
            self._logger.info(f"Подключение к модели llama3.1 с температурой {temperature}")
            ollama = OllamaLLM(
                model="llama3.1",
                temperature=temperature,
                num_predict=1024,
            )
            
            self._logger.info("Подключение к Ollama успешно")
            return ollama
            
        except FileNotFoundError:
            self._logger.error("Ollama не установлена или не найдена в PATH")
            raise RuntimeError(
                "Ollama не найдена. Убедитесь, что Ollama установлена и запущена.\n"
                "Скачать: https://ollama.ai/"
            )
        except subprocess.TimeoutExpired:
            self._logger.warning("Таймаут при проверке Ollama")
            return OllamaLLM(
                model="llama3.1",
                temperature=temperature,
                num_predict=1024,
            )

    def _create_prompts(self) -> dict[Prompts, PromptTemplate]:
        """Загрузка промптов с проверкой файлов"""
        self._logger.info("Загрузка шаблонов промптов...")
        
        prompts = {}
        
        prompt_files = [
            (self.Prompts.BASE, self.FILE_BASE_PROMPT),
            (self.Prompts.FEW_SHOT, self.FILE_FEW_SHOT_PROMPT),
            (self.Prompts.COT, self.FILE_COT_PROMPT),
        ]
        
        missing_files = []
        
        for prompt_type, filename in prompt_files:
            filepath = os.path.join(self.current_dir, filename)
            
            if not os.path.exists(filepath):
                missing_files.append(filename)
                self._logger.error(f"Файл промпта не найден: {filepath}")
                continue
            
            try:
                self._logger.debug(f"Загрузка промпта из {filename}")
                prompts[prompt_type] = PromptTemplate.from_file(
                    filepath, 
                    encoding="UTF-8"
                )
                self._logger.info(f"Загружен промпт {prompt_type.name} из {filename}")
            except Exception as e:
                self._logger.error(f"Ошибка загрузки {filename}: {e}")
                raise
        
        if missing_files:
            error_msg = f"Отсутствуют файлы промптов: {', '.join(missing_files)}"
            self._logger.error(error_msg)
            raise FileNotFoundError(
                f"{error_msg}\n"
                f"Убедитесь, что они находятся в директории: {self.current_dir}"
            )
        
        self._logger.info(f"Загружено {len(prompts)} шаблонов промптов")
        return prompts

    def _setup_logger(self, filename: str | None = None, verbose=DEFAULT_VERBOSE):
        """Настройка логирования"""
        format_pattern = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
        formatter = logging.Formatter(
            fmt=format_pattern,
            datefmt="%d-%m-%YT%H:%M:%S",
        )

        # Создание логгера для класса
        logger = logging.getLogger(__name__)
        
        # Очистка существующих обработчиков
        logger.handlers.clear()
        
        # Установка уровня логирования
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Добавление файлового обработчика
        log_file = filename or os.path.join(self.current_dir, "rag_bot.log")
        
        # Создание директории для логов, если нужно
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Создана директория для логов: {log_dir}")
        
        file_handler = logging.FileHandler(
            filename=log_file,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(file_handler)
        
        # Добавление обработчика для консоли, если verbose
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
        
        # Подавление лишних логов от библиотек
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        
        logger.debug("Логгер успешно настроен")
        return logger


def main():
    """Основная функция с обработкой аргументов командной строки"""
    parser = argparse.ArgumentParser(
        prog="rag_client",
        description="RAG клиент для вопросов о Шерлоке Холмсе",
        epilog="Пример: python rag_bot.py 'Кто такой Шерлок Холмс?' -d 5 -p cot -v",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "question", 
        type=str, 
        help="Вопрос к RAG боту"
    )
    
    parser.add_argument(
        "-d",
        "--documents",
        type=int,
        default=RAGBot.DEFAULT_MAX_DOCUMENTS,
        help=f"Количество документов для RAG (по умолчанию: {RAGBot.DEFAULT_MAX_DOCUMENTS})",
    )
    
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        choices=[el.value for el in RAGBot.Prompts],
        default=RAGBot.Prompts.BASE.value,
        help=f"Тип промпта: {', '.join([el.value for el in RAGBot.Prompts])}",
    )
    
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="Температура для LLM (0.0-1.0, по умолчанию: 0.1)",
    )
    
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true",
        help="Подробный вывод (логи в консоль)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Путь к файлу лога (по умолчанию: rag_bot.log в текущей директории)",
    )
    
    parser.add_argument(
        "--db-dir",
        type=str,
        help="Путь к директории с FAISS БД (переопределяет стандартный)",
    )
    
    args = parser.parse_args()

    try:
        # Проверка аргументов
        if args.documents <= 0:
            print("Ошибка: количество документов должно быть положительным числом")
            sys.exit(1)
            
        if args.temperature < 0 or args.temperature > 1:
            print("Ошибка: температура должна быть между 0 и 1")
            sys.exit(1)
        
        # Создание бота
        print("Инициализация RAG бота...")
        bot = RAGBot(
            rag_max_results=args.documents,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        
        # Если указан другой путь к БД, переопределяем
        if args.db_dir:
            print(f"Использование альтернативной БД: {args.db_dir}")
            bot.faiss_db_dir = args.db_dir
            bot.vector_db = bot._load_db()
        
        # Получение ответа
        print(f"Отправка запроса: '{args.question}'")
        print(f"Тип промпта: {args.prompt}")
        
        response = bot.ask(args.question, RAGBot.Prompts(args.prompt))
        
        # Вывод результата
        if response.get("success", False):
            print("\n" + "="*50)
            print("ОТВЕТ:")
            print("="*50)
            print(response["response"])
            print("="*50)
            print(f"Найдено источников: {response['num_sources']}")
            if response['sources']:
                print("\nИСТОЧНИКИ:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"{i}. {source['source']} (чанк: {source['chunk_id']})")
        else:
            print(f"\nОШИБКА: {response.get('error', 'Неизвестная ошибка')}")
            print(response["response"])
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
        print("\nПроверьте:")
        print("1. Существует ли директория с FAISS БД")
        print("2. Есть ли файлы промптов в текущей директории")
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"\nОшибка выполнения: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    