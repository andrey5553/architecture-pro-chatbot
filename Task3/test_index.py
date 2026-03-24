#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from build_index import KnowledgeBaseIndexer


def search_example(test_queries: list[str] | None = None):
    if not test_queries:
        return

    # Создаем индексер и указываем путь к индексу в текущей папке
    indexer = KnowledgeBaseIndexer()
    indexer.faiss_db_dir = "faiss_db"  # просто имя папки в текущем каталоге
    print(f"Загружаем индекс из: {indexer.faiss_db_dir}")
    
    faiss_index = indexer.create_faiss_index()

    for query in test_queries:
        print(f"query: {query}")
        results = faiss_index.similarity_search(query, k=3)

        for i, result in enumerate(results, 1):
            print(f"Source {i}. {result.metadata['source']}:")
            print(f"{result.page_content[:300]}...\n\n")


if __name__ == "__main__":
    test_queries = [
        "Кто такой Леонард Уитмор?",
        "Кто такой Майкрофт Холмс?",
        "Что такое собака баскервилей?",
        "Кто такой Доктор Артур Кейн?",
        "Найди все упоминания о Докторе Артуре Кейне",
        "перечисли все книги где есть доктор Кейн",
        "перечисли все книги где есть Ватсон", 
    ]

    search_example(test_queries)
