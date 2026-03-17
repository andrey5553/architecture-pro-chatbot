import sys
import os
import csv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from Task4.rag_bot import RAGBot


class CoverageTester:

    def __init__(self) -> None:
        self.bot = RAGBot(
            rag_max_results=5,
            temperature=0.01,
        )
        self.filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs.csv"
        )
        self.csv_headers = [
            "timestamp",
            "query",
            "num_sources",
            "response_length",
            "success",
            "sources",
        ]
        with open(self.filepath, "w", encoding="utf-8") as file:
            csv.writer(file).writerow(self.csv_headers)

        # Два теста: валидный и невалидный
        self.test_data: dict[str, dict] = {
            "Виктор Стоун": {
                "should_find": True,  # Должен найти!
                "expected_keywords": ["сыщик", "детектив", "расследование"],  # Ожидаемые слова
                "unexpected_keywords": []  # Слов, которых НЕ должно быть
            },
            "Джон Траволта": {
                "should_find": False,  # Не должен найти!
                "expected_keywords": [],  # Не ожидаем конкретных слов
                "unexpected_keywords": ["шерлок", "холмс", "детектив", "сыщик", "вампир"]  # Слова, которых быть НЕ должно
            },
        }
        self.success_counter = 0

    def make_request(self, search: str, test_config: dict):
        """
        Делает запрос к боту и проверяет результат
        test_config: {
            "should_find": bool, - должен ли найти информацию
            "expected_keywords": list, - ожидаемые ключевые слова
            "unexpected_keywords": list - нежелательные ключевые слова
        }
        """
        print(f"\n  Запрос: {search}")
        print(f"  Ожидается найти информацию: {test_config['should_find']}")
        
        response = self.bot.ask(f"Расскажи о персонаже {search}", RAGBot.Prompts.BASE)
        sources = response["sources"]
        response_text = response["response"].lower()
        
        # Проверяем наличие источников
        has_sources = response["num_sources"] > 0
        
        # Проверяем наличие ожидаемых слов (для позитивного теста)
        has_expected = any(
            expect.lower() in response_text 
            for expect in test_config["expected_keywords"]
        ) if test_config["expected_keywords"] else True
        
        # Проверяем наличие нежелательных слов
        has_unexpected = any(
            unexpected.lower() in response_text 
            for unexpected in test_config["unexpected_keywords"]
        )
        
        # Определяем успех в зависимости от типа теста
        if test_config["should_find"]:
            # Для позитивного теста: должны быть источники И ожидаемые слова И не должно быть нежелательных
            success = has_sources and has_expected and not has_unexpected
        else:
            # Для негативного теста: не должно быть источников И не должно быть нежелательных слов
            success = not has_sources and not has_unexpected
        
        print(f"  Найдено источников: {response['num_sources']}")
        print(f"  Длина ответа: {len(response['response'])}")
        print(f"  Успех: {success}")
        
        # Детальный вывод для отладки
        if test_config["should_find"]:
            if not has_sources:
                print(f"  ❌ Не найдено источников!")
            elif not has_expected:
                print(f"  ❌ Не найдены ожидаемые слова: {test_config['expected_keywords']}")
            elif has_unexpected:
                print(f"  ❌ Найдены нежелательные слова: {[w for w in test_config['unexpected_keywords'] if w in response_text]}")
            else:
                print(f"  ✅ Найдены источники и ожидаемые слова!")
        else:
            if has_sources:
                print(f"  ❌ Найдены источники, хотя их быть не должно!")
                if sources:
                    print(f"  Первый источник: {sources[0]['source']}")
            elif has_unexpected:
                print(f"  ❌ Найдены нежелательные слова: {[w for w in test_config['unexpected_keywords'] if w in response_text]}")
            else:
                print(f"  ✅ Тест пройден: нет источников и нет нежелательных слов")
        
        # Показываем первые 200 символов ответа для понимания
        print(f"  Ответ (первые 200 символов): {response['response'][:200]}...")
        
        self._write_log(
            response["timestamp"],
            response["query"],
            response["num_sources"],
            len(response["response"]),
            success,
            "\n".join([source["source"] for source in sources]),
            f"has_sources={has_sources}, has_expected={has_expected}, has_unexpected={has_unexpected}"
        )
        
        return success

    def _write_log(
        self,
        timestamp,
        query: str,
        num_sources: int,
        response_length: int,
        success: bool,
        sources: str,
        extra_info: str = ""
    ):
        if success:
            self.success_counter += 1
        with open(self.filepath, "a", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    timestamp,
                    query,
                    num_sources,
                    response_length,
                    success,
                    sources,
                    extra_info
                ]
            )

    def run_tests(self):
        """Запускает оба теста"""
        print("\n🔍 ТЕСТ 1: Персонаж В базе (ДОЛЖЕН найти)")
        print("-" * 50)
        success1 = self.make_request("Виктор Стоун", self.test_data["Виктор Стоун"])
        
        print("\n🔍 ТЕСТ 2: Персонаж НЕ в базе (НЕ должен находить)")
        print("-" * 50)
        success2 = self.make_request("Джон Траволта", self.test_data["Джон Траволта"])
        
        return success1, success2


if __name__ == "__main__":
    print("="*60)
    print("ТЕСТИРОВАНИЕ ПОКРЫТИЯ RAG-БОТА")
    print("="*60)
    
    coverage_tester = CoverageTester()
    success1, success2 = coverage_tester.run_tests()
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"  ✅ Тест 1 (Виктор Стоун): {'ПРОЙДЕН' if success1 else 'ПРОВАЛЕН'}")
    print(f"  ✅ Тест 2 (Джон Траволта): {'ПРОЙДЕН' if success2 else 'ПРОВАЛЕН'}")
    print(f"  Успешных ответов: {coverage_tester.success_counter} из 2")
    print(f"  Процент успеха: {coverage_tester.success_counter/2*100:.1f}%")
    print(f"  Лог сохранен: {coverage_tester.filepath}")
    print("="*60)