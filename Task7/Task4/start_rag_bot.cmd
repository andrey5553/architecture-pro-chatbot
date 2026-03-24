rem пример запуска скрипта с параметрами
rem python rag_bot.py "Кто такой Шерлок Холмс?" -d 5 -p cot -v
rem запуск скрипта

rem установка зависимостей
pip install -r requirements.txt

rem start
@echo off
chcp 65001 > nul
title RAG Bot

@echo test illama
C:\Users\mikhailenko.a\AppData\Local\Programs\Ollama\ollama.exe --version

REM ===================================================
REM Примеры запуска:
REM ===================================================

@echo off
chcp 65001 > nul

REM Функция для получения текущего времени в формате для имени файла
setlocal enabledelayedexpansion

echo ================================================
echo ЗАПУСК ВСЕХ СКРИПТОВ
echo ================================================
echo Начало работы: %DATE% %TIME%
echo.

REM Вариант 1: Запрос про Шерлока Холмса
echo [%DATE% %TIME%] Запуск: Запрос про Шерлока Холмса
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Кто такой Шерлок Холмс?" > base.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про Шерлока Холмса
echo.

REM Вариант 1: Запрос про Виктора Стоуна
echo [%DATE% %TIME%] Запуск: Запрос про Виктора Стоуна
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Кто такой Виктор Стоун?" > base_stoun.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про Виктора Стоуна
echo.

REM Вариант 2: С параметрами
echo [%DATE% %TIME%] Запуск: Запрос про Стоуна с параметрами
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Где живет Стоун?" -d 5 -p cot -v > cot.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про Стоуна с параметрами
echo.

REM Вариант 3: Другой вопрос
echo [%DATE% %TIME%] Запуск: Запрос про доктора Кейн
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Расскажи о докторе Кейн" -d 3 -p few-shot > few_shot.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про доктора Кейн
echo.

echo ================================================
echo ВСЕ СКРИПТЫ ВЫПОЛНЕНЫ
echo ================================================
echo Окончание работы: %DATE% %TIME%
echo ================================================

REM Создаем лог-файл со всеми временными метками
(
echo ================================================
echo ЛОГ ВЫПОЛНЕНИЯ СКРИПТОВ
echo ================================================
echo Начало: %DATE% %TIME%
echo.
echo 1. Шерлок Холмс - Завершен
echo 2. Виктор Стоун - Завершен
echo 3. Стоун с параметрами - Завершен
echo 4. Доктор Кейн - Завершен
echo.
echo Окончание: %DATE% %TIME%
echo ================================================
) > execution_log.txt

echo Лог сохранен в файл execution_success_log.txt

pause
pause