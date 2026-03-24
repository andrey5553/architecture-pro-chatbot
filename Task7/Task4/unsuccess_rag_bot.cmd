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

echo [%DATE% %TIME%] Запуск: Запрос про Вавилена Татарского 
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Кто такой Вавилен Татарский ?" > base.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про Вавилена Татарского 
echo.

echo [%DATE% %TIME%] Запуск: Запрос про Вавилена Татарского с параметрами
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Где живет Вавилен Татарский?" -d 5 -p cot -v > cot.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про  Вавилена Татарского с параметрами
echo.

echo [%DATE% %TIME%] Запуск: Запрос про  Вавилена Татарского
set PYTHONIOENCODING=utf-8 && python rag_bot.py "Расскажи о Вавилене Татарском" -d 3 -p few-shot > few_shot.txt 2>&1
echo [%DATE% %TIME%] Завершен: Запрос про  Вавилена Татарского
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

echo Лог сохранен в файл execution_log.txt

pause
pause