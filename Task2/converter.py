#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import json

DIR_LOCATION = ""
DIR_KNOWLEDGE_BASE = "knowledge_data"
DIR_SOURCE = "source_data"

FILE_MAP_RENAME = "terms_map.json"
ENCODING_DEFAULT = "UTF-8"


def read_mapper():
    if not os.path.isfile(FILE_MAP_RENAME):
        print(f"ERROR: no '{FILE_MAP_RENAME}' file!")
        exit(1)
    with open(FILE_MAP_RENAME, "r", encoding=ENCODING_DEFAULT) as file:
        data = json.load(file)

    print("INFO: Using content remapper:")
    print(json.dumps(data, indent=4))
    return data


def copy_remapped_content(filepath_src, filepath_dst, mapper):
    with open(filepath_dst, "w", encoding=ENCODING_DEFAULT) as output_file:
        with open(filepath_src, "r", encoding=ENCODING_DEFAULT) as input_file:
            for line in input_file:
                for lhv, rhv in mapper.items():
                    regex_replace = re.compile(re.escape(lhv), re.IGNORECASE)
                    line = regex_replace.sub(rhv, line)
                output_file.write(line)


def convert_data():
    if not os.path.isdir(DIR_SOURCE):
        print(f"ERROR: no {DIR_SOURCE} directory!")
        exit(1)
    if not os.path.isdir(DIR_KNOWLEDGE_BASE):
        os.makedirs(DIR_KNOWLEDGE_BASE)

    remapper = read_mapper()

    for address, _, files in os.walk(DIR_SOURCE):
        for name in files:
            src_path = os.path.join(address, name)
            dest_dir = address.replace(DIR_SOURCE, DIR_KNOWLEDGE_BASE)
            dest_path = os.path.join(dest_dir, name)
            os.makedirs(dest_dir, exist_ok=True)

            print(
                f"INFO: Copy remapped file content "
                f"from \n\t{src_path}\nto\n\t{dest_path}"
            )
            copy_fb2_file_content(src_path, dest_path, remapper)


def copy_fb2_file_content(filepath_src, filepath_dst, mapper):
    supported_codecs = [ENCODING_DEFAULT, "windows-1251", "cp866"]
    content = None
    used_encoding = None
    
    # Пробуем разные кодировки
    for codec in supported_codecs:
        try:
            with open(filepath_src, "r", encoding=codec) as input_file:
                content = input_file.read()
                used_encoding = codec
                print(f"  Успешно прочитан в кодировке: {codec}")
                print(f"  Первые 200 символов файла: {content[:200]}")
                break
        except UnicodeDecodeError:
            print(f"  Не удалось прочитать в {codec}, пробуем дальше...")
            continue
        except Exception as error:
            print("WARNING: ", error)
            continue
    
    if content is None:
        print(f"ERROR: could not read file {filepath_src} in any supported encoding")
        exit(1)
    
    # Применяем замены из маппера
    print("\n  Применяем замены из маппера:")
    replacements_count = 0
    
    # СОХРАНЯЕМ результат замен
    modified_content = content
    
    for old, new in mapper.items():
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        
        # Проверяем, есть ли вхождения
        if pattern.search(modified_content):
            matches = len(pattern.findall(modified_content))
            replacements_count += matches
            print(f"    Найдено '{old}' -> '{new}': {matches} совпадений")
            modified_content = pattern.sub(new, modified_content)
        else:
            print(f"    НЕ найдено '{old}'")
    
    if replacements_count == 0:
        print("    ВНИМАНИЕ: Не найдено ни одного совпадения для замены!")
        print("    Проверьте написание терминов в исходном файле")
    else:
        print(f"  Всего сделано замен: {replacements_count}")
    
    # ВАЖНО: Записываем результат в выходной файл!
    with open(filepath_dst, "w", encoding=ENCODING_DEFAULT) as output_file:
        output_file.write(modified_content)
    
    print(f"  Файл сохранен: {filepath_dst}")
    print(f"  Размер файла: {len(modified_content)} символов")
            
if __name__ == "__main__":
    DIR_LOCATION = os.path.dirname(os.path.abspath(__file__))
    DIR_KNOWLEDGE_BASE = os.path.join(DIR_LOCATION, DIR_KNOWLEDGE_BASE)
    DIR_SOURCE = os.path.join(DIR_LOCATION, DIR_SOURCE)
    FILE_MAP_RENAME = os.path.join(DIR_LOCATION, FILE_MAP_RENAME)

    convert_data()
