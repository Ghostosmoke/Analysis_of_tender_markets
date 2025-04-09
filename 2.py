import json


# Функция для преобразования словаря в нужный формат
def convert_to_custom_format(data):
    # Преобразуем все строки в нужный формат
    for key in data:
        if isinstance(data[key], str):
            data[key] = data[key].replace('"', "'")
            # Убираем лишние символы из ktru
            if key == 'ktru':
                # Преобразуем строку вида "['...']" в список
                data[key] = eval(data[key])  # Используем eval для преобразования строки в список

        elif isinstance(data[key], list):
            for item in data[key]:
                convert_to_custom_format(item)

    return data


# Функция для вывода словаря без кавычек вокруг ключей
def print_dict_without_quotes(d):
    result = []
    for key, value in d.items():
        if isinstance(value, list):
            value_str = ', '.join(str(v) for v in value)
            result.append(f"{key}: '[{value_str}]'")
        elif isinstance(value, dict):
            result.append(f"{key}: '{{{print_dict_without_quotes(value)}}}'")
        else:
            result.append(f"{key}: '{value}'")

    return ', '.join(result)


# Чтение данных из JSON-файла
with open('output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Преобразуем данные
for i in data:
    converted_data = convert_to_custom_format(i)

    # Выводим результат без кавычек вокруг ключей
    formatted_output = print_dict_without_quotes(converted_data)
    print(f"{{ {formatted_output} }},")
    # print(converted_data)