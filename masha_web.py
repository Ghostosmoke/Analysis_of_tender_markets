import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import networkx as nx
from collections import defaultdict
from main import G

app = dash.Dash(__name__)

# Стили для интерфейса
styles = {
    'container': {
        'margin': '20px',
        'fontFamily': 'Arial, sans-serif'
    },
    'search-box': {
        'padding': '10px',
        'margin-bottom': '20px',
        'border': '1px solid #ddd',
        'border-radius': '5px'
    },
    'info-panel': {
        'padding': '15px',
        'margin': '10px 0',
        'background': '#f9f9f9',
        'border': '1px solid #eee',
        'border-radius': '5px'
    },
    'relations-panel': {
        'margin': '20px 0'
    },
    'table': {
        'margin': '10px 0',
        'max-height': '400px',
        'overflow-y': 'auto'
    }
}

app.layout = html.Div(style=styles['container'], children=[
    html.H1("Анализ связей в закупочной деятельности"),

    # Поисковая панель
    html.Div(style=styles['search-box'], children=[
        dcc.Input(
            id='search-input',
            type='text',
            placeholder='Введите название заказчика, закупки или поставщика...',
            style={'width': '60%', 'margin-right': '10px'}
        ),
        html.Button('Найти', id='search-button'),
        dcc.Dropdown(
            id='entity-type-filter',
            options=[
                {'label': 'Все типы', 'value': 'all'},
                {'label': 'Заказчики', 'value': 'customer'},
                {'label': 'Закупки', 'value': 'procurement'},
                {'label': 'КТРУ', 'value': 'ktru'},
                {'label': 'Поставщики', 'value': 'executor'}
            ],
            value='all',
            style={'width': '30%', 'display': 'inline-block'}
        )
    ]),

    # Панель информации о найденной сущности
    html.Div(id='entity-info', style=styles['info-panel']),

    # Панель связей
    html.Div(id='relations-panel', style=styles['relations-panel'])
])

# Русскоязычные названия типов связей
RELATION_NAMES = {
    # Заказчики
    'customer_procurement': 'Закупки',
    'shared_executor': 'Совместные исполнители',
    'shared_methods': 'Совпадение методов закупок',
    'price_correlation': 'Корреляция бюджетов',
    'behavioral_cluster': 'Поведенческие кластеры',
    'ktru_rule': 'Ассоциативные правила по КТРУ',

    # Закупки
    'procurement_ktru': 'КТРУ',
    'procurement_executor': 'Поставщики',
    'same_ktru': 'Связь по одинаковому КТРУ',
    'same_method': 'Связь по методу закупки',
    'text_similarity': 'Неявные связи через текст закупок (TF-IDF)',

    # КТРУ
    'ktru_executor': 'Поставщики',
    'co_occurrence': 'Совместное использование в закупках',
    'price_group': 'Ценовые диапазоны (группировка по квантилям)',
    'status_pattern': 'Статусные паттерны',
    'seasonal_pattern': 'Временные кластеры (сезонность)',

    # Поставщики
    'shared_customer': 'Общие заказчики',
    'shared_ktru': 'Специализация по КТРУ',
    'co_execution': 'Сеть коисполнителей (через общие закупки)',
    'name_similarity': 'Текстовая схожесть названий',
    'behavior_cluster': 'Кластеры по поведению'
}


# Callback для поиска сущности
@app.callback(
    [Output('entity-info', 'children'),
     Output('relations-panel', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value'),
     State('entity-type-filter', 'value')]
)
def search_entity(n_clicks, search_term, entity_type):
    if not n_clicks or not search_term:
        return dash.no_update, dash.no_update

    # Поиск сущности в графе
    found_entities = []
    for node in G.nodes():
        node_data = G.nodes[node]
        if (entity_type == 'all' or node_data.get('type') == entity_type) and \
                search_term.lower() in str(node).lower():
            found_entities.append((node, node_data))

    if not found_entities:
        return "Ничего не найдено", dash.no_update

    # Берем первую найденную сущность (можно добавить выбор, если найдено несколько)
    entity_id, entity_data = found_entities[0]

    # Формируем информацию о сущности
    info_content = generate_entity_info(entity_id, entity_data)

    # Формируем информацию о связях
    relations_content = generate_relations_info(entity_id, entity_data)

    return info_content, relations_content


def generate_entity_info(entity_id, entity_data):
    """Генерирует информацию о сущности в виде таблицы"""

    # Фильтруем атрибуты - оставляем type и убираем layer
    filtered_data = {k: v for k, v in entity_data.items() if k != 'layer'}

    # Создаем упорядоченный список атрибутов (ID и type сначала)
    ordered_attrs = ['ID', 'type'] + [k for k in filtered_data.keys() if k not in ['ID', 'type']]

    info_df = pd.DataFrame({
        'Атрибут': ordered_attrs,
        'Значение': [entity_id if attr == 'ID' else
                     filtered_data.get(attr, '') for attr in ordered_attrs]
    })

    return html.Div([
        html.H3(f"Информация о сущности: {entity_id}"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in info_df.columns],
            data=info_df.to_dict('records'),
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold'}
        )
    ])


def generate_relations_info(entity_id, entity_data):
    """Генерирует информацию о связях сущности"""
    relations = defaultdict(list)

    # Собираем все связи для данной сущности
    for neighbor in G.neighbors(entity_id):
        edge_data = G.get_edge_data(entity_id, neighbor)
        relation_type = edge_data.get('type', 'unknown')
        target_type = G.nodes[neighbor].get('type')

        # Добавляем связь в словарь
        relations[relation_type].append({
            'target': neighbor,
            'target_type': target_type
        })

    # Создаем панели для каждого типа связи
    relation_panels = []
    for relation_type, connections in relations.items():
        # Преобразуем в DataFrame
        df = pd.DataFrame(connections)

        # Определяем отображаемое название типа связи
        display_relation_type = RELATION_NAMES.get(relation_type, relation_type)

        # Если target_type = customer, заменяем на "Заказчики"
        if any(conn['target_type'] == 'customer' for conn in connections):
            display_relation_type = 'Заказчики'
        else:
            display_relation_type = RELATION_NAMES.get(relation_type, relation_type)

        # Добавляем столбец с типом связи
        df['Тип связи'] = display_relation_type
        print(df)
        relation_panels.append(html.Div([
            html.H4(display_relation_type),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df.columns if i != 'target_type'],
                # Убираем target_type из отображения
                data=df.to_dict('records'),
                style_table=styles['table'],
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                row_selectable='single',
                id={'type': 'relation-table', 'index': relation_type}
            )
        ]))


    return html.Div([
        html.H3("Связи сущности"),
        *relation_panels
    ])

    # Callback для навигации по связям


@app.callback(
    [Output('entity-info', 'children', allow_duplicate=True),
     Output('relations-panel', 'children', allow_duplicate=True)],
    [Input({'type': 'relation-table', 'index': dash.ALL}, 'selected_rows')],
    [State({'type': 'relation-table', 'index': dash.ALL}, 'data'),
     State({'type': 'relation-table', 'index': dash.ALL}, 'id')],
    prevent_initial_call=True
)
def navigate_to_related_entity(selected_rows_list, data_list, id_list):
    # Находим первую выбранную строку
    for selected_rows, data, id_dict in zip(selected_rows_list, data_list, id_list):
        if selected_rows:
            selected_row = data[selected_rows[0]]
            target_id = selected_row['target']
            target_data = G.nodes[target_id]

            # Генерируем информацию о новой сущности
            info_content = generate_entity_info(target_id, target_data)
            relations_content = generate_relations_info(target_id, target_data)

            return info_content, relations_content

    return dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run(debug=True)