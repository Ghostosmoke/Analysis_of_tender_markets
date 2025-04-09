import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import heapq

# Загрузка данных
data = pd.read_csv('df_for_work.csv')

# columns_to_drop - список столбцов, которые нужно удалить
columns_to_drop = ['tt_url', 'tt_text', 'tt_analysis', 'word_count', 'okpd', 'ktru_details', 'proc_url', 'document_url',
                   'customer_url']
data = data.drop(columns=columns_to_drop)

# Заполнение пропущенных значений
data['proc_object'] = data['proc_object'].fillna('').astype(str)
data['ktru'] = data['ktru'].fillna('').astype(str)
data['executor'] = data['executor'].fillna('').astype(str)

# Создаем многослойный граф
G = nx.Graph()

# Добавляем вершины всех типов
for _, row in data.iterrows():
    # Слой заказчиков
    


    G.add_node(row['customer'], layer='customer', type='customer')

    # Слой закупок
    G.add_node(row['proc_id'], layer='procurement', type='procurement',
               customer=row['customer'], ktru=row['ktru'], executor=row['executor'],
               price=row['price'], post_date=row['post_date'])

    # Слой КТРУ
    if row['ktru']:  # Добавляем только если значение не пустое
        G.add_node(row['ktru'], layer='ktru', type='ktru')

    # Слой поставщиков
    if row['executor']:  # Добавляем только если значение не пустое
        G.add_node(row['executor'], layer='executor', type='executor')

######## Добавляем связи МЕЖДУ слоями #########
for _, row in data.iterrows():
    # Связь заказчик -> закупка
    G.add_edge(row['customer'], row['proc_id'], type='customer_procurement')

    # Связь закупка -> КТРУ
    if row['ktru']:
        G.add_edge(row['proc_id'], row['ktru'], type='procurement_ktru')

    # Связь закупка -> поставщик
    if row['executor']:
        G.add_edge(row['proc_id'], row['executor'], type='procurement_executor')

    # Связь КТРУ -> поставщик
ktru_executor_counts = data.groupby('ktru')['executor'].apply(
    lambda x: x.value_counts().to_dict() if not x.empty else {}
).to_dict()

for ktru, exec_counts in ktru_executor_counts.items():
    if pd.notna(ktru) and isinstance(exec_counts, dict):
        for executor, count in exec_counts.items():
            if pd.notna(executor):
                if ktru in G.nodes() and executor in G.nodes():  # Проверка существования узлов
                    G.add_edge(ktru, executor,
                               type='ktru_executor',
                               weight=count,
                               label=f"Совместно в {count} закупках")

# Определение всех параметров и вспомогательных переменных
price_similarity_threshold = 0.1  # 10% разницы в ценах
min_support = 0.05  # для ассоциативных правил
confidence_threshold = 0.7  # для ассоциативных правил
n_clusters = 5  # для кластеризации

# Подготовка данных для связей
customer_executors = data.groupby('customer')['executor'].apply(set).to_dict()
customer_methods = data.groupby('customer')['determination_method'].value_counts().unstack(fill_value=0)
customer_avg_price = data.groupby('customer')['price'].mean()

# ===== ЯВНЫЕ СВЯЗИ ДЛЯ СЛОЯ ЗАКАЗЧИКОВ =====

# 1. Совместные исполнители
for cust1, exec1 in customer_executors.items():
    for cust2, exec2 in customer_executors.items():
        if cust1 < cust2 and exec1 & exec2:
            common_execs = exec1 & exec2
            G.add_edge(cust1, cust2, type='shared_executor', weight=len(common_execs))

# 2. Совпадение методов закупок
for cust1 in customer_methods.index:
    for cust2 in customer_methods.index:
        if cust1 < cust2:
            common_methods = (customer_methods.loc[cust1] * customer_methods.loc[cust2]).sum()
            if common_methods > 0:
                G.add_edge(cust1, cust2, type='shared_methods', weight=common_methods)

# 3. Корреляция бюджетов
for cust1 in customer_avg_price.index:
    for cust2 in customer_avg_price.index:
        if cust1 < cust2:
            price_diff = abs(customer_avg_price[cust1] - customer_avg_price[cust2])
            avg_price = (customer_avg_price[cust1] + customer_avg_price[cust2]) / 2
            if price_diff / avg_price < price_similarity_threshold:
                G.add_edge(cust1, cust2, type='price_correlation', weight=1 - price_diff / avg_price)

# ===== НЕЯВНЫЕ СВЯЗИ ДЛЯ СЛОЯ ЗАКАЗЧИКОВ =====

# 1. Поведенческие кластеры (оптимизированная версия)
methods_ktru = data.groupby('customer').agg({
    'determination_method': lambda x: x.mode()[0],
    'ktru': lambda x: ','.join(x.value_counts().nlargest(3).index)
})

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(methods_ktru.apply(' '.join, axis=1))
clusters = KMeans(n_clusters=5).fit_predict(features)

for i, cust1 in enumerate(methods_ktru.index):
    for j, cust2 in enumerate(methods_ktru.index):
        if i < j and clusters[i] == clusters[j]:
            G.add_edge(cust1, cust2,
                       type='behavioral_cluster',
                       cluster=int(clusters[i]))

# 2. Ассоциативные правила по КТРУ
try:
    # Создаем кросс-таблицу и преобразуем в бинарный формат
    ktru_matrix = pd.crosstab(data['customer'], data['ktru'])
    ktru_matrix = (ktru_matrix > 0).astype(int)  # Преобразуем в 0/1

    # Проверяем, что матрица содержит только 0 и 1
    if not set(ktru_matrix.values.flatten()).issubset({0, 1}):
        raise ValueError("Матрица должна содержать только 0 и 1")

    # Применяем алгоритм Apriori
    frequent_itemsets = apriori(ktru_matrix,
                                min_support=min_support,
                                use_colnames=True,
                                max_len=3)  # Ограничение длины

    # Генерируем правила
    rules = association_rules(frequent_itemsets,
                              metric="lift",
                              min_threshold=1.5)

    # Добавляем связи в граф
    for _, rule in rules.iterrows():
        ant = list(rule['antecedents'])[0]
        cons = list(rule['consequents'])[0]

        # Получаем списки заказчиков
        ant_customers = ktru_matrix.index[ktru_matrix[ant] == 1]
        cons_customers = ktru_matrix.index[ktru_matrix[cons] == 1]

        # Добавляем связи между заказчиками
        for cust1 in ant_customers:
            for cust2 in cons_customers:
                if cust1 != cust2:
                    G.add_edge(cust1, cust2,
                               type='ktru_rule',
                               rule=f"{ant}→{cons}",
                               support=rule['support'],
                               confidence=rule['confidence'],
                               lift=rule['lift'])

except Exception as e:
    print(f"Ошибка при генерации ассоциативных правил: {str(e)}")

# Явные связи между закупками
for i, row1 in data.iterrows():
    for j, row2 in data.iterrows():
        if i < j:
            # 1. Связь по одинаковому КТРУ
            if row1['ktru'] and row2['ktru'] and row1['ktru'] == row2['ktru']:
                G.add_edge(row1['proc_id'], row2['proc_id'],
                           type='same_ktru',
                           ktru=row1['ktru'])

            # 2. Связь по методу закупки
            if row1['determination_method'] == row2['determination_method']:
                G.add_edge(row1['proc_id'], row2['proc_id'],
                           type='same_method',
                           method=row1['determination_method'])

# Неявные связи через текст закупок (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(data['proc_object'])
similarity_matrix = cosine_similarity(tfidf_matrix)
threshold = 0.7  # Повышаем порог для более релевантных связей

for i in range(len(data)):
    for j in range(i + 1, len(data)):
        if similarity_matrix[i, j] > threshold:
            G.add_edge(data.iloc[i]['proc_id'], data.iloc[j]['proc_id'],
                       type='text_similarity',
                       score=similarity_matrix[i, j])

        #  Явные связи между КТРУ

# Подготовка данных
data['ktru'] = data['ktru'].astype(str)
ktru_data = data[data['ktru'] != ''].copy()

# 1. Совместное использование в закупках (оптимизированная версия)
ktru_co_occurrence = ktru_data.groupby('proc_id')['ktru'].apply(list).to_dict()
for procs in ktru_co_occurrence.values():
    for i in range(len(procs)):
        for j in range(i + 1, len(procs)):
            if procs[i] != procs[j]:
                if not G.has_edge(procs[i], procs[j]):
                    G.add_edge(procs[i], procs[j], type='co_occurrence', weight=0)
                G[procs[i]][procs[j]]['weight'] += 1

# 2. Ценовые диапазоны (группировка по квантилям)
ktru_prices = ktru_data.groupby('ktru')['price'].agg(['median', 'count'])
ktru_prices = ktru_prices[ktru_prices['count'] > 3]  # Фильтр по количеству закупок
ktru_prices['price_group'] = pd.qcut(ktru_prices['median'], q=4, labels=['low', 'medium', 'high', 'very_high'])

for ktru1 in ktru_prices.index:
    for ktru2 in ktru_prices.index:
        if ktru1 < ktru2 and ktru_prices.loc[ktru1, 'price_group'] == ktru_prices.loc[ktru2, 'price_group']:
            G.add_edge(ktru1, ktru2, type='price_group', group=ktru_prices.loc[ktru1, 'price_group'])

# 3. Статусные паттерны
status_ratio = ktru_data.groupby('ktru')['proc_status'].apply(
    lambda x: (x == 'Завершена').mean()).to_dict()

for ktru1, ratio1 in status_ratio.items():
    for ktru2, ratio2 in status_ratio.items():
        if ktru1 < ktru2 and abs(ratio1 - ratio2) < 0.1:  # Разница менее 10%
            G.add_edge(ktru1, ktru2, type='status_pattern',
                       ratio_diff=abs(ratio1 - ratio2))

# Неявные связи между КТРУ

from collections import defaultdict
from itertools import combinations

# 1. Временные кластеры (сезонность)
ktru_data['quarter'] = pd.to_datetime(ktru_data['post_date']).dt.quarter
seasonal_patterns = ktru_data.groupby(['ktru', 'quarter']).size().unstack(fill_value=0)

for ktru1, ktru2 in combinations(seasonal_patterns.index, 2):
    similarity = cosine_similarity([seasonal_patterns.loc[ktru1]],
                                   [seasonal_patterns.loc[ktru2]])[0][0]
    if similarity > 0.7:
        G.add_edge(ktru1, ktru2,
                   type='seasonal_pattern',
                   similarity=round(similarity, 2))

# 2. Граф ассоциаций (совместное использование)
ktru_cooc = ktru_data.groupby('proc_id')['ktru'].apply(set)
cooc_counts = defaultdict(int)

for ktru_set in ktru_cooc:
    for ktru1, ktru2 in combinations(ktru_set, 2):
        cooc_counts[(min(ktru1, ktru2), max(ktru1, ktru2))] += 1

for (ktru1, ktru2), count in cooc_counts.items():
    if count >= 3:  # Фильтр по минимальному количеству совместных закупок
        if not G.has_edge(ktru1, ktru2):
            G.add_edge(ktru1, ktru2,
                       type='co_occurrence',
                       weight=count)
        else:
            # Обновляем вес, если связь уже существует
            G[ktru1][ktru2]['weight'] += count

# ===== ЯВНЫЕ СВЯЗИ ДЛЯ СЛОЯ ПОСТАВЩИКОВ =====

# Подготовка данных (с обработкой пропусков)
executor_data = data[data['executor'] != ''].copy()
executor_data['ktru'] = executor_data['ktru'].fillna('').astype(str)
executor_data = executor_data[executor_data['ktru'] != '']

# 1. Общие заказчики
customer_executors = executor_data.groupby('customer')['executor'].apply(set).to_dict()
executor_customers = executor_data.groupby('executor')['customer'].apply(set).to_dict()

for exec1, custs1 in executor_customers.items():
    for exec2, custs2 in executor_customers.items():
        if exec1 < exec2 and custs1 & custs2:
            common_custs = custs1 & custs2
            G.add_edge(exec1, exec2,
                       type='shared_customer',
                       weight=len(common_custs),
                       label=f"Общие заказчики: {len(common_custs)}")

# 2. Специализация по КТРУ
executor_ktru = executor_data.groupby('executor')['ktru'].apply(
    lambda x: x.value_counts().nlargest(3).to_dict() if not x.empty else {}
)

for exec1, ktru1 in executor_ktru.items():
    for exec2, ktru2 in executor_ktru.items():
        if exec1 < exec2 and isinstance(ktru1, dict) and isinstance(ktru2, dict):
            common_ktru = set(ktru1.keys()) & set(ktru2.keys())
            if common_ktru:
                G.add_edge(exec1, exec2,
                           type='shared_ktru',
                           ktru=list(common_ktru),
                           count=len(common_ktru))

# ===== НЕЯВНЫЕ СВЯЗИ ДЛЯ СЛОЯ ПОСТАВЩИКОВ =====

# 1. Сеть коисполнителей (через общие закупки)
proc_executors = executor_data.groupby('proc_id')['executor'].apply(set).to_dict()
coexec_counts = defaultdict(int)

for proc, execs in proc_executors.items():
    for exec1, exec2 in combinations(execs, 2):
        coexec_counts[min(exec1, exec2), max(exec1, exec2)] += 1

for (exec1, exec2), count in coexec_counts.items():
    if count >= 2:
        if not G.has_edge(exec1, exec2):
            G.add_edge(exec1, exec2,
                       type='co_execution',
                       weight=count)
        else:
            G[exec1][exec2]['weight'] += count

# 2. Текстовая схожесть названий
executor_names = executor_data['executor'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
name_matrix = vectorizer.fit_transform(executor_names)
name_sim = cosine_similarity(name_matrix)

threshold = 0.9  # Повышаем порог с 0.85 до 0.9
for i in range(len(executor_names)):
    for j in range(i + 1, len(executor_names)):
        if name_sim[i, j] > threshold:
            G.add_edge(executor_names[i], executor_names[j],
                       type='name_similarity',
                       score=round(name_sim[i, j], 2))

# 3. Кластеры по поведению
executor_features = executor_data.groupby('executor').agg({
    'ktru': lambda x: ','.join(x.value_counts().nlargest(3).index),
    'price': 'median'
}).fillna(0)

# Упрощенная кластеризация (только по КТРУ и ценам)
features = pd.get_dummies(executor_features['ktru'].str.get_dummies(','))
clusters = KMeans(n_clusters=5).fit_predict(features)

for i, exec1 in enumerate(executor_features.index):
    for j, exec2 in enumerate(executor_features.index):
        if i < j and clusters[i] == clusters[j]:
            G.add_edge(exec1, exec2,
                       type='behavior_cluster',
                       cluster=int(clusters[i]))

######### ВИЗУАЛИЗАЦИЯ ############

# Определяем стили для разных типов связей
edge_styles = {
    # Межслойные связи
    'customer_procurement': 'solid',
    'procurement_ktru': 'solid',
    'procurement_executor': 'solid',
    'ktru_executor': 'solid',

    # Внутрислойные связи заказчиков
    'shared_executor': 'dashed',
    'shared_methods': 'dotted',
    'price_correlation': 'dashed',
    'behavioral_cluster': 'dotted',
    'ktru_rule': 'dashed',

    # Внутрислойные связи закупок
    'same_ktru': 'dashed',
    'same_method': 'dotted',
    'text_similarity': 'dashed',

    # Внутрислойные связи КТРУ
    'co_occurrence': 'dashed',
    'price_group': 'dotted',
    'status_pattern': 'dashed',
    'seasonal_pattern': 'dotted',

    # Внутрислойные связи поставщиков
    'shared_customer': 'solid',
    'shared_ktru': 'dashed',
    'co_execution': 'dotted',
    'name_similarity': 'dashed',
    'behavior_cluster': 'dotted'
}

# Цвета для разных типов связей
edge_colors = {
    'Заказчик_закупка': 'darkblue',
    'Закупка_ктру': 'green',
    'Закупка_поставщик': 'purple',
    'Ктру_поставщик': 'orange',

    # Остальные связи серого цвета с разными оттенками
    'Совместные_исполнители': 'gray',
    'Совпадение_методов_закупок': 'darkgray',
    'Корреляция_бюджетов': 'lightgray',
    'Поведенчиские_кластеры': 'gray',
    'Ассоциативные_правила_по_ктру': 'darkgray',

    'Связь_по_одинаковому_ктру': 'gray',
    'Связь_по_методу_закупки': 'darkgray',
    'Связь_через_текст_закупок': 'lightgray',

    'Граф_ассоциаций': 'red',
    'Совместное_использование_в_закупках': 'gray',
    'Ценовые_диапозоны_группировка_по_кватилям': 'darkgray',
    'Статусные_паттерны': 'lightgray',
    'Временные_кластеры': 'gray',

    'Общие_заказчики': 'blue',
    'Специализация_по_ктру': 'gray',
    'Сеть_коисполнителей_через_общие_закупки': 'darkgray',
    'Текстовоя_схожесть_названий': 'lightgray',
    'Кластер_по_поведению': 'gray'
}

# Создаем позиционирование узлов по слоям
plt.figure(figsize=(20, 15))
pos = {}

# Координаты слоев
layer_positions = {
    'customer': (0, 0.5),
    'procurement': (1, 0.5),
    'ktru': (2, 0.5),
    'executor': (3, 0.5)
}

# Распределяем узлы по слоям с учетом их степени (важности)
for layer, (x, y) in layer_positions.items():
    layer_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == layer]

    # Сортируем узлы по степени (количеству связей)
    layer_nodes_sorted = sorted(layer_nodes, key=lambda n: G.degree(n), reverse=True)

    # Распределяем узлы по вертикали с учетом их важности
    for i, node in enumerate(layer_nodes_sorted):
        pos[node] = x, y + (i - len(layer_nodes) / 2) / max(1, len(layer_nodes) / 4)


def dijkstra_analysis(G, start_node, weight_attribute='weight'):
    """
    Анализ графа с помощью алгоритма Дейкстры для нахождения кратчайших путей
    от заданного узла до всех остальных узлов.

    Возвращает словарь расстояний и словарь предшественников.
    """
    # Инициализация
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = 0
    predecessors = {node: None for node in G.nodes()}

    # Очередь с приоритетами
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Пропускаем если нашли более короткий путь ранее
        if current_distance > distances[current_node]:
            continue

        for neighbor, edge_attrs in G[current_node].items():
            # Используем вес ребра, если он есть, иначе 1
            weight = edge_attrs.get(weight_attribute, 1)
            distance = current_distance + weight

            # Если нашли более короткий путь
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


def find_important_paths(G, start_node, num_paths=5):
    """
    Находит наиболее важные/короткие пути из начального узла.
    """
    distances, predecessors = dijkstra_analysis(G, start_node)

    # Фильтруем достижимые узлы
    reachable_nodes = {k: v for k, v in distances.items() if v != float('infinity')}

    # Сортируем по расстоянию
    sorted_nodes = sorted(reachable_nodes.items(), key=lambda x: x[1])

    # Выбираем топ-N самых близких и топ-N самых далеких
    top_closest = sorted_nodes[:num_paths]
    top_farthest = sorted_nodes[-num_paths:] if len(sorted_nodes) > num_paths else []

    # Восстанавливаем пути
    def reconstruct_path(target):
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = predecessors[node]
        return path[::-1]

    closest_paths = [(node, dist, reconstruct_path(node)) for node, dist in top_closest]
    farthest_paths = [(node, dist, reconstruct_path(node)) for node, dist in top_farthest]

    return closest_paths, farthest_paths


def visualize_paths(G, pos, paths, title="Кратчайшие пути"):
    """
    Визуализирует найденные пути на графе.
    """
    plt.figure(figsize=(15, 10))

    # Рисуем весь граф (светлее)
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.2)
    nx.draw_networkx_edges(G, pos, alpha=0.1)

    # Рисуем пути с выделением
    for i, (node, dist, path) in enumerate(paths):
        # Рисуем узлы пути
        path_nodes = {n: pos[n] for n in path if n in pos}
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=path_nodes.keys(),
            node_color=f'C{i}',
            node_size=100,
            alpha=0.8
        )

        # Рисуем ребра пути
        path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color=f'C{i}',
            width=2,
            alpha=0.8
        )

        # Подписи для узлов пути
        labels = {n: n for n in path_nodes if len(str(n)) < 20}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Пример использования:
if __name__ == "__main__":
    # Выбираем интересный узел для анализа (например, крупного заказчика)
    customers = [n for n in G.nodes() if G.nodes[n].get('type') == 'customer']
    important_customer = max(customers, key=lambda x: G.degree(x))

    print(f"\nАнализ кратчайших путей для заказчика: {important_customer}")

    # Находим пути
    closest_paths, farthest_paths = find_important_paths(G, important_customer)

    # Выводим результаты
    print("\nСамые близкие узлы:")
    for node, dist, path in closest_paths:
        print(f"{node} (расстояние: {dist}): {' -> '.join(str(p) for p in path)}")

    print("\nСамые далекие узлы:")
    for node, dist, path in farthest_paths:
        print(f"{node} (расстояние: {dist}): {' -> '.join(str(p) for p in path)}")

    # Визуализируем пути
    visualize_paths(G, pos, closest_paths, f"Самые близкие пути от {important_customer}")
    visualize_paths(G, pos, farthest_paths, f"Самые далекие пути от {important_customer}")


# Рисуем узлы с разными стилями
node_colors = {
    'customer': 'lightblue',
    'procurement': 'lightgreen',
    'ktru': 'salmon',
    'executor': 'gold'
}

node_sizes = {
    'customer': 300,
    'procurement': 200,
    'ktru': 250,
    'executor': 300,
    'ktru_level1': 150,
    'ktru_level2': 100,
    'ktru_level3': 70
}

for node_type, color in node_colors.items():
    nodes = [n for n in G.nodes()
             if G.nodes[n].get('type') == node_type or
             isinstance(node_type, str) and node_type.startswith('ktru_') and
             G.nodes[n].get('type', '').startswith('ktru_')]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color=color,
        node_size=[node_sizes.get(G.nodes[n].get('type', ''), 100) for n in nodes],
        alpha=0.8,
        label=node_type.replace('_', ' ').capitalize()
    )

# Рисуем связи с разными стилями и прозрачностью
for edge_type, style in edge_styles.items():
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]

    # Определяем ширину линии в зависимости от веса
    widths = [d.get('weight', 0.5) * 0.5 + 0.5 for u, v, d in G.edges(data=True) if d.get('type') == edge_type]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors.get(edge_type, 'gray'),
        style=style,
        width=widths if len(widths) == len(edges) else 0.5,
        alpha=0.6,
        label=edge_type.replace('_', ' ').capitalize()
    )

# Подписи для ключевых узлов
labels = {}
for node in G.nodes():
    node_type = G.nodes[node].get('type', '')
    if node_type in ['customer', 'executor'] or \
            (node_type == 'ktru' and G.degree(node) > 2) or \
            node.startswith('ktru_'):

        label = str(node)
        if len(label) > 20:
            if '_' in label:
                # Для КТРУ оставляем только последнюю часть кода
                label = label.split('_')[-1]
            else:
                label = label[:15] + '...'
        labels[node] = label

nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

print(G.edges)
# Легенда
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1, 1),
    fontsize=10,
    ncol=2
)
#print(G.nodes,G.edges)
plt.title("Многослойный граф закупочной деятельности\n(Заказчики → Закупки → КТРУ → Поставщики)", fontsize=14)
plt.axis('off')

# Сохраняем граф в файл
plt.tight_layout()
plt.savefig('procurement_multilayer_graph.png', dpi=300, bbox_inches='tight')
plt.show()
# # Получаем все номера заказов из графа
# procurement_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'procurement']
#
# for proc_id in procurement_nodes:
#     print(f"\nНомер заказа: {proc_id}")
#     print("Связи:")
#
#     # Инициализируем словари для хранения связей по типам
#     connections = {
#         'По заказчику': [],
#         'По КТРУ': [],
#         'По поставщику': [],
#         'По заказу': []
#     }
#
#     # Проходим по всем связям данного заказа
#     for neighbor in G.neighbors(proc_id):
#         edge_data = G.get_edge_data(proc_id, neighbor)
#         edge_type = edge_data.get('type', '')
#
#         # Определяем тип узла-соседа
#         neighbor_type = G.nodes[neighbor].get('type', '')
#
#         # Классифицируем связи
#         if neighbor_type == 'customer':
#             connections['По заказчику'].append((neighbor, edge_type))
#         elif neighbor_type == 'ktru':
#             connections['По КТРУ'].append((neighbor, edge_type))
#         elif neighbor_type == 'executor':
#             connections['По поставщику'].append((neighbor, edge_type))
#         elif neighbor_type == 'procurement':
#             connections['По заказу'].append((neighbor, edge_type))
#
#     # Выводим результаты
#     for connection_type, links in connections.items():
#         print(f"\n{connection_type}:")
#         if not links:
#             print("  Нет связей")
#         else:
#             for node, edge_type in links:
#                 # Форматируем вывод для длинных идентификаторов
#                 display_node = str(node)
#                 if len(display_node) > 30:
#                     display_node = display_node[:15] + "..." + display_node[-15:]
#
#                 # Получаем дополнительные атрибуты
#                 edge_attrs = {k: v for k, v in G.get_edge_data(proc_id, node).items()
#                               if k not in ['type', 'weight']}
#
#                 print(f"  - {display_node} (тип связи: {edge_type})", end='')
#                 if edge_attrs:
#                     print(f", атрибуты: {edge_attrs}")
#                 else:
#                     print()
#
#     # Добавляем разделитель между заказами
#     print("\n" + "=" * 80 + "\n")
# Функция для форматирования вывода связи
def format_connection(proc_id, node, edge_data):
    display_node = str(node)
    if len(display_node) > 30:
        display_node = display_node[:15] + "..." + display_node[-15:]

    edge_type = edge_data.get('type', '')
    edge_attrs = {k: v for k, v in edge_data.items()
                  if k not in ['type', 'weight']}

    connection_str = f"  - {display_node} (тип связи: {edge_type})"
    if edge_attrs:
        connection_str += f", атрибуты: {edge_attrs}"
    return connection_str


# Открываем файл для записи
with open('procurement_connections.txt', 'w', encoding='utf-8') as f:
    # Получаем все номера заказов из графа
    procurement_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'procurement']

    for proc_id in procurement_nodes:
        f.write(f"\nНомер заказа: {proc_id}\n")
        f.write("Связи:\n")

        # Инициализируем словари для хранения связей по типам
        connections = {
            'По заказчику': [],
            'По КТРУ': [],
            'По поставщику': [],
            'По заказу': []
        }

        # Проходим по всем связям данного заказа
        for neighbor in G.neighbors(proc_id):
            edge_data = G.get_edge_data(proc_id, neighbor)

            # Определяем тип узла-соседа
            neighbor_type = G.nodes[neighbor].get('type', '')

            # Классифицируем связи
            if neighbor_type == 'customer':
                connections['По заказчику'].append((neighbor, edge_data))
            elif neighbor_type == 'ktru':
                connections['По КТРУ'].append((neighbor, edge_data))
            elif neighbor_type == 'executor':
                connections['По поставщику'].append((neighbor, edge_data))
            elif neighbor_type == 'procurement':
                connections['По заказу'].append((neighbor, edge_data))

        # Записываем результаты в файл
        for connection_type, links in connections.items():
            f.write(f"\n{connection_type}:\n")
            if not links:
                f.write("  Нет связей\n")
            else:
                for node, edge_data in links:
                    f.write(format_connection(proc_id, node, edge_data) + "\n")

        # Добавляем разделитель между заказами
        f.write("\n" + "=" * 80 + "\n\n")

print("Анализ связей успешно сохранен в файл procurement_connections.txt")