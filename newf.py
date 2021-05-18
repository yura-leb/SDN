from math import acos, cos, sin, radians
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import csv
import sys
import os.path

Earth_radius = 6371


# обработка поля node
def get_node(graph, file):
    id = 0
    has_label, has_lat, has_long = False, False, False # флаги существования нужных полей
    line = file.readline()
    while line.strip(' \t\n') != ']':
        first_word, value = line.strip(' \t\n').split(sep=' ', maxsplit=1)
        if first_word == 'id':
            id = int(value)
        elif first_word == 'label':
            graph.add_node(id)
            graph.nodes[id]['label'] = str(value).strip('"')
            has_label = True
        elif first_word == 'Longitude':
            graph.nodes[id]['Longitude'] = float(value)
            has_lat = True
        elif first_word == 'Latitude':
            graph.nodes[id]['Latitude'] = float(value)
            has_long = True
        line = file.readline()
    if has_label and has_lat and has_long:
        pass
    else:  # если какого-то поля не было, удаление данного узла
        graph.remove_node(id)

# обработка поля edge
def get_edge(graph, file):
    nodes = list(graph.nodes)
    line = file.readline()
    has_source = False
    source = 0
    while line.strip(' \t\n') != ']': # пока не конец поля edge
        first_word, value = line.strip(' \t\n').split(sep=' ', maxsplit=1)
        if first_word == 'source':
            source = int(value)
            has_source = True
        elif first_word == 'target':
            # если был задан источник и источнику и цели соответствуют какие-то вершины
            if has_source and source in nodes and int(value) in nodes:
                graph.add_edge(source, int(value))
        line = file.readline()

def delete_equal_nodes(graph, node_numbers, node_id):
    for j in node_numbers:
        lat_1 = graph.nodes[node_id]['Latitude']
        lat_2 = graph.nodes[j]['Latitude']
        if lat_1 == lat_2:
            long_1 = graph.nodes[node_id]['Longitude']
            long_2 = graph.nodes[j]['Longitude']
            if long_1 == long_2:
                graph.remove_node(j)
                node_numbers.remove(j)
                for edge in list(graph.edges):
                    if edge[0] == j or edge[1] == j:
                        graph.remove_edge(edge[0], edge[1])

def parse(filename):
    graph = nx.Graph()
    with open(filename, mode='r') as file:
        line = file.readline()
        while line and line.strip(' \t\n') != ']':  # пока не конец файла считываем вершину или грань
            if line.strip(' \t\n') == 'node [':  # если поле node
                get_node(graph, file)
            elif line.strip(' \t\n') == 'edge [':  # если поле edge
                get_edge(graph, file)
            line = file.readline()
    nodes = list(graph.nodes)
    for i in nodes:  # удаление изолированных узлов
        if graph.degree[i] == 0:
            graph.remove_node(i)
    if (list(graph.nodes)) == list():
        return None
    else:
        node_numbers = list(graph.nodes)
        node_id = node_numbers[0]
        node_numbers.remove(node_id)
        while node_numbers != list():
            delete_equal_nodes(graph, node_numbers, node_id)
            if node_numbers:
                node_id = node_numbers[0]
                node_numbers.remove(node_id)
        return graph

def write_labels(file_writer):
    file_writer.writerow(['Node 1 (id)', 'Node 1 (label)', 'Node 1 (longitude)',
                          'Node 1 (latitude)', 'Node 2 (id)', 'Node 2 (label)',
                          'Node 2 (longitude)', 'Node 2 (latitude)',
                          'Distance (km)', 'Delay (mks)'])

def write_string(file_writer, graph, i, j, edgeattr):
    file_writer.writerow([i, graph.nodes[i]['label'],graph.nodes[i]['Longitude'],
                          graph.nodes[i]['Latitude'], j, graph.nodes[j]['label'],
                          graph.nodes[j]['Longitude'],
                          graph.nodes[j]['Latitude'],
                          int(edgeattr['distance']),
                          int(edgeattr['distance'] * 4.8)])

def write_csv_k1(csv_filename, graph):
    with open(csv_filename + '_topo.csv', mode='w', encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\r')
        file_writer.writerow(['Node 1 (id)', 'Node 1 (label)',           # строка заголовков
                              'Node 1 (longitude)', 'Node 1 (latitude)',
                              'Node 2 (id)', 'Node 2 (label)',
                              'Node 2 (longitude)', 'Node 2 (latitude)',
                              'Distance (km)', 'Delay (mks)'])
        used_nodes = set()
        for i in sorted(list(graph.nodes)):
            used_nodes.add(i)
            for j, edgeattr in graph.adj[i].items():    # для номера соседа, атрибутов грани
                if not j in used_nodes:
                    write_string(file_writer, graph, i, j, edgeattr)


def get_dict_of_paths(graph, node, distance=0, path=None, used_nodes=None, dict_of_paths=None):
    if dict_of_paths is None:
        dict_of_paths = dict()  # словарь путей с расстояниями
    if path is None:
        path = list()
        path.append(node)
    if used_nodes is None:
        used_nodes = set()
        used_nodes.add(node)
    for neighbour, edgeattr in graph.adj[node].items(): # для каждого соседа, атрибутов ребра вершины node
        if not neighbour in used_nodes:     # если не использовался
            distance += edgeattr['distance']
            path.append(neighbour)
            dict_of_paths[neighbour] = (tuple(path), distance)  # записываем в словарь путь и расстояние
            used_nodes.add(neighbour)
            if graph.degree[neighbour] == 1:    # если висячая вершина
                path.remove(neighbour)          # удаляем её из пути
                distance -= edgeattr['distance']
            else:   # если есть потомки, рекурсивный вызов для соседней вершины
                dict_of_paths = get_dict_of_paths(graph, neighbour, distance, path,
                                                  used_nodes, dict_of_paths)
                path.remove(neighbour)  # подсчет для соседней вершины прошёл, убираем её из пути
                distance -= edgeattr['distance']
    return dict_of_paths

def write_csv_k2(csv_filename, graph, node):
    with open(csv_filename + '_routes.csv', mode='w', encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=';', lineterminator='\r')
        file_writer.writerow(['Node 1 (id)', 'Node 1 (label)', 'Path', 'Delay (mks)'])
        path_dict = get_dict_of_paths(graph, node)
        node_numbers = sorted(path_dict)
        for target in node_numbers:
            file_writer.writerow([node, target, path_dict[target][0],
                                  int(path_dict[target][1] * 4.8)])


def make_distances(graph):          # сопоставление граням расстояний между
    for edge in list(graph.edges):  # вершинамидля каждой грани
        id_1 = edge[0]
        id_2 = edge[1]
        lat_1 = graph.nodes.data()[id_1].get("Latitude")    # широта первой вершины в грани
        long_1 = graph.nodes.data()[id_1].get("Longitude")  # долгота первой вершины в грани
        lat_2 = graph.nodes.data()[id_2].get("Latitude")    # широта второй вершины в грани
        long_2 = graph.nodes.data()[id_2].get("Longitude")  # долгота второй вершины в грани
        lat_1, lat_2, long_1, long_2 = map(radians, (lat_1, lat_2, long_1, long_2))
        alpha = acos(sin(lat_1) * sin(lat_2) +                        # угол между двумя вершинами
                     cos(lat_1) * cos(lat_2) * cos(long_1 - long_2))  # с вершиной угла в центре Земли
        distance = Earth_radius * alpha
        graph[id_1][id_2]['distance'] = distance
    return graph

def dijkstra_make_weight(graph, neighbour, edgeattr, node_number, new_nodes):
    if graph.nodes[neighbour]['is_met']:
        # если встречалась ранее, присваиваем вершине минимальный из весов по алгоритму Дейкстры
        graph.nodes[neighbour]['weight'] = min(graph.nodes[neighbour]['weight'],
                                               graph.nodes[node_number]['weight'] + \
                                               edgeattr['distance'])
    else:  # иначе присваиваем сумму весов node_number и расстояния между ним и соседней вершиной
        new_nodes.add(neighbour)  # добавление в множество доступных вершин
        graph.nodes[neighbour]['weight'] = graph.nodes[node_number]['weight'] + \
                                           edgeattr['distance']
        graph.nodes[neighbour]['is_met'] = True

def not_connected(graph, node_number, used_nodes):
    new_nodes = set(graph.nodes)
    new_nodes.difference_update(used_nodes)
    new_nodes = list(sorted(new_nodes))  # список неиспользованных вершин
    for edge in list(graph.edges):  # удаление граней, в которых есть неиспользованные вершины
        if edge[0] in new_nodes or edge[1] in new_nodes:
            graph.remove_edge(edge[0], edge[1])
    for node in new_nodes:  # удаление неиспользованных вершин
        graph.remove_node(node)

def find_min_weight_number(graph, new_nodes):
    min_weight_number = new_nodes.pop()  # присвоили номеру с минимальным весом
    new_nodes.add(min_weight_number)  # случайный номер из доступных вершин
    min_weight = graph.nodes[min_weight_number]['weight']
    for number in new_nodes:  # поиск номера с минимальным весом
        if graph.nodes[number]['weight'] < min_weight:
            min_weight = graph.nodes[number]['weight']
            min_weight_number = number
    return min_weight_number

# поиск максимального пути в графе от конкретной вершины node_number
def dijkstra(graph, node_number, new_nodes=None, used_nodes=None):
    if new_nodes is None:   # множество вершин, доступных для следующего использования
        new_nodes = set()
    if used_nodes is None:  # множество использованных вершин
        used_nodes = set()
    used_nodes.add(node_number)     # перенос используемой на данном шаге
    new_nodes.discard(node_number)  # вершины в использованные
    if len(used_nodes) != graph.number_of_nodes():  # если использовались не все
        # для каждого номера соседа вершины node_number, атрибутов грани
        for neighbour, edgeattr in graph.adj[node_number].items():
            if not (neighbour in used_nodes):   # если соседняя вершина не использовалась
                dijkstra_make_weight(graph, neighbour, edgeattr, node_number, new_nodes)
        # если множество доступных вершин пусто, но ещё не все использованы,
        # значит граф не связный
        if new_nodes == set():
            not_connected(graph, node_number, used_nodes)
            # возврат максимальной длины пути
            return graph.nodes[node_number]['weight']
        # если ещё есть доступные вершины
        min_weight_number = find_min_weight_number(graph, new_nodes)
        # рекурсивный переход к доступной вершине с минимальным весом
        return dijkstra(graph, min_weight_number, new_nodes, used_nodes)
    else: # если использовалтсь все
        # возврат длины максимального пути
        return graph.nodes[node_number]['weight']

# функция, задающая начальные условия для dijkstra
def max_path(graph, node_number):
    for node in graph.nodes.data():
        node[1]['weight'] = 200000000   # присваивание бесконечных весов
        node[1]['is_met'] = False
    graph.nodes[node_number]['weight'] = 0
    return dijkstra(graph, node_number, set(), set())

# поиск лучшей вершины по критерию к1
def best_node(graph):
    node_ids = list(graph.nodes)
    best_id = node_ids[0]
    best_path = max_path(graph, best_id)
    node_ids = list(graph.nodes)  # обновление списка узлов на случай, если граф не связный
    node_ids.remove(best_id)
    for id in node_ids:
        path = max_path(graph, id)
        if best_path > path:
            best_path = path
            best_id = id
    return best_id

# поиск минимального ребра из доступных
def min_edge(graph, used_nodes, list_of_nodes, min_dist=20000000, min_dist_source=0, min_dist_target=0):
    for node in used_nodes:  # для каждой использованной вершины
    # для каждого номера соседа и атрибутов грани из вершины node
        for neighbour, edgeattr in graph.adj[node].items():
            if not edgeattr['in_skelet'] and neighbour in (list_of_nodes):
                if (min_dist > edgeattr['distance']):
                    min_dist = edgeattr['distance']
                    min_dist_source = node
                    min_dist_target = neighbour
    return (min_dist_source, min_dist_target)

# создание остовного дерева по алгоритму Прима
def skelet_tree(graph):
    new_graph = nx.create_empty_copy(graph)  # создание графа без граней
    for edge in graph.edges.data():   # присваивание граням флагов использования в остовном дереве
        edge[2]['in_skelet'] = False
    list_of_nodes = list(graph.nodes)  # список доступных вершин
    used_nodes = [list_of_nodes[0]]  # начинаем с некоторой нулевой вершины
    list_of_nodes.remove(list_of_nodes[0])
    while list_of_nodes:
        # поиск минимального ребра
        min_dist_source, min_dist_target = min_edge(graph, used_nodes, list_of_nodes)
        # перенос вершины минимального ребра в список использованных
        list_of_nodes.remove(min_dist_target)
        used_nodes.append(min_dist_target)
        # флаг о нахождении ребра в остовном дереве
        graph.edges[min_dist_source, min_dist_target]['in_skelet'] = True
        new_graph.add_edge(min_dist_source, min_dist_target) # добавление в остовное дерево
        new_graph.edges[min_dist_source, min_dist_target]['distance'] = \
            graph.edges[min_dist_source, min_dist_target]['distance']
    return new_graph


if __name__ == '__main__':
    filename = sys.argv[2]
    mode = int(sys.argv[4])
    if len(sys.argv) == 5:
        if sys.argv[1] != '-t' or sys.argv[3] != '-k':
            print('Input error. Wrong flags')
        elif mode != 1 and mode != 2:
            print('Input error. Wrong -k mode')
        elif not os.path.exists(filename):
            print('Input error. File does not exist')
        else:
            graph = parse(filename)
            filename = filename[:-4]
            if graph:
                make_distances(graph)
                if mode == 1:
                    best_number = best_node(graph)
                    write_csv_k1(filename, graph)
                else:
                    graph = skelet_tree(graph)
                    best_number = best_node(graph)
                    write_csv_k2(filename, graph, best_number)
                ax = plt.subplot()
                for edge in list(graph.edges.data()):
                    x0 = graph.nodes[edge[0]]['Longitude']
                    y0 = graph.nodes[edge[0]]['Latitude']
                    x1 = graph.nodes[edge[1]]['Longitude']
                    y1 = graph.nodes[edge[1]]['Latitude']
                    line = mlines.Line2D([x0, x1], [y0, y1], color='darkblue', linewidth=1, linestyle='-')
                    ax.add_line(line)
                for node in graph.nodes.data():
                    plt.scatter(node[1]['Longitude'], node[1]['Latitude'], c='blue', linewidths=1)
                    ax.annotate(node[1]['label'], (node[1]['Longitude'], node[1]['Latitude']))
                plt.scatter(graph.nodes[best_number]['Longitude'],
                            graph.nodes[best_number]['Latitude'], c='red', linewidths=5)
                plt.show()
            else:
                print("Input error. Graph is empty")
    else:
        print('Input error. Wrong number of arguments')
