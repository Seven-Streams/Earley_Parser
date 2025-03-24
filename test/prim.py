def initialize_graph(vertices):
    graph = {}
    i = 1
    while i <= vertices:
        graph[i] = {}
        i = i + 1
    return graph

def add_edge(graph, u, v, weight):
    if u in graph:
        graph[u][v] = weight
    if v in graph:
        graph[v][u] = weight

def find_min_key(keys, mst_set, vertices):
    min_value = float('inf')
    min_vertex = -1
    i = 1
    while i <= vertices:
        if not mst_set[i] and keys[i] < min_value:
            min_value = keys[i]
            min_vertex = i
        i = i + 1
    return min_vertex

def prim_mst(graph, vertices):
    keys = {}
    mst_set = {}
    parent = {}
    i = 1
    while i <= vertices:
        keys[i] = float('inf')
        mst_set[i] = False
        parent[i] = -1
        i = i + 1
    keys[1] = 0
    count = 0
    while count < vertices:
        u = find_min_key(keys, mst_set, vertices)
        mst_set[u] = True
        for neighbor in graph[u]:
            if not mst_set[neighbor] and graph[u][neighbor] < keys[neighbor]:
                keys[neighbor] = graph[u][neighbor]
                parent[neighbor] = u
        count = count + 1
    return parent

vertices = 5
graph = initialize_graph(vertices)
add_edge(graph, 1, 2, 2)
add_edge(graph, 1, 3, 3)
add_edge(graph, 2, 3, 1)
add_edge(graph, 2, 4, 4)
add_edge(graph, 3, 4, 5)
add_edge(graph, 3, 5, 6)
add_edge(graph, 4, 5, 7)
parent = prim_mst(graph, vertices)
i = 2
while i <= vertices:
    if parent[i] != -1:
        print("Edge from", parent[i], "to", i, "is in the MST")
    i = i + 1