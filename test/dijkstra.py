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

def find_min_distance(distances, visited, vertices):
    min_distance = float('inf')
    min_vertex = -1
    i = 1
    while i <= vertices:
        if not visited[i] and distances[i] < min_distance:
            min_distance = distances[i]
            min_vertex = i
        i = i + 1
    return min_vertex

def dijkstra(graph, start, vertices):
    distances = {}
    visited = {}
    i = 1
    while i <= vertices:
        distances[i] = float('inf')
        visited[i] = False
        i = i + 1
    distances[start] = 0
    count = 0
    while count < vertices:
        current = find_min_distance(distances, visited, vertices)
        if current == -1:
            break
        visited[current] = True
        for neighbor in graph[current]:
            if not visited[neighbor]:
                new_distance = distances[current] + graph[current][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
        count = count + 1
    return distances

vertices = 5
graph = initialize_graph(vertices)
add_edge(graph, 1, 2, 10)
add_edge(graph, 1, 3, 5)
add_edge(graph, 2, 3, 2)
add_edge(graph, 2, 4, 1)
add_edge(graph, 3, 4, 9)
add_edge(graph, 3, 5, 2)
add_edge(graph, 4, 5, 4)
start = 1
distances = dijkstra(graph, start, vertices)
i = 1
while i <= vertices:
    if distances[i] == float('inf'):
        print("Vertex", i, "is unreachable from vertex", start)
    else:
        print("Shortest distance from vertex", start, "to vertex", i, "is", distances[i])
    i = i + 1