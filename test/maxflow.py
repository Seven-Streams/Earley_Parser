def initialize_graph(vertices):
    graph = {}
    i = 1
    while i <= vertices:
        graph[i] = {}
        i = i + 1
    return graph

def add_edge(graph, u, v, capacity):
    if u in graph:
        graph[u][v] = capacity
    if v in graph:
        graph[v][u] = 0

def bfs(graph, source, sink, parent, vertices):
    visited = {}
    i = 1
    while i <= vertices:
        visited[i] = False
        i = i + 1
    queue = [source]
    visited[source] = True
    while len(queue) > 0:
        current = queue.pop(0)
        for neighbor in graph[current]:
            if not visited[neighbor] and graph[current][neighbor] > 0:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
                if neighbor == sink:
                    return True
    return False

def edmonds_karp(graph, source, sink, vertices):
    parent = {}
    max_flow = 0
    while bfs(graph, source, sink, parent, vertices):
        path_flow = float('inf')
        current = sink
        while current != source:
            path_flow = min(path_flow, graph[parent[current]][current])
            current = parent[current]
        max_flow = max_flow + path_flow
        current = sink
        while current != source:
            prev = parent[current]
            graph[prev][current] = graph[prev][current] - path_flow
            graph[current][prev] = graph[current][prev] + path_flow
            current = prev
    return max_flow

vertices = 6
graph = initialize_graph(vertices)
add_edge(graph, 1, 2, 16)
add_edge(graph, 1, 3, 13)
add_edge(graph, 2, 3, 10)
add_edge(graph, 2, 4, 12)
add_edge(graph, 3, 2, 4)
add_edge(graph, 3, 5, 14)
add_edge(graph, 4, 3, 9)
add_edge(graph, 4, 6, 20)
add_edge(graph, 5, 4, 7)
add_edge(graph, 5, 6, 4)
source = 1
sink = 6
max_flow = edmonds_karp(graph, source, sink, vertices)
if max_flow > 0:
    print("The maximum possible flow is", max_flow)
else:
    print("No flow is possible from source to sink")