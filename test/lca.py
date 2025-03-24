def initialize_tree(vertices):
    tree = {}
    i = 1
    while i <= vertices:
        tree[i] = {}
        i = i + 1
    return tree

def add_edge(tree, u, v):
    if u in tree:
        tree[u][v] = True
    if v in tree:
        tree[v][u] = True

def dfs(tree, node, parent, depth, depths, parents):
    depths[node] = depth
    parents[node] = parent
    for neighbor in tree[node]:
        if neighbor != parent:
            dfs(tree, neighbor, node, depth + 1, depths, parents)

def preprocess_lca(tree, vertices, root):
    depths = {}
    parents = {}
    i = 1
    while i <= vertices:
        depths[i] = -1
        parents[i] = -1
        i = i + 1
    dfs(tree, root, -1, 0, depths, parents)
    return depths, parents

def find_lca(u, v, depths, parents):
    if depths[u] < depths[v]:
        u, v = v, u
    while depths[u] > depths[v]:
        u = parents[u]
    while u != v:
        u = parents[u]
        v = parents[v]
    return u

vertices = 7
tree = initialize_tree(vertices)
add_edge(tree, 1, 2)
add_edge(tree, 1, 3)
add_edge(tree, 2, 4)
add_edge(tree, 2, 5)
add_edge(tree, 3, 6)
add_edge(tree, 3, 7)
root = 1
depths, parents = preprocess_lca(tree, vertices, root)
u = 4
v = 5
lca = find_lca(u, v, depths, parents)
if lca != -1:
    print("The LCA of", u, "and", v, "is", lca)
else:
    print("No LCA found for", u, "and", v)