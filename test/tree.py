def create_node(value, parent, depth, size, heavy, top, chain_index):
    return {"value": value, "parent": parent, "depth": depth, "size": size, "heavy": heavy, "top": top, "chain_index": chain_index, "children": {}}

def initialize_tree(n):
    tree = {}
    i = 1
    while i <= n:
        tree[i] = create_node(i, None, 0, 0, None, None, None)
        i = i + 1
    return tree

def add_edge(tree, u, v):
    if u in tree and v in tree:
        tree[u]["children"][v] = tree[v]
        tree[v]["children"][u] = tree[u]

def dfs_size(tree, node, parent):
    node["parent"] = parent
    node["size"] = 1
    max_size = 0
    for child in node["children"].values():
        if child != parent:
            child["depth"] = node["depth"] + 1
            dfs_size(tree, child, node)
            node["size"] = node["size"] + child["size"]
            if child["size"] > max_size:
                max_size = child["size"]
                node["heavy"] = child

def decompose(tree, node, top, chain_index):
    node["top"] = top
    node["chain_index"] = chain_index
    if node["heavy"] is not None:
        decompose(tree, node["heavy"], top, chain_index)
    for child in node["children"].values():
        if child != node["parent"] and child != node["heavy"]:
            decompose(tree, child, child, chain_index + 1)

def lca(tree, u, v):
    while u["top"] != v["top"]:
        if u["top"]["depth"] < v["top"]["depth"]:
            v = v["top"]["parent"]
        else:
            u = u["top"]["parent"]
    if u["depth"] < v["depth"]:
        return u
    else:
        return v

tree = initialize_tree(7)
add_edge(tree, 1, 2)
add_edge(tree, 1, 3)
add_edge(tree, 2, 4)
add_edge(tree, 2, 5)
add_edge(tree, 3, 6)
add_edge(tree, 3, 7)
dfs_size(tree, tree[1], None)
decompose(tree, tree[1], tree[1], 0)
u = tree[4]
v = tree[5]
ancestor = lca(tree, u, v)
print("LCA of", u["value"], "and", v["value"], "is", ancestor["value"])