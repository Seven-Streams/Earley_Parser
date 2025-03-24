def build_segment_tree(tree, data, node, start, end):
    if start == end:
        tree[node] = data[start]
    else:
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        build_segment_tree(tree, data, left_child, start, mid)
        build_segment_tree(tree, data, right_child, mid + 1, end)
        tree[node] = tree[left_child] + tree[right_child]

def update_segment_tree(tree, node, start, end, idx, value):
    if start == end:
        tree[node] = value
    else:
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        if idx <= mid:
            update_segment_tree(tree, left_child, start, mid, idx, value)
        else:
            update_segment_tree(tree, right_child, mid + 1, end, idx, value)
        tree[node] = tree[left_child] + tree[right_child]

def query_segment_tree(tree, node, start, end, l, r):
    if r < start or l > end:
        return 0
    if l <= start and end <= r:
        return tree[node]
    mid = (start + end) // 2
    left_child = 2 * node + 1
    right_child = 2 * node + 2
    left_sum = query_segment_tree(tree, left_child, start, mid, l, r)
    right_sum = query_segment_tree(tree, right_child, mid + 1, end, l, r)
    return left_sum + right_sum

def initialize_tree(size):
    tree = {}
    i = 0
    while i < size:
        tree[i] = 0
        i = i + 1
    return tree

data = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
n = len(data)
tree = initialize_tree(4 * n)
build_segment_tree(tree, data, 0, 0, n - 1)
update_segment_tree(tree, 0, 0, n - 1, 2, 6)
result = query_segment_tree(tree, 0, 0, n - 1, 1, 3)
if result > 0:
    print("The sum of the range is", result)
else:
    print("The range sum is zero or negative")