def create_priority_queue():
    return {"data": {}, "size": 0}

def enqueue(queue, priority, value):
    if priority in queue["data"]:
        queue["data"][priority].append(value)
    else:
        queue["data"][priority] = [value]
    queue["size"] = queue["size"] + 1

def dequeue(queue):
    if queue["size"] == 0:
        return None
    highest_priority = None
    for priority in queue["data"]:
        if highest_priority is None or priority < highest_priority:
            highest_priority = priority
    value = queue["data"][highest_priority].pop(0)
    if len(queue["data"][highest_priority]) == 0:
        del queue["data"][highest_priority]
    queue["size"] = queue["size"] - 1
    return value

def is_empty(queue):
    return queue["size"] == 0

queue = create_priority_queue()
enqueue(queue, 2, "task2")
enqueue(queue, 1, "task1")
enqueue(queue, 3, "task3")
enqueue(queue, 1, "task1-2")
while not is_empty(queue):
    task = dequeue(queue)
    if task is not None:
        print("Processing", task)