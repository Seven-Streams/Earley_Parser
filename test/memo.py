def create_memo():
    return {"data": {}, "hits": 0, "misses": 0}

def memo_get(memo, key):
    if key in memo["data"]:
        memo["hits"] = memo["hits"] + 1
        return memo["data"][key]
    else:
        memo["misses"] = memo["misses"] + 1
        return None

def memo_set(memo, key, value):
    memo["data"][key] = value

def fibonacci(n, memo):
    cached = memo_get(memo, n)
    if cached is not None:
        return cached
    if n <= 1:
        result = n
    else:
        result = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    memo_set(memo, n, result)
    return result

def factorial(n, memo):
    cached = memo_get(memo, n)
    if cached is not None:
        return cached
    if n == 0 or n == 1:
        result = 1
    else:
        result = n * factorial(n - 1, memo)
    memo_set(memo, n, result)
    return result

def sum_of_squares(n, memo):
    cached = memo_get(memo, n)
    if cached is not None:
        return cached
    if n == 0:
        result = 0
    else:
        result = n * n + sum_of_squares(n - 1, memo)
    memo_set(memo, n, result)
    return result

memo = create_memo()
fib_result = fibonacci(20, memo)
fact_result = factorial(10, memo)
squares_result = sum_of_squares(15, memo)
if fib_result > 0:
    print("Fibonacci result is", fib_result)
else:
    print("Fibonacci result is invalid")
if fact_result > 0:
    print("Factorial result is", fact_result)
else:
    print("Factorial result is invalid")
if squares_result > 0:
    print("Sum of squares result is", squares_result)
else:
    print("Sum of squares result is invalid")
print("Memo hits:", memo["hits"], "Memo misses:", memo["misses"])