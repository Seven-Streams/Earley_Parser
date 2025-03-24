def factorial(n):
    result = 1
    i = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False

x = 5
y = 4

if is_even(x):
    print("x is even")
else:
    print("x is odd")

if is_even(y):
    print("y is even")
else:
    print("y is odd")
n = 6
print("Factorial of", n, "is", factorial(n))