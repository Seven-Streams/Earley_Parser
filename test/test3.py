def gcd(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def is_prime(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i = i + 1
    return True

def next_prime(n):
    n = n + 1
    while not is_prime(n):
        n = n + 1
    return n

def sum_of_primes(limit):
    total = 0
    num = 2
    while num <= limit:
        if is_prime(num):
            total = total + num
        num = num + 1
    return total

def factorial(n):
    result = 1
    i = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

x = 15
y = 20
g = gcd(x, y)
l = lcm(x, y)
if g > 1:
    print("GCD is", g)
else:
    print("GCD is 1")
if l > 100:
    print("LCM is large")
else:
    print("LCM is small")

n = 10
f = factorial(n)
if f % 2 == 0:
    print("Factorial is even")
else:
    print("Factorial is odd")

limit = 30
s = sum_of_primes(limit)
if s > 100:
    print("Sum of primes is large")
else:
    print("Sum of primes is small")

p = 7
np = next_prime(p)
if np > 10:
    print("Next prime is greater than 10")
else:
    print("Next prime is 10 or less")