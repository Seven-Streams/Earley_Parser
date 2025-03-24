def power(base, exp):
    result = 1
    i = 0
    while i < exp:
        result = result * base
        i = i + 1
    return result

def is_palindrome(number):
    original = number
    reverse = 0
    while number > 0:
        digit = number % 10
        reverse = reverse * 10 + digit
        number = number // 10
    if original == reverse:
        return True
    else:
        return False

def count_palindromes(limit):
    count = 0
    num = 1
    while num <= limit:
        if is_palindrome(num):
            count = count + 1
        num = num + 1
    return count

def sum_of_powers(limit, exp):
    total = 0
    num = 1
    while num <= limit:
        total = total + power(num, exp)
        num = num + 1
    return total

def nested_loops(limit):
    total = 0
    i = 1
    while i <= limit:
        j = 1
        while j <= i:
            k = 1
            while k <= j:
                total = total + i * j * k
                k = k + 1
            j = j + 1
        i = i + 1
    return total

limit = 100
exp = 3
palindrome_count = count_palindromes(limit)
power_sum = sum_of_powers(limit, exp)
nested_result = nested_loops(10)

if palindrome_count > 50:
    print("Many palindromes")
else:
    print("Few palindromes")

if power_sum > 10000:
    print("Sum of powers is large")
else:
    print("Sum of powers is small")

if nested_result > 500:
    print("Nested result is large")
else:
    print("Nested result is small")