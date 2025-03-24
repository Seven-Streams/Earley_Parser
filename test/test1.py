def count_even_numbers(limit):
    count = 0
    number += 1
    while number <= limit:
        if number % 2 == 0:
            count = count + 1
        number = number + 1
    return count

limit = 10
result = count_even_numbers(limit)

if result > 0:
    print("There are",result, "even numbers.")
else:
    print("No even numbers found.")