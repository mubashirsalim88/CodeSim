def is_palindrome_string(num):
    return str(num) == str(num)[::-1]

# Test
number = 121
print(f"{number} is palindrome? {is_palindrome_string(number)}")
