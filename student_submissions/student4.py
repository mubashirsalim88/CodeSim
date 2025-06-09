def is_prime_recursive(num, divisor=2):
    if num < 2:
        return False
    if divisor > int(num ** 0.5):
        return True
    if num % divisor == 0:
        return False
    return is_prime_recursive(num, divisor + 1)

def find_primes_recursive(n):
    return [x for x in range(2, n+1) if is_prime_recursive(x)]

print(find_primes_recursive(50))
