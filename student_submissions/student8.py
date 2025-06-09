prime_cache = {}

def is_prime_memo(num):
    if num in prime_cache:
        return prime_cache[num]
    if num < 2:
        prime_cache[num] = False
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            prime_cache[num] = False
            return False
    prime_cache[num] = True
    return True

def find_primes_memo(n):
    return [x for x in range(2, n+1) if is_prime_memo(x)]

print(find_primes_memo(50))
