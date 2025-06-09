def is_prime(num):
    return num > 1 and all(num % i != 0 for i in range(2, int(num**0.5) + 1))

def find_primes_functional(n):
    return list(filter(lambda x: is_prime(x), range(2, n+1)))

print(find_primes_functional(50))
