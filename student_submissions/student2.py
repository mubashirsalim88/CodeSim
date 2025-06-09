def is_prime_list_comp(n):
    return [x for x in range(2, n+1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]

print(is_prime_list_comp(50))
