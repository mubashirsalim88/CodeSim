import numpy as np

def primes_numpy(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0].tolist()

print(primes_numpy(50))
