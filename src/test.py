def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

primes = []
for num in range(101, 201):
    if is_prime(num):
        primes.append(num)

print("101到200之间的素数有：")
print(primes)
print(f"素数的个数: {len(primes)}")