import math
from fractions import Fraction

def comb(n, k):
    return math.comb(n, k)

def p_hyper(N, K, n, k):
    return Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))

def p_at_least(N, K, n, m):
    tot = comb(N, n)
    less = sum(comb(K, k) * comb(N - K, n - k) for k in range(m))
    return Fraction(tot - less, tot)

def p_2way(N, K1, K2, n):
    tot = comb(N, n)
    fav = tot - comb(N-K1, n) - comb(N-K2, n) + comb(N-K1-K2, n)
    return Fraction(fav, tot)

def p_3way(N, K1, K2, K3, n):
    tot = comb(N, n)
    fav = tot - (comb(N-K1,n) + comb(N-K2,n) + comb(N-K3,n)) + (comb(N-K1-K2,n) + comb(N-K1-K3,n) + comb(N-K2-K3,n)) - comb(N-K1-K2-K3,n)
    return Fraction(fav, tot)

configs = [
    ("Config A: Heavy Basic (14B / 8S / 8D / 12E)", 14, 8, 8, 16, 12),
    ("Config B: Balanced (12B / 10S / 8D / 12E)", 12, 10, 8, 18, 12),
    ("Config C: Standard Engine (10B / 10S / 8D / 12E)", 10, 10, 8, 18, 12),
    ("Config D: High Accel (10B / 10S / 8D / 14E)", 10, 10, 8, 18, 14),
    ("Config E: Turbo Search (8B / 12S / 8D / 12E)", 8, 12, 8, 20, 12),
]

print("="*105)
print(f"{'Configuration':<42} | {'P(Mul)':<8} | {'P(Set7)':<8} | {'P(Set8)':<8} | {'P(B&S 7)':<8} | {'P(B&Eng7)':<9} | {'P(Ideal 7)':<10} | {'P(Ideal 8)':<10}")
print("="*105)

for name, kb, ks, kd, keng, ke in configs:
    pm = float(p_hyper(60, kb, 7, 0)) * 100
    ps7 = float(p_at_least(60, kb, 7, 1)) * 100
    ps8 = float(p_at_least(60, kb, 8, 1)) * 100
    pbs7 = float(p_2way(60, kb, ks, 7)) * 100
    pbeng7 = float(p_2way(60, kb, keng, 7)) * 100
    pideal7 = float(p_3way(60, kb, keng, ke, 7)) * 100
    pideal8 = float(p_3way(60, kb, keng, ke, 8)) * 100
    print(f"{name:<42} | {pm:6.2f}% | {ps7:6.2f}% | {ps8:6.2f}% | {pbs7:6.2f}% | {pbeng7:7.2f}% | {pideal7:8.2f}% | {pideal8:8.2f}%")
