import numpy as np
# p = 22 / 7
# p = 355/113
p = 103638/32989
print(f"p = {p}")

acc = abs(p - np.pi)
relative_acc = acc / np.pi
print(f"absolute accuracy = {acc}")
print(f"relative accuracy = {relative_acc}")

num_accurate_digits = -np.log10(acc / np.pi)
print(f"Number of accurate digits = {num_accurate_digits}")