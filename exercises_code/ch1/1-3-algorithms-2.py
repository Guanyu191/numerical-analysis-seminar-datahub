import numpy as np

def samplevar(x):
    n = len(x)
    mean_x = np.mean(x)
    variance = np.sum((x - mean_x) ** 2) / (n - 1)
    return variance

if __name__ == "__main__":
    # Test case 1: A vector of ones with length 100
    test_ones = np.ones(100)
    our_result_ones = samplevar(test_ones)
    numpy_result_ones = np.var(test_ones, ddof=1)
    print("The variance of the ones - vector (Ours):", our_result_ones)
    print("The variance of the ones - vector (NumPy):", numpy_result_ones)

    # Test case 2: A random vector with length 200
    test_rand = np.random.rand(200)
    our_result_rand = samplevar(test_rand)
    numpy_result_rand = np.var(test_rand, ddof=1)
    print("The variance of the random - vector (Ours):", our_result_rand)
    print("The variance of the random - vector (NumPy):", numpy_result_rand)
