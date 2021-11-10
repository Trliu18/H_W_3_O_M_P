import random
import numpy as np
from numpy.linalg import norm


def normalized_error(x, x_):
    return norm(x - x_, ord=2) / norm(x, ord=2)


def generate_sparse_x(N, s_max, know_s=False):
    """
    :param M: number of maximal non-zero entries in x
    :param s: maximum support size
    :output: x: a vector of length M with uniformly distributed entries
    """
    x = np.zeros(N)
    entry_range = list(range(-10, 0)) + list(range(1, 11))
    if not know_s:
        cardinality = random.randint(1, s_max)  # generate a random int number from 1 to 10
    else:
        cardinality = s_max
    randomlist = random.choices(entry_range, k=cardinality)
    x[0: cardinality] = randomlist
    x = np.array(x).reshape(N, 1)
    return x


def generate_data(M, N, s, noise_sigma=0, know_s=False):
    """
    Generate data for experiment
    : input M: number of measurements
    : input N: number of atoms
    : input s: support size
    : noise_sigma: if True, add additive noise to the measurements
    
    : output A: measurement matrix M*N
    : output x: sparse vector  N*1    
    : output y: measurement vector M*1
    : output noise: additive noise to the measurements
    """
    # generate A, whose columns are normalized
    A = np.random.randn(M, N)
    A = A / norm(A, ord=2, axis=0)
    x = generate_sparse_x(N, s, know_s)
    y = A.dot(x)
    noise = None
    if noise_sigma != 0:
        noise = np.random.normal(loc=0, scale=noise_sigma, size=M).reshape(M, 1)
        y += noise
    return A, x, y, noise


def OMP(A, y, iteration, acc_tolerance=0.001):
    """
    Orthogonal Matching pursuit algorithm
    :input A: measurement matrix M*N
    :input y: measurement vector M*1
    :output x: sparse vector  N*1
    """
    r = y    # initialize the residual as y
    M, N = A.shape
    x = np.zeros(N)
    Lambdas = []
    i = 0
    # Control stop interation with norm thresh or sparsity

    while norm(r, ord=2) > acc_tolerance and i < iteration:
        scores = A.T.dot(r)  # Compute the score of each atoms
        Lambda = np.argmax(abs(scores))  # Select the atom with the max score
        Lambdas.append(Lambda)
        An = A[:, Lambdas]  # All selected atoms form a basis

        # least square solution: x = (A^T A)^(-1) A^T y
        x[Lambdas] = np.linalg.pinv(An).dot(y)
        x = x.reshape(N, -1)
        r = y - A.dot(x)  # Calculate the residual
        i += 1
    return x


def repeated_expermients(s, M, N, repeated_time=2000, noise_sigma=0, know_s=False):
    """
    repeat the experiments for many times and see the error rate
    :param s: support size
    :param M: number of measurements
    :param N: number of atoms
    :param repeated_time: number of experiments
    :param noise: additive noise to the measurements
    :return error_list: list of error rate (if under noise, return error rate, else return normalized error)
    """

    error_list = []

    iteration = s if know_s else N

    for _ in range(repeated_time):
        A, x, y, noise = generate_data(M, N, s, noise_sigma=noise_sigma, know_s=know_s)
        if (noise is not None) and (not know_s):
            acc_tolerance = norm(noise, ord=2)
        else:
            acc_tolerance = 0.001
        x_pred = OMP(A, y, iteration=iteration, acc_tolerance=acc_tolerance)
        err = normalized_error(x, x_pred)
        error_list.append(err)
    return error_list


if __name__ == "__main__":

    A = np.array([[-0.707, 0.8, 0],
                  [0.707, 0.6, -1]])
    x = np.array([[-1.2],
                  [1],
                  [0]])
    y = A.dot(x)

    x_ = OMP(A, y, iteration=3)
    print(norm(x - x_, ord=2)/norm(x, ord=2))
    # expected output: 0.0
    
    # M = 44
    # N = 128
    # s = 24
    
    # n = 0.01
    # res_list = repeated_expermients(s, M, N, repeated_time=200, noise_sigma=n, know_s=False, acc_tolerance=0.001)
    # if n != 0:
    #     res_list = [1 if err < 1e-3 else 0 for err in res_list]
    #     res = sum(res_list) / len(res_list)
    #     print("Success rate: ", res)
    # else:
    #     res = sum(res_list) / len(res_list)
    #     print("error: ", res)


