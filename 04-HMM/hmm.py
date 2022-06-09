# Author: Wang peng
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    a = np.zeros((N, T))
    for i in range(0, N):
        a[i][0] = pi[i] * B[i][O[0]] #初值
    for t in range(0, T-1):
        for i in range(0, N):
            for j in range(0, N):
                a[i][t+1] = a[i][t+1] + a[j][t] * A[j][i] * B[i][O[t+1]]
    print(a)
    for i in range(0, N):
        prob = prob + a[i][T-1]

    return prob

def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm. 
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    beita = np.zeros((N, T))
    for i in range(0, N):
        beita[i][T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(0, N):
            for j in range(0, N):
                beita[i][t] = beita[i][t] + A[i][j] * B[j][O[t+1]] * beita[j][t+1]
    for i in range(0, N):
        prob = prob + pi[i] * B[i][O[0]] * beita[i][0]
    print(beita)
    return prob

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob = 0.0
    best_path = np.zeros(T)
    deita = np.zeros((N, T))
    fai = np.zeros((N, T))
    #初始化
    for i in range(0, N):
        deita[i][0] = pi[i] * B[i][O[0]]
        fai[i][0] = 0
    for t in range(1,T):
        for i in range(0, N):
            for j in range(0, N):
                max = deita[0][t-1] * A[0][i] *B[i][O[t]]
                if(deita[j][t-1] * A[j][i] *B[i][O[t]] > max):
                    deita[i][t] = deita[j][t-1] * A[j][i] *B[i][O[t]]
                    fai[i][t] = j
    for i in range(0, N):
        max = deita[0][T-1]
        if(deita[i][T-1] > max):
            best_prob = deita[i][T-1]
            best_path[T-1] = i
    for t in range(T-2, -1, -1):
        i = best_path[t+1]
        best_path[t] = fai[int(i)][t+1]
	best_path = best_path + 1
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)
