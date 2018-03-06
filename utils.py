# March 5th @DP

import numpy as np

def seq2pairwise(seq):
    batch_size, a,b = seq.shape
    matrix = np.zeros((batch_size,a,a,b,3))
    #print(matrix.shape)

    # TODO: replace loop with a np.meshgrid for efficiency
    for ba in range(0,batch_size):
        for i in range(0,a):
            for j in range(0,a):
                for k in range(0,b):
                    matrix[ba][i][j][k] = np.array([
                        seq[ba][i][k],
                        seq[ba][int((i+j)/2)][k],
                        seq[ba][j][k]
                        ])
    return matrix.reshape(batch_size,a,a,b*3)

if __name__ == "__main__":
    res = seq2pairwise(np.random.rand(20,10,30))
    print(res.shape)
