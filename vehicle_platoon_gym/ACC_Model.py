import numpy as np


def model(model_type, **kwargs):
    # Matrices for the Journal paper. Switch between ACC and CACC. ACC with t = 1.4
    if model_type == 'ACC_CACC':
        A = np.zeros((2, 15, 15), dtype=float)
        for i in [0, 1, 5, 10]:
            A[0][0 + i][[0 + i, 1 + i]] = [1.0, 0.1]
        for i in [0, 5]:
            A[0][6 + i][[2 + i, 6 + i, 7 + i]] = [-0.1, 1.0, 0.1]
            A[0][8 + i][[7 + i, 8 + i, 9 + i]] = [0.1, 1.0, -0.1]
            A[0][9 + i][2] = 1.0
        A[0][2][[0, 1, 2]] = [-0.17857143, -0.60714286, 0.5]
        A[1] = A[0]
        A[0][7][1] = -0.25
        A[0][12][[1, 6]] = [-0.25, -0.25]
        for i in [0, 5]:
            A[0][7 + i][[5 + i, 6 + i, 7 + i]] = [-0.17857143, -0.60714286, 0.5]
            A[1][7 + i][[2 + i, 5 + i, 6 + i, 7 + i]] = [0.5, -0.245, -0.7, 0.5]

        B = np.zeros((2, 15, 2), dtype=float)
        B[0:2, 2, 0] = 0.17857143  # (0.25/h) h = 1.4
        B[0:2, 2, 1] = 0.35714286  # (0.5/h)

    if model_type == 'ACC_time_gap':
        # Switch between different time gap on the ACC controller. t=1.0 and t=3.0
        A = np.zeros((2, 15, 15), dtype=float)
        for i in [0, 1, 5, 10]:
            A[0][0 + i][[0 + i, 1 + i]] = [1.0, 0.1]
        for i in [0, 5]:
            A[0][6 + i][[2 + i, 6 + i, 7 + i]] = [-0.1, 1.0, 0.1]
            A[0][8 + i][[7 + i, 8 + i, 9 + i]] = [0.1, 1.0, -0.1]
            A[0][9 + i][2] = 1.0
        A[0][2][[0, 1, 2]] = [-0.0833333, -0.4166667, 0.5]
        A[0][7][1] = -0.25
        A[0][12][[1, 6]] = [-0.25, -0.25]
        A[1] = A[0]
        for i in [0, 5]:
            A[0][7 + i][[5 + i, 6 + i, 7 + i]] = [-0.0833333, -0.4166667, 0.5]
            A[1][7 + i][[5 + i, 6 + i, 7 + i]] = [-0.25, -0.75, 0.5]

        B = np.zeros((2, 15, 2), dtype=float)
        B[0:2, 2, 0] = 0.0833333  # (0.25/h) h = 3
        B[0:2, 2, 1] = 0.1666667  # (0.5/h)

    if model_type == 'ACC':
        '''
        Ts = 0.1
        lamb = 0.5
        tal = 0.2
        h = 1, 1.4 or 3
        '''
        Ts = kwargs['Ts']
        lamb = kwargs['lamb']
        tal = kwargs['tal']
        h = kwargs['h']  # h = [h1, h2, h3]
        k = 3 * ['None']
        for i in [0, 1, 2]:
            k[i] = (Ts/tal) * np.array([-lamb/h[i], -1/h[i], -lamb])

        A = np.zeros((9, 9), dtype=float)
        for i in [0, 1, 3, 4, 6, 7]:
            A[i][[i, i + 1]] = [1.0, Ts]
        for i in [0, 1, 2]:
            A[3*i+2][3*i+2] = 1 - Ts/tal
            A[3*i+2][[3*i, 3*i+1]] = [k[i][0], k[i][1] + k[i][2]]
        for i in [4, 7]:
            A[i][i-2] = -Ts
        A[5][1] = k[1][2]
        A[8][[1, 4]] = [k[2][2], k[2][2]]

        B = np.zeros((9, 2), dtype=float)
        B[2] = (Ts/tal) * np.array([lamb/h[0], 1/h[0]])

    return A, B


if __name__ == '__main__':
    A1, B1 = model('ACC', Ts=0.1, lamb=0.5, tal=0.2, h=[3, 3, 3])
    A2, B2 = model('ACC', Ts=0.1, lamb=0.5, tal=0.2, h=[3, 1, 1])
    A3, B3 = model('ACC_time_gap', Ts=0.1, lamb=0.5, tal=0.2, h=[3, 3, 3])
    print(A)

