import numpy as np
import scipy as sp

path = "/Users/minima/Nutstore Files/Documents/Python_file/Light Harvesting/G_ITS/"

def linear_solve(V, L_rho):
    S = V.conj().T @ V;
    b = V.conj().T @ L_rho
    a = sp.linalg.lstsq(S, b)[0]  # 用最小二乘法解线性方程组(更精确) #
    # a = sp.linalg.pinv(S) @ b #用伪逆解线性方程组
    # 计算 秩
    print(S.shape)
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    print("rank(S) =", rank_S)
    print("rank([S | b]) =", rank_augmented)
    print(np.linalg.norm(S @ a - b))
    return np.real(a)

S = np.load(path+'S_matrix.npy')
b = np.load(path+'b.npy')
V = np.load(path+'V.npy')
L_rho = np.load(path+'L_rho.npy')

print(S.shape)
print(b.shape)
print(V.shape)
print(L_rho.shape)

a = linear_solve(V,L_rho)

print(np.linalg.norm(S @ a - b))


