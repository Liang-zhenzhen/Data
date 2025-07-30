import numpy as np
import scipy as sp

path = "D:/7_30data/"

# linear_solve
def linear_solve(V, L_rho):
    S = V.conj().T @ V
    b = V.conj().T @ L_rho
    a = sp.linalg.lstsq(S, b)[0]  # 用最小二乘法解线性方程组(更精确) #
    # a = sp.linalg.pinv(S) @ b #用伪逆解线性方程组
    # 计算 秩
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    print("rank(S) =", rank_S)
    print("rank([S | b]) =", rank_augmented)
    print("Residual norm for S @ a - b:",np.linalg.norm(S @ a - b))
    return np.real(a)

# complex_linear_solve
def complex_linear_solve(V, L_rho):
    S = V.conj().T @ V
    b = V.conj().T @ L_rho
    a = sp.linalg.lstsq(S, b,lapack_driver='gelsy')[0]  # 用最小二乘法解线性方程组(更精确) #
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    print("rank(S) =", rank_S)
    print("rank([S | b]) =", rank_augmented)
    print("Residual norm for S @ a - b:",np.linalg.norm(S @ a - b))
    return np.real(a)

# improved_linear_solve
def improved_linear_solve(V, L_rho):
    # 直接对 V 和 L_rho 进行最小二乘求解
    a = sp.linalg.lstsq(V, L_rho)[0] # 直接传入V和L_rho，不再先计算S和b
    rank_V = np.linalg.matrix_rank(V)
    augmented = np.hstack((V, L_rho.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    print("rank(V) =", rank_V)
    print("rank([V | L_rho]) =", rank_augmented)
    # 计算残差（与原始问题一致）
    print("Residual norm for V @ a - L_rho:", np.linalg.norm(V @ a - L_rho))
    return np.real(a)

# improved_linear_solve
def complex_improved_linear_solve(V, L_rho):
    a, residuals, rank, s = sp.linalg.lstsq(V, L_rho, lapack_driver='gelsy')
    rank_V = np.linalg.matrix_rank(V)
    augmented = np.hstack((V, L_rho.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    print("rank(V) =", rank_V)
    print("rank([V | L_rho]) =", rank_augmented)
    print("Residual norm for V @ a - L_rho:", np.linalg.norm(V @ a - L_rho))
    print("Rank of V:", rank)
    return a

# 加载数据
S = np.load(path+'S_matrix.npy').astype(np.complex128)
b = np.load(path+'b.npy').astype(np.complex128)
V = np.load(path+'V.npy').astype(np.complex128)
L_rho = np.load(path+'L_rho.npy').astype(np.complex128)

# print(S.shape)
# print(b.shape)
# print(V.shape)
# print(L_rho.shape)

a = linear_solve(V,L_rho)
print(np.linalg.norm(S @ a - b))

x=complex_linear_solve(V, L_rho)

y = improved_linear_solve(V, L_rho)
print(np.linalg.norm(V @ x - L_rho))

z=complex_improved_linear_solve(V, L_rho)