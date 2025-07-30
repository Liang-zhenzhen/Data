import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

path = "D:/7_30data/"
def tikhonov_regularized_solve(V, L_rho, lam=1e-3):
    VT_V = V.T @ V
    identity = np.eye(V.shape[1])
    a = np.linalg.solve(VT_V + lam * identity, V.T @ L_rho)
    residual = np.linalg.norm(V @ a - L_rho)
    print(f"λ = {lam:.1e}, Residual norm = {residual:.4e}")
    return np.real(a)


# 读取数据
V = np.load(path+'V.npy').astype(np.complex128)
L_rho = np.load(path+'L_rho.npy').astype(np.complex128)

lambdas = np.logspace(-10, 0, 11)  # 从 1e-10 到 1e0 的正则化强度
residuals = []

for lam in lambdas:
    a = tikhonov_regularized_solve(V, L_rho, lam)
    res = np.linalg.norm(V @ a - L_rho)
    residuals.append(res)


plt.semilogx(lambdas, residuals, marker='o')
plt.xlabel('λ (Regularization strength)')
plt.ylabel('Residual norm ||Va - L_rho||')
plt.title('Tikhonov Regularization Effect')
plt.grid(True)
plt.show()
