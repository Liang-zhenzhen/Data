import numpy as np
import scipy as sp
from scipy.linalg import lstsq

def iterative_refinement_solve(V, L_rho, max_iter=15, tolerance=1e-15, verbose=True):
    """
    迭代精化求解复数线性系统 S @ a = b，其中 S = V^H @ V, b = V^H @ L_rho
    
    Parameters:
    -----------
    V : ndarray, shape (m, n)
        输入复数矩阵
    L_rho : ndarray, shape (m,)
        右端复数向量
    max_iter : int
        最大迭代次数
    tolerance : float
        收敛容差
    verbose : bool
        是否输出详细信息
    
    Returns:
    --------
    a : ndarray
        解向量 (复数形式)
    """
    
    # 构建正规方程 S @ a = b
    S = V.conj().T @ V  # V^H @ V
    b = V.conj().T @ L_rho  # V^H @ L_rho
    
    # 计算并输出矩阵的秩信息
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    if verbose:
        print(f"rank(S) = {rank_S}")
        print(f"rank([S | b]) = {rank_augmented}")
        
        # 输出条件数信息
        cond_S = np.linalg.cond(S)
        print(f"Condition number of S: {cond_S:.4e}")
        print("\n=== Iterative Refinement Process ===")
    
    # 检查系统的相容性
    if rank_S != rank_augmented:
        print("Warning: System is inconsistent (rank(S) ≠ rank([S|b]))")
        print("Using least squares solution...")
    
    # 步骤1: 获得初始解 (使用最小二乘法)
    try:
        a = lstsq(S, b, lapack_driver='gelsy')[0]
        if verbose:
            print("Initial solution computed using least squares")
    except Exception as e:
        print(f"Failed to compute initial solution: {e}")
        return np.zeros(S.shape[1], dtype=complex)
    
    # 初始残差
    initial_residual = np.linalg.norm(S @ a - b)
    if verbose:
        print(f"Initial residual norm: {initial_residual:.4e}")
    
    # 迭代精化过程 - 记录残差历史用于更智能的停止判断
    residual_history = []
    
    for i in range(max_iter):
        # 步骤2: 计算当前残差 r = b - S @ a
        r = b - S @ a
        residual_norm = np.linalg.norm(r)
        residual_history.append(residual_norm)
        
        if verbose:
            print(f"Iteration {i+1}: Residual norm = {residual_norm:.4e}")
        
        # 步骤3: 检查收敛
        if residual_norm < tolerance:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break
        
        # 步骤4: 求解修正方程 S @ da = r
        try:
            da = lstsq(S, r, lapack_driver='gelsy')[0]
        except Exception as e:
            if verbose:
                print(f"Iteration {i+1} failed: {e}")
            break
        
        # 步骤5: 更新解 a = a + da
        a = a + da
        
        # 更智能的停滞检测：只有在残差连续增加或完全停滞时才停止
        if i >= 5:  # 至少迭代5次再考虑停止
            recent_residuals = residual_history[-4:]  # 最近4次的残差
            
            # 检查是否连续3次残差都没有显著改善
            if len(recent_residuals) >= 4:
                improvements = []
                for j in range(1, len(recent_residuals)):
                    if recent_residuals[j-1] > 0:
                        improvement = (recent_residuals[j-1] - recent_residuals[j]) / recent_residuals[j-1]
                        improvements.append(improvement)
                
                # 如果连续3次改善都小于1%，且残差仍然远大于机器精度，则可能真的停滞了
                if (len(improvements) >= 3 and 
                    all(imp < 0.01 for imp in improvements[-3:]) and
                    residual_norm > 1e-14):  # 只有远大于机器精度时才考虑停滞
                    
                    if verbose:
                        print(f"Convergence stalled after {i+1} iterations (minimal improvement)")
                    break
    
    # 计算最终残差
    final_residual = np.linalg.norm(S @ a - b)
    
    if verbose:
        print(f"Final residual norm for S @ a - b: {final_residual:.4e}")
        
        # 输出改进情况
        improvement_factor = initial_residual / final_residual if final_residual > 0 else float('inf')
        print(f"Improvement factor: {improvement_factor:.2e}")
        
        # 检查解的性质
        max_imag = np.max(np.abs(np.imag(a)))
        if max_imag > 1e-12:
            print(f"Warning: Solution has significant imaginary parts: {max_imag:.4e}")
    
    # 输出必要的信息
    print(f"rank(S) = {rank_S}")
    print(f"rank([S | b]) = {rank_augmented}")
    print(f"Residual norm for S @ a - b: {final_residual:.4e}")
    
    return a

def iterative_refinement_solve_real_output(V, L_rho, max_iter=15, tolerance=1e-15, verbose=True):
    """
    迭代精化求解，返回实数解 (如果虚部可忽略)
    
    Parameters:
    -----------
    V : ndarray, shape (m, n)
        输入复数矩阵
    L_rho : ndarray, shape (m,)
        右端复数向量
    max_iter : int
        最大迭代次数
    tolerance : float
        收敛容差
    verbose : bool
        是否输出详细信息
    
    Returns:
    --------
    a : ndarray
        解向量 (如果虚部可忽略则返回实数，否则返回复数)
    """
    
    # 调用复数版本
    a_complex = iterative_refinement_solve(V, L_rho, max_iter, tolerance, verbose)
    
    # 检查是否可以转换为实数
    max_imag = np.max(np.abs(np.imag(a_complex)))
    
    if max_imag < 1e-12:  # 虚部可忽略
        if verbose:
            print("Converting to real solution (imaginary parts negligible)")
        return np.real(a_complex)
    else:
        if verbose:
            print(f"Keeping complex solution (max imaginary part: {max_imag:.4e})")
        return a_complex

# 高精度版本 - 完全按照您原始成功的算法
def high_precision_iterative_solve(V, L_rho, max_iter=20, tolerance=1e-16, verbose=True):
    """
    高精度迭代精化求解 - 完全复制您原始成功的算法逻辑
    不使用过早的停滞检测，让算法充分迭代到机器精度
    """
    
    # 构建正规方程 S @ a = b
    S = V.conj().T @ V  # V^H @ V
    b = V.conj().T @ L_rho  # V^H @ L_rho
    
    # 计算并输出矩阵的秩信息
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    if verbose:
        print(f"rank(S) = {rank_S}")
        print(f"rank([S | b]) = {rank_augmented}")
    
    # 步骤1: 获得初始解 (使用最小二乘法)
    a = lstsq(S, b, lapack_driver='gelsy')[0]
    if verbose:
        print("Initial solution computed")
    
    # 迭代精化过程 - 简单版本，不做复杂的停滞检测
    for i in range(max_iter):
        # 计算当前残差
        r = b - S @ a
        residual_norm = np.linalg.norm(r)
        
        if verbose:
            print(f"Iteration {i+1}: Residual norm = {residual_norm:.4e}")
        
        # 简单的收敛检查
        if residual_norm < tolerance:
            if verbose:
                print(f"Converged to tolerance after {i+1} iterations")
            break
            
        # 求解修正方程并更新
        try:
            da = lstsq(S, r, lapack_driver='gelsy')[0]
            a = a + da
        except:
            if verbose:
                print("Iterative refinement failed, using current solution")
            break
    
    # 计算最终残差
    final_residual = np.linalg.norm(S @ a - b)
    
    if verbose:
        print(f"Final residual norm for S @ a - b: {final_residual:.4e}")
    
    # 输出必要的信息 (与您的需求格式一致)
    print(f"rank(S) = {rank_S}")
    print(f"rank([S | b]) = {rank_augmented}")
    print(f"Residual norm for S @ a - b: {final_residual:.4e}")
    
    return a

# 推荐使用的函数 - 直接替换您原来的函数
def improved_iterative_solve(V, L_rho):
    """
    推荐版本：使用高精度迭代精化求解
    完全复制您原始成功算法的逻辑，应该能达到 2e-16 的精度
    """
    solution = high_precision_iterative_solve(V, L_rho, max_iter=20, tolerance=1e-16, verbose=True)
    
    # 检查是否可以转换为实数
    max_imag = np.max(np.abs(np.imag(solution)))
    if max_imag < 1e-12:
        return np.real(solution)
    else:
        return solution