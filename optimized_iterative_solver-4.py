import numpy as np
from scipy.linalg import lstsq

def iterative_refinement_solve(V, L_rho, max_iter=25, tolerance=1e-15, verbose=True):
    """
    高精度迭代精化求解复数线性系统 S @ a = b，其中 S = V^H @ V, b = V^H @ L_rho
    专门追求机器精度，允许更多迭代次数，更严格的收敛标准

    Parameters:
    -----------
    V : ndarray, shape (m, n)
        输入复数矩阵
    L_rho : ndarray, shape (m,)
        右端复数向量
    max_iter : int, default=25
        最大迭代次数（增加到25次以确保充分迭代）
    tolerance : float, default=1e-15
        收敛容差（机器精度级别）
    verbose : bool, default=True
        是否输出详细信息

    Returns:
    --------
    a : ndarray
        解向量（如果虚部可忽略则返回实数，否则返回复数）
    """
    
    # 确保输入数据类型为complex128以获得最高精度
    V = V.astype(np.complex128)
    L_rho = L_rho.astype(np.complex128)
    
    # 构建正规方程 S @ a = b
    S = V.conj().T @ V  # V^H @ V
    b = V.conj().T @ L_rho  # V^H @ L_rho
    
    # 计算矩阵的秩信息
    rank_S = np.linalg.matrix_rank(S)
    augmented = np.hstack((S, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    if verbose:
        print(f"rank(S) = {rank_S}")
        print(f"rank([S | b]) = {rank_augmented}")
        
        # 输出条件数信息用于诊断
        cond_S = np.linalg.cond(S)
        print(f"Condition number of S: {cond_S:.4e}")
        print("\n=== High Precision Iterative Refinement ===")
    
    # 检查系统的相容性
    if rank_S != rank_augmented:
        if verbose:
            print("Warning: System is inconsistent (rank(S) ≠ rank([S|b]))")
            print("Using least squares solution...")
    
    # 步骤1: 获得初始解（使用最高精度的最小二乘法）
    try:
        a = lstsq(S, b, lapack_driver='gelsy')[0]
        if verbose:
            print("Initial solution computed using high-precision least squares")
    except Exception as e:
        if verbose:
            print(f"Failed to compute initial solution: {e}")
        return np.zeros(S.shape[1], dtype=np.complex128)
    
    # 计算初始残差
    initial_residual = np.linalg.norm(S @ a - b)
    if verbose:
        print(f"Initial residual norm: {initial_residual:.16e}")
    
    # 迭代精化过程 - 追求机器精度，不做过早的停滞检测
    best_residual = initial_residual
    best_solution = a.copy()
    consecutive_no_improvement = 0
    
    for i in range(max_iter):
        # 步骤2: 计算当前残差 r = b - S @ a
        r = b - S @ a
        residual_norm = np.linalg.norm(r)
        
        if verbose:
            print(f"Iteration {i+1:2d}: Residual norm = {residual_norm:.16e}")
        
        # 记录最佳解
        if residual_norm < best_residual:
            best_residual = residual_norm
            best_solution = a.copy()
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
        
        # 步骤3: 严格的收敛检查
        if residual_norm <= tolerance:
            if verbose:
                print(f"Converged to tolerance after {i+1} iterations")
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
        
        # 只在连续多次无改善且远未达到机器精度时才考虑停止
        # 这里设置更严格的条件：连续8次无改善且残差仍大于10倍机器精度
        if (consecutive_no_improvement >= 8 and 
            best_residual > 10 * tolerance and 
            i >= 10):  # 至少迭代10次
            if verbose:
                print(f"Stopping: No improvement for {consecutive_no_improvement} consecutive iterations")
                print("Using best solution found so far")
            a = best_solution.copy()
            break
    
    # 计算最终残差（使用最佳解）
    final_residual = np.linalg.norm(S @ a - b)
    
    if verbose:
        print(f"\nFinal residual norm for S @ a - b: {final_residual:.16e}")
        
        # 输出改进情况
        improvement_factor = initial_residual / final_residual if final_residual > 0 else float('inf')
        print(f"Improvement factor: {improvement_factor:.2e}")
        
        # 检查解的性质
        max_imag = np.max(np.abs(np.imag(a)))
        if max_imag > 1e-12:
            print(f"Solution has complex components (max imaginary part: {max_imag:.4e})")
        else:
            print("Solution is effectively real (converting to real)")
    
    # 输出必要的最终信息（不截断，显示完整精度）
    print(f"rank(S) = {rank_S}")
    print(f"rank([S | b]) = {rank_augmented}")
    print(f"Residual norm for S @ a - b: {final_residual:.16e}")
    
    # 智能返回类型：如果虚部可忽略则返回实数
    max_imag = np.max(np.abs(np.imag(a)))
    if max_imag < 1e-12:
        return np.real(a)
    else:
        return a