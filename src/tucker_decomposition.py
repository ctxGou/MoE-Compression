# /home/Anonymous/TD-MoE/src/tucker_decomposition.py

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorly as tl
tl.set_backend('pytorch')

import math
import time
import torch

from tensorly.decomposition import tucker
from typing import List
import numpy as np
from tqdm.auto import tqdm

def calculate_tucker_ranks(n0, n1, n2, ratio, fix_r0=True) -> List[int]:
    """
    计算 Tucker 分解的 rank [r0, r1, r2]，固定 r0 = n_expert，
    并根据目标压缩比例 ratio 估算 r1 和 r2。
    
    参数:
        W: 输入张量，shape = [n_expert, n_out, n_in]
        ratio: 目标压缩比例，比如 0.2 表示保留 20%参数
    
    返回:
        ranks: [r0, r1, r2]
    """
    print(f"DEBUG(calculate_tucker_ranks):{n0=}, {n1=}, {n2=}, {ratio=}, {fix_r0=}")
    r0 = n0 if fix_r0 else None
    original_params = n0 * n1 * n2
    target_params = int(original_params * (1-ratio))

    best_r1, best_r2 = None, None
    min_error = float('inf')

    # 精确搜索，保证能找到接近 target_params 的解
    for r1 in range(1, n1 + 1):
        numerator = target_params - (n0 * r0 + n1 * r1)
        denominator = r0 * r1 + n2
        if denominator <= 0:
            continue
        r2 = numerator // denominator
        if 1 <= r2 <= n2:
            compressed = r0 * r1 * r2 + n0 * r0 + n1 * r1 + n2 * r2
            error = abs(compressed - target_params)
            if error < min_error:
                min_error = error
                best_r1, best_r2 = r1, r2
                
    # 核心与因子形状
    r1, r2 = best_r1, best_r2
    core_shape = (r0, r1, r2)
    U0_shape = (n0, r0)
    U1_shape = (n1, r1)
    U2_shape = (n2, r2)

    original_params = original_params /1e6 
    core_params = r0 * r1 * r2 /1e6 
    U0_params = n0 * r0 /1e6 
    U1_params = n1 * r1 /1e6 
    U2_params = n2 * r2 /1e6 
    total_params = (core_params + U0_params + U1_params + U2_params)

    print(f"Original tensor shape  = {(n0,n1,n2)}, params = {original_params:2f}M -> Tucker ranks={(r0, r1, r2)}  params = {total_params:2f}M \n")
    print(f"Core tensor shape  = {core_shape}, params = {core_params:2f}M")
    print(f"Factor U0 shape    = {U0_shape}, params = {U0_params:2f}M")
    print(f"Factor U1 shape    = {U1_shape}, params = {U1_params:2f}M")
    print(f"Factor U2 shape    = {U2_shape}, params = {U2_params:2f}M")
    print(f"Total compressed params = {total_params:2f}M")
    print(f"Original params         = {original_params:2f}M")
    print(f"Actual compression ratio= {1-total_params/original_params:.4f}")
    
    return [r0, best_r1, best_r2]


def calculate_tucker_ranks_equal(n0, n1, n2, ratio, fix_r0=True) -> List[int]:
    """
    计算 Tucker ranks [r0, r1, r2]，其中强制 r1 == r2。
    目标是在参数预算内选择最大的对称瓶颈 r。

    约束:
        P_tucker = r0*r^2 + n0*r0 + n1*r + n2*r <= P_orig*(1-ratio)
    """
    print(f"DEBUG(calculate_tucker_ranks_equal):{n0=}, {n1=}, {n2=}, {ratio=}, {fix_r0=}")
    if not fix_r0:
        raise AssertionError("calculate_tucker_ranks_equal requires fix_r0=True.")

    r0 = n0
    original_params = n0 * n1 * n2
    target_params = int(original_params * (1 - ratio))

    max_rank = min(n1, n2)
    best_r = 1
    for r in range(1, max_rank + 1):
        compressed = r0 * r * r + n0 * r0 + n1 * r + n2 * r
        if compressed <= target_params:
            best_r = r
        else:
            break

    r1 = best_r
    r2 = best_r

    core_shape = (r0, r1, r2)
    U0_shape = (n0, r0)
    U1_shape = (n1, r1)
    U2_shape = (n2, r2)

    original_params_m = original_params / 1e6
    core_params = r0 * r1 * r2 / 1e6
    U0_params = n0 * r0 / 1e6
    U1_params = n1 * r1 / 1e6
    U2_params = n2 * r2 / 1e6
    total_params = core_params + U0_params + U1_params + U2_params

    print(f"Original tensor shape  = {(n0,n1,n2)}, params = {original_params_m:2f}M -> Tucker ranks={(r0, r1, r2)}  params = {total_params:2f}M \n")
    print(f"Core tensor shape  = {core_shape}, params = {core_params:2f}M")
    print(f"Factor U0 shape    = {U0_shape}, params = {U0_params:2f}M")
    print(f"Factor U1 shape    = {U1_shape}, params = {U1_params:2f}M")
    print(f"Factor U2 shape    = {U2_shape}, params = {U2_params:2f}M")
    print(f"Total compressed params = {total_params:2f}M")
    print(f"Original params         = {original_params_m:2f}M")
    print(f"Actual compression ratio= {1-total_params/original_params_m:.4f}")

    return [r0, r1, r2]





def calculate_tucker_ranks_balanced(n0: int, n1: int, n2: int, ratio: float, fix_r0: bool=True):
    """
    计算 Tucker 分解的 rank [r0, r1, r2]，固定 r0 = n_expert，
    根据目标压缩比例 ratio 搜索 (r1, r2)，并偏向均衡解。
    
    参数:
        n0, n1, n2: 原始张量的三个维度 (experts, out, in)
        ratio: 压缩比例，比如 0.6 表示压缩 60% 参数
        fix_r0: 是否固定 r0 = n0 (默认 True)

    返回:
        ranks: (r0, r1, r2)
    """

    # r0 是否固定
    r0 = n0 if fix_r0 else n0 // 2  # 如果不固定，可以考虑减半
    
    # 原始和目标参数量
    original_params = n0 * n1 * n2
    target_params = int(original_params * (1-ratio) )

    best_r1, best_r2 = None, None
    min_score = float("inf")

    # 搜索范围（为了加速，这里只取 5% ~ 100%）
    for r1 in range(max(1, n1 // 50), n1 + 1, max(1, n1 // 200)):  
        for r2 in range(max(1, n2 // 50), n2 + 1, max(1, n2 // 200)):
            compressed = r0 * r1 * r2 + n0 * r0 + n1 * r1 + n2 * r2
            # 参数量误差
            param_error = abs(compressed - target_params) / target_params
            # 均衡性惩罚: 希望 r1/n1 ≈ r2/n2
            balance_error = abs(r1 / n1 - r2 / n2)
            # 总分：参数量优先，其次考虑均衡性
            score = param_error * 10 + balance_error  
            
            if score < min_score:
                min_score = score
                best_r1, best_r2 = r1, r2

    # 输出结果
    core_shape = (r0, best_r1, best_r2)
    U0_shape = (n0, r0)
    U1_shape = (n1, best_r1)
    U2_shape = (n2, best_r2)

    core_params = r0 * best_r1 * best_r2 / 1e6 
    U0_params = n0 * r0 / 1e6 
    U1_params = n1 * best_r1 / 1e6 
    U2_params = n2 * best_r2 / 1e6 
    total_params = core_params + U0_params + U1_params + U2_params

    print(f"Original tensor shape  = {(n0,n1,n2)}, params = {original_params/1e6:.2f}M")
    print(f"Tucker ranks          = {core_shape}, params = {total_params:.2f}M")
    print(f"Core tensor shape     = {core_shape}, params = {core_params:.2f}M")
    print(f"Factor U0 shape       = {U0_shape}, params = {U0_params:.2f}M")
    print(f"Factor U1 shape       = {U1_shape}, params = {U1_params:.2f}M")
    print(f"Factor U2 shape       = {U2_shape}, params = {U2_params:.2f}M")
    print(f"Actual compression ratio = {1-total_params/(original_params/1e6):.4f}")
    
    return (r0, best_r1, best_r2)


def calculate_tucker_ranks_ratio_optimized(n0, n1, n2, ratio, fix_r0=True):
    """
    优化版 Tucker rank 计算：
    - 固定 r0 = n0
    - 保证 total_params ≈ target_params（严格匹配 ratio）
    - 在可行解中选择 r1:r2 尽量接近 n1:n2 比例
    """
    print("calculate_tucker_ranks_ratio_optimized...")
    r0 = n0 if fix_r0 else None
    original_params = n0 * n1 * n2
    target_params = int(original_params * ratio)

    best_r1, best_r2 = None, None
    min_error = float('inf')
    best_ratio_diff = float('inf')

    for r1 in range(1, n1 + 1):
        numerator = target_params - (n0 * r0 + n1 * r1)
        denominator = r0 * r1 + n2
        if denominator <= 0:
            continue
        r2 = numerator // denominator
        if 1 <= r2 <= n2:
            compressed = r0 * r1 * r2 + n0 * r0 + n1 * r1 + n2 * r2
            error = abs(compressed - target_params)
            # r1:r2 与 n1:n2 比例差异
            ratio_diff = abs((r1/r2) - (n1/n2))
            # 首先保证压缩率误差最小，其次选择比例最接近 n1:n2
            if error < min_error or (error == min_error and ratio_diff < best_ratio_diff):
                min_error = error
                best_ratio_diff = ratio_diff
                best_r1, best_r2 = r1, r2

    r1, r2 = best_r1, best_r2

    # 输出信息
    core_shape = (r0, r1, r2)
    U0_shape = (n0, r0)
    U1_shape = (n1, r1)
    U2_shape = (n2, r2)

    original_params_m = original_params / 1e6
    core_params = r0 * r1 * r2 / 1e6
    U0_params = n0 * r0 / 1e6
    U1_params = n1 * r1 / 1e6
    U2_params = n2 * r2 / 1e6
    total_params_m = core_params + U0_params + U1_params + U2_params

    print(f"Original tensor shape  = {(n0,n1,n2)}, params = {original_params_m:2f}M -> Tucker ranks={(r0, r1, r2)}  params = {total_params_m:2f}M \n")
    print(f"Core tensor shape  = {core_shape}, params = {core_params:2f}M")
    print(f"Factor U0 shape    = {U0_shape}, params = {U0_params:2f}M")
    print(f"Factor U1 shape    = {U1_shape}, params = {U1_params:2f}M")
    print(f"Factor U2 shape    = {U2_shape}, params = {U2_params:2f}M")
    print(f"Total compressed params = {total_params_m:2f}M")
    print(f"Original params         = {original_params_m:2f}M")
    print(f"Target compression ratio = {ratio:.4f}, Actual ratio = {total_params_m/original_params_m:.4f}")

    return [r0, r1, r2]


# def nearest_positive_definite(A):
#     # 对称化矩阵
#     B = (A + A.T) / 2
#     # 计算特征值和特征向量
#     eigenvalues, eigenvectors = torch.linalg.eigh(B)
#     # 将特征值中的负值裁剪为微小的正数
#     min_eig = torch.min(eigenvalues)
#     if min_eig < 0:
#         eigenvalues = eigenvalues + (-min_eig + 1e-8)
#     # 重构正定矩阵
#     A_pd = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
#     # 确保矩阵对称
#     A_pd = (A_pd + A_pd.T) / 2
#     return A_pd

# def make_positive_definite(matrix, initial_adjustment=1e-6, max_attempts=12, adjustment_factor=6):
#     attempts = 0
    
#     while attempts < max_attempts:
#         try:
#             #  Cholesky decomposition
#             chol_matrix = torch.linalg.cholesky(matrix)
#             # 检查是否存在NaN
#             if torch.isnan(chol_matrix).any() or torch.isinf(chol_matrix).any():
#                 print("Warning: NaN or Inf detected in Cholesky decomposition result.")
#                 # 使用Higham算法调整矩阵
#                 matrix_pd = nearest_positive_definite(matrix)
#                 # 重新进行Cholesky分解
#                 chol_matrix = torch.linalg.cholesky(matrix_pd)
#             if torch.isnan(chol_matrix).any() or torch.isinf(chol_matrix).any():
#                 print("nan")
#             return chol_matrix
#         except torch._C._LinAlgError:
#             # Fail, try again
#             attempts += 1
#             eigenvalues = torch.linalg.eigvalsh(matrix)
#             adjustment = max(initial_adjustment, -eigenvalues[0] * 1e-3)
#             matrix += adjustment * torch.eye(matrix.shape[0]).to(matrix.device)
#             initial_adjustment *= adjustment_factor

    # raise ValueError("Failed")


# def svd_positive_definite(A, eps=1e-6):
#     # 对称化矩阵
#     A = (A + A.T) / 2
#     # SVD
#     U, S, Vh = torch.linalg.svd(A)
#     # 截断小特征值
#     S_clamped = torch.clamp(S, min=eps)
#     # 重构矩阵
#     A_pd = (U * S_clamped) @ Vh
#     # 确保对称
#     A_pd = (A_pd + A_pd.T) / 2
#     return A_pd

def _ensure_positive_definite(matrix: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    通过特征值分解和截断来确保矩阵是正定的。

    Args:
        matrix (torch.Tensor): 输入的对称矩阵。
        eps (float): 特征值的最小允许值。

    Returns:
        torch.Tensor: 一个保证为正定的矩阵。
    """
    # 1. 确保矩阵对称
    matrix = (matrix + matrix.T) / 2
    # 2. 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # 3. 将所有小于 eps 的特征值替换为 eps
    clamped_eigenvalues = torch.clamp(eigenvalues, min=eps)
    # 4. 使用处理过的特征值重构矩阵
    matrix_pd = eigenvectors @ torch.diag(clamped_eigenvalues) @ eigenvectors.T
    # 5. 再次确保对称性以消除可能的浮点误差
    return (matrix_pd + matrix_pd.T) / 2


def whiten_tensor(W: torch.Tensor, Sigma2: torch.Tensor, Sigma3: torch.Tensor = None, eps: float = 1e-6, dev: torch.device = None) -> torch.Tensor:
    """
    对堆叠专家权重 W 进行白化。
    W: [n_expert, d_out, d_in]
    Sigma2: [d_in, d_in] 输入协方差
    Sigma3: [d_out, d_out]  输出协方差
    whiten_output: 是否使用 Sigma3 对输入方向白化
    eps: 调整最小特征值，防止奇异
    dev: 指定设备
    """
    W_whitened = W.clone()
    S2_final, S3_final = None, None
    device = torch.device(dev)

    # --- 输入方向白化 ---
    if Sigma2 is not None:
        print(f"Applying input whitening with Sigma2 shape: {Sigma2.shape}")
        try:
            Sigma2_pd = _ensure_positive_definite(Sigma2.to(device).float(), eps)
            S2 = torch.linalg.cholesky(Sigma2_pd)
            whitened_experts = [(W_whitened[i].to(device).float() @ S2) for i in range(W_whitened.shape[0])]
            W_whitened = torch.stack([w.cpu() for w in whitened_experts], dim=0)
            S2_final = S2.cpu()
        except Exception as e:
            print(f"An error occurred during input whitening: {e}. Skipping input whitening.")

            
    #  --- 输出方向白化 ---
    if Sigma3 is not None:
        print(f"Applying output whitening with Sigma3 shape: {Sigma3.shape}")
        try:
            Sigma3_pd = _ensure_positive_definite(Sigma3.to(device).float(), eps)
            S3 = torch.linalg.cholesky(Sigma3_pd)
            S3_inv = torch.linalg.inv(S3)
            final_whitened_experts = [(S3_inv @ W_whitened[i].to(device).float()) for i in range(W_whitened.shape[0])]
            W_whitened = torch.stack([w.cpu() for w in final_whitened_experts], dim=0)
            S3_final = S3.cpu()
        except Exception as e:
            print(f"An error occurred during output whitening: {e}. Skipping output whitening.")

            
    Cholesky_matrices = {'S2': S2_final, 'S3': S3_final}
    return W_whitened, Cholesky_matrices



# def tucker_decomposition(tensor: torch.Tensor, ranks: List[int]):
#     """
#     对一个给定的PyTorch张量执行Tucker分解。
#     """
#     # 确保TensorLy使用PyTorch后端
#     tl.set_backend('pytorch')

#     # 记录原始设备，以便在CPU回退后将结果移回
#     original_device = tensor.device
    
#     # 确保输入张量在内存中是连续的
#     tensor_contiguous = tensor.contiguous()
    
#     # 优先尝试在GPU上执行
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tensor_to_use = tensor_contiguous.to(device)    

#     # 优先尝试在GPU上执行
#     if device.type == 'cuda':
#         try:
#             print("Attempting Tucker decomposition on GPU using Randomized SVD...")
#             core, factors = tucker(tensor_to_use, rank=ranks, random_state=3,n_iter_max=10,verbose=True)
#             print("GPU decomposition successful.")
#             return core, factors
            
#         except (torch.cuda.OutOfMemoryError, torch._C._LinAlgError) as e:
#             print(f"\n!!! WARNING: GPU decomposition failed with error: {e}")
#             print("!!! Automatically falling back to CPU. This will be much slower... !!!\n")
#             # 如果GPU失败，则将张量移至CPU，准备进行CPU计算
#             tensor_to_use = tensor_contiguous.cpu()
#             core, factors = tucker(tensor_to_use, rank=ranks, svd='truncated_svd', init='svd',n_iter_max=10,verbose=True)
#     else:
#         # GPU不可用，直接在CPU上
#         print("Executing Tucker decomposition on CPU...")        
#         core, factors = tucker(tensor_to_use, rank=ranks, svd='truncated_svd', init='svd',n_iter_max=10,verbose=True)    
#         print("CPU decomposition successful.")
    
#     print("Moved results back to the original GPU device.")
        
#     return core, factors


def hosvd_decomposition(tensor: torch.Tensor, ranks: list):
    """
    使用高阶SVD（HOSVD）算法对张量进行Tucker分解 (纯PyTorch版本)。
    
    此函数在PyTorch环境中运行，避免了与NumPy之间的数据转换。

    Args:
        tensor (torch.Tensor): 需要被分解的输入PyTorch张量。
        ranks (list or tuple): 每个模式的目标秩。

    Returns:
        tuple: 一个包含四个元素的元组:
            - core (torch.Tensor): 分解后的核心张量 G。
            - factors (list): 包含所有因子矩阵的列表 [A, B, C, ...]。
            - reconstructed_tensor (torch.Tensor): 重构后的张量。
            - error (float): 重构张量与原始张量之间的相对误差。
    """
    # print(f"{tensor[0]=}")
    print(f"{ranks=}")
    
    original_device = tensor.device
    print(f"{original_device=}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"tensor is moving to {device} ...")
    tensor = tensor.to(device)
    
    num_modes = tensor.ndim
    if len(ranks) != num_modes:
        raise ValueError(f"Ranks列表的长度({len(ranks)})必须与张量的阶数({num_modes})相同。")

    factors = []

    pbar = tqdm(range(num_modes), desc="HOSVD Factor Calculation", unit="factor")
    for mode in pbar:
    # for mode in range(num_modes):
        unfolded_tensor = tl.unfold(tensor, mode)
        
        print(f"Processing mode {mode}, unfolded shape: {unfolded_tensor.shape}")
        try:    
            print(f"-> Processing SVD on GPU.")
            U, S, V = torch.linalg.svd(unfolded_tensor.float(), full_matrices=False)
            rank = ranks[mode]
            factor_matrix = U[:, :rank]                        
        except:            
            print(f"-> Shape exceeds threshold. Falling back to CPU for SVD.")
            # 1. 将问题矩阵移至CPU
            unfolded_cpu = unfolded_tensor.cpu()
            # 2. 在CPU上执行SVD
            U_cpu, S_cpu, V_cpu = torch.linalg.svd(unfolded_cpu.float(), full_matrices=False)
            # 3. 将结果因子矩阵移回原始设备
            rank = ranks[mode]
            factor_matrix = U_cpu[:, :rank].to(device)            

        factors.append(factor_matrix)    

    # TensorLy在PyTorch后端下会自动使用PyTorch函数
    core = tl.tenalg.multi_mode_dot(tensor.float(), [f.T for f in factors], modes=list(range(num_modes)))
    core = core.to(original_device)
    factors = [f.to(original_device) for f in factors]
       
    return core, factors



def nmode_product_compress(X: torch.Tensor, U: torch.Tensor, mode: int) -> torch.Tensor:
    """ Y = X ×_mode U^T """
    Y = torch.tensordot(X, U, dims=([mode], [0]))
    return Y.movedim(-1, mode)

def nmode_product_expand(X: torch.Tensor, U: torch.Tensor, mode: int) -> torch.Tensor:
    """ Y = X ×_mode U """
    Y = torch.tensordot(X, U, dims=([mode], [1]))
    return Y.movedim(-1, mode)


# ---------------------------
# 不同基分解方法（核心修改点）
# ---------------------------
def svd_basis(unfolded: torch.Tensor, rank: int, device: str) -> torch.Tensor:
    """
    使用 SVD 计算基矩阵。
    """
    try:
        if unfolded.device.type != 'cpu':
            # print("-> Processing SVD on GPU.")
            U, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
            return U[:, :rank]
        else:
            # print("-> Processing SVD on CPU (as requested).")
            U, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
            return U[:, :rank]
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"-> SVD on GPU failed due to OOM: {e}. Falling back to CPU.")
        unfolded_cpu = unfolded.cpu()
        U_cpu, _, _ = torch.linalg.svd(unfolded_cpu, full_matrices=False)
        # 将结果因子矩阵移回目标设备
        return U_cpu[:, :rank].to(device)
    except Exception as e:
        print(f"-> SVD on GPU failed with a general error: {e}. Falling back to CPU.")
        unfolded_cpu = unfolded.cpu()
        U_cpu, _, _ = torch.linalg.svd(unfolded_cpu, full_matrices=False)
        return U_cpu[:, :rank].to(device)


def qr_basis(unfolded: torch.Tensor, rank: int, device: str) -> torch.Tensor:
    """使用 QR 分解计算基矩阵。"""
    try:
        # <-- 开始修改 -->
        if unfolded.device.type != 'cpu':
            Q, _ = torch.linalg.qr(unfolded)
            return Q[:, :rank]
        else:
            Q, _ = torch.linalg.qr(unfolded)
            return Q[:, :rank]
    except torch.cuda.OutOfMemoryError as e:
        print(f"-> QR on GPU failed due to OOM: {e}. Falling back to CPU.")
        unfolded_cpu = unfolded.cpu()
        Q_cpu, _ = torch.linalg.qr(unfolded_cpu)
        return Q_cpu[:, :rank].to(device) # 别忘了移回原设备
    except Exception as e:
        print(f"-> QR on GPU failed with a general error: {e}. Falling back to CPU.")
        unfolded_cpu = unfolded.cpu()
        Q_cpu, _ = torch.linalg.qr(unfolded_cpu)
        return Q_cpu[:, :rank].to(device)
        # <-- 结束修改 -->


def randomized_svd_basis(unfolded: torch.Tensor, rank: int, device: str, oversample: int = 10, n_iter: int = 2) -> torch.Tensor:
    """
    使用手动实现的随机SVD算法计算基矩阵，兼容旧版PyTorch。
    该算法通过随机投影找到一个近似的列空间，然后通过QR分解获得正交基。
    """
    try:
        # 目标维度
        target_dim = rank + oversample
        m, n = unfolded.shape

        # 随机投影矩阵
        # 当n维度很大时，这个矩阵会很大，但在GPU上生成通常很快
        P = torch.randn(n, target_dim, device=unfolded.device, dtype=unfolded.dtype)

        # "素描"矩阵 Y = A @ P
        # 这是计算瓶颈，但比完整的SVD快得多
        Y = torch.matmul(unfolded, P)

        # 对素描矩阵进行多次迭代，以提高精度 (Power Iteration)
        for _ in range(n_iter):
            Q, _ = torch.linalg.qr(Y)
            Y = torch.matmul(unfolded.T, Q)
            Q, _ = torch.linalg.qr(Y)
            Y = torch.matmul(unfolded, Q)
        
        # 从最终的素描矩阵中提取正交基
        Q, _ = torch.linalg.qr(Y)
        
        return Q[:, :rank]

    except torch.cuda.OutOfMemoryError as e:
        print(f"-> Manual RandSVD on GPU failed due to OOM: {e}. Falling back to CPU.")
        unfolded_cpu = unfolded.cpu()
        target_dim = rank + oversample
        m, n = unfolded_cpu.shape
        P = torch.randn(n, target_dim, device='cpu', dtype=unfolded_cpu.dtype)
        Y = torch.matmul(unfolded_cpu, P)
        for _ in range(n_iter):
            Q, _ = torch.linalg.qr(Y)
            Y = torch.matmul(unfolded_cpu.T, Q)
            Q, _ = torch.linalg.qr(Y)
            Y = torch.matmul(unfolded_cpu, Q)
        Q, _ = torch.linalg.qr(Y)
        return Q[:, :rank].to(device) # 别忘了移回原设备
    except Exception as e:
        print(f"-> Manual RandSVD on GPU failed with a general error: {e}. Re-raising.")
        raise e

# ---------------------------
# 新的、统一的 Tucker 分解函数 (替换 hosvd_decomposition)
# ---------------------------
def tucker_decomposition(tensor: torch.Tensor,
                         ranks: list,
                         method: str = "svd",
                         verbose: bool = False) -> tuple[torch.Tensor, list]:
    """
    对张量进行Tucker分解，并支持多种基底生成方法。

    Args:
        tensor (torch.Tensor): 需要被分解的输入PyTorch张量。
        ranks (list): 每个模式的目标秩。
        method (str): "svd", "qr", 或 "rand"。
        verbose (bool): 是否打印中间过程信息。

    Returns:
        tuple: (core, factors)
            - core (torch.Tensor): 分解后的核心张量，设备与原始张量一致。
            - factors (list): 因子矩阵列表，设备与原始张量一致。
    """
    original_device = tensor.device
    
    # 智能选择计算设备，优先GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_compute = tensor.to(device)
    
    if verbose:
        print(f"Starting Tucker decomposition with method='{method}' on device='{device}'")
        print(f"Original tensor shape: {tensor.shape}, ranks: {ranks}")

    num_modes = tensor.ndim
    if len(ranks) != num_modes:
        raise ValueError(f"Ranks列表的长度({len(ranks)})必须与张量的阶数({num_modes})相同。")

    factors = []
    
    pbar = tqdm(range(num_modes), desc=f"Tucker ({method.upper()})", unit="factor", disable=verbose)
    for mode in pbar:
        unfolded_tensor = tl.unfold(tensor_compute, mode)
        
        if verbose:
            print(f"Processing mode {mode}, unfolded shape: {unfolded_tensor.shape}")

        if method == "svd":
            factor_matrix = svd_basis(unfolded_tensor.float(), ranks[mode], device)
        elif method == "qr":
            factor_matrix = qr_basis(unfolded_tensor.float(), ranks[mode], device)
        elif method == "rand":
            # 随机SVD对float32更稳定
            factor_matrix = randomized_svd_basis(unfolded_tensor.float(), ranks[mode], device)
        else:
            raise ValueError(f"Unknown method: {method}")

        factors.append(factor_matrix)

    # 计算核心张量 G = T x_1 U_1^T x_2 U_2^T ...
    # 使用 tensorly 的 multi_mode_dot 依然是最高效的方式
    core = tl.tenalg.multi_mode_dot(tensor_compute.float(), [f.T for f in factors], modes=list(range(num_modes)))
    
    # 将结果移回原始设备，确保接口一致性
    core = core.to(original_device)
    factors = [f.to(original_device) for f in factors]
    
    if verbose:
        print(f"Decomposition finished. Core shape: {core.shape}")
        
    return core, factors




def tucker_decomposition_eval(W,core, factors):
    reconstructed_W = tl.tucker_to_tensor((core, factors))
    W = W.cpu()
    reconstructed_W = reconstructed_W.cpu()
    print(f"{reconstructed_W[0]=}")
    reconstruction_error = torch.linalg.norm(W - reconstructed_W) / torch.linalg.norm(W)
    print(f"\nReconstruction error: {reconstruction_error.item():.6f}")

    original_params = W.numel()
    compressed_params = core.numel() + sum(f.numel() for f in factors)
    print(f"Original parameters: {original_params:,}")
    print(f"Compressed parameters: {compressed_params:,}")
    print(f"Compression ratio: {original_params / compressed_params :.2f}x")








if __name__ == '__main__':

    dim1 = 14336 # 14336
    dim2 = 4096
    n_expert = 2
    ratio = 0.2
    
    weight_list = []
    shape = (dim1, dim2)  # 示例形状
    for i in range(n_expert):  # 创建两个张量
        weight_list.append(torch.randn(shape, dtype=torch.float32))    
        
    stacked_weights = torch.stack(weight_list, dim=0)
    print(f"Stacked tensor shape: {stacked_weights.shape}")        
    print(f"{stacked_weights[0]=}")
    

    # r0, r1, r2 = calculate_tucker_ranks(n_expert,dim1,dim2, ratio)
    # print(f"Tucker ranks for ~{ratio*100:.0f}% compression: r0={r0}, r1={r1}, r2={r2}")        
    # target_ranks = [r0, r1, r2]
    
    target_ranks = [n_expert,8271,3661]
    
    start_time = time.time()
    core, factors = tucker_decomposition(stacked_weights, target_ranks)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTucker decomposition elapsed time: {elapsed_time:.2f} seconds")
        
    print("\n--- Decomposition Results ---")
    print(f"Core tensor shape: {core.shape}")
    print("Factor matrices shapes:")    
    for i, factor in enumerate(factors):
        print(f"  Factor {i}: {factor.shape}")    

    tucker_decomposition_eval(stacked_weights, core, factors)
