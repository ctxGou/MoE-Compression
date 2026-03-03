import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import torch
print("!!! Switching CUDA linear algebra backend to MAGMA !!!")
try:
    torch.backends.cuda.preferred_linalg_library("magma")
    print("Successfully set MAGMA as the preferred backend.")
except RuntimeError as e:
    print(f"Could not set MAGMA as backend: {e}")
    print("Please ensure PyTorch was compiled with MAGMA support.")

import tensorly as tl
from tensorly.decomposition import tucker
from torch.linalg import svd

# 设置TensorLy后端为PyTorch
tl.set_backend('pytorch')
# torch.set_num_threads(32)
# torch.backends.cuda.preferred_linalg_library = 'cutlass'

def make_positive_definite(matrix, epsilon=1e-6, max_attempts=5, device=None):
    """
    将矩阵调整为正定矩阵，以便进行 Cholesky 分解。
    针对大矩阵 + GPU + float32，兼顾速度和数值稳定性。
    
    参数:
        matrix: 待处理矩阵，torch.Tensor，shape [N, N]
        epsilon: 对角线扰动初始值
        max_attempts: 最大尝试次数
        device: 可选，指定设备

    返回:
        torch.Tensor: 正定矩阵，可用于 Cholesky
    """
    if device is not None:
        matrix = matrix.to(device)
    # 保证 float32
    matrix = matrix.to(torch.float32)
    # 对称化
    matrix = (matrix + matrix.T) / 2

    for attempt in range(max_attempts):
        try:
            # 尝试 Cholesky
            L = torch.linalg.cholesky(matrix)
            if torch.isnan(L).any() or torch.isinf(L).any():
                raise RuntimeError("NaN or Inf in Cholesky")
            return matrix
        except RuntimeError:
            # 对角线加小扰动
            diag_adjust = epsilon * (2 ** attempt)
            matrix = matrix + diag_adjust * torch.eye(matrix.shape[0], device=matrix.device)
            # 对称化
            matrix = (matrix + matrix.T) / 2

    # 最后尝试 Higham 方法
    # 特征值裁剪
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    min_eig = torch.min(eigenvals)
    if min_eig < 0:
        eigenvals = eigenvals - min_eig + 1e-8                      # 所有特征值平移，最小值变成 1e-8
    matrix_pd = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T     # 重建矩阵
    matrix_pd = (matrix_pd + matrix_pd.T) / 2   
    return matrix_pd


def calculate_tucker_ranks(tensor_shape, ratio):
    """
    tensor_shape: [k, d_out, d_in]
    返回值顺序与 tucker 的 mode 对应: [rank_k, rank_out, rank_in]
    """
    k, d_out, d_in = tensor_shape  # 保持 W 本身顺序

    rank_k = k  # 专家维度不压缩
    dim_ratio = ratio ** 0.5
    rank_out = max(1, int(d_out * dim_ratio))
    rank_in  = max(1, int(d_in * dim_ratio))

    print(f"Original shape: [{k}, {d_out}, {d_in}], Target Tucker ranks: [{rank_k}, {rank_out}, {rank_in}]")
    return [rank_k, rank_out, rank_in]

def condition_number(matrix):
    singular_values = torch.svd(matrix).S
    return singular_values[0] / singular_values[-1]



def tucker_decomposition(W, rank, device='cuda'):
    """
    使用 PyTorch 自定义的 SVD 实现 Tucker 分解，支持 GPU 加速
    :param W: 输入张量
    :param rank: 所需的秩 (rank)
    :param device: 计算设备，默认为 'cuda' (GPU)，否则为 'cpu'
    :return: core tensor 和因子矩阵
    """
    # 确保输入张量在正确的设备上
    W = W.to(device)
    
    # 获取张量的维度
    modes = W.shape
    U = []
    
    for i in range(len(modes)):
        # 对每个模式进行 SVD 分解
        unfolded = W.reshape(-1, modes[i])  # 用 reshape 代替 view
        U_i, _, _ = svd(unfolded)
        U.append(U_i[:, :rank[i]].to(device))  # 确保因子矩阵 U 也在 GPU 上
    
    # 构造核心张量 (通过 multi-mode multiplication)
    core = W
    for i, u in enumerate(U):
        core = torch.matmul(core, u.T)

    return core, U


# def wt_moe_compress(W, Sigma2, Sigma3, ranks, epsilon=1e-3, whiten_output=True):
#     """
#     对单个权重张量W执行完整的WT-MoE压缩流程。
#     W的维度应为 (k, d_out, d_in)
#     """    
    
#     device = W.device
#     k, d_out, d_in = W.shape
    
#     # 保证协方差矩阵正定
#     Sigma2 = Sigma2.to(device, dtype=torch.float32)
#     Sigma3 = Sigma3.to(device, dtype=torch.float32)    
    
#     Sigma2 = make_positive_definite(Sigma2, epsilon=epsilon)
#     Sigma3 = make_positive_definite(Sigma3, epsilon=epsilon)  
    
#     # 推导白化矩阵
#     I2 = torch.eye(d_in, device=device, dtype=torch.float32)
#     I3 = torch.eye(d_out, device=device, dtype=torch.float32)
#     S2 = torch.linalg.cholesky(Sigma2 + epsilon * I2)
#     S3 = torch.linalg.cholesky(Sigma3 + epsilon * I3)

#     # print(f"{S2=}")
#     print(f"{S2.shape=}") # hidden_size × hidden_size
#     print(f"{S3.shape=}") # intermediate_size × intermediate_size
    
#     # 白化权重
#     S2_inv_T = torch.linalg.inv(S2).T
#     S3_inv_T = torch.linalg.inv(S3).T
    
#     S2_inv_T = S2_inv_T.to(W.dtype)
#     S3_inv_T = S3_inv_T.to(W.dtype)    
    
#     W_whitened = tl.tenalg.mode_dot(W, S2_inv_T, mode=2)  # mode=2 对应 d_in
    
#     if whiten_output:
#         W_whitened = tl.tenalg.mode_dot(W_whitened, S3_inv_T, mode=1)  # mode=1 对应 d_out

 
 
#     if torch.isnan(W_whitened).any() or torch.isinf(W_whitened).any():
#         print("W_whitened contains NaN/Inf!")

#     # HOSVD与截断
#     # core, factors = tucker(W_whitened.to(torch.float32), rank=ranks, init='random', tol=1e-3, n_iter_max=20,svd='truncated_svd')
    
#     W_whitened_cpu = W_whitened.to('cpu', dtype=torch.float32)
#     core, factors = tucker(W_whitened_cpu, rank=ranks, init='svd', tol=1e-3, n_iter_max=20,svd='truncated_svd')    
    
#     # 返回压缩组件和白化矩阵用于后续重构
#     compressed_components = {'core': core.cpu(),'factors': [f.cpu() for f in factors]}
#     whitening_matrices = {'S2': S2.cpu(),'S3': S3.cpu()}
    
#     return compressed_components, whitening_matrices

# W, Sigma2, Sigma3, ranks, args.whiten_output, epsilon=1e-1
def wt_moe_compress(W, Sigma2, Sigma3, ranks, epsilon=1e-1, whiten_output=True):
    """
    对单个权重张量W执行完整的WT-MoE压缩流程。
    W的维度应为 (k, d_out, d_in)
    GPU执行
    """    
    # device
    # device = W.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 默认为GPU, 如果可用
    k, d_out, d_in = W.shape
    
    # 保证协方差矩阵正定
    Sigma2 = Sigma2.to(device, dtype=torch.float32)
    Sigma3 = Sigma3.to(device, dtype=torch.float32)    
    
    Sigma2 = make_positive_definite(Sigma2, epsilon=epsilon)
    Sigma3 = make_positive_definite(Sigma3, epsilon=epsilon)  
    
    print(f"{Sigma2=}")
    
    # 推导白化矩阵
    I2 = torch.eye(d_in, device=device, dtype=torch.float32)
    I3 = torch.eye(d_out, device=device, dtype=torch.float32)
    S2 = torch.linalg.cholesky(Sigma2 + epsilon * I2)
    S3 = torch.linalg.cholesky(Sigma3 + epsilon * I3)

    # 打印矩阵形状
    print(f"{Sigma2 + epsilon * I2=}")
    print(f"{S2=}")
    
    
    print(f"{S2.shape=}")  # hidden_size × hidden_size
    print(f"{S3.shape=}")  # intermediate_size × intermediate_size
    
    # 白化权重
    # 使用伪逆替代普通逆，以提高数值稳定性
    S2_inv_T = torch.linalg.pinv(S2).T
    S3_inv_T = torch.linalg.pinv(S3).T
    
    W = W.to(dtype=torch.float32).to(device)
    S2_inv_T = S2_inv_T.to(dtype=torch.float32).to(device)
    S3_inv_T = S3_inv_T.to(dtype=torch.float32).to(device)    
    
    print(f"{S2_inv_T=}")
    
    # 白化权重
    W_whitened = tl.tenalg.mode_dot(W, S2_inv_T, mode=2)  # mode=2 对应 d_in    
    cond_W = condition_number(W_whitened)
    print(f"Condition number of W_whitened: {torch.sort(cond_W)}")
    
    if whiten_output:
        W_whitened = tl.tenalg.mode_dot(W_whitened, S3_inv_T, mode=1)  # mode=1 对应 d_out
    
    # 检查是否存在 NaN 或 Inf
    if torch.isinf(S2_inv_T).any() or torch.isnan(S2_inv_T).any():
        print("S2_inv_T contains NaN/Inf after inversion!")
    if torch.isinf(S3_inv_T).any() or torch.isnan(S3_inv_T).any():
        print("S3_inv_T contains NaN/Inf after inversion!")    
    if torch.isnan(W_whitened).any() or torch.isinf(W_whitened).any():
        print("W_whitened contains NaN/Inf!")
    
    print(f"{W_whitened=}")
    
    # 防止数值异常，强制转换 NaN/Inf 为数值
    finfo = torch.finfo(W_whitened.dtype)
    W_whitened = torch.nan_to_num(
        W_whitened, 
        nan=0.0, 
        posinf=finfo.max,  # 替换为该数据类型的最大值
        neginf=finfo.min   # 替换为该数据类型的最小值
    )    
    
    
        
    # 执行 Tucker 分解
    try:
        W_whitened_contiguous = W_whitened.contiguous()
        # print("Switching to float64 for Tucker decomposition...")
        # W_whitened_double = W_whitened_contiguous.to(torch.float64)       
         
        core, factors = tucker(W_whitened_contiguous.to(device), rank=ranks, init='svd', tol=1e-3, n_iter_max=20, svd='randomized_svd')
        
        # 分解完成后，将结果转换回 float32 以节省内存
        core = core.to(torch.float32)
        factors = [f.to(torch.float32) for f in factors]
                
        print(f"{core},{factors=}")
                
    except Exception as e:
        raise ValueError(f"Error during Tucker decomposition: {e}")
        # return None, None
    
    # 返回压缩组件和白化矩阵用于后续重构
    compressed_components = {'core': core.to(device), 'factors': [f.to(device) for f in factors]}
    whitening_matrices = {'S2': S2.to(device), 'S3': S3.to(device)}
    
    return compressed_components, whitening_matrices




def reconstruct_from_components(components_path, whitening_path, device, original_dtype=torch.bfloat16, whiten_output=True):
    """从存储的文件中加载并重建权重"""
    components = torch.load(components_path, map_location=device)
    whitening = torch.load(whitening_path, map_location=device)

    core = components['core']
    factors = components['factors']
    S2 = whitening['S2']
    S3 = whitening['S3']

    # 重构白化后的权重
    W_prime_whitened = tl.tucker_to_tensor((core, factors))
    
    # "去白化"
    S2_T = S2.T.to(W_prime_whitened.dtype)
    S3_T = S3.T.to(W_prime_whitened.dtype)
    
    W_prime = W_prime_whitened
    if whiten_output:
        W_prime = tl.tenalg.mode_dot(W_prime, S3_T, mode=1) # mode=1 对应 d_out
    W_prime = tl.tenalg.mode_dot(W_prime, S2_T, mode=2) # mode=2 对应 d_in
    
    return W_prime.to(original_dtype)


if __name__ == "__main__":
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # model_path="./models/Mixtral-8x7B-Instruct-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     # device_map="cpu",
    #     offload_folder="./offload",
    #     low_cpu_mem_usage=True
    # )
    # model.eval()

    

    # import torch
    # file_path = "./output/covariances/Mixtral-8x7B-Instruct-v0.1_layers_14_16_wsamples_256.pt"
    # all_covariances = torch.load(file_path, map_location='cpu')

    # role='gate'
    # ranks=[8,6411,1831]
    # moe_layer_name = "model.layers.14.block_sparse_moe"
    # Sigma2 = all_covariances[moe_layer_name][role]['Sigma2']
    # Sigma3 = all_covariances[moe_layer_name][role]['Sigma3']
    
    # expert_weights = []
    # for i in range(model.config.num_local_experts):
    #     expert_weight_name = f"{moe_layer_name}.experts.{i}.w1.weight"
    #     expert_weights.append(dict(model.named_parameters())[expert_weight_name].data)
    # W = torch.stack(expert_weights, dim=0)
    
    # compressed_components, whitening_matrices = wt_moe_compress(W, Sigma2, Sigma3, ranks, epsilon=1e-3, whiten_output=False)
    # print(f'{compressed_components=}')
    # print(f'{whitening_matrices=}')
    # print(f"finish!")
    
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pathlib import Path
    
    # --- 1. 配置: 请修改这里以匹配您要验证的文件 ---
    MODEL_PATH = "./models/Mixtral-8x7B-Instruct-v0.1"
    SAVE_PATH = "./output"
    DECOMPOSITION_MODE = "global_whitening"
    LAYER_IDX = 14
    ROLE = "down" 
    RATIO = 0.2
    WHITEN_OUTPUT_FLAG_BOOL = False
    # ----------------------------------------------------

    whiten_output_flag_str = "whitened_outputs" if WHITEN_OUTPUT_FLAG_BOOL else "unwhitened_outputs"
    
    print("="*60)
    print("--- Running Verification for a Specific Decomposition Result ---")
    print(f"Layer: {LAYER_IDX}, Role: {ROLE}, Ratio: {RATIO}, WhitenOutput: {WHITEN_OUTPUT_FLAG_BOOL}")
    print("="*60)

    # 确定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Verification will run on CPU.")

    # --- 2. 加载分解后的文件 ---
    print("\n--- Step 1: Loading decomposed components and whitening matrices ---")
    base_dir = Path(SAVE_PATH) / "decomposition_results" / Path(MODEL_PATH).name / DECOMPOSITION_MODE / f"layer_{LAYER_IDX}" / ROLE
    components_path = base_dir / f"compressed_components_ratio_{RATIO}_{whiten_output_flag_str}.pt"
    whitening_path = base_dir / f"whitening_matrices_ratio_{RATIO}_{whiten_output_flag_str}.pt"

    if not components_path.exists() or not whitening_path.exists():
        raise FileNotFoundError(f"Decomposition files not found at: {base_dir}")
    
    # --- 3. 加载原始模型并提取对应的原始权重 W ---
    print("\n--- Step 2: Loading original model to extract ground truth weight W ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    
    moe_layer_pattern = 'model.layers.{}.block_sparse_moe' # Mixtral的模式
    moe_layer_name = moe_layer_pattern.format(LAYER_IDX)
    param_name_map = {'gate': 'w1', 'up': 'w3', 'down': 'w2'}
    param_name = param_name_map[ROLE]

    print(f"Extracting and stacking weights for '{param_name}'...")
    expert_weights = []
    for i in range(model.config.num_local_experts):
        expert_weight_name = f"{moe_layer_name}.experts.{i}.{param_name}.weight"
        # 直接将权重提取到目标设备上
        expert_weights.append(dict(model.named_parameters())[expert_weight_name].data.to(device))
    
    W_original = torch.stack(expert_weights, dim=0)
    original_dtype = W_original.dtype
    print(f"{W_original[0]=}")
    print(f"Original W shape: {W_original.shape}, Dtype: {original_dtype}, Device: {W_original.device}")

    # 释放大模型，节省内存
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model deleted and cache cleared.")

    # --- 4. 调用函数，从文件重建权重 ---
    print(f"\n--- Step 3: Reconstructing W_prime from saved files ---")
    W_reconstructed = reconstruct_from_components(
        components_path=components_path,
        whitening_path=whitening_path,
        device=device,
        original_dtype=original_dtype,
        whiten_output=WHITEN_OUTPUT_FLAG_BOOL
    )
    print(f"{W_reconstructed[0]=}")
    print(f"Reconstructed W_prime shape: {W_reconstructed.shape}, Dtype: {W_reconstructed.dtype}")
    
    # --- 5. 验证结果 ---
    print("\n--- Step 4: Verifying results ---")
    
    assert W_reconstructed.shape == W_original.shape
    assert W_reconstructed.dtype == W_original.dtype
    print("Shape and Dtype match.")

    # 在float32下计算相对重构误差
    original_norm = torch.linalg.norm(W_original.to(torch.float32)).item()
    diff_norm = torch.linalg.norm(W_original.to(torch.float32) - W_reconstructed.to(torch.float32)).item()
    relative_error = diff_norm / original_norm if original_norm > 0 else float('inf')

    print("\n" + "="*50)
    print("                 VERIFICATION REPORT")
    print("="*50)
    print(f"Layer: {LAYER_IDX}, Role: '{ROLE}', Ratio: {RATIO}")
    print(f"Relative Reconstruction Error: {relative_error:.6f}")
    
    if relative_error < 0.1:
        print("SUCCESS: The reconstruction error is low. The cycle is working correctly.")
    elif relative_error < 0.5:
        print("OK: The reconstruction error is moderate. Check if this is acceptable.")
    else:
        print("WARNING: The reconstruction error is high. There might be an issue.")
    print("="*50)