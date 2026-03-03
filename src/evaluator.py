import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5'
import torch
import torch.backends.cudnn as cudnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import json
import pandas as pd


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            input_ids = self.tensors[index]
            return input_ids

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        try:
            test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                # Fallback for newer datasets versions that disallow script-based loading.
                test_data = load_dataset(
                    'ptb_text_only',
                    'penn_treebank',
                    split='test',
                    revision='refs/convert/parquet'
                )
            else:
                raise
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    # elif 'c4' in name:
    #     test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
    #     test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    elif 'c4' in name:
        test_data = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation').select(range(2000))
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def print_memory_usage():
    total_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    
    for i in range(total_gpus):
        allocated = torch.cuda.memory_allocated(device=i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device=i) / 1024 / 1024
        total_allocated += allocated
        total_reserved += reserved
        # print(f"GPU {i} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    
    # print(f"Total - Allocated: {total_allocated:.2f} MiB, Reserved: {total_reserved:.2f} MiB")
    
    return total_allocated, total_reserved


@torch.no_grad()
def run_lm_eval(model, tokenizer, batch_size=16, task_names=["openbookqa", "arc_easy", "winogrande",
             "arc_challenge", "piqa", "mathqa", "hellaswag"], output_csv="results.csv"):
    # Import the correct task loading function
    from lm_eval import tasks, evaluator
    from lm_eval.models.huggingface import HFLM
    import json
    from datetime import datetime
    import os
    import torch

    # 如果已有 CSV，加载已完成的任务
    finished_tasks = set()
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            finished_tasks = set(prev_df["task"].tolist())
            print(f"已完成任务: {finished_tasks}")
        except Exception as e:
            print(f"读取 {output_csv} 失败，将重新生成: {e}")
            
                
    # lm-eval>=0.4 expects an LM object (or model string), not tokenizer arg in simple_evaluate.
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        trust_remote_code=True,
    )

    all_results = []
    for task in task_names:
        if task in finished_tasks:
            print(f"跳过任务 {task} (已存在结果)")
            continue        
        
        print(f"\n===== 开始评估任务: {task} =====")
        
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=batch_size,
            write_out=False,
            log_samples=False,
            verbosity="INFO",
            num_fewshot=0,
            task_manager=tasks.TaskManager(),
        )

        # Remove samples from results to reduce file size
        if 'samples' in results:
            del results['samples']

        # # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
        # class CustomEncoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, torch.Tensor):
        #             return obj.detach().cpu().tolist()
        #         elif isinstance(obj, torch.device):
        #             return str(obj)
        #         elif isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         return super().default(obj)

        # # Create a timestamped filename
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # # Create output directory if it doesn't exist
        # if output_dir:
        #     os.makedirs(output_dir, exist_ok=True)
        # filename = os.path.join(output_dir, f"results_{timestamp}.json")

        # # Save the results dictionary to a JSON file using the custom encoder
        # with open(filename, 'w') as f:
        #     json.dump(results, f, indent=4, cls=CustomEncoder)

        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none", None)
        acc_norm = task_results.get("acc_norm,none", None)

        row = {
            "task": task,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "—",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "—"
        }

        print(f"结果: {row}")
        all_results.append(row)

        # 追加写入 CSV
        df = pd.DataFrame([row])
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode="w")
        else:
            df.to_csv(output_csv, index=False, mode="a", header=False)
        
        print(f"{task} finish! save to {output_csv}")

    return pd.DataFrame(all_results)

@torch.no_grad()
def ppl_eval_sharing(model, tokenizer, dev, experiment_name, datasets=['wikitext2'], model_seq_len=2048, batch_size=4, params_only=False):
    """
    评估模型的困惑度 (Perplexity)。
    
    Args:
        model: 要评估的模型。
        tokenizer: 对应的分词器。
        dev: 评估设备 (例如 "cuda")。
        experiment_name (str): 实验名称，用于报告。
        datasets (list): 要评估的数据集列表 (例如 ['wikitext2', 'ptb'])。
        model_seq_len (int): 模型的序列长度。
        batch_size (int): 评估时使用的批处理大小。
        params_only (bool): 如果为True，则只计算参数量，跳过PPL评估。

    Returns:
        str: 包含评估结果的格式化字符串。
    """
    print(f"\n--- 开始 PPL 评估 (实验: {experiment_name}) ---")
    
    model.eval()  # 确保模型处于评估模式
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # 自动获取模型所在的主要设备
    main_device = next(model.parameters()).device
    print(f"模型主要运行在设备: {main_device}")

    if not params_only:
        for dataset in datasets:
            # 1. 加载测试数据
            # get_test_data 返回一个 torch.utils.data.DataLoader 对象
            data_loader = get_test_data(
                dataset, 
                tokenizer, 
                seq_len=model_seq_len, 
                batch_size=batch_size
            )
            
            nlls = []
            total_tokens = 0

            # 2. 正确的迭代循环
            progress_bar = tqdm(data_loader, desc=f"正在评估 {dataset}", leave=False)
            for batch in progress_bar:
                batch = batch.to(main_device)
                
                # 基于Token的精确计算
                shift_labels = batch[:, 1:].contiguous()
                tokens_in_batch = shift_labels.numel()
                total_tokens += tokens_in_batch

                # 监控显存使用
                allocated, reserved = print_memory_usage()
                total_allocated_list.append(allocated)
                total_reserved_list.append(reserved)

                # 3. 模型前向传播和损失计算
                outputs = model(batch)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                neg_log_likelihood = loss.float() * tokens_in_batch
                nlls.append(neg_log_likelihood)

                # 动态更新 PPL 到进度条
                current_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
                progress_bar.set_description(f"正在评估 {dataset} | 当前 PPL: {current_ppl:.4f}")

            # 计算最终的 PPL
            final_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
            ppls[dataset] = final_ppl.item()
            print(f"数据集 '{dataset}' 评估完成, 最终 PPL: {ppls[dataset]:.4f}")

    # 4. 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 5. 格式化并返回最终结果
    result_str = f"\n--- 评估报告: {experiment_name} ---\n"
    if not params_only:
        result_str += f"困惑度 (PPL): {ppls}\n"
        if total_allocated_list:
            avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
            avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
            result_str += f"评估期间平均已分配显存: {avg_allocated:.2f} MiB\n"
            result_str += f"评估期间平均保留显存: {avg_reserved:.2f} MiB\n"
    
    result_str += f"模型总参数量: {total_params / 1e9:.3f} B\n"
    result_str += f"可训练参数量: {trainable_params / 1e9:.3f} B\n"
    result_str += f"非训练参数量: {non_trainable_params / 1e9:.3f} B\n"
    result_str += "--- 报告结束 ---\n"

    return result_str
    

if __name__ == "__main__":
    # main()
    
    base_model_path="./models/Mixtral-8x7B-v0.1"
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                device_map="auto", 
                                                trust_remote_code=True, 
                                                torch_dtype=torch.bfloat16)


    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    
    
    model_name = Path(base_model_path).name
    experiment_name = f"{model_name}_origin"    

    save_path = "./output"
    eval_save_dir = Path(save_path) / "evaluation_results" / model_name
    eval_save_dir.mkdir(parents=True, exist_ok=True)    
    
    result_str = ppl_eval_sharing(model, tokenizer, "cuda", experiment_name, datasets=['wikitext2'], model_seq_len=2048, batch_size=4, params_only=False)
    
    ppl_save_path = eval_save_dir / f"ppl_{experiment_name}.txt"
    with open(ppl_save_path, 'w') as f:
        f.write(result_str)
    
    task_names=["openbookqa", "arc_easy", "winogrande","arc_challenge", "piqa", "mathqa", "hellaswag"]
    run_lm_eval(model, tokenizer, batch_size=16, task_names=task_names, output_csv=f"{eval_save_dir}/acc_origin.csv")
