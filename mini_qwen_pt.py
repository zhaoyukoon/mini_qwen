import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载分词器与模型
output_path = "results/pt"
model_path = "models/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_path)
# 调整模型配置
config.num_attention_heads = 16
config.num_key_value_heads = 4
config.hidden_size = 1024
config.num_hidden_layers = 48
# print(config)
logger.info(f'load model from {config}')
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
logger.info('load tokenizer: {model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path)

logger.info('load tokenizer: {model_path} done')
# # 计算参数量
# num_params = sum(p.numel() for p in model.parameters())
# print(f"模型参数量: {num_params}")


def find_files(dirs, max_size_in_MB=1000):
    files = []
    for dir in dirs:
        dir_size_MB = 0
        base_path = os.path.join("data/pt", dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    file_size_bytes = os.path.getsize(full_path)
                    file_size_kb = int(file_size_bytes / 1024)
                    file_size_mb = int(file_size_kb / 1024)
                    if dir_size_MB < max_size_in_MB:
                        dir_size_MB += file_size_mb
                        files.append(full_path)
                    else:
                        logger.info(f'discard {filename} due to exceed size')
        logger.info(f'file {dir}:\t{dir_size_MB} MB')
    return files


# 加载数据集并进行预处理
directories = [
    "accommodation_catering_hotel",
    "artificial_intelligence_machine_learning",
    "computer_communication",
    "computer_programming_code",
    "film_entertainment",
    "literature_emotion",
    "news_media",
    "tourism_geography",
    "current_affairs_government_administration",
    "mathematics_statistics",
]
data_files = find_files(directories)
columns_to_load = ['text', 'alnum_ratio', 'avg_line_length', 'char_rep_ratio', 'max_line_length', 'num_words', 'quality_score', 'special_char_ratio', 'industry_type']
dataset = load_dataset("parquet", data_files=data_files, split="train",columns=columns_to_load)
#dataset = load_dataset("parquet", data_files=data_files, split="train")
dataset = dataset.shuffle(seed=42)
# dataset = dataset.shuffle(seed=42).select(range(20))
# print(dataset[:3]);input()


def preprocess_dataset(examples):
    """预处理预训练数据集，将文本分词并分块"""
    eos_token = "<|im_end|>"
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

    # 将分词结果拼接并分块
    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = 1024  # 分块大小
    total_length = (total_length // block_size) * block_size  # 对齐块大小

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


# 应用预处理函数
train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=dataset.column_names,
    num_proc=16,
)

# 数据整理器
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=16,
    save_steps=100_000,  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    # save_only_model=True,
    logging_steps=20,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()
trainer.save_model()  # 保存模型
tokenizer.save_pretrained(output_path)  # 保存分词器


def plot_loss(save_directory, log_history):
    """绘制训练损失曲线并保存图像"""
    plt.switch_backend("agg")  # 使用非交互式后端
    key = "loss"  # 默认使用 'loss' 作为绘图的关键字
    steps, metrics = [], []

    # 提取损失数据
    for log_entry in log_history:
        if key in log_entry:
            steps.append(log_entry["step"])
            metrics.append(log_entry[key])

    # 绘制图像
    plt.figure()
    plt.plot(steps, metrics, color="#1f77b4", label="original")
    plt.title(f"Training {key} of {save_directory}")
    plt.xlabel("Step")
    plt.ylabel(key.capitalize())
    plt.legend()

    # 保存图像
    figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print(f"Figure saved at: {figure_path}")


# 绘制并保存损失曲线
plot_loss(output_path, trainer.state.log_history)
