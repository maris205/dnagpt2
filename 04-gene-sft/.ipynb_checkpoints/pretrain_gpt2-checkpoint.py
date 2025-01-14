# import os

# # 设置环境变量
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # 打印环境变量以确认设置成功
# print(os.environ.get('HF_ENDPOINT'))
import subprocess
import os
# 设置环境变量, autodl一般区域
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value



import math
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer,AutoConfig


# 加载 OpenWebText 数据集
dataset = load_dataset("text", data_files="../01-data_env/data/dna_1g.txt")["train"].train_test_split(test_size=0.01, shuffle=True)

# 定义最大输入长度
max_length = 256


# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)



# 初始化 GPT-2 分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2_tokenizer")
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=5)

# 4. 创建一个数据收集器，用于动态填充和遮蔽
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)



# 加载并调整 GPT-2 配置
config = AutoConfig.from_pretrained(
    "gpt2",  # 加载 GPT-2 的默认配置
    vocab_size=len(tokenizer),  # 更新词汇表大小为自定义分词器的词汇表大小
    n_ctx=max_length,  # 最大上下文长度（序列长度）
    n_positions=max_length,  # 最大位置编码长度，通常与 n_ctx 一致
)

# 初始化 GPT-2 模型
model = GPT2LMHeadModel(config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-small",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    save_steps=10000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=20000,
    evaluation_strategy="steps",
    eval_steps=10000,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # 启用混合精度训练
    #deepspeed="ds_zero2_no_offload.json"  
)


# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()


# 评估 perplexity
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")



out_model_path = "gpt2-small-gene-openweb"
trainer.save_model(out_model_path)
tokenizer.save_pretrained(out_model_path)