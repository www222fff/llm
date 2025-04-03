import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(15)
torch.set_num_interop_threads(15)

# 指定本地模型路径
model_name = "/models/CodeLlama-13b-Instruct-hf"

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载本地模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    use_safetensors=True,
    device_map="cpu"
)
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable()

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("--------load done")

dataset = load_dataset("json", data_files="dataset.json")
dataset = dataset["train"]

# 预处理数据集

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_function(example):
    instruction = clean_text(example['instruction'])
    response = clean_text(example['response'])
    text = f"<s>[INST] {instruction} [/INST] {response} </s>"

    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512, add_special_tokens=True)
    tokenized["labels"] = tokenized["input_ids"].copy()

    print("===== Debug:=====")
    print(f"Raw Text: {text}")
    print(f"input_ids: {tokenized['input_ids'][:20]}")
    print(f"Decoded: {tokenizer.decode(tokenized['input_ids'])}")
    print(f"labels: {tokenized['labels'] if 'labels' in tokenized else 'Not Set'}")
    print("------------------------------------------------")

    return tokenized


dataset = dataset.map(preprocess_function, remove_columns=["instruction", "response"])
tokenized_dataset = dataset.shuffle(seed=42)

print("--------preprocess done")

# 分割数据集
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print("--------split done")

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./CodeLlama-13b-finetuned",
    logging_dir="./logs",
    logging_steps=1,  # 每步记录日志
    num_train_epochs=1,
    dataloader_num_workers=0,
    per_device_train_batch_size=1,  # 减小以适应内存
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    max_grad_norm=1.0,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    use_cpu=True,
    seed=42
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 开始训练
print("--------train start")
trainer.train()

# 保存微调后的模型和 tokenizer
model.save_pretrained("./CodeLlama-13b-finetuned")
tokenizer.save_pretrained("./CodeLlama-13b-finetuned")
print("--------Complete")
