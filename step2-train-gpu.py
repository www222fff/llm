import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 指定本地模型路径
model_name = "/models/CodeLlama-13b-Instruct-hf"

# 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.eos_token_id

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

# 加载和预处理数据集
dataset = load_dataset("json", data_files="dataset.json")

def preprocess_function(example):
    return {"text": f"<s>[INST] {example['instruction']} [/INST] {example['response']} </s>"}

dataset = dataset.map(preprocess_function, remove_columns=["instruction", "response"])
dataset = dataset["train"]

# Tokenize 数据集（截断模式）
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    #return tokenizer(example["text"], truncation=True, max_length=512, return_overflowing_tokens=True, stride=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

# 分割数据集
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 训练参数
training_args = TrainingArguments(
    output_dir="./codellama-13b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # A100 20GB 支持
    gradient_accumulation_steps=2,  # 有效 batch size = 16
    optim="adamw_torch",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    seed=42
)

# 初始化和运行 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer.train()

# 保存微调后的模型和 tokenizer
model.save_pretrained("/models/CodeLlama-13b-finetuned")
tokenizer.save_pretrained("/models/CodeLlama-13b-finetuned")
