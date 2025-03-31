from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 基础模型路径
model_name = "/models/CodeLlama-13b-Instruct-hf"

# 微调权重路径
finetuned_path = "/models/CodeLlama-13b-finetuned"

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型并应用量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 加载微调权重
model = PeftModel.from_pretrained(model, finetuned_path)

# 输入测试文本
input_text = "<s>[INST] Verify Nudm_SDM_Get can be triggered when ULR initial reg[/INST]"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 生成输出
outputs = model.generate(**inputs, max_new_tokens=3600)

# 解码并打印结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))