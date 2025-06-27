# !pip install transformers==4.44.1
# !pip install accelerate
# !pip install bitsandbytes==0.43.3
# !pip install -U "huggingface_hub[cli]"

import os

os.environ["HF_TOKEN"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_model_with_quantization():
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # 8-bit 量化配置（大幅减少内存占用）
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,  # 兼容 Apple Neural Engine
    )

    # 加载模型（自动分配到 mps/cpu）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model_name, quantized_model, tokenizer


def show_data_type_of_model_parameters_and_memory_footprints(quantized_model):
    param_dtypes = [param.dtype for param in quantized_model.parameters()]
    print("Parameter dtypes:", param_dtypes)
    print("Memory footprints:", quantized_model.get_memory_footprint())


def inference(model_name, quantized_model, tokenizer):
    print("hit inference")
    # 不要对输入进行.to()操作，让模型自己处理设备分配
    input = tokenizer("Portugal is", return_tensors="pt")
    print("input", input)

    # 使用generate时也不要指定设备
    response = quantized_model.generate(**input, max_new_tokens=50)
    print("response", response)
    print(tokenizer.batch_decode(response, skip_special_tokens=True))


print("=== start ===")
model_name, quantized_model, tokenizer = load_model_with_quantization()
print("model_name", model_name)
print("model", quantized_model)
show_data_type_of_model_parameters_and_memory_footprints(quantized_model)
inference(model_name, quantized_model, tokenizer)
print("=== end ===")
