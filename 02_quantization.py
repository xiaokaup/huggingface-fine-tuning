from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_model_with_quantization():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # 4-bit 量化配置（大幅减少内存占用）
    # BitsAndBytesConfig not support MacOS M1
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,  # 兼容 Apple Neural Engine
    )

    # 加载模型（自动分配到 mps/cpu）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 自动选择 mps/cpu
    )
    return model_name, model


def show_data_type_of_model_parameters_and_memory_footprints(model):
    param_dtypes = [param.dtype for param in model.parameters()]
    print("Parameter dtypes:", param_dtypes)
    print("Memory footprints:", model.get_memory_footprint())


def inference(model_name, model):
    print("hit inference")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = tokenizer("Portugal is", return_tensors="pt").to("cuda")
    print("input", input)

    response = model.generate(**input, max_new_tokens=50)
    print("response", response)
    print(tokenizer.batch_decode(response, skip_special_tokens=True))


if __name__ == "__main__":
    print("=== start ===")
    model_name, model = load_model_with_quantization()
    print("model_name", model_name)
    print("model", model)
    show_data_type_of_model_parameters_and_memory_footprints(model)
    inference(model_name, model)
    print("=== end ===")
