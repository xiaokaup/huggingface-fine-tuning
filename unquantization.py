from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model_name, model


def show_data_type_of_model_parameters_and_memory_footprints(model):
    param_dtypes = [param.dtype for param in model.parameters()]
    print("Parameter dtypes:", param_dtypes)
    print("Memory footprints:", model.get_memory_footprint())


def inference(model_name, model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = tokenizer("Portugal is", return_tensors="pt").to("cuda")
    print("input", input)

    response = model.generate(**input, max_new_tokens=50)
    print("response", response)
    print(tokenizer.batch_decode(response, skip_special_tokens=True))


if __name__ == "__main__":
    model_name, model = load_model()
    show_data_type_of_model_parameters_and_memory_footprints(model_name, model)
    # inference(model)
