from unsloth import FastLanguageModel
import torch


def download_base_model():

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distilled-Llama-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def inference_base_model(model, tokenizer):
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        Please answer the following medical question.

        ### Question:
        {}

        ### Response:
        <think>{}"""
    question = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"

    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to(
        "mps" if torch.backends.mps.is_available() else "cuda"
    )  # mps/cuda
    outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


if __name__ == "__main__":
    print("=== start ===")
    model, tokenizer = download_base_model()
    inference_base_model(model, tokenizer)
    print("=== end ===")
