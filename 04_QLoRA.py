# 01-Get Quantized Model
## NF4 (4-bit NormalFloat) quantization
## Bitsandbytes configuration
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

os.environ["HF_TOKEN"] = ""


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)


print(quantized_model.get_memory_footprint())


param_dtypes = [param.dtype for param in quantized_model.parameters()]
print("param_dtypes", param_dtypes)


tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to(
    "mps"
)  # cuda

response = quantized_model.generate(**input, max_new_tokens=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))


# 02-Train Model
from datasets import load_dataset
import peft
from peft import LoraConfig
import transformers
from transformers import TrainingArguments
import os
from trl import SFTTrainer


## Process the dataset

dataset = "openai/gsm8k"
data = load_dataset(dataset, "main")

tokenizer.pad_token = tokenizer.eos_token
data = data.map(
    lambda samples: tokenizer(
        samples["question"],
        samples["answer"],
        truncation=True,
        padding="max_length",
        max_length=100,
        batched=True,
    )
)
train_sample = data["train"].select(range(400))


## LoRA configurations

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


## Setting the training arguments

working_dir = "./"
output_directory = os.path.join(working_dir, "qlora")

training_args = TrainingArguments(
    output_dir=output_directory,
    auto_find_batch_size=True,
    learning_rate=3e-4,
    num_train_epochs=5,
)


## Setting the trainer

trainer = SFTTrainer(
    model=quantized_model,
    args=training_args,
    train_dataset=train_sample,
    peft_config=lora_config,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


## Train the model
trainer.train()


## Save the model
model_path = os.path.join(output_directory, f"qulora_model")
trainer.model.save_pretrained(model_path)


# 03-Load the fine-tuned model
model_path = "/trained_models/lora/lora_model"

from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

loaded_model = AutoPeftModelForCausalLM.from_pretrained(
    model_path, quantization_config=bnb_config, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to(
    "mps"
)  # cuda

response = loaded_model.generate(**input, max_new_tokens=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))
