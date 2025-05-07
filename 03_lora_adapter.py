# 01-Load model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"  # ""
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as \
many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to(
    "mps"
)  # mps/cuda


response = quantized_model.generate(**input, max_new_tokens=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))


# 02-Load dataset
from datasets import load_dataset

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
    ),
    batched=True,
)
train_sample = data["train"].select(range(400))

# display(train_sample) # display isn't defined


# 03-Preprocess the dataset (option)


# 04-Training cnofig
## Lora config
import peft
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


## Set training arguments
from transformers import TrainingArguments
import os

working_dir = "./"
output_directory = os.path.join(working_dir, "lora")

training_args = TrainingArguments(
    output_dir=output_directory,
    auto_find_batch_size=True,
    learning_rate=3e-4,
    num_train_epochs=5,
)


# $ Set the trainer
import transformers
from trl import SFTTrainer

trainer = SFTTrainer(
    model=quantized_model,
    args=training_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


# 05-Train
trainer.train()

## Save model
model_path = os.path.join(output_directory, f"lora_model")
trainer.model.save_pretrained(model_path)


# 06-Evaluation then optimization
model_path = "/trained_models/lora/lora_model"

from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig


bnb_confg = BitsAndBytesConfig(load_in_8bit=True)

loaded_model = AutoPeftModelForCausalLM.from_pretrained(
    model_path, quantization_config=bnb_config, devise_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to(
    "mps"
)  # cuda

response = loaded_model.generate(**input, max_new_token=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))
