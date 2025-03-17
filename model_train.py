from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling
import os

def cpu_train(qas_pairs, train_model_name, sequences_lenght, finetuned_model_name):
    data_list = []
    for article, qa_pairs in qas_pairs.items():
        for pair in qa_pairs:
            instruction = f"[{article}] {pair['question']}"
            output = pair["answer"]
            data_list.append({"instruction": instruction, "output": output})

    print(f"Dataset size: {len(data_list)}")

    dataset = Dataset.from_list(data_list)
    dataset = dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(train_model_name)

    def tokenize_function(sample):
        prompt = f"Instruction: {sample['instruction']}\nAnswer: {sample['output']}\n"
        return tokenizer(prompt, truncation=True, max_length=sequences_lenght)

    tokenized_datasets = dataset.map(tokenize_function, batched=False)

    model = AutoModelForCausalLM.from_pretrained(
        train_model_name,
        device_map="cpu"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to("cpu")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=50,
        learning_rate=2e-4,
        fp16=False,
        no_cuda=True,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    os.makedirs("finetuned_models", exist_ok=True)
    model.save_pretrained(f"finetuned_models/{finetuned_model_name}")

def gpu_train(qas_pairs, train_model_name, sequences_lenght, finetuned_model_name):   
    data_list = []
    for article, qa_pairs in qas_pairs.items():
        for pair in qa_pairs:
            instruction = f"[{article}] {pair['question']}"
            output = pair["answer"]
            data_list.append({"instruction": instruction, "output": output})
    
    print(f"Dataset size: {len(data_list)}")
    
    dataset = Dataset.from_list(data_list)
    dataset = dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(train_model_name)

    def tokenize_function(sample):
        prompt = f"Instruction: {sample['instruction']}\nAnswer: {sample['output']}\n"
        tokenized = tokenizer(prompt, truncation=True, max_length=sequences_lenght, padding="max_length", return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0]
        }

    tokenized_datasets = dataset.map(tokenize_function, batched=False)

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        train_model_name,
        device_map="auto",
        quantization_config=quant_config
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=50,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    os.makedirs("finetuned_models", exist_ok=True)
    model.save_pretrained(f"finetuned_models/{finetuned_model_name}")