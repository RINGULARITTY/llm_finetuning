from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def load_optimized_model(model_name, adaptator_name, compile_model=True):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quant_config
    )

    model = PeftModel.from_pretrained(base_model, f"finetuned_models/{adaptator_name}")
    if compile_model:
        model = torch.compile(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  
    
    return model, tokenizer

def question_model(model, tokenizer, user_input, sys_prompt=""):
    chat_prompt = f"<|system|>\n{sys_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text[len(chat_prompt):].strip()
