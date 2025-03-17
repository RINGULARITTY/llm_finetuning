import requests
import time
from llama_cpp import Llama

LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "bartowski/llama-3.2-3b-instruct"
MAX_INPUT_TOKENS = 16384
TIMEOUT_SECONDS = 10*60
MAX_RETRIES = 3

llm = Llama(model_path="models/Llama-3.2-3B-Instruct-Q6_K.gguf", n_ctx=MAX_INPUT_TOKENS, verbose=False)

def get_tokens_amount(text):
    return len(llm.tokenize(text.encode("utf-8"), add_bos=True))

def call_llm(sys_prompt, usr_prompt, temperature=0.2) -> str:
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.15,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(LM_STUDIO_API, json=payload, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            response_data = response.json()
            output_text = response_data['choices'][0]['message']['content'].strip()
            return output_text.split("</think>")[-1]
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout ({attempt + 1}/{MAX_RETRIES})...")
        except requests.RequestException as e:
            print(f"🚨 Error : {e}")
        time.sleep(1)

    return "[ERROR] Failure after multiple try"