import json
from tqdm import tqdm
from prompts import build_qa_pairs_prompt
from lm_studio_caller import call_llm

def change_key_to_lower(d, selected_key):
    for key in list(d.keys()):
        if key.lower() == selected_key.lower():
            if key != key.lower():
                d[key.lower()] = d.pop(key)
            return True
    
    return False

def extract_and_check_qa_pairs(output: str):
    sections = output.split("```")
    if len(sections) != 3:
        return None

    json_content = sections[1].replace("\\", "\\\\").replace("\n", "").strip()
    if json_content[:4] == "json":
        json_content = json_content[4:]

    try:
        qas = json.loads(json_content)
    except:
        return None

    if type(qas) != list or len(qas) == 0:
        return None

    for qa in qas:
        if type(qa) != dict or len(qa) != 2:
            return None

    for qa in qas:
        if change_key_to_lower(qa, "question") and change_key_to_lower(qa, "answer"):
            return qas
    
    return None

def generate_all_qa_pairs(condensed_papers, max_retry=5, amount=None):
    articles_qas = {}
    for title, content in tqdm(list(condensed_papers.items())[:amount]):
        qas = None
        retries = 0
        while qas is None:
            if retries == max_retry:
                break
            
            sys_prompt, usr_prompt = build_qa_pairs_prompt(title, content)
            output = call_llm(sys_prompt, usr_prompt, 0.35)
            
            if "[ERROR]" in output:
                retries += 1
                continue
            
            qas = extract_and_check_qa_pairs(output)
            retries += 1

        if qas is not None:
            articles_qas[title] = qas
    
    return articles_qas