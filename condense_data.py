from lm_studio_caller import call_llm, MAX_INPUT_TOKENS
from prompts import build_condensed_prompt
from tqdm import tqdm

def merge_summaries(summaries: list) -> str:
    merged_summary = ""
    for summary in summaries:
        merged_summary += f"{summary}\n\n"

    return merged_summary.strip()

def condensed_paper(paper: dict) -> dict:
    sys_prompt, usr_prompts = build_condensed_prompt(paper)
    
    summaries = []
    for usr_prompt in usr_prompts:
        output = None
        while not output:
            if len(sys_prompt) + len(usr_prompt) >= MAX_INPUT_TOKENS:
                return None

            output = call_llm(sys_prompt, usr_prompt)
            if "[ERROR]" in output:
                return None

        summaries.append(output.strip())

    condensed_summary = merge_summaries(summaries)

    return condensed_summary

def condensed_papers(papers, amount=None):
    condensed_papers_data = {}
    
    for p in tqdm(papers[:amount]):
        paper = condensed_paper(p)
        if paper is not None:
            condensed_papers_data[p["title"]] = paper
    
    return condensed_papers_data