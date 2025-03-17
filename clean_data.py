import io
import tarfile
import os
import re
import requests
from PyPDF2 import PdfReader
from html import unescape
import shutil
from tqdm import tqdm
from fetch_data import download_pdf

def tokenize_latex(latex, pattern):
    tokens = []
    pos = 0
    for match in pattern.finditer(latex):
        start, end = match.span()
        if start > pos:
            text = latex[pos:start]
            tokens.append({
                "type": "text",
                "content": text,
                "pos": pos
            })
        if match.group("env"):
            envname = match.group("envname").strip().lower()
            envcontent = match.group("envcontent").strip()
            tokens.append({
                "type": "env",
                "envname": envname,
                "title": envname.capitalize(),
                "content": envcontent,
                "pos": start
            })
        elif match.group("command"):
            cmdname = match.group("cmdname").strip().lower()
            cmdtitle = match.group("cmdtitle").strip()
            tokens.append({
                "type": "command",
                "cmdname": cmdname,
                "title": cmdtitle,
                "content": "",
                "pos": start
            })
        pos = end
    if pos < len(latex):
        tokens.append({
            "type": "text",
            "content": latex[pos:],
            "pos": pos
        })
    return tokens

def create_node(node_type, title, level, pos):
    return {
        "node_type": node_type,
        "title": title,
        "level": level,
        "pos": pos,
        "content": "",
        "children": []
    }

def build_hierarchy(tokens, command_levels, env_levels):
    root = {
        "node_type": "root",
        "title": "root",
        "level": 0,
        "content": "",
        "children": []
    }
    stack = [root]
    
    for token in tokens:
        if token["type"] in ("command", "env"):
            if token["type"] == "command":
                level = command_levels.get(token["cmdname"], 100)
            else:
                level = env_levels.get(token["envname"], 100)
            new_node = create_node(token["type"], token["title"], level, token["pos"])
            if token["type"] == "env":
                new_node["content"] = token["content"]
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["children"].append(new_node)
            stack.append(new_node)
        elif token["type"] == "text":
            stack[-1]["content"] += token["content"]
    return root["children"]

def flatten_hierarchy(nodes, flat_dict):
    for node in nodes:
        title = node["title"]
        node_content = node["content"].strip()
        if title in flat_dict:
            flat_dict[title] += "\n" + node_content
        else:
            flat_dict[title] = node_content
        if node["children"]:
            flatten_hierarchy(node["children"], flat_dict)

def extract_flat_sections(latex, pattern, command_levels, env_levels):
    tokens = tokenize_latex(latex, pattern)
    hierarchy = build_hierarchy(tokens, command_levels, env_levels)
    flat_dict = {}
    flatten_hierarchy(hierarchy, flat_dict)
    return flat_dict

def get_and_extract_paper_segmented_content(paper, folder):
    arxiv_id = paper["pdf_url"].split("/")[-1]
    url = f"https://arxiv.org/e-print/{arxiv_id}"

    response = requests.get(url)
    if response.status_code != 200:
        print("Download error")
        return None
    if "reCAPTCHA" in str(response.content):
        raise Exception("CAPTCHA required")
    
    tar_bytes = io.BytesIO(response.content)
    try:
        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
            tar.extractall(path=folder)
    except:
        return None
        
    tex_file = os.path.join(folder, "main.tex")
    if not os.path.exists(tex_file):
        tex_files = [f for f in os.listdir(folder) if f.endswith(".tex")]
        if tex_files:
            tex_file = os.path.join(folder, tex_files[0])
        else:
            print("No .tex found in the archive")
            return None
    
    command_levels = {"part": 1, "chapter": 2, "section": 3, "subsection": 4, "subsubsection": 5, "paragraph": 6, "subparagraph": 7}
    env_levels = {"abstract": 1, "keywords": 1, "acknowledgments": 1, "acknowledgements": 1, "résumé": 1, "resume": 1, "preface": 1}
    
    env_pattern = (r"(?P<env>\\begin\{(?P<envname>" + "|".join(env_levels.keys()) + r")\}(?P<envcontent>.*?)\\end\{\2\})")
    cmd_pattern = (r"(?P<command>\\(?P<cmdname>" + "|".join(command_levels.keys()) + r")\*?\{(?P<cmdtitle>[^}]+)\})")
    pattern = re.compile(env_pattern + "|" + cmd_pattern, re.DOTALL | re.IGNORECASE)
    
    with open(tex_file, encoding="utf-8", errors="ignore") as f:
        latex_content = f.read()
    
    return extract_flat_sections(latex_content, pattern, command_levels, env_levels)

def extract_titles(outlines, additional_titles):
    extracted_titles = []

    def process_outline(outline):
        if isinstance(outline, dict) and "/Title" in outline:
            title = outline["/Title"]
            cleaned_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
            extracted_titles.append(cleaned_title)
        elif isinstance(outline, list):
            for sub_outline in outline:
                process_outline(sub_outline)

    for outline in outlines:
        process_outline(outline)

    for title in additional_titles:
        if title not in extracted_titles:
            extracted_titles.append(title)

    return extracted_titles


def extract_pdf_outlines(url, pdf_path, additional_titles):
    download_pdf(url, pdf_path)
    
    try:
        reader = PdfReader(pdf_path)
        outlines = reader.outline
    except Exception as e:
        return None
    
    outlines = extract_titles(outlines, additional_titles)

    return outlines

def filter_sections_by_outlines(flat_sections, outline_titles):
    if not outline_titles:
        return {}
    
    valid_sections = {}
    current_valid_title = None
    for title, content in flat_sections.items():
        is_valid = any(title.lower() == ot.lower() for ot in outline_titles)
        if is_valid:
            current_valid_title = title
            if title in valid_sections:
                valid_sections[title] += "\n" + content
            else:
                valid_sections[title] = content
        else:
            if current_valid_title is not None:
                valid_sections[current_valid_title] += "\n" + content
            else:
                pass
    return valid_sections

def clean_latex(text: str) -> str:
    text = re.sub(r'%.*', '', text)
    
    text = re.sub(r'\\(documentclass|usepackage|maketitle|tableofcontents|bibliographystyle|cite|label|footnote)\{.*?\}', '', text)
    
    text = re.sub(r'\\title\{.*?\}', '', text)
    text = re.sub(r'\\author\{.*?\}', '', text)
    text = re.sub(r'\\maketitle', '', text)
    
    text = re.sub(r'\\(section|subsection|subsubsection)\*?\{(.*?)\}', r'\n\n\2\n\n', text)
    
    text = re.sub(r'\\(textbf|textit|underline|emph)\{(.*?)\}', r'\2', text)
    
    text = text.replace(r'\%', '%').replace(r'\_', '_').replace(r'\&', '&')
    
    text = text.replace('---', '—').replace('--', '–').replace('``', '“').replace("''", '”')
    
    text = re.sub(r'\\cite\{.*?\}', '[citation]', text)
    
    text = re.sub(r'\$(.*?)\$', r' \1 ', text)
    text = re.sub(r'\\\[(.*?)\\\]', r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r' \1 ', text, flags=re.DOTALL)
    
    text = re.sub(r'\\begin\{figure\*?\}.*?\\caption\{(.*?)\}.*?\\end\{figure\*?\}', r'\n[FIGURE: \1]\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{table\*?\}.*?\\caption\{(.*?)\}.*?\\end\{table\*?\}', r'\n[TABLE: \1]\n', text, flags=re.DOTALL)
    
    text = re.sub(r'\\caption\{(.*?)\}', r'\n[FIGURE: \1]\n', text)
    
    text = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', text, flags=re.DOTALL)
    
    text = re.sub(r'\\ref\{.*?\}', '[reference]', text)
    
    text = re.sub(r'\\begin\{itemize\}|\\begin\{enumerate\}', '', text)
    text = re.sub(r'\\end\{itemize\}|\\end\{enumerate\}', '', text)
    text = re.sub(r'\\item\s+', '- ', text)
    
    text = re.sub(r'\\href\{.*?\}\{(.*?)\}', r'\1', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    text = unescape(text)
    
    return text.strip()

def get_filtered_sections(paper, folder, pdf_path):
    flat_sections = get_and_extract_paper_segmented_content(paper, folder)
    if flat_sections is None:
        shutil.rmtree(folder, ignore_errors=True)
        return None
    
    additional_titles = ["Abstract", "Conclusion", "Conclusions"]
    outline_titles = extract_pdf_outlines(paper["pdf_url"], pdf_path, additional_titles)
    if outline_titles is None:
        shutil.rmtree(folder, ignore_errors=True)
        return None
    
    if outline_titles == additional_titles:
        outline_titles = list(flat_sections.keys())


    filtered_sections = filter_sections_by_outlines(flat_sections, outline_titles)

    for fs in filtered_sections:
        filtered_sections[fs] = clean_latex(filtered_sections[fs])

    paper["sections_content"] = filtered_sections
    
    shutil.rmtree(folder, ignore_errors=True)
    
    return paper

def get_filtered_sections_papers(papers):
    clean_selections_papers = []
    
    bar = tqdm(papers)
    for p in bar:
        bar.set_description(p["title"])
        paper = get_filtered_sections(p, "temp", "temp/paper.pdf")
        if paper:
            clean_selections_papers.append(paper)
    
    return clean_selections_papers