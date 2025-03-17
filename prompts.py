from lm_studio_caller import MAX_INPUT_TOKENS, get_tokens_amount

def build_condensed_prompt(paper: dict) -> list:
    sys_prompt = (
        "You are an advanced AI assistant specializing in **extreme compression of AI research papers** while retaining **100% of the key technical content**. "
        "Your primary goal is to **minimize the token count** to the absolute lowest possible while **ensuring no critical information is lost**.\n\n"
        
        "🔹 **Compression Strategy:**\n"
        "• Convert full sentences into **ultra-condensed representations**: use bullet points, mathematical notation, and structured data where possible.\n"
        "• Retain **ALL mathematical formulas, algorithms, key findings, datasets, and experimental results EXACTLY** as presented in the original paper.\n"
        "• Use **symbolic notation**, abbreviations, and compact formulations **only when they do not reduce clarity**.\n"
        "• Eliminate all **redundant phrasing, filler words, and narrative-style explanations**.\n"
        "• When summarizing explanations, **preserve the full logical reasoning chain** while reducing word count.\n\n"

        "🔹 **What NOT to Remove:**\n"
        "✅ Keep all **technical explanations, dataset details, hyperparameters, evaluation metrics, and results**.\n"
        "✅ Maintain the **structural integrity** of the paper (e.g., sections should be recognizable).\n"
        "❌ Do NOT alter **any** formulas, figures, or critical findings.\n"
        "❌ Do NOT remove **any methodology steps or mathematical derivations**.\n"
        "❌ Do NOT replace precise terms with vague summaries.\n\n"

        "Your final output should be a **highly compressed yet fully informative representation** of the research paper."
    )

    base_prompt = (
        "Your goal is to **compress the provided AI research paper** while ensuring **100% retention of key technical content, formulas, methodologies, and results**. "
        "Follow these precise guidelines:\n\n"

        "🔹 **Compression Techniques:**\n"
        "• Use **bullet points, compact notations, and ultra-condensed language** instead of full sentences.\n"
        "• Retain **ALL mathematical formulas exactly as they appear** (no paraphrasing or reformatting).\n"
        "• Use **symbolic notation** for complex equations and algorithmic expressions where possible.\n"
        "• Replace **wordy explanations with direct, factual, structured data**.\n"
        "• Prioritize **core contributions and methodologies**, removing only redundant wording.\n\n"

        "🔹 **Essential Content to Preserve:**\n"
        "✅ **Core Hypothesis & Research Goals**\n"
        "✅ **Mathematical Formulations, Proofs, and Equations**\n"
        "✅ **Algorithms & Methodological Steps** (maintain structure and key decisions)\n"
        "✅ **Datasets, Hyperparameters, and Experimental Setup**\n"
        "✅ **Evaluation Metrics & Results** (including tables and figures as references)\n\n"

        "🔹 **Strict Prohibitions:**\n"
        "❌ Do NOT remove or modify **any formulas, equations, or notation**.\n"
        "❌ Do NOT simplify content in a way that **loses critical reasoning or alters meaning**.\n"
        "❌ Do NOT include **non-essential sections** (extended references, acknowledgments, bibliographies, generic introductions).\n\n"

        f"**Paper Title:** {paper['title']}\n\n"
        "🔹 **Condense the following AI research paper while strictly maintaining ALL mathematical integrity, algorithms, and key findings**:\n"
    )

    total_tokens = get_tokens_amount(sys_prompt + base_prompt)
    sections_text = []
    prompts = []

    for section_title, section_content in paper.get("sections_content", {}).items():
        section_text = f"### {section_title}:\n{section_content}\n\n"
        section_tokens = get_tokens_amount(section_text)

        if total_tokens + section_tokens < 0.98 * MAX_INPUT_TOKENS:
            sections_text.append(section_text)
            total_tokens += section_tokens
        else:
            prompts.append(base_prompt + "".join(sections_text))
            sections_text = [section_text]
            total_tokens = get_tokens_amount(base_prompt) + section_tokens

    if sections_text:
        prompts.append(base_prompt + "".join(sections_text))

    return sys_prompt, prompts

def build_qa_pairs_prompt(title: str, condensed_text: str) -> list:
    sys_prompt = (
        "You are an expert AI research assistant specializing in the deep analysis and explanation of condensed scientific texts. "
        "Your primary task is to generate 0 to 5 high-quality question-and-answer pairs strictly based on the provided condensed paper text. "
        "For each QA pair, you must adhere to the following guidelines:\n\n"
        
        "1. **No Redundant QA Pairs**: If the question is something you already confidently know from your pre-training, do **not** generate it. "
        "Only create QA pairs that are genuinely **novel** or require interpretation based on the given text. The goal is to generate fine-tuning new data "
        "that improves knowledge, not to reinforce already known concepts.\n\n"
        
        "2. **Prioritize Conceptual Depth**: Focus on **fundamental concepts, algorithms, mathematical formulas (in LaTeX), and methodologies** present in the text. "
        "Your questions should demand **deep understanding** and encourage step-by-step explanations of reasoning.\n\n"
        
        "3. **Avoid Generic or Obvious Questions**: Do not generate questions that can be trivially answered from general AI knowledge. "
        "For instance, avoid 'What is backpropagation?' unless the paper introduces a unique perspective on it.\n\n"
        
        "4. **Detailed and Step-by-Step Explanations**: Provide comprehensive, structured answers that clearly demonstrate reasoning (chain-of-thought). "
        "Any formulas or algorithms must be precisely replicated as they appear in the text, ensuring technical precision.\n\n"
        
        "5. **Strict Source Fidelity**: Base all questions and answers solely on the provided condensed text. Do not reference any external knowledge "
        "or details not explicitly present in the text.\n\n"
        
        "6. **Formal and Academic Tone**: Maintain a scholarly tone with rigorous academic language. Use best practices from technical pedagogy and "
        "chain-of-thought methodologies to enhance transparency and instruction.\n\n"
        
        "7. **JSON Output Format**: Your final output must be a JSON array of objects. Each object should have exactly two keys: 'question' and 'answer'.\n\n"
        
        "8. **Self-Check Mechanism**: Before finalizing a QA pair, critically evaluate: 'Does this question add value for fine-tuning, or is it something "
        "the model already understands well?'. If it is redundant, **discard it and generate a better one.**\n\n"
        
        "9. **Quality Over Quantity**: Prioritize in-depth exploration over sheer volume. Each QA pair should be carefully crafted to reveal key ideas "
        "and nuanced interconnections between concepts.\n\n"
    )

    usr_prompt = (
        "You will receive a condensed version of an AI research paper that is dense with technical details, including key concepts, "
        "algorithms, mathematical formulas (in LaTeX), and technical methodologies.\n\n"

        "Your task is to generate between 5 and 10 **high-quality** question-and-answer pairs that **thoroughly assess and deepen** understanding "
        "of the core concepts in the provided text. Please follow these guidelines:\n\n"

        "1. **Generate Only Useful Questions**: If you already confidently know the answer to a question **without needing the provided text**, "
        "**do not generate it**. Instead, focus on insights that require interpretation of the specific paper.\n\n"

        "2. **Question Requirements**: Each question should target a core element such as a **fundamental concept, an algorithm, or a mathematical formula**. "
        "They should demand detailed explanations that break down reasoning in a structured, step-by-step manner.\n\n"

        "3. **Answer Requirements**: Each answer must be **detailed and systematic**, precisely reproducing any formulas or algorithms as they appear in the text. "
        "The explanation should reveal the **chain-of-thought process**, making clear the underlying reasoning and logical connections between ideas.\n\n"

        "4. **Reject Generic Questions**: Avoid simplistic questions like 'What is a neural network?' unless the paper provides a novel insight. "
        "Prioritize questions that explore **specific contributions, mathematical derivations, or theoretical implications.**\n\n"

        "5. **Strict Adherence**: Do not incorporate external knowledge. Base everything solely on the provided text.\n\n"

        "6. **Formal, Academic Style**: Use an academic tone, following best practices from technical pedagogy. Ensure rigor and clarity in your explanations.\n\n"

        f"**Article Title:** {title}\n\n"
        "Here is the condensed paper text:\n\n"
        f"{condensed_text}\n"
    )
    
    return sys_prompt, usr_prompt

def build_evaluation_prompt(question, answer, llm_output):
    sys_prompt = (
        "You are an AI assistant designed to evaluate the quality of responses generated by another LLM. "
        "Your task is to compare a reference answer with a generated answer based on the following criteria:\n"
        "1. **Accuracy (0-10)**: How factually correct is the generated answer compared to the reference?\n"
        "2. **Relevance (0-10)**: How well does the generated answer address the question?\n"
        "3. **Clarity (0-10)**: Is the generated answer well-structured and easy to understand?\n"
        "4. **Completeness (0-10)**: Does the generated answer provide sufficient detail without unnecessary fluff?\n\n"

        "Provide your judgment in the exact JSON format:\n"
        "```json\n"
        "{\n"
        "    'accuracy': ...,\n"
        "    'relevance': ...,\n"
        "    'clarity': ...,\n"
        "    'completeness': ...,\n"
        "    'comments': ...\n"
        "}\n"
        "```"
    )

    usr_prompt = (
        "### Question:\n"
        f"{question.replace("\n\n", "\n")}\n\n"
        
        "### Reference Answer:\n"
        f"{answer.replace("\n\n", "\n")}\n\n"
        
        "### Generated Answer:\n"
        f"{llm_output.replace("\n\n", "\n")}\n\n"
        
        "Evaluate the generated answer based on the criteria provided."
    )

    return sys_prompt, usr_prompt