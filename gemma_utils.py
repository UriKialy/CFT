"""
Gemma VLM utilities
"""
import re
from collections import defaultdict, Counter

import torch
import numpy as np
from tqdm.auto import tqdm

def is_structured(task_name):
    return task_name in STRUCTURED_TASK_CONFIG

# =============================================================================
# 2. PROMPT BUILDERS — 3 strategies
# =============================================================================

def build_prompt_strategy3(task_name):
    """Strategy 3: 'Classify this image into one of: [c1, c2, ...]. Answer with the class name only.'"""
    if is_structured(task_name):
        cfg = STRUCTURED_TASK_CONFIG[task_name]
        opts = ", ".join(cfg["class_names"])
        return f"{cfg['question']} Choose from: {opts}."
    classes = TASK_CLASS_NAMES[task_name]
    opts = ", ".join(classes)
    return f"Classify this image into one of: [{opts}]. Answer with the class name only."


# =============================================================================
# 3. ANSWER MATCHING — fuzzy match VLM output to class name
# =============================================================================

def match_answer_to_class(answer, class_names):
    """Match VLM free-text answer to closest class name."""
    answer_lower = answer.strip().lower()

    # 1) Exact match
    for i, cn in enumerate(class_names):
        if cn.lower() == answer_lower:
            return i

    # 2) Class name contained in answer (longest first to avoid partial matches)
    sorted_by_len = sorted(enumerate(class_names), key=lambda x: -len(x[1]))
    for i, cn in sorted_by_len:
        if cn.lower() in answer_lower:
            return i

    # 3) Answer contained in class name
    for i, cn in enumerate(class_names):
        if answer_lower in cn.lower():
            return i

    # 4) Word overlap score
    answer_words = set(answer_lower.split())
    best_i, best_score = 0, 0
    for i, cn in enumerate(class_names):
        cn_words = set(cn.lower().split())
        overlap = len(answer_words & cn_words)
        if overlap > best_score:
            best_score = overlap
            best_i = i
    if best_score > 0:
        return best_i

    # 5) For numeric answers (structured tasks)
    nums_in_answer = re.findall(r'\d+', answer_lower)
    if nums_in_answer:
        for num_str in nums_in_answer:
            for i, cn in enumerate(class_names):
                if cn.strip() == num_str:
                    return i

    return -1  # no match

def get_class_names(task_name):
    if is_structured(task_name):
        return STRUCTURED_TASK_CONFIG[task_name]["class_names"]
    return TASK_CLASS_NAMES[task_name]

def get_answer_token(task_name, label_idx):
    """Get the expected text answer for a given label index."""
    names = get_class_names(task_name)
    if label_idx < len(names):
        return names[label_idx]
    return str(label_idx)
def build_messages(image, question):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]

@torch.no_grad()
def classify_image(image, question):
    """Send one image + question to Gemma and get text answer."""
    messages = build_messages(image, question)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()
