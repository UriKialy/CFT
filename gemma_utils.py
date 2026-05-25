"""
Gemma VLM utilities — extracted verbatim from CFT_Gemma3_4B_IT_CUB200.ipynb
cells 8 and 9 (prompt builders, answer matching, zero-shot eval, confused-class).
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


# ── The actual VLM question prompts (what we send with each image) ──

def get_question_for_image(task_name, strategy):
        return build_prompt_strategy3(task_name)


# =============================================================================
# 3. ANSWER MATCHING — fuzzy match VLM output to class name
# =============================================================================
def extract_tagged_answer(raw_answer):
    """Extract answer from [CLASSIFICATION_START]...[CLASSIFICATION_END] tags."""
    match = re.search(
        r'\[CLASSIFICATION_START\](.*?)\[CLASSIFICATION_END\]',
        raw_answer, re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return raw_answer.strip()  # fallback: use full answer if no tags found

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
from collections import defaultdict, Counter
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


def match_answer_to_class(answer, class_names):
    """Fuzzy match VLM output to closest class name."""
    answer_lower = answer.strip().lower()

    # Exact
    for i, cn in enumerate(class_names):
        if cn.lower() == answer_lower:
            return i
    # Class in answer (longest first)
    for i, cn in sorted(enumerate(class_names), key=lambda x: -len(x[1])):
        if cn.lower() in answer_lower:
            return i
    # Answer in class
    for i, cn in enumerate(class_names):
        if answer_lower in cn.lower():
            return i
    # Word overlap
    answer_words = set(answer_lower.split())
    best_i, best_score = 0, 0
    for i, cn in enumerate(class_names):
        overlap = len(answer_words & set(cn.lower().split()))
        if overlap > best_score:
            best_score = overlap
            best_i = i
    if best_score > 0:
        return best_i
    # Numeric fallback
    nums = re.findall(r'\d+', answer_lower)
    for n in nums:
        for i, cn in enumerate(class_names):
            if cn.strip() == n:
                return i
    return -1


@torch.no_grad()
def evaluate_zero_shot(dataset, task_name, return_confusion=False, batch_size=8):
    question = build_prompt_strategy3(task_name)
    class_names = get_class_names(task_name)

    correct = 0
    predictions = []
    labels_list = []

    indices = list(range(len(dataset)))

    for batch_start in tqdm(range(0, len(indices), batch_size), desc=f"  ZS {task_name}", leave=False):
        batch_idx = indices[batch_start:batch_start + batch_size]
        batch_imgs = [dataset.get_pil_image(i) for i in batch_idx]
        batch_labels = [dataset.get_label(i) for i in batch_idx]

        # Build messages for each image
        batch_messages = []
        for img in batch_imgs:
            batch_messages.append([{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]}])

        # Process batch
        batch_inputs = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(model.device)

        output_ids = model.generate(
            **batch_inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,
        )

        # Decode each answer
        for i, idx in enumerate(batch_idx):
            input_len = batch_inputs["input_ids"][i].ne(processor.tokenizer.pad_token_id).sum()
            new_tokens = output_ids[i, input_len:]
            answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
            pred = match_answer_to_class(answer, class_names)
            predictions.append(pred)
            labels_list.append(batch_labels[i])
            if pred == batch_labels[i]:
                correct += 1

    acc = 100.0 * correct / len(dataset)

    if return_confusion:
        confusion = defaultdict(Counter)
        for true, pred in zip(labels_list, predictions):
            if pred >= 0:
                confusion[true][pred] += 1
        return acc, confusion

    return acc


def get_most_confused_class(confusion, num_classes):
    """For each class, find which OTHER class the model most often predicts.
    Returns dict: class_idx -> most_confused_class_idx."""
    most_confused = {}
    for c in range(num_classes):
        if c in confusion:
            # Remove self-predictions
            other_preds = {k: v for k, v in confusion[c].items() if k != c and k >= 0}
            if other_preds:
                most_confused[c] = max(other_preds, key=other_preds.get)
            else:
                # Fallback: random other class
                most_confused[c] = (c + 1) % num_classes
        else:
            most_confused[c] = (c + 1) % num_classes
    return most_confused