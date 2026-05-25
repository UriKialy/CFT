"""
Gemma generative training loop — extracted verbatim from
CFT_Gemma3_4B_IT_CUB200.ipynb cell 15.

Requires gemma_utils helpers (build_prompt_strategy3, classify_image,
match_answer_to_class, get_answer_token, get_class_names, evaluate_zero_shot)
to be in scope when train_generative is called.
"""
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

# =============================================================================
# TRAINING: Generative fine-tuning (Full FT, CFT) + Linear Probe
# =============================================================================

def train_generative(model, train_dataset, test_dataset, task_name, config, method_name, num_epochs, lr):
    model.train()
    question = build_prompt_strategy3(task_name)
    batch_size = config.get("batch_size_train", 4)
    if len(get_class_names(task_name)) > 200:
        batch_size = 1

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    grad_accum = config["gradient_accumulation_steps"]
    best_acc = 0.0

    for epoch in range(num_epochs):
        if method_name == "cft" and epoch > 50:
            print("CFT no more than 50 epochs")
            break
        total_loss = 0.0
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        optimizer.zero_grad()
        num_steps = 0

        for batch_start in tqdm(range(0, len(indices), batch_size),
                                 desc=f"  {method_name} E{epoch+1}", leave=False):
            batch_idx = indices[batch_start:batch_start + batch_size]

            batch_messages = []
            for idx in batch_idx:
                img = train_dataset.get_pil_image(idx)
                label = train_dataset.get_label(idx)
                answer_text = get_answer_token(task_name, label)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer_text},
                    ]},
                ]
                batch_messages.append(messages)

            inputs = processor.apply_chat_template(
                batch_messages, tokenize=True, return_dict=True,
                return_tensors="pt", padding=True,
            ).to(model.device)

            labels = inputs["input_ids"].clone()
            for i in range(len(batch_idx)):
                answer_text_i = batch_messages[i][1]["content"][0]["text"]
                answer_ids = processor.tokenizer.encode(answer_text_i, add_special_tokens=False)
                non_pad = inputs["attention_mask"][i].sum().item()
                seq = inputs["input_ids"][i, :int(non_pad)].tolist()

                answer_start = -1
                for j in range(len(seq) - len(answer_ids), -1, -1):
                    if seq[j:j+len(answer_ids)] == answer_ids:
                        answer_start = j
                        break

                labels[i, :] = -100
                if answer_start >= 0:
                    labels[i, answer_start:answer_start + len(answer_ids)] = inputs["input_ids"][i, answer_start:answer_start + len(answer_ids)]

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            num_steps += 1

            if num_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if num_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)
        scheduler.step()

        # Train acc every epoch
        model.eval()
        train_correct = 0
        train_check = min(50, len(train_dataset))
        check_indices = np.random.choice(len(train_dataset), train_check, replace=False)
        for ci in check_indices:
            img = train_dataset.get_pil_image(ci)
            lbl = train_dataset.get_label(ci)
            answer = classify_image(img, build_prompt_strategy3(task_name))
            pred = match_answer_to_class(answer, get_class_names(task_name))
            if pred == lbl:
                train_correct += 1
        train_acc = 100.0 * train_correct / train_check

        # Debug predictions first 3 epochs
        if epoch < 3:
            for di in check_indices[:3]:
                img = train_dataset.get_pil_image(di)
                lbl = train_dataset.get_label(di)
                answer = classify_image(img, build_prompt_strategy3(task_name))
                expected = get_answer_token(task_name, lbl)
                print(f"      Expected: '{expected}' | Got: '{answer[:80]}'")

        # Test acc only selectively
        should_eval_test = (epoch == 0) or (epoch > 2 and epoch % 2 == 0) or (epoch + 1 == num_epochs)
        if should_eval_test:
            acc = evaluate_zero_shot(test_dataset, task_name)
            print(f"    Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Test: {acc:.1f}% {'(best!)' if acc > best_acc else ''}")
            if acc > best_acc:
                best_acc = acc
        else:
            print(f"    Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f} | Train: {train_acc:.1f}%")

        model.train()

    model.eval()
    print(f"    Best acc: {best_acc:.1f}%")
    return model, best_acc


@torch.no_grad()
def extract_features(model, dataset, task_name):
    """Extract last hidden state features from frozen Gemma for linear probe."""
    question = build_prompt_strategy3(task_name)
    features, labels = [], []

    for idx in tqdm(range(len(dataset)), desc="  Extracting features", leave=False):
        img = dataset.get_pil_image(idx)
        label = dataset.get_label(idx)
        messages = build_messages(img, question)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0, -1, :]
        features.append(last_hidden.cpu())
        labels.append(label)

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels


@torch.no_grad()
def evaluate_generative(model, test_dataset, task_name):
    """Evaluate fine-tuned model with Strategy 3 generative inference."""
    return evaluate_zero_shot(test_dataset, task_name, return_confusion=False)


print("✅ Training functions defined.")