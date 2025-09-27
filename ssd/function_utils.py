import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
import torch.nn as nn
from transformers import PreTrainedTokenizer

def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    ## shuffle and append the index of the true statement
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }


def _join_nonempty(parts: List[str]) -> str:
    """Join strings with single spaces, skipping empties."""
    return " ".join(p for p in parts if p and str(p).strip())


def _tokens_to_text(tokenizer: PreTrainedTokenizerBase, tokens: List[str]) -> str:
    """Convert tokens back to text in a tokenizer-appropriate way."""
    if hasattr(tokenizer, "convert_tokens_to_string"):
        return tokenizer.convert_tokens_to_string(tokens)
    return " ".join(tokens)


def iter_transformer_blocks(model, part: str = "auto"):
    """
    Yields (path, idx, block_module) for each Transformer block.

    part: "auto" | "encoder" | "decoder"
    """
    m = model

    # ---- decoder-only families ----
    # GPT-2 / Falcon
    if hasattr(m, "transformer") and isinstance(getattr(m.transformer, "h", None), nn.ModuleList):
        for i, blk in enumerate(m.transformer.h):
            if part in ("auto", "decoder"):
                yield "transformer.h", i, blk
        return

    # LLaMA / Mistral / many Meta-based
    if hasattr(m, "model") and isinstance(getattr(m.model, "layers", None), nn.ModuleList):
        for i, blk in enumerate(m.model.layers):
            if part in ("auto", "decoder"):
                yield "model.layers", i, blk
        return

    # GPT-NeoX
    if hasattr(m, "gpt_neox") and isinstance(getattr(m.gpt_neox, "layers", None), nn.ModuleList):
        for i, blk in enumerate(m.gpt_neox.layers):
            if part in ("auto", "decoder"):
                yield "gpt_neox.layers", i, blk
        return

    # MPT
    if hasattr(m, "transformer") and isinstance(getattr(m.transformer, "blocks", None), nn.ModuleList):
        for i, blk in enumerate(m.transformer.blocks):
            if part in ("auto", "decoder"):
                yield "transformer.blocks", i, blk
        return

    # ---- encoder–decoder families ----
    # T5
    if hasattr(m, "encoder") and isinstance(getattr(m.encoder, "block", None), nn.ModuleList):
        if part in ("auto", "encoder"):
            for i, blk in enumerate(m.encoder.block):
                yield "encoder.block", i, blk
    if hasattr(m, "decoder") and isinstance(getattr(m.decoder, "block", None), nn.ModuleList):
        if part in ("auto", "decoder"):
            for i, blk in enumerate(m.decoder.block):
                yield "decoder.block", i, blk
        return

    # BART / Marian
    if hasattr(m, "model"):
        enc = getattr(m.model, "encoder", None)
        dec = getattr(m.model, "decoder", None)
        if enc is not None and isinstance(getattr(enc, "layers", None), nn.ModuleList):
            if part in ("auto", "encoder"):
                for i, blk in enumerate(enc.layers):
                    yield "model.encoder.layers", i, blk
        if dec is not None and isinstance(getattr(dec, "layers", None), nn.ModuleList):
            if part in ("auto", "decoder"):
                for i, blk in enumerate(dec.layers):
                    yield "model.decoder.layers", i, blk
        if enc or dec:
            return

    # ---- encoder-only families ----
    # BERT / RoBERTa / DeBERTa / ViTText-like
    # (BERT/Roberta: bert.encoder.layer / roberta.encoder.layer)
    for root_name in ("bert.encoder.layer", "roberta.encoder.layer", "deberta.encoder.layer",
                      "encoder.layer", "vision_model.encoder.layers"):
        try:
            mod = m.get_submodule(root_name)
            if isinstance(mod, nn.ModuleList):
                for i, blk in enumerate(mod):
                    yield root_name, i, blk
                return
        except Exception:
            pass

    # Fallback: find any ModuleList whose length matches known layer counts
    candidates = []
    wanted_counts = {
        "num_hidden_layers": getattr(m.config, "num_hidden_layers", None),
        "encoder_layers": getattr(m.config, "encoder_layers", None),
        "decoder_layers": getattr(m.config, "decoder_layers", None),
        "num_layers": getattr(m.config, "num_layers", None),            # T5 encoder
        "num_decoder_layers": getattr(m.config, "num_decoder_layers", None),
    }
    for name, module in m.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            candidates.append((name, module))
    for name, module in candidates:
        if len(module) in {v for v in wanted_counts.values() if isinstance(v, int)}:
            for i, blk in enumerate(module):
                yield name, i, blk
            return
            



def build_honesty_pairs(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    user_tag: str = "",
    assistant_tag: str = "",
    template: str = "Pretend you're {type} person making statements about the world.",
    seed: int = 0,
    min_tail_tokens: int = 5,
    min_prefix_tokens: int = 3,
    prefixes_per_statement: int = 4,
    n_train_pairs: int = 512,
    n_test_pairs: Optional[int] = None,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Build (text0, text1, label) pairs for train and test.
      - label is the index of the *honest* option in the pair (0 or 1).
      - Train pairs are [honest, untruthful] but randomly swapped per-example.
      - Test pairs are [honest[i], untruthful[i+1]] so the first is always honest (label=0).

    Notes:
      - We only need 'true' statements to construct truncated prefixes.
      - The CSV must contain columns: 'statement' and 'label' (1 for true, 0 for false).
        False statements are not required by this construction.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Load data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1]["statement"].dropna().astype(str).tolist()

    honest_texts: List[str] = []
    untruthful_texts: List[str] = []

    for stmt in true_statements:
        tokens = tokenizer.tokenize(stmt)
        L = len(tokens)
        # Need room for a tail to avoid trivial completions (like your original -5)
        if L - min_tail_tokens <= min_prefix_tokens:
            continue

        # Choose prefix cut points
        valid_idxs = np.arange(min_prefix_tokens, L - min_tail_tokens)
        k = min(prefixes_per_statement, len(valid_idxs))
        if k <= 0:
            continue
        idxs = rng.choice(valid_idxs, size=k, replace=False)
        idxs.sort()

        for idx in idxs:
            truncated_tokens = tokens[: int(idx)]
            truncated_text = _tokens_to_text(tokenizer, truncated_tokens)

            honest_prompt = _join_nonempty([
                user_tag,
                template.format(type="an honest"),
                assistant_tag,
                truncated_text,
            ])
            untruthful_prompt = _join_nonempty([
                user_tag,
                template.format(type="an untruthful"),
                assistant_tag,
                truncated_text,
            ])

            honest_texts.append(honest_prompt)
            untruthful_texts.append(untruthful_prompt)

    N = len(honest_texts)
    if N < 2:
        raise ValueError(
            f"Not enough prefixes created ({N}). Try lowering min_prefix_tokens, "
            f"min_tail_tokens, or increasing prefixes_per_statement."
        )

    # shuffle in a reproducible way (but keep honest/untruthful aligned)
    order = rng.permutation(N)
    honest_texts = [honest_texts[i] for i in order]
    untruthful_texts = [untruthful_texts[i] for i in order]

    # ----- TRAIN -----
    n_train = min(n_train_pairs, N - 1)  # leave at least 1 for test alignment
    train_pairs: List[Tuple[str, str, int]] = []
    for i in range(n_train):
        h = honest_texts[i]
        u = untruthful_texts[i]
        # Randomly decide whether to swap, label is index of honest
        if rng.integers(2) == 0:
            pair = (h, u, 0)  # honest first
        else:
            pair = (u, h, 1)  # honest second
        train_pairs.append(pair)

    # ----- TEST -----
    # If n_test_pairs isn't provided, try to mirror n_train but cap by availability
    if n_test_pairs is None:
        n_test_pairs = min(n_train, N - n_train - 1)
    else:
        n_test_pairs = min(n_test_pairs, N - n_train - 1)

    test_pairs: List[Tuple[str, str, int]] = []
    start = n_train
    end = start + n_test_pairs
    for i in range(start, end):
        j = i + 1  # misalignment; guaranteed < N by construction above
        # honest first, untruthful from next index
        test_pairs.append((honest_texts[i], untruthful_texts[j], 0))

    return train_pairs, test_pairs


# ---------- dataset & collate ----------

class PairsTextDataset(Dataset):
    """
    Holds pairs of texts and a label indicating which option (0 or 1) is the honest one.
    Returns dicts with 'text0', 'text1', 'label' (int).
    """
    def __init__(self, pairs: List[Tuple[str, str, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        t0, t1, lbl = self.pairs[idx]
        return {"text0": t0, "text1": t1, "label": int(lbl)}


def make_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: Optional[int] = None,
    return_text: bool = False,
):
    """
    Collate into (B, 2, L) padded tensors + labels of shape (B,).
    """
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        labels: List[int] = []
        for item in batch:
            texts.append(item["text0"])
            texts.append(item["text1"])
            labels.append(int(item["label"]))

        enc = tokenizer(
            texts,
            padding=True,
            truncation=(max_length is not None),
            max_length=max_length,
            return_tensors="pt",
        )
        B = len(batch)
        # reshape to (B, 2, L)
        out = {
            "input_ids": enc["input_ids"].view(B, 2, -1),
            "attention_mask": enc["attention_mask"].view(B, 2, -1),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # include token_type_ids if present (e.g., BERT-like tokenizers)
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].view(B, 2, -1)

        if return_text:
            out["text0"] = [item["text0"] for item in batch]
            out["text1"] = [item["text1"] for item in batch]

        return out
    return _collate


# ---------- top-level convenience ----------

def make_honesty_dataloaders(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    user_tag: str = "",
    assistant_tag: str = "",
    template: str = "Pretend you're {type} person making statements about the world.",
    seed: int = 0,
    min_tail_tokens: int = 5,
    min_prefix_tokens: int = 3,
    prefixes_per_statement: int = 4,
    n_train_pairs: int = 512,
    n_test_pairs: Optional[int] = None,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    return_text_in_batch: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders for the honesty-pair task.
    Each batch contains:
      - input_ids:     (B, 2, L)
      - attention_mask:(B, 2, L)
      - labels:        (B,)   (0 or 1; index of honest option)
    """
    train_pairs, test_pairs = build_honesty_pairs(
        data_path=data_path,
        tokenizer=tokenizer,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        template=template,
        seed=seed,
        min_tail_tokens=min_tail_tokens,
        min_prefix_tokens=min_prefix_tokens,
        prefixes_per_statement=prefixes_per_statement,
        n_train_pairs=n_train_pairs,
        n_test_pairs=n_test_pairs,
    )

    train_ds = PairsTextDataset(train_pairs)
    test_ds = PairsTextDataset(test_pairs)

    collate = make_collate_fn(
        tokenizer,
        max_length=max_length,
        return_text=return_text_in_batch,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )

    return train_loader, test_loader



############## USAGE ####################
##### TODO: MAKE PYTESTS #####


# from transformers import AutoTokenizer

# # tok = AutoTokenizer.from_pretrained("gpt2")  # or any HF tokenizer
# # train_loader, test_loader = make_honesty_dataloaders(
# #     data_path="honesty.csv",
# #     tokenizer=tok,
# #     user_tag="<|user|>",
# #     assistant_tag="<|assistant|>",
# #     seed=42,
# #     prefixes_per_statement=3,
# #     n_train_pairs=512,
# #     batch_size=16,
# #     max_length=256,
# # )

# for batch in train_loader:
#     # batch["input_ids"]: (B, 2, L)
#     # batch["labels"]: (B,) -> index of the honest option
#     # Feed each option through your model, score them, and apply a 2-way loss.
#     pass
 