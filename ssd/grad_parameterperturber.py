import math
import os
import sys
from typing import Optional

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from repe import repe_pipeline_registry

from function_utils import make_emotion_dataloaders
from perturber_shared import (
    BaseParameterPerturber,
    BaseSSSD,
    ConceptLoss,
    ParamConfig,
    analyze_average_negative_gradients,
    analyze_importance_tensors,
    build_emotion_concept_vector,
    default_test_data_path,
    load_tensor_vector,
    move_vector_dict_to_device,
    parse_common_args,
)


repe_pipeline_registry()


class ParameterPerturber(BaseParameterPerturber):
    def init_extra_statistics(self):
        return self.__zero_params__()

    def accumulate_extra_statistics(self, extra_statistics, name: str, grad: torch.Tensor):
        if extra_statistics is not None:
            extra_statistics[name] += -1 * grad

    def finalize_extra_statistics(self, extra_statistics, num_accumulation_steps: float):
        if extra_statistics is None:
            return None
        for tensor in extra_statistics.values():
            tensor /= num_accumulation_steps
        return extra_statistics

    def format_importance_result(self, importances, extra_statistics):
        return importances, extra_statistics

    def compute_weight_tensor(self, original_importance, forget_importance, extra_state=None):
        grad_factor = torch.exp(extra_state) if extra_state is not None else 1
        forget_importance_safe = forget_importance + 1e-12
        return (
            (original_importance.mul(self.paramconfig.dampening_constant) * grad_factor)
            .div(forget_importance_safe)
            .pow(self.exponent)
        )

    def modify_weight(self, original_importances, new_importances, extra_state=None):
        super().modify_weight(original_importances, new_importances, extra_state=extra_state)
        if extra_state is not None:
            analyze_average_negative_gradients(extra_state, self.param_names_to_modify)


class LowRankParameterPerturber(ParameterPerturber):
    pass


def forget_retain_signal_tuning(
    model: torch.nn.Module,
    dataloader: DataLoader,
    concept_vector: Optional[Tensor],
    reading_vector: Optional[Tensor],
    device: str = "cuda",
    layer_ids: list | None = None,
    forget_loss_class: torch.nn.Module = ConceptLoss,
    retain_loss_class: Optional[torch.nn.Module] = CrossEntropyLoss,
    config: ParamConfig = ParamConfig(dampening_constant=0.1, selection_weighting=0.1),
):
    move_vector_dict_to_device(concept_vector, device)
    move_vector_dict_to_device(reading_vector, device)

    forget_loss = forget_loss_class(concept_vector, reading_vector)
    retain_loss = retain_loss_class() if retain_loss_class is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    perturber = ParameterPerturber(
        model,
        device,
        optimizer,
        config,
        layer_indices_to_modify=layer_ids,
    )

    print("Calculating FORGET importances...")
    forget_importances, neg_forget_grad_sign = perturber.calc_importance(
        dataloader,
        loss_=forget_loss,
        accum_steps=config.forget_threshold,
    )

    if retain_loss is not None:
        print("Calculating RETAIN importances...")
        retain_importances, _neg_retain_grad_sign = perturber.calc_importance(
            dataloader,
            loss_=retain_loss,
            accum_steps=config.forget_threshold,
        )
    else:
        print("No retain loss. Using zero importances.")
        retain_importances = perturber.__zero_params__()

    analyze_importance_tensors(retain_importances, forget_importances, perturber.param_names_to_modify)

    print("Modifying weights...")
    perturber.modify_weight(retain_importances, forget_importances, extra_state=neg_forget_grad_sign)
    print("Weight modification complete.")
    return model


def compute_perplexity(model, test_loader, device):
    total_nll = 0.0
    total_tokens = 0
    model.eval()

    with tqdm(total=len(test_loader), desc="Evaluating Perplexity") as pbar:
        for batch in test_loader:
            input_ids = batch["input_ids"][:, 0].to(device)
            attention_mask = batch["attention_mask"][:, 0].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                num_valid_tokens = attention_mask.sum().item()
                total_nll += outputs.loss.item() * num_valid_tokens
                total_tokens += num_valid_tokens
            pbar.update(1)

    return math.exp(total_nll / total_tokens)


def finetune_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    concept_vector: Optional[Tensor],
    reading_vector: Optional[Tensor] = None,
    epochs: int = 3,
    num_epochs: Optional[int] = None,
):
    if num_epochs is not None:
        epochs = num_epochs

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = ConceptLoss(concept_vector, reading_vector)

    with tqdm(total=epochs, desc="Fine-tuning Model") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"][:, 0].to(device)
                attention_mask = batch["attention_mask"][:, 0].to(device)
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states
                activations = [hidden_states[i + 1] for i in range(model.config.num_hidden_layers)]
                loss = loss_fn(activations)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}")
            pbar.update(1)

    perplexity = compute_perplexity(model, test_loader, device)
    print("Fine-tuning complete. Test Perplexity:", perplexity)
    return model


class SSSD(BaseSSSD):
    def __init__(self, optimizer, model, config: ParamConfig, device: str, layer_ids: list):
        super().__init__(optimizer, model, config, device, layer_ids, forget_retain_signal_tuning)


def parse_args():
    return parse_common_args("SSSD Unlearning (Single GPU)", include_finetune_save_dir=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print("Model loaded.")

    print("Setting up datasets...")
    train_loader, _ = make_emotion_dataloaders(
        data_path=args.data_path,
        tokenizer=tokenizer,
        user_tag="USER:",
        assistant_tag="ASSISTANT:",
        emotion=args.concept,
        seed=args.seed,
        n_train_pairs=args.n_train_pairs,
        n_test_pairs=args.n_test_pairs,
        batch_size=args.batch_size,
    )
    test_path = default_test_data_path(args.data_path)
    test_loader, _ = make_emotion_dataloaders(
        data_path=test_path,
        tokenizer=tokenizer,
        user_tag="USER:",
        assistant_tag="ASSISTANT:",
        emotion=args.concept,
        seed=args.seed,
        n_train_pairs=args.n_train_pairs,
        n_test_pairs=args.n_test_pairs,
        batch_size=args.batch_size,
    )
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    layer_ids = list(range(-5, -13, -1))
    concept_vector = load_tensor_vector(args.concept_vector_path)
    reading_vector = load_tensor_vector(args.reading_vector_path)

    if reading_vector is None:
        concept_vector = build_emotion_concept_vector(
            model=model,
            tokenizer=tokenizer,
            data_path=args.data_path,
            concept=args.concept,
            seed=args.seed,
            coeff=args.coeff,
            layer_ids=layer_ids,
        )
        print("Saving Concept Vector to a file: honesty_rep_reading_vector.pt")
        torch.save(concept_vector, "honesty_rep_reading_vector.pt")

    if concept_vector is None and reading_vector is None:
        print("ERROR: No concept_vector or reading_vector provided.")
        print("Please specify at least one in your config file.")
        sys.exit(1)

    try:
        sssd_config = ParamConfig(**args.param_config)
        print(f"SSSD Config: {sssd_config.dict()}")
    except Exception as exc:
        print("ERROR: Failed to parse param_config.")
        print(f"Error: {exc}")
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    print("Evaluating model perplexity before unlearning...")
    print(f"Perplexity before unlearning: {compute_perplexity(model, test_loader, device)}")

    unlearner = SSSD(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        model=model,
        config=sssd_config,
        device=device,
        layer_ids=layer_ids,
    )
    unlearner.set_model(model)

    print(f"Concept vector: {concept_vector}")
    print("Starting unlearning process...")
    unlearned_model = unlearner.unlearn(
        train_loader=train_loader,
        test_loader=test_loader,
        concept_vector=concept_vector,
        reading_vector=reading_vector,
    )
    print("Unlearning completed.")

    print("Evaluating model perplexity after unlearning...")
    print(f"Perplexity after unlearning: {compute_perplexity(unlearned_model, test_loader, device)}")

    print("HuggingFace save")
    os.makedirs(args.save_dir, exist_ok=True)
    unlearned_model.save_pretrained(args.save_dir)
    del model
    del unlearned_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
