import os
import sys
from typing import Optional

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from repe import repe_pipeline_registry

from function_utils import make_emotion_dataloaders
from perturber_shared import (
    BaseParameterPerturber,
    BaseSSSD,
    ConceptLoss,
    ParamConfig,
    analyze_importance_tensors,
    build_emotion_concept_vector,
    load_tensor_vector,
    move_vector_dict_to_device,
    parse_common_args,
)


repe_pipeline_registry()


class ParameterPerturber(BaseParameterPerturber):
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
    forget_importances = perturber.calc_importance(
        dataloader,
        loss_=forget_loss,
        accum_steps=config.forget_threshold,
    )

    if retain_loss is not None:
        print("Calculating RETAIN importances...")
        retain_importances = perturber.calc_importance(
            dataloader,
            loss_=retain_loss,
            accum_steps=config.forget_threshold,
        )
    else:
        print("No retain loss. Using zero importances.")
        retain_importances = perturber.__zero_params__()

    analyze_importance_tensors(retain_importances, forget_importances, perturber.param_names_to_modify)

    print("Modifying weights...")
    perturber.modify_weight(retain_importances, forget_importances)
    print("Weight modification complete.")
    return model


class SSSD(BaseSSSD):
    def __init__(self, optimizer, model, config: ParamConfig, device: str, layer_ids: list):
        super().__init__(optimizer, model, config, device, layer_ids, forget_retain_signal_tuning)


def parse_args():
    return parse_common_args("SSSD Unlearning (Single GPU)")


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
    train_loader, test_loader = make_emotion_dataloaders(
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

    print(f"Saving unlearned model to: {args.save_path}")
    torch.save(unlearned_model.state_dict(), args.save_path)
    print("HuggingFace save")
    os.makedirs(args.save_dir, exist_ok=True)
    unlearned_model.save_pretrained(args.save_dir)
    print("Model saved.")


if __name__ == "__main__":
    main()
