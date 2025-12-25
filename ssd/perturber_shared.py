import argparse
import sys
import time
from typing import Optional

import torch
import yaml
from pydantic import BaseModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from function_utils import emotion_function_dataset, iter_transformer_blocks


def analyze_importance_tensors(retain_importances, forget_importances, param_names_to_modify):
    print("\n" + "=" * 50)
    print("      START: Importance Tensor Analysis      ")
    print("=" * 50)

    overall_metrics = {"mse": [], "cosine_sim": []}

    for name in sorted(param_names_to_modify):
        if name not in retain_importances or name not in forget_importances:
            print(f"Skipping {name} as it's missing from one of the importance dictionaries.")
            continue

        retain_tensor = retain_importances[name].detach().to(torch.bfloat16)
        forget_tensor = forget_importances[name].detach().to(torch.bfloat16)

        retain_mean = retain_tensor.mean().item()
        retain_std = retain_tensor.std().item()
        forget_mean = forget_tensor.mean().item()
        forget_std = forget_tensor.std().item()

        mse = torch.nn.functional.mse_loss(retain_tensor, forget_tensor).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            retain_tensor.flatten(),
            forget_tensor.flatten(),
            dim=0,
        ).item()

        overall_metrics["mse"].append(mse)
        overall_metrics["cosine_sim"].append(cosine_sim)

        print(f"\nParameter: {name}")
        print(f"  - Shape: {list(retain_tensor.shape)}")
        print(
            "  - Retain Imp | Mean: "
            f"{retain_mean:.4e}, Std: {retain_std:.4e}, Min: {retain_tensor.min():.4e}, Max: {retain_tensor.max():.4e}"
        )
        print(
            "  - Forget Imp | Mean: "
            f"{forget_mean:.4e}, Std: {forget_std:.4e}, Min: {forget_tensor.min():.4e}, Max: {forget_tensor.max():.4e}"
        )
        print(f"  - Comparison | MSE: {mse:.4e}, Cosine Similarity: {cosine_sim:.4f}")

    avg_mse = sum(overall_metrics["mse"]) / len(overall_metrics["mse"]) if overall_metrics["mse"] else 0
    avg_cosine = (
        sum(overall_metrics["cosine_sim"]) / len(overall_metrics["cosine_sim"])
        if overall_metrics["cosine_sim"]
        else 0
    )

    print("\n" + "-" * 50)
    print("Overall Average Statistics:")
    print(f"  - Average MSE: {avg_mse:.4e}")
    print(f"  - Average Cosine Similarity: {avg_cosine:.4f}")
    print("=" * 50)
    print("       END: Importance Tensor Analysis       ")
    print("=" * 50 + "\n")


def analyze_average_negative_gradients(neg_grad_signs, param_names_to_modify):
    print("\n" + "=" * 50)
    print("   START: Average Negative Gradient Analysis   ")
    print("=" * 50)

    total_percent = 0.0
    sorted_param_names = sorted(param_names_to_modify)
    for name in sorted_param_names:
        if name not in neg_grad_signs:
            print(f"Skipping {name} as it's missing from the negative gradient signs dictionary.")
            continue

        neg_grad = neg_grad_signs[name].detach().to(torch.bfloat16)
        num_negative = (neg_grad < 0).sum().item()
        total_params = neg_grad.numel()
        total_percent += num_negative / total_params

        print("Collecting Number of parameters with Negative Average Gradient Signs...")
        print(f"  - {num_negative} out of {total_params} parameters have positive average gradient signs.")
        print(f"  - Percentage: {(num_negative / total_params) * 100:.2f}%")
        print(f"\nParameter: {name}")
        print(f"  - Shape: {list(neg_grad.shape)}")
        print(
            "  - Avg Negative Gradient | Mean: "
            f"{neg_grad.mean().item():.4e}, Std: {neg_grad.std().item():.4e}, "
            f"Min: {neg_grad.min().item():.4e}, Max: {neg_grad.max().item():.4e}"
        )

    if sorted_param_names:
        total_percent /= len(sorted_param_names)
    print("Percentage of parameters with Positive Average Gradient Signs across all tensors:", total_percent * 100)
    print("\n" + "=" * 50)
    print("     END: Average Negative Gradient Analysis     ")
    print("=" * 50 + "\n")


class ParamConfig(BaseModel):
    lower_bound: int = 1
    exponent: int = 1
    magnitude_diff: Optional[float] = None
    min_layer: int = -1
    max_layer: int = -1
    forget_threshold: int = 1
    dampening_constant: float
    selection_weighting: float


class ConceptLoss(torch.nn.Module):
    def __init__(self, concept_vector: Optional[Tensor], reading_vector: Optional[Tensor]):
        super().__init__()
        assert concept_vector is not None or reading_vector is not None, (
            "At least one of concept_vector or reading_vector must be provided."
        )
        self.l_vec = concept_vector if concept_vector is not None else reading_vector
        print(self.l_vec)

    def forward(self, activations: list):
        loss = 0
        for layer_number in range(len(self.l_vec)):
            act = activations[layer_number]
            batch_size, seq_len, dim = act.size()
            act = act.view(batch_size * seq_len, dim)
            act_norm = torch.nn.functional.normalize(act, p=2, dim=1)
            vec_norm = torch.nn.functional.normalize(self.l_vec[layer_number], p=2, dim=0)
            cos_sim = torch.matmul(act_norm, vec_norm.T)
            loss = loss + cos_sim.pow(2).mean()

        return loss / len(activations)


def move_vector_dict_to_device(vector_dict, device: str):
    if vector_dict is None:
        return None
    for key in vector_dict.keys():
        vector_dict[key] = vector_dict[key].to(device=device)
    return vector_dict


def load_tensor_vector(path: Optional[str]):
    if not path:
        return None
    return torch.load(path, map_location="cpu").to(torch.bfloat16)


def build_emotion_concept_vector(model, tokenizer, data_path: str, concept: str, seed: int, coeff: float, layer_ids: list[int]):
    user_tag = "USER:"
    assistant_tag = "ASSISTANT:"
    data = emotion_function_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        seed=seed,
        emotion=concept,
    )
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    rep_reader = rep_reading_pipeline.get_directions(
        train_inputs=data["train"]["data"],
        rep_token=-1,
        hidden_layers=hidden_layers,
        n_difference=1,
        train_labels=data["train"]["labels"],
        direction_method="pca",
        batch_size=32,
    )

    concept_activations = {}
    for layer in hidden_layers:
        concept_activations[layer] = torch.tensor(
            coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        ).to(torch.bfloat16).to(model.device)

    concept_vector = {}
    idx = 0
    for layer, value in concept_activations.items():
        if layer in layer_ids:
            concept_vector[idx] = value
            idx += 1

    print(f"Reading Vector obtained via REP reader: {concept_vector}")
    print("Generated reading vector using REP reader.")
    return concept_vector


def default_test_data_path(train_path: str):
    return train_path.replace("train", "test", 1) if "train" in train_path else train_path


def parse_common_args(description: str, include_finetune_save_dir: bool = False):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to the YAML configuration file.")
    config_args, remaining_argv = config_parser.parse_known_args()

    config_data = {}
    if config_args.config:
        print(f"Loading configuration from: {config_args.config}")
        with open(config_args.config, "r") as handle:
            try:
                config_data = yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
    else:
        print("No --config file provided. Using command-line args or defaults.")

    parser = argparse.ArgumentParser(description=description, parents=[config_parser])

    model_args = config_data.get("model_args", {})
    data_args = config_data.get("data_args", {})
    vector_args = config_data.get("vector_args", {})

    parser.add_argument("--model_name", type=str, default=model_args.get("model_name", "gpt2"))
    parser.add_argument("--save_path", type=str, default=model_args.get("save_path", "unlearned_model.pth"))
    parser.add_argument("--save_dir", type=str, default=model_args.get("save_dir", "save_directory"))
    if include_finetune_save_dir:
        parser.add_argument(
            "--finetune_save_dir",
            type=str,
            default=model_args.get("finetune_save_dir", "finetuned_model"),
        )

    parser.add_argument("--data_path", type=str, default=data_args.get("data_path", "./data/"))
    parser.add_argument("--concept", type=str, default=data_args.get("concept", "happiness"))
    parser.add_argument("--n_train_pairs", type=int, default=data_args.get("n_train_pairs", 1000))
    parser.add_argument("--n_test_pairs", type=int, default=data_args.get("n_test_pairs", 100))
    parser.add_argument("--batch_size", type=int, default=data_args.get("batch_size", 1))
    parser.add_argument("--seed", type=int, default=data_args.get("seed", 42))

    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default=vector_args.get("concept_vector_path", None),
    )
    parser.add_argument(
        "--reading_vector_path",
        type=str,
        default=vector_args.get("reading_vector_path", None),
    )
    parser.add_argument("--coeff", type=float, default=vector_args.get("coeff", 1.0))

    args = parser.parse_args(remaining_argv)
    args.param_config = config_data.get("param_config", {})
    return args


class BaseParameterPerturber(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        optimizer: torch.optim.Optimizer,
        parameters: ParamConfig,
        layer_indices_to_modify: list,
        use_contrast_vector: bool = False,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.paramconfig = parameters
        self.layer_indices_to_modify = sorted(layer_indices_to_modify, reverse=True)
        self.use_contrast_vector = use_contrast_vector
        self.lower_bound = self.paramconfig.lower_bound
        self.exponent = self.paramconfig.exponent

        num_blocks = sum(1 for _ in iter_transformer_blocks(self.model))
        print(f"Total model blocks: {num_blocks}")
        positive_indices_to_modify = {num_blocks + idx if idx < 0 else idx for idx in self.layer_indices_to_modify}
        print(f"Targeting specific layers for modification. Positive indices: {sorted(positive_indices_to_modify)}")

        self.param_names_to_modify = set()
        for path, index, block in iter_transformer_blocks(self.model):
            if index in positive_indices_to_modify:
                for name, _param in block.named_parameters():
                    self.param_names_to_modify.add(f"{path}.{index}.{name}")

        self.num_params_to_modify = len(self.param_names_to_modify)
        print(
            f"Identified {self.num_params_to_modify} parameters in "
            f"{len(positive_indices_to_modify)} targeted blocks for modification."
        )

    def __zero_params__(self):
        return {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if name in self.param_names_to_modify
        }

    def init_extra_statistics(self):
        return None

    def accumulate_extra_statistics(self, extra_statistics, name: str, grad: torch.Tensor):
        return None

    def finalize_extra_statistics(self, extra_statistics, num_accumulation_steps: float):
        return extra_statistics

    def format_importance_result(self, importances, extra_statistics):
        return importances

    def compute_weight_tensor(self, original_importance, forget_importance, extra_state=None):
        forget_importance_safe = forget_importance + 1e-12
        return (
            (original_importance.mul(self.paramconfig.dampening_constant)).div(forget_importance_safe)
        ).pow(self.exponent)

    def _warn_on_bad_importances(self, importances):
        print("Running NaN checks on importances...")
        for param, importance in importances.items():
            if torch.isnan(importance).any():
                print(f"Warning: NaNs found in importance for param {param}")
            if (importance < 1e-10).all():
                print(f"Warning: All near-zero importance for param {param}")

    def _loss_from_batch(self, batch, loss_fn):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pos_input_ids = input_ids[:, 0]
        pos_attention_mask = attention_mask[:, 0]

        if isinstance(loss_fn, ConceptLoss):
            outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            num_model_layers = self.model.config.num_hidden_layers
            activations_for_loss = []
            for neg_idx in self.layer_indices_to_modify:
                model_idx = num_model_layers + neg_idx
                hidden_state_idx = model_idx + 1
                if 0 <= hidden_state_idx < len(hidden_states):
                    activations_for_loss.append(hidden_states[hidden_state_idx])
                else:
                    print(
                        f"Warning: Calculated index {hidden_state_idx} for neg_idx {neg_idx} is out of bounds."
                    )
            return loss_fn(activations_for_loss) if activations_for_loss else None

        if isinstance(loss_fn, CrossEntropyLoss):
            outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                labels=pos_input_ids,
            )
            return outputs.loss

        return None

    def calc_importance(self, dataloader: DataLoader, loss_: torch.nn.Module, accum_steps: int = 4):
        importances = self.__zero_params__()
        if not importances:
            raise RuntimeError("Importance tensor is empty.")

        self.model.train()
        total_batches = len(dataloader)
        extra_statistics = self.init_extra_statistics()

        with tqdm(total=total_batches, desc="Calculating Importances") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                loss = self._loss_from_batch(batch, loss_)
                if loss is None:
                    pbar.update(1)
                    continue

                (loss / accum_steps).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
                    for name, param in self.model.named_parameters():
                        if name in importances and param.grad is not None:
                            grad = param.grad.detach()
                            importances[name] += grad.pow(2)
                            self.accumulate_extra_statistics(extra_statistics, name, grad)

                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                pbar.update(1)

        num_accumulation_steps = total_batches / accum_steps
        for importance in importances.values():
            importance.data /= num_accumulation_steps
        extra_statistics = self.finalize_extra_statistics(extra_statistics, num_accumulation_steps)

        self._warn_on_bad_importances(importances)
        return self.format_importance_result(importances, extra_statistics)

    def modify_weight(self, original_importances, new_importances, extra_state=None):
        param_dict = dict(self.model.named_parameters())
        params_modified_per_tensor = {}
        relative_mse = {}

        with torch.no_grad():
            for name in self.param_names_to_modify:
                original_importance = original_importances[name]
                forget_importance = new_importances[name]
                locations = torch.where(
                    forget_importance > original_importance.mul(self.paramconfig.selection_weighting)
                )
                params_modified_per_tensor[name] = locations[0].numel()
                if locations[0].numel() == 0:
                    print("No parameters selected for modification in this tensor.")
                    continue

                param_extra_state = extra_state[name] if extra_state is not None else None
                weight = self.compute_weight_tensor(original_importance, forget_importance, param_extra_state)
                update = weight[locations]
                update[torch.where(update > self.lower_bound)] = self.lower_bound

                print(
                    f"Shape of place going to be modified: {param_dict[name][locations].shape}, "
                    f"Shape of update: {update.shape}"
                )
                pre_params = param_dict[name].clone()
                param_dict[name][locations] = param_dict[name][locations].mul(update)
                mse = torch.nn.functional.mse_loss(pre_params, param_dict[name]).item()
                relative_mse[name] = mse / torch.mean(pre_params[locations].pow(2)).item()

        print("Total Number of parameters modified:", sum(params_modified_per_tensor.values()))
        print("Parameters modified per tensor:")
        print(params_modified_per_tensor)
        print("Relative Mean Squared Difference per tensor after modification:")
        print(relative_mse)
        if relative_mse:
            print("Average Relative MSE across all modified tensors:", sum(relative_mse.values()) / len(relative_mse))


class BaseSSSD:
    def __init__(self, optimizer, model, config: ParamConfig, device: str, layer_ids: list, tuning_fn):
        self.optimizer = optimizer
        self.model = model
        self.config = config
        self.device = device
        self.layer_ids = layer_ids
        self.tuning_fn = tuning_fn
        self.save_files = {"train_time_taken": 0.0}

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def unlearn(self, train_loader, test_loader, concept_vector, reading_vector, eval_loaders=None):
        time_start = time.process_time()
        self.best_model = self.tuning_fn(
            self.model,
            train_loader,
            concept_vector=concept_vector,
            reading_vector=reading_vector,
            device=self.device,
            layer_ids=self.layer_ids,
            config=self.config,
        )
        self.save_files["train_time_taken"] += time.process_time() - time_start
        return self.best_model
