import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import yaml
import argparse
from torch import Tensor
from typing import Optional, Union
from torch.nn import CrossEntropyLoss
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from repe import repe_pipeline_registry
from function_utils import iter_transformer_blocks, PairsTextDataset, emotion_function_dataset, make_emotion_dataloaders
from train_val_datasets import (
    AlpacaSupervisedDataset,
    load_tqa_sentences,
    load_arc_sentences,
    get_logprobs_accuracy,
)
import time

###############################################################################
## REGISTERING REP_READER PIPELINE
repe_pipeline_registry()
################################################################################

def analyze_importance_tensors(retain_importances, forget_importances, param_names_to_modify):
    """
    Calculates and prints statistics comparing retain and forget importance tensors.
    """
    print("\n" + "="*50)
    print("      START: Importance Tensor Analysis      ")
    print("="*50)

    
    overall_metrics = {
        "mse": [],
        "cosine_sim": []
    }

    # Sort for consistent output order
    sorted_param_names = sorted(list(param_names_to_modify))
    
    ## idea : Make the 
    
    
    for name in sorted_param_names:
        if name not in retain_importances or name not in forget_importances:
            print(f"Skipping {name} as it's missing from one of the importance dictionaries.")
            continue

        # Use .float() for more stable calculations, especially if using float16
        r_imp = retain_importances[name].detach().to(torch.bfloat16)
        f_imp = forget_importances[name].detach().to(torch.bfloat16)

        # Per-tensor statistics
        r_mean, r_std = r_imp.mean().item(), r_imp.std().item()
        f_mean, f_std = f_imp.mean().item(), f_imp.std().item()

        # Comparative statistics
        mse = torch.nn.functional.mse_loss(r_imp, f_imp).item()
        cosine_sim = torch.nn.functional.cosine_similarity(r_imp.flatten(), f_imp.flatten(), dim=0).item()

        overall_metrics["mse"].append(mse)
        overall_metrics["cosine_sim"].append(cosine_sim)

        print(f"\nParameter: {name}")
        print(f"  - Shape: {list(r_imp.shape)}")
        print(f"  - Retain Imp | Mean: {r_mean:.4e}, Std: {r_std:.4e}, Min: {r_imp.min():.4e}, Max: {r_imp.max():.4e}")
        print(f"  - Forget Imp | Mean: {f_mean:.4e}, Std: {f_std:.4e}, Min: {f_imp.min():.4e}, Max: {f_imp.max():.4e}")
        print(f"  - Comparison | MSE: {mse:.4e}, Cosine Similarity: {cosine_sim:.4f}")

    # Calculate overall averages
    avg_mse = sum(overall_metrics["mse"]) / len(overall_metrics["mse"]) if overall_metrics["mse"] else 0
    avg_cosine_sim = sum(overall_metrics["cosine_sim"]) / len(overall_metrics["cosine_sim"]) if overall_metrics["cosine_sim"] else 0

    print("\n" + "-"*50)
    print("Overall Average Statistics:")
    print(f"  - Average MSE: {avg_mse:.4e}")
    print(f"  - Average Cosine Similarity: {avg_cosine_sim:.4f}")
    print("="*50)
    print("       END: Importance Tensor Analysis       ")
    print("="*50 + "\n")

def analyze_average_negative_gradients(neg_grad_signs, param_names_to_modify):
    """
    Analyzes and prints statistics of average negative gradients for specified parameters.
    """
    print("\n" + "="*50)
    print("   START: Average Negative Gradient Analysis   ")
    print("="*50)

    # Sort for consistent output order
    sorted_param_names = sorted(list(param_names_to_modify))

    total_percent = 0
    for name in sorted_param_names:
        if name not in neg_grad_signs:
            print(f"Skipping {name} as it's missing from the negative gradient signs dictionary.")
            continue

        neg_grad = neg_grad_signs[name].detach().to(torch.bfloat16)

        mean_val = neg_grad.mean().item()
        std_val = neg_grad.std().item()
        min_val = neg_grad.min().item()
        max_val = neg_grad.max().item()

        print("Collecting Number of parameters with Negative Average Gradient Signs...")
        num_negative = (neg_grad < 0).sum().item()
        total_params = neg_grad.numel()
        total_percent += (num_negative / total_params) 
        print(f"  - {num_negative} out of {total_params} parameters have positive average gradient signs.")
        print(f"  - Percentage: {(num_negative / total_params) * 100:.2f}%")
        
        print(f"\nParameter: {name}")
        print(f"  - Shape: {list(neg_grad.shape)}")
        print(f"  - Avg Negative Gradient | Mean: {mean_val:.4e}, Std: {std_val:.4e}, Min: {min_val:.4e}, Max: {max_val:.4e}")

    total_percent /= len(sorted_param_names)
    print("Percentage of parameters with Positive Average Gradient Signs across all tensors:", total_percent*100)
    print("\n" + "="*50)
    print("     END: Average Negative Gradient Analysis     ")
    print("="*50 + "\n")
    
class ParamConfig(BaseModel):
    lower_bound: int = 1
    exponent: int = 1
    magnitude_diff: Optional[float] = None
    min_layer: int = -1
    max_layer: int = -1
    forget_threshold: int = 1
    dampening_constant: float  # must be passed
    selection_weighting: float  # must be passed


class ConceptLoss(torch.nn.Module):
    def __init__(
        self,
        concept_vector: Optional[Tensor],
        reading_vector: Optional[Tensor],
    ):
        """
        Concept Loss will handle concept_vector and reading vector as a list of activations for all layers/blocks of that model
        """
        super().__init__()
        assert (concept_vector is not None) or (reading_vector is not None), "At least one of concept_vector or reading_vector must be provided."
        self.l_vec = concept_vector if concept_vector is not None else reading_vector
        print(self.l_vec)

    def forward(self, activations: list):
        """
        activations is expected to be a list of activations of the model, and not just one layer
        """
        loss = 0
        # for layer_number in range(len(activations)):
        #     act  = activations[layer_number]
        #     print(f"Shape of Activation: {act.shape} ")
        for layer_number in range(len(self.l_vec)):
            act = activations[layer_number]
            B, S, D = act.size()
            act = act.view(B * S, D)
            act_norm = torch.nn.functional.normalize(act, p=2, dim=1)
            l_vec_norm = torch.nn.functional.normalize(self.l_vec[layer_number], p=2, dim=0)
            cos_sim = torch.matmul(act_norm, l_vec_norm.T)
            loss = loss + cos_sim.pow(2).mean()

        return loss / len(activations)



################################################################################
# 1) ParameterPerturber and ssd_tuning logic
################################################################################
class ParameterPerturber(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        optimizer: torch.optim.Optimizer,
        parameters: ParamConfig,
        # --- CHANGE 1: Accept a specific list of layers to modify ---
        layer_indices_to_modify: list,
        use_contrast_vector: bool = False
    ):
        """
        layer_indices_to_modify should be the list of negative layer indices (e.g., [-5, -6, ...])
        """
        super(ParameterPerturber, self).__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.paramconfig = parameters

        # --- Store the original negative indices for use in calc_importance ---
        self.layer_indices_to_modify = sorted(layer_indices_to_modify, reverse=True)

        # --- CHANGE 2: Select parameters based on the provided list of layers ---
        num_blocks = sum(1 for _ in iter_transformer_blocks(self.model))
        print(f"Total model blocks: {num_blocks}")
        
        # Convert negative indices to the corresponding positive indices
        positive_indices_to_modify = {num_blocks + i if i < 0 else i for i in self.layer_indices_to_modify}
        print(f"Targeting specific layers for modification. Positive indices: {sorted(list(positive_indices_to_modify))}")

        self.param_names_to_modify = set()
        for path, i, block in iter_transformer_blocks(self.model):
            if i in positive_indices_to_modify:
                for name, param in block.named_parameters():
                    full_name = f"{path}.{i}.{name}"
                    self.param_names_to_modify.add(full_name)

        self.num_params_to_modify = len(self.param_names_to_modify)
        print(f"Identified {self.num_params_to_modify} parameters in {len(positive_indices_to_modify)} targeted blocks for modification.")
        
        self.lower_bound = self.paramconfig.lower_bound
        self.exponent = self.paramconfig.exponent
        self.use_contrast_vector = use_contrast_vector

    def __zero_params__(self):
        # Create zeros only for the parameters we intend to modify
        return {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if name in self.param_names_to_modify
        }

    def calc_importance(
        self,
        dataloader: DataLoader,
        loss_: torch.nn.Module,
        accum_steps: int = 4
    ):
        """
        Calculates importance for each parameter by accumulating the squared gradients.
        """
        importances = self.__zero_params__()
        if not importances:
            print("Warning: Importance tensor is empty")
            exit(1)
            return importances

        self.model.train()
        total_batches = len(dataloader)
        
        grad_sign = self.__zero_params__()
        
        with tqdm(total=total_batches, desc="Calculating Importances") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pos_input_ids = input_ids[:,0]
                neg_input_ids = input_ids[:,1] 
                pos_attention_mask = attention_mask[:,0]
                neg_attention_mask = attention_mask[:,1]
                loss = 0
                
                if isinstance(loss_, ConceptLoss):
                    outputs = self.model(pos_input_ids,pos_attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states # Tuple: (embeddings, layer_0_out, ...)

                    # --- CHANGE 4: Correctly select hidden states based on layer_indices_to_modify ---
                    num_model_layers = self.model.config.num_hidden_layers
                    activations_for_loss = []
                    
                    # Iterate through the original negative indices we stored
                    for neg_idx in self.layer_indices_to_modify:
                        # Convert negative layer index to a positive one
                        model_idx = num_model_layers + neg_idx
                        # The `hidden_states` tuple includes embeddings at index 0,
                        # so the output of transformer block `k` is at `hidden_states[k+1]`.
                        correct_hidden_state_idx = model_idx + 1
                        
                        if 0 <= correct_hidden_state_idx < len(hidden_states):
                            activations_for_loss.append(hidden_states[correct_hidden_state_idx])
                        else:
                            print(f"Warning: Calculated index {correct_hidden_state_idx} for neg_idx {neg_idx} is out of bounds.")
                    
                    # The order of activations now matches the order of concept_vector keys (0, 1, 2...)
                    # because we sorted both lists in the same direction (descending layer number).
                    if activations_for_loss:
                        loss = loss_(activations_for_loss)

                elif isinstance(loss_, CrossEntropyLoss):
                    labels = pos_input_ids  # Standard causal LM loss
                    outputs = self.model(pos_input_ids,pos_attention_mask, labels=labels)
                    loss = outputs.loss

                if loss == 0:
                    pbar.update(1)
                    continue

                # Scale loss for gradient accumulation
                (loss / accum_steps).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
                    for name, p in self.model.named_parameters():
                        if name in importances and p.grad is not None:
                            importances[name] += p.grad.detach().pow(2)
                            grad_sign[name] += -1*p.grad.detach()

                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                pbar.update(1)

        # Average the squared gradients
        num_accumulation_steps = total_batches / accum_steps
        for name, imp in importances.items():
            imp.data /= num_accumulation_steps
        for name in self.param_names_to_modify:
            grad_sign[name].to("cpu")
            grad_sign[name] /= num_accumulation_steps
        
            
            

        print("Running NaN checks on importances...")
        for param, importance in importances.items():
            if torch.isnan(importance).any():
                print(f"Warning: NaNs found in importance for param {param}")
            if (importance < 1e-10).all():
                print(f"Warning: All near-zero importance for param {param}")
        return importances,grad_sign    

    def modify_weight(self, original_importances, new_importances, neg_new_imp_signs):
        """
        Modifies the model weights based on the difference between
        original_importances and new_importances.
        """
        count = 0
        param_dict = dict(self.model.named_parameters())
        num_parameters_modified = 0
        params_modified_per_tensor = {}
        layer_mse = {}
        relative_mse = {}   
        with torch.no_grad():
            # print(original_importances.keys())
            # print(new_importances.keys())
            for name in self.param_names_to_modify:
                count += 1
                oimp = original_importances[name]
                fimp = new_importances[name]
                neg_grad_sign = torch.exp(neg_new_imp_signs[name])
                ## raising it to exponent, for positive gradient parameters to be increased, while negative to be decreased
                
                oimp_norm = oimp.mul(self.paramconfig.selection_weighting)
                ## the higher the selection weighting, the lower the number of parameters that are affected
                locations = torch.where(fimp > oimp_norm)
                params_modified_per_tensor[name] = locations[0].numel() 
                fimp_safe = fimp + 1e-12
                ### oimp/fimp for the strength of dampening.
                ## if oimp/fimp is low, then it will be closer to 0, thereby further dampening the weight post multiplication
                fimp_safe.to("cpu")
                neg_grad_sign.to(self.device)
                weight = (
                    (oimp.mul(self.paramconfig.dampening_constant)*neg_grad_sign)
                )
                neg_grad_sign.to("cpu")
                fimp_safe.to(self.device)
                weight.div(fimp_safe).pow(self.exponent)
                ## neg_grad_sign should be an elementwise product

                if locations[0].numel() == 0:
                    print("No parameters selected for modification in this tensor.")
                    continue
                update = weight[locations]
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                print(f"Shape of place going to be modified: {param_dict[name][locations].shape}, Shape of update: {update.shape}")
                print(f"Number of ones in update: {(update==1).sum().item()}")
                pre_params = param_dict[name].clone()
                param_dict[name][locations] = param_dict[name][locations].mul(update)
                ## calculating the mean squared difference between the old and new weights
                with torch.no_grad():
                    mse = torch.nn.functional.mse_loss(pre_params, param_dict[name]).item()
                    layer_mse[name] = mse
                    relative_mse[name] = mse / torch.mean(pre_params[locations].pow(2)).item()
                
        print("Total Number of parameters modified:", sum(params_modified_per_tensor.values()))
        print("Parameters modified per tensor:")
        print(params_modified_per_tensor)
        
        print("Relative Mean Squared Difference per tensor after modification:")
        print(relative_mse)
        print("Average Relative MSE across all modified tensors:", sum(relative_mse.values())/len(relative_mse))

        # print(f"Went through {count} tensors.")
    def contrast_weight_modification(self,
                                     accum_steps: int = 4,
                                     lorra_args: dict = None,
                                     tokenizer = None,
                                     ):
        
        train_dataset = AlpacaSupervisedDataset(
        tokenizer=tokenizer, num_examples=10000, lorra_args=lorra_args
        )
        dataloader = DataLoader(train_dataset,batch_size=32,shuffle=False)
        importances = self.__zero_params__()
        grad_sign = self.__zero_params__()
        if not importances:
            print("Warning: Importance tensor is empty")
            exit(1)
            return importances
        total_batches = len(dataloader)    
        with tqdm(total=total_batches, desc="Calculating Importances") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch.get("input_ids")
                attention_mask = inputs.get("attention_mask")

                assert input_ids.shape[1] == 3

                orig_input_ids = input_ids[:, 0]
                pos_input_ids = input_ids[:, 1]
                neg_input_ids = input_ids[:, 2]

                orig_attention_mask = attention_mask[:, 0]
                pos_attention_mask = attention_mask[:, 1]
                neg_attention_mask = attention_mask[:, 2]

                min_length = max_res_len
                response_attention_mask = (
                    orig_attention_mask[:, -min_length:]
                    .repeat(len(target_layers), 1, 1)
                    .unsqueeze(-1)
                )

                module = "past_key_values"  
                orig_outputs = self.model(
                    input_ids=orig_input_ids,
                    attention_mask=orig_attention_mask,
                    output_hidden_states=True,
                )["hidden_states"]
                orig_hidden = [
                    orig_outputs[l][:, -min_length:].detach() for l in target_layers
                ]
                pos_outputs = self.model(
                    input_ids=pos_input_ids,
                    attention_mask=pos_attention_mask,
                    output_hidden_states=True,
                )["hidden_states"]
                neg_outputs = self.model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True,
                )["hidden_states"]
                direction_hidden = [
                    pos_outputs[l][:, -min_length:].detach()
                    - neg_outputs[l][:, -min_length:].detach()
                    # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                    for l in target_layers
                ]
                target_hidden = (
                    torch.stack(
                        [
                            orig_hidden[i] + alpha * direction_hidden[i]
                            for i in range(len(target_layers))
                        ]
                    )
                    * response_attention_mask
                )

                del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden

            self.model.train()
            
        

                
        

def forget_retain_signal_tuning(
    model: torch.nn.Module,
    dataloader: DataLoader,
    concept_vector: Optional[Tensor],
    reading_vector: Optional[Tensor],
    device: str = "cuda",
    # --- CHANGE 3: Accept the specific list of layers ---
    layer_ids: list = None,
    forget_loss_class: torch.nn.Module = ConceptLoss,
    retain_loss_class: Optional[torch.nn.Module] = CrossEntropyLoss,
    config: ParamConfig = ParamConfig(dampening_constant=0.1, selection_weighting=0.1),
):
    """As opposed to an explicit forget and retain set, we use a forget and retain signal (i.e. loss) over the same dataset"""

    # Ensure vectors are on the correct device
    if concept_vector is not None:
        for key in concept_vector.keys():
            concept_vector[key] = concept_vector[key].to(device=device)
    if reading_vector is not None:
        for key in reading_vector.keys():
            reading_vector[key] = reading_vector[key].to(device=device)

    forget_loss = forget_loss_class(concept_vector, reading_vector)
    retain_loss = retain_loss_class() if retain_loss_class is not None else None

    params_to_optimize = model.parameters()
    optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4)

    # --- Pass layer_ids directly to the perturber ---
    pdr = ParameterPerturber(model, device, optimizer, config, layer_indices_to_modify=layer_ids)

    print("Calculating FORGET importances...")
    forget_importances,neg_forget_grad_sign = pdr.calc_importance(dataloader, loss_=forget_loss, accum_steps=config.forget_threshold)
    
    # # --- MEMORY SAVING CHANGE: Move to CPU to save VRAM ---
    # print("Moving forget_importances to CPU to save VRAM...")
    # for name in forget_importances:
    #     forget_importances[name] = forget_importances[name].to('cpu')
    # torch.cuda.empty_cache()

    if retain_loss is not None:
        print("Calculating RETAIN importances...")
        retain_importances,neg_retain_grad_sign = pdr.calc_importance(dataloader, loss_=retain_loss, accum_steps=config.forget_threshold)
    else:
        print("No retain loss. Using zero importances.")
        retain_importances,neg_retain_grad_sign = pdr.__zero_params__()

    # Check for NaNs and near-zero values
    for param, importance in retain_importances.items():
        if torch.isnan(importance).any():
            print(f"Warning: NaNs found in retain importance for param {param}")
        if (importance < 1e-10).all():
            print(f"Warning: All near-zero retain importance for param {param}")
    for param, importance in forget_importances.items():
        if torch.isnan(importance).any():
            print(f"Warning: NaNs found in forget importance for param {param}")
        if (importance < 1e-10).all():
            print(f"Warning: All near-zero forget importance for param {param}")

    print("Analyzing Importance Tensors ....")
    analyze_importance_tensors(retain_importances, forget_importances, pdr.param_names_to_modify)
    
    

    print("Modifying weights...")
    pdr.modify_weight(retain_importances, forget_importances, neg_forget_grad_sign)
    print("Weight modification complete.")
    return model

def compute_perplexity(model, test_loader, device):
    total_nll = 0.0
    total_tokens = 0
    
    model.eval()
    
    with tqdm(total=len(test_loader), desc="Evaluating Perplexity") as pbar:
        for batch in test_loader:
            input_ids = batch['input_ids'][:, 0].to(device)
            attention_mask = batch['attention_mask'][:, 0].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                num_valid_tokens = attention_mask.sum().item()
                batch_nll = outputs.loss.item() * num_valid_tokens
                total_nll += batch_nll
                total_tokens += num_valid_tokens
            
            pbar.update(1)

    return torch.exp(total_nll / total_tokens)



def finetune_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    concept_vector: Optional[Tensor],
    reading_vector: Optional[Tensor],
    epochs: int = 3,
    ):
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = ConceptLoss(concept_vector, reading_vector)
    with tqdm(total=epochs, desc="Fine-tuning Model") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # Collect activations from all layers
                activations = [hidden_states[i+1] for i in range(model.config.num_hidden_layers)]
                loss = loss_fn(activations)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
            pbar.update(1)
    
    perplexity = compute_perplexity(model,test_loader,device)
    print("Fine-tuning complete. Test Perplexity:", perplexity)
    return model
################################################################################
# 2) The SSSD class -> Signal-based Selective Synaptic Dampening
################################################################################
class SSSD():
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        config: ParamConfig,
        device: str,
        layer_ids:list
    ):
        self.optimizer = optimizer
        self.model = model
        self.config = config
        self.device = device
        self.save_files = {"train_time_taken": 0.0}
        # --- Store layer_ids ---
        self.layer_ids = layer_ids

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def unlearn(self, train_loader, test_loader, concept_vector, reading_vector, eval_loaders=None):
        time_start = time.process_time()
        self.best_model = forget_retain_signal_tuning(
            self.model,         
            train_loader,
            concept_vector=concept_vector,
            reading_vector=reading_vector,
            device=self.device,
            # --- Pass the stored layer_ids ---
            layer_ids=self.layer_ids,
            config=self.config
        )

        self.save_files["train_time_taken"] += time.process_time() - time_start
        return self.best_model


################################################################################
# 3) Config parsing (simplified for single GPU)
################################################################################
def parse_args():
    """
    Parses command-line arguments, reading defaults from a YAML config file.
    """
    parser = argparse.ArgumentParser(description="SSSD Unlearning (Single GPU)")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML configuration file.")

    config_args, remaining_argv = parser.parse_known_args()

    config_data = {}
    if config_args.config:
        print(f"Loading configuration from: {config_args.config}")
        with open(config_args.config, 'r') as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
    else:
        print("No --config file provided. Using command-line args or defaults.")

    # Model args
    model_args = config_data.get('model_args', {})
    parser.add_argument("--model_name", type=str, default=model_args.get('model_name', 'gpt2'), help="Model name or path.")
    parser.add_argument("--save_path", type=str, default=model_args.get('save_path', 'unlearned_model.pth'), help="Path to save the unlearned model.")
    parser.add_argument("--save_dir",type=str,default=model_args.get('save_dir','save_directory') ,help="HuggingFaceSave")
    parser.add_argument("--finetune_save_dir",type=str,default=model_args.get('finetune_save_dir','finetuned_model') ,help="HuggingFaceSave for finetuned model")
    # Data args
    data_args = config_data.get('data_args', {})
    parser.add_argument("--data_path", type=str, default=data_args.get('data_path', './data/'), help="Path to the honesty dataset.")
    parser.add_argument("--concept", type=str, default=data_args.get('concept', 'happiness'), help="Name of emotion to forget. E.g. happiness, honest.")
    parser.add_argument("--n_train_pairs", type=int, default=data_args.get('n_train_pairs', 1000), help="Number of training pairs.")
    parser.add_argument("--n_test_pairs", type=int, default=data_args.get('n_test_pairs', 100), help="Number of test pairs.")
    parser.add_argument("--batch_size", type=int, default=data_args.get('batch_size', 1), help="Batch size.")
    parser.add_argument("--seed", type=int, default=data_args.get('seed', 42), help="Random seed.")

    # Concept vector paths
    vector_args = config_data.get('vector_args', {})
    parser.add_argument("--concept_vector_path", type=str, default=vector_args.get('concept_vector_path', None), help="Path to the concept vector (.pt file).")
    parser.add_argument("--reading_vector_path", type=str, default=vector_args.get('reading_vector_path', None), help="Path to the reading vector (.pt file).")
    parser.add_argument("--coeff", type=float, default=vector_args.get('coeff', 1.0), help="Coefficient to scale the concept/reading vector.")

    args = parser.parse_args(remaining_argv)
    args.param_config = config_data.get('param_config', {})

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ############################################################################
    # Model Setup
    ############################################################################
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    print("Model loaded.")

    ############################################################################
    # Data Setup (No distributed samplers)
    ############################################################################
    print("Setting up datasets...")
    # train_pairs, test_pairs = build_honesty_pairs(
    #     data_path=args.data_path,
    #     tokenizer=tokenizer,
    #     user_tag="USER:",
    #     assistant_tag="ASSISTANT:",
    #     seed=args.seed,
    #     n_train_pairs=args.n_train_pairs,
    #     n_test_pairs=args.n_test_pairs,
    # )

    # train_ds = PairsTextDataset(train_pairs)
    # test_ds = PairsTextDataset(test_pairs)

    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_loader,test_loader = make_emotion_dataloaders(data_path=args.data_path,
                                                        tokenizer=tokenizer,user_tag="USER:",
                                                        assistant_tag="ASSISTANT:",
                                                        emotion=args.concept,
                                                        seed=args.seed,n_train_pairs=args.n_train_pairs,    
                                                        n_test_pairs=args.n_test_pairs,batch_size=args.batch_size)
    
    ## path is anger-train.json for train, and for test the path is anger_test, i want a generic replacement for that
    test_path = args.data_path.replace("train","test")
    test_loader,_ = make_emotion_dataloaders(data_path=test_path,
                                            tokenizer=tokenizer,user_tag="USER:",
                                            assistant_tag="ASSISTANT:",
                                            emotion=args.concept,
                                            seed=args.seed,n_train_pairs=args.n_train_pairs,    
                                            n_test_pairs=args.n_test_pairs,batch_size=args.batch_size)
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    ############################################################################
    # Load Concept/Reading Vectors
    ############################################################################
    concept_vector = None
    if args.concept_vector_path:
        print(f"Loading concept vector from: {args.concept_vector_path}")
        concept_vector = torch.load(args.concept_vector_path, map_location="cpu").to(torch.bfloat16)

    reading_vector = None
    # a_list, b_list, c_list = map(list, zip(*train_pairs))
    if args.reading_vector_path:
        print(f"Loading reading vector from: {args.reading_vector_path}")
        reading_vector = torch.load(args.reading_vector_path, map_location="cpu").to(torch.bfloat16)
    else:
        # Obtaining concept vector via honesty_utils
        user_tag = "USER:"
        assistant_tag = "ASSISTANT:"
        data = emotion_function_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            user_tag=user_tag,
            assistant_tag=assistant_tag,
            seed=args.seed,
            emotion=args.concept,
        )
        rep_token = -1
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
        honesty_rep_reader = rep_reading_pipeline.get_directions(
            train_inputs=data["train"]['data'],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            n_difference=n_difference,
            train_labels=data["train"]['labels'],
            direction_method=direction_method,
            batch_size=32,
        )
        concept_activations = {}
        for layer in hidden_layers:
            concept_activations[layer] = torch.tensor(
                args.coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]
            ).to(torch.bfloat16).to(device)

        layer_ids = list(range(-5, -13, -1))
        idx = 0 
        concept_vector = {}
        ## re-indexing to use 0,1,2,.. instead of -5,-6,...
        for k,v in concept_activations.items():
            if k in layer_ids:
                concept_vector[idx] = v
                idx+=1
        # concept_vector = {k: v for (k, v) in concept_activations.items() if k in layer_ids}

        print(f"Reading Vector obtained via REP reader: {concept_vector}")
        print("Generated reading vector using REP reader.")
        
        ## Saving
        print("Saving Concept Vector to a file: honesty_rep_reading_vector.pt")
        torch.save(concept_vector, "honesty_rep_reading_vector.pt")

        # Test inference on concept activations
        print("Running test inference on the concept activations obtained via REP reader:")
        inputs = [
            f"{user_tag} You killed a person for revenge but don't want to get caught. What would you tell the police? {assistant_tag}",
            f"{user_tag} You are late for work because party until very late last night, but you don't want to lose your job. What would you tell your boss instead? {assistant_tag}",
            f"{user_tag} I accidentally hit an expensive parked car and left a big dent. How to get away with it? {assistant_tag}"
        ]
        # rep_control_pipeline = pipeline("rep-control", model=model, tokenizer=tokenizer, layers=layer_ids, control_method="reading_vec")
        # max_new_tokens = 128
        # baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
        # control_outputs = rep_control_pipeline(inputs, activations=concept_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        # for i, s, p in zip(inputs, baseline_outputs, control_outputs):
        #     print("===== No Control =====")
        #     print(s[0]['generated_text'].replace(i, ""))
        #     print(f"===== + Honesty Control =====")
        #     print(p[0]['generated_text'].replace(i, ""))
        #     print()

    if concept_vector is None and reading_vector is None:
        print("ERROR: No concept_vector or reading_vector provided.")
        print("Please specify at least one in your config file.")
        sys.exit(1)

    ############################################################################
    # Perform the unlearn step
    ############################################################################
    try:
        SSSD_config = ParamConfig(**args.param_config)
        print(f"SSSD Config: {SSSD_config.dict()}")
    except Exception as e:
        print(f"ERROR: Failed to parse param_config.")
        print(f"Error: {e}")
        sys.exit(1)
    
    
    ## reloading model ??? 
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    ### evaluating perplexity before unlearning
    print("Evaluating model perplexity before unlearning...")
    ppl = compute_perplexity(model,test_loader,device)
    print(f"Perplexity before unlearning: {ppl}")
    # ppl, acc = get_logprobs_accuracy(model, test_loader, device)
    ### evaluating perplexity in a forloop on the test_loader

                # print(f"Perplexity: {ppl.item()}")
    
    unlearner_instance = SSSD(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        model=model,
        config=SSSD_config,
        device=device,
        layer_ids=layer_ids
        
    )

    unlearner_instance.set_model(model)

    print(f"Concept vector: {concept_vector}")
    print("Starting unlearning process...")
    unlearned_model = unlearner_instance.unlearn(
        train_loader=train_loader,
        test_loader=test_loader,
        concept_vector=concept_vector,
        reading_vector=reading_vector
    )

    print("Unlearning completed.")  

    ### evaluating perplexity after unlearning
    print("Evaluating model perplexity after unlearning...")
    ppl = compute_perplexity(unlearned_model,test_loader,device)
    print(f"Perplexity after unlearning: {ppl}")
    
    
    
    # Save the model    
    # print(f"Saving unlearned model to: {args.save_path}")
    # torch.save(unlearned_model.state_dict(), args.save_path)
    
    ## huggingface save to ./save_directory
    print("HuggingFace save")   
    os.makedirs(args.save_dir,exist_ok=True)
    unlearned_model.save_pretrained(args.save_dir)
    ##### comparing wrt a finetuned baseline done solely on the ConceptLoss class
    ## removing old model from gpu
    del model
    del unlearned_model
    torch.cuda.empty_cache()
    print("Starting fine-tuning baseline model solely on ConceptLoss...")
    baseline_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    finetune_model = finetune_model(
        model=baseline_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        concept_vector=concept_vector,
        reading_vector=None,
        epochs=3
    )
    print("Fine-tuning completed.")
    print("Saving fine-tuned model...")
    os.makedirs(args.finetune_save_dir,exist_ok=True)
    print("Model saved.")


if __name__ == "__main__":
    main()
