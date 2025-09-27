import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import builtins as __builtins__
import sys
import copy
import time
import itertools
import numpy as np
import yaml # Added for config file parsing
import argparse # Added for arg parsing
from function_utils import iter_transformer_blocks, make_honesty_dataloaders, build_honesty_pairs, PairsTextDataset,honesty_function_dataset
from torch import Tensor
from typing import Optional,Union
from torch.nn import CrossEntropyLoss
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from repe import repe_pipeline_registry

###############################################################################
## REGISTERING REP_READER PIPELINE
repe_pipeline_registry()
################################################################################





class ParamConfig(BaseModel):
    lower_bound: int = 1
    exponent: int = 1
    magnitude_diff: Optional[float] = None
    min_layer: int = -1
    max_layer: int = -1
    forget_threshold: int = 1
    dampening_constant: float # must be passed
    selection_weighting: float # must be passed

class ConceptLoss(torch.nn.Module):
    def __init__(self,
                 concept_vector:Optional[Tensor],
                 reading_vector:Optional[Tensor],):
        '''
        Concept Loss will handle concept_vector and reading vector as a list of activations for all layers/blocks of that model
        
        '''
        super().__init__() # Added super() call
        assert (concept_vector is not None) or (reading_vector is not None), "At least one of concept_vector or reading_vector must be provided."
        self.l_vec = concept_vector if concept_vector is not None else reading_vector
    
    def forward(self,activations:list):
        '''
        activations is expected to be a list of activatios of the model, and not just one layer
        '''
        # Reshape activations: (B, S, D) -> (B*S, D)
        loss=0
        for layer_number in len(activations):
            act = activations[layer_number]
            B, S, D = act.size()
            act = act.view(B * S, D)
            act_norm = torch.nn.functional.normalize(act, p=2, dim=1)
            l_vec_norm = torch.nn.functional.normalize(self.l_vec[layer_number], p=2, dim=0)
            cos_sim = torch.matmul(act_norm, l_vec_norm)
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
        layer_number:Union[int,slice]=-1,
    ):
        '''
        layer number can be an int for a specific layer, or a slice, for the range of layers that it wants to be
        '''
        super(ParameterPerturber, self).__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.paramconfig = parameters  # Store config dict

        self.layer_number = layer_number
        
        ## we only include till the layer_number : layer_number is negative with 0 as the first layer
        self.paramset = {}
        num_blocks = sum(1 for _ in iter_transformer_blocks(self.model.module)) # Use .module for DDP
        print(f"Total model blocks: {num_blocks}")

        # Determine the target layer index
        # layer_number = -1 means the last block, -2 means second to last, etc.
        # 0 means the first block (index 0)
        if layer_number < 0:
            target_layer_index = num_blocks + layer_number
        else:
            target_layer_index = layer_number
        
        print(f"Targeting layers up to (but not including) index: {target_layer_index} (layer_number arg: {layer_number})")

        self.param_names_to_modify = set()
        count = 0
        for path, i, block in iter_transformer_blocks(self.model.module): # Use .module for DDP
            if i < target_layer_index:
                for name, param in block.named_parameters():
                    full_name = f"{path}.{name}"
                    self.param_names_to_modify.add(full_name)
                    count += 1
            
        self.num_params_to_modify = len(self.param_names_to_modify)
        print(f"Identified {self.num_params_to_modify} parameters in {target_layer_index} blocks for modification.")

        self.lower_bound = self.paramconfig.get("lower_bound", 1)
        self.exponent = self.paramconfig.get("exponent", 1)

    def __zero_params__(self):
        # Create zeros only for the parameters we intend to modify
        return {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if name in self.param_names_to_modify
        }

    def calc_importance(self,
                        dataloader: DataLoader,
                        loss_:torch.nn.Module, ## this can be either concept loss or cross entropy loss(retain)
                        accum_steps:int=4):
        """
        Calculates importance for each parameter by accumulating the squared
        gradients across the data in the dataloader.
        """
        importances = self.__zero_params__()
        if not importances:
            print("Warning: No parameters selected for importance calculation. Returning empty importances.")
            return importances
            
        self.model.train()
        batch_counter = 0
        total_batches = len(dataloader)
        
        with tqdm(total=total_batches, desc="Calculating Importances", disable=(dist.get_rank() != 0)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # self.optimizer.zero_grad(set_to_none=True) # Zero grad inside loop
                input_ids = batch['input_ids'].to(self.device)
                
                # Note: DDP handles the all-reduce automatically
                loss = 0
                if isinstance(loss_, ConceptLoss):
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    
                    ## TODO: huh ?  
                    # Ensure layer_number indexing is correct
                    # hidden_states[0] is embeddings, [1] is block 0, ... [-1] is final output
                    # If user passes -1, they mean the *last* hidden state (before LM head)
                    # If they pass 0, they mean the *first* block's output (index 1)
                    # Let's clarify:
                    # hidden_states index 0: input embeddings
                    # hidden_states index 1: output of block 0
                    # ...
                    # hidden_states index N: output of block N-1a
                    # hidden_states index N+1 (or -1): final output (if model returns it)
                    
                    # The original code used self.layer_number directly.
                    # outputs.hidden_states has length num_layers + 1 (embeddings + N layers)
                    # So hidden_states[self.layer_number] should work if -1 means last layer output
                    
                    
                    selected_hidden_state = hidden_states[self.layer_number]
                    ## pass in a list of the hidden_states to optimize ove
                    loss = loss_(selected_hidden_state)
                
                elif isinstance(loss_, CrossEntropyLoss):
                    labels = input_ids # Standard causal LM loss
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss
                
                if loss == 0:
                    pbar.update(1)
                    continue

                # Scale loss for gradient accumulation
                (loss / accum_steps).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
                    for name, p in self.model.named_parameters():
                        if name in importances and p.grad is not None:
                            # DDP averages gradients, so pow(2) is on the averaged grad
                            importances[name] += p.grad.detach().pow(2)
                            
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                pbar.update(1)

        # Average the squared gradients
        # We accumulated sums of squared *averaged* gradients
        # We need to average this sum
        num_accumulation_steps = total_batches / accum_steps
        for name, imp in importances.items():
            imp.data /= num_accumulation_steps

        return importances

    def modify_weight(self, original_importances, new_importances):
        """
        Modifies the model weights based on the difference between
        original_importances and new_importances, following your SSD scheme.
        """
        count = 0
        with torch.no_grad():
            for (name, p) in self.model.named_parameters():
                if name in self.param_names_to_modify:
                    count += 1
                    oimp = original_importances[name]
                    fimp = new_importances[name]
                    
                    # Synapse Selection with parameter alpha
                    oimp_norm = oimp.mul(self.paramconfig["selection_weighting"])
                    locations = torch.where(fimp > oimp_norm)

                    # Dampening
                    # Add epsilon to fimp denominator to avoid div by zero
                    fimp_safe = fimp + 1e-12 
                    weight = (
                        (oimp.mul(self.paramconfig["dampening_constant"])).div(fimp_safe)
                    ).pow(self.exponent)
                    
                    if locations[0].numel() == 0: # Check if locations is empty
                        continue
                    update = weight[locations]
                    min_locs = torch.where(update > self.lower_bound)
                    update[min_locs] = self.lower_bound

                    p[locations] = p[locations].mul(update)
        
        print(f"Modified {count} parameter tensors.")


def forget_retain_signal_tuning(
    model:torch.nn.Module,
    dataloader:DataLoader,
    concept_vector:Optional[Tensor],
    reading_vector:Optional[Tensor],
    device:str="cuda",
    layer_number:int=-1,
    forget_loss_class:torch.nn.Module=ConceptLoss,
    retain_loss_class:Optional[torch.nn.Module]=CrossEntropyLoss,
    config:ParamConfig=ParamConfig(dampening_constant=0.1,selection_weighting=0.1),):
    '''As oppposed to an explicit forget and retain set, we use a forget and retain signal (ie:loss) over the same dataset'''

    # Ensure vectors are on the correct device *before* passing to loss
    
    if concept_vector is not None:
        for keys in concept_vector.keys():
            concept_vector[keys] = concept_vector[keys].to(device=device)        
    if reading_vector is not None:
        for keys in reading_vector.keys():
            reading_vector[keys] = reading_vector[keys].to(device=device)  

    forget_loss = forget_loss_class(concept_vector, reading_vector)
    retain_loss = retain_loss_class() if retain_loss_class is not None else None
    
    # Pass model.module.parameters() to optimizer if it's a DDP model
    params_to_optimize = model.parameters()
    optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4) # lr doesn't matter for SSD
    
    pdr = ParameterPerturber(model, device, optimizer, config.dict(), layer_number)
    
    print("Calculating FORGET importances...")
    forget_importances = pdr.calc_importance(dataloader, loss_=forget_loss, accum_steps=config.forget_threshold)
    
    if retain_loss is not None:
        print("Calculating RETAIN importances...")
        retain_importances = pdr.calc_importance(dataloader, loss_=retain_loss, accum_steps=config.forget_threshold)
    else:
        print("No retain loss. Using zero importances.")
        retain_importances = pdr.__zero_params__()
        
    print("Modifying weights...")
    pdr.modify_weight(retain_importances, forget_importances)
    print("Weight modification complete.")
    return model



################################################################################
# 3) The SSSD class -> Signal-based Selective Synaptic Dampening
################################################################################
class SSSD():
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        config: ParamConfig,
        device: str
    ):
        self.optimizer = optimizer
        self.model = model
        self.config = config
        self.device = device
        # Track times, etc.
        self.save_files = {"train_time_taken": 0.0}

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        
    def unlearn(self, 
                train_loader, 
                test_loader, 
                concept_vector, 
                reading_vector, 
                eval_loaders=None):
        
        time_start = time.process_time()
        self.best_model = forget_retain_signal_tuning(
            self.model,
            train_loader,
            concept_vector=concept_vector,
            reading_vector=reading_vector,
            device=self.device,
            layer_number=self.config.min_layer,
            config=self.config
        )
        
        #### POSSIBLE POST-PROCESSING: EVALUATE ON TEST SETS, SAVE BEST MODEL, ETC.#####
        ## TODO: Add evaluation logic here if needed
        ################################################################################
        self.save_files["train_time_taken"] += time.process_time() - time_start
        return self.best_model
    
################################################################################
# 3) Distributed main
################################################################################
def parse_args():
    """
    Parses command-line arguments, reading defaults from a YAML config file
    if specified.
    """
    
    ##### TODO: REMOVE THE CLI FORM OF PASSING ARGS, ONLY PASS VIA YAML #####
     ############################################################################
    parser = argparse.ArgumentParser(description="Distributed SSSD Unlearning")
    
    # This argument points to the config file
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML configuration file.")
    
    # --- Parse config file path ---
    # We parse known args first to get the config path
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

    # --- Re-create parser to add all arguments ---
    # Now we add all other arguments.
    # Their defaults will be set by config_data if it exists.
    
    # Distributed args
    dist_args = config_data.get('distributed_args', {})
    parser.add_argument("--local_rank", type=int, default=dist_args.get('local_rank', -1), help="Local process rank (usually set by launcher).")
    parser.add_argument("--world_size", type=int, default=dist_args.get('world_size', 1), help="Number of total processes (usually set by launcher).")
    parser.add_argument("--dist_backend", type=str, default=dist_args.get('dist_backend', 'nccl'), help="Distributed backend.")
    parser.add_argument("--dist_url", type=str, default=dist_args.get('dist_url', 'env://'), help="URL for init.")

    # Model args
    model_args = config_data.get('model_args', {})
    parser.add_argument("--model_name", type=str, default=model_args.get('model_name', 'gpt2'), help="Model name or path.")
    parser.add_argument("--save_path", type=str, default=model_args.get('save_path', 'unlearned_model.pth'), help="Path to save the unlearned model.")

    # Data args
    data_args = config_data.get('data_args', {})
    parser.add_argument("--data_path", type=str, default=data_args.get('data_path', './data/'), help="Path to the honesty dataset.")
    parser.add_argument("--n_train_pairs", type=int, default=data_args.get('n_train_pairs', 1000), help="Number of training pairs.")
    parser.add_argument("--n_test_pairs", type=int, default=data_args.get('n_test_pairs', 100), help="Number of test pairs.")
    parser.add_argument("--batch_size", type=int, default=data_args.get('batch_size', 1), help="Batch size PER GPU.")
    parser.add_argument("--seed", type=int, default=data_args.get('seed', 42), help="Random seed for data splitting.")
    # Unlearn config (SSSD)
    # We pass this as a dictionary
    # parser.add_argument("--param_config", type=dict, default=config_data.get('param_config', {}), help="Dictionary of ParamConfig settings (loaded from YAML).")

    # Concept vector paths
    vector_args = config_data.get('vector_args', {})
    parser.add_argument("--concept_vector_path", type=str, default=vector_args.get('concept_vector_path', None), help="Path to the concept vector (.pt file).")
    parser.add_argument("--reading_vector_path", type=str, default=vector_args.get('reading_vector_path', None), help="Path to the reading vector (.pt file).")
    parser.add_argument("--coeff", type=float, default=vector_args.get('coeff', 1.0), help="Coefficient to scale the concept/reading vector.")

    # Now, parse all arguments (including the ones in remaining_argv)
    # This will correctly override defaults from config_data with any
    # command-line arguments provided.
    args = parser.parse_args(remaining_argv)
    args.param_config = None
    
    # Manually merge the param_config dict, as argparse doesn't handle
    # dict defaults perfectly when passed as cmd-line override.
    # The default loaded from YAML will be used.
    if not args.param_config:
         args.param_config = config_data.get('param_config', {})

    return args


def setup_for_distributed(is_master: bool):
    """
    This utility disables tqdm/bar updates for non-master processes to avoid
    console spam.
    """
    if not is_master:
        # Disable printing entirely on non-master ranks
        def print_pass(*args, **kwargs):
            pass
        __builtins__.print = print_pass
        
        # Disable tqdm
        global tqdm
        tqdm = lambda *args, **kwargs: args[0] if args else None # Return iterator if exists


def main_worker(local_rank: int, world_size: int, args):
    # 1) Initialize process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=local_rank,
    )

    # 2) Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 3) (Optional) silence logs if not master
    is_master = (local_rank == 0)
    setup_for_distributed(is_master)
    
    print(f"[{local_rank}] Process started. World size: {world_size}.")
    print(f"[{local_rank}] Loading model: {args.model_name}")

    ############################################################################
    # Model Setup
    ############################################################################
    model_name = args.model_name
    # Load model with float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # DDP wrapper
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print(f"[{local_rank}] Model loaded and wrapped in DDP.")

    ############################################################################
    # Data Setup (Distributed Samplers)
    ############################################################################
    print(f"[{local_rank}] Setting up datasets...")
    # These args (user_tag, assistant_tag) should be in your config if needed
    train_pairs, test_pairs = build_honesty_pairs(
        data_path=args.data_path,
        tokenizer=tokenizer,
        user_tag="USER:",
        assistant_tag="ASSISTANT:",
        seed=args.seed,
        n_train_pairs=args.n_train_pairs,
        n_test_pairs=args.n_test_pairs,
    )
    
    train_ds = PairsTextDataset(train_pairs)
    test_ds = PairsTextDataset(test_pairs)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    print(f"[{local_rank}] Dataloaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    ############################################################################
    # Load Concept/Reading Vectors
    ############################################################################
    concept_vector = None
    if args.concept_vector_path:
        print(f"[{local_rank}] Loading concept vector from: {args.concept_vector_path}")
        concept_vector = torch.load(args.concept_vector_path, map_location="cpu").to(torch.float16)
    
        
    ## the label tells you which onem in the tuple is honest or not
    reading_vector = None
    a_list, b_list, c_list = map(list, zip(*train_pairs))
    if args.reading_vector_path:
        print(f"[{local_rank}] Loading reading vector from: {args.reading_vector_path}")
        reading_vector = torch.load(args.reading_vector_path, map_location="cpu").to(torch.float16)
    else:
        ## obtaining concept vector via honesty_utils.
        data = honesty_function_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            user_tag="USER:",
            assistant_tag="ASSISTANT:",
            seed=args.seed
        )
        
        rep_token = -1
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
        honesty_rep_reader = rep_reading_pipeline.get_directions(
        train_inputs=data["train"]['data'], ## this is train and test pair, column 3 is the label column 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=data["train"]['labels'], 
        direction_method=direction_method,
        batch_size=32,
        )
        concept_activations = {}
        for layer in hidden_layers:
            concept_activations[layer] = torch.tensor(args.coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()
        concept_vector = concept_activations
        print(f"Reading Vector obtained via REP reader : {concept_vector}  ")
        print(f"[{local_rank}] Generated reading vector using REP reader.")
        
        
    
    if concept_vector is None and reading_vector is None:
        print(f"[{local_rank}] ERROR: No concept_vector or reading provided. ConceptLoss will fail.")
        print(f"[{local_rank}] Please specify at least one in your config file.")
        dist.destroy_process_group()
        sys.exit(1)


    ############################################################################
    # Perform the "unlearn" step
    ############################################################################
    
    # Instantiate config from the dictionary
    try:
        SSSD_config = ParamConfig(**args.param_config)
        print(f"[{local_rank}] SSSD Config: {SSSD_config.dict()}")
    except Exception as e:
        print(f"[{local_rank}] ERROR: Failed to parse param_config. Make sure all required fields are in the YAML.")
        print(f"[{local_rank}] Error: {e}")
        dist.destroy_process_group()
        sys.exit(1)
        
    unlearner_instance = SSSD(
        optimizer=torch.optim.Adam(ddp_model.parameters(), lr=1e-4), # Dummy optimizer
        model=ddp_model,
        config=SSSD_config,
        device=device
    )
    
    unlearner_instance.set_model(ddp_model)
    
    print(f"[{local_rank}] Starting unlearning process...")
    ## concept_vector is a Dict[Layer_Number: Tensor]
    unlearned_model = unlearner_instance.unlearn(
        train_loader=train_loader,
        test_loader=test_loader,
        concept_vector=concept_vector,
        reading_vector=reading_vector
    )
    
    print(f"[{local_rank}] Unlearning completed.")
    
    # Save the model only on the master rank
    if is_master:
        print(f"Saving unlearned model to: {args.save_path}")
        # We save the underlying model's state_dict (not the DDP wrapper)
        torch.save(obj=unlearned_model.module.state_dict(), f=args.save_path)
        print("Model saved.")

    dist.destroy_process_group()
    print(f"[{local_rank}] Process finished.")

def main():
    args = parse_args()

    # Get rank and world size from environment variables
    # This is the standard way torchrun/mp.spawn sets them
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Override with args if they were somehow set differently (e.g., manual launch)
    # But typically, env vars are the source of truth.
    if args.local_rank != -1:
        local_rank = args.local_rank
    if args.world_size != 1:
        world_size = args.world_size
        
    if world_size > 1:
        print(f"Initializing distributed training with {world_size} GPUs.")
        # mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
        # The mp.spawn is not needed if you use `torchrun`
        # The script is designed to be launched with:
        # torchrun --nproc_per_node=NUM_GPUS unlearn.py --config config.yaml
        main_worker(local_rank, world_size, args)
    else:
        print("Running in single-process mode (world_size=1).")
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()


############################# command to run:##################################################################
####  torchrun --standalone --nproc_per_node=1 parameterperturber.py --config configs/ssd_config.yaml #########
###############################################################################################################