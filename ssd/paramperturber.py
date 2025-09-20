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
from function_utils import iter_transformer_blocks
from torch import Tensor
from typing import Optional
from torch.nn import CrossEntropyLoss
from pydantic import BaseModel
from function_utils import make_honesty_dataloaders,build_honesty_pairs,PairsTextDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        assert (concept_vector is not None) or (reading_vector is not None), "At least one of concept_vector or reading_vector must be provided."
        self.l_vec = concept_vector if concept_vector is not None else reading_vector
    
    def forward(self,activations):
        return (activations.matmul(self.l_vec)).pow(2).mean()
            

################################################################################
# 1) ParameterPerturber and ssd_tuning logic
################################################################################
class ParameterPerturber(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        optimizer: torch.optim.Optimizer,
        parameters: dict,
        layer_number:int
    ):
        super(ParameterPerturber, self).__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer


        # For clarity:

        self.layer_number = layer_number
        all_params = self.model.named_parameters()
        ## we only include till the layer_number : layer_number is negative with 0 as the first layer
        self.paramset = {}
        num_blocks  = sum(1 for _ in iter_transformer_blocks(self.model))
        print(f"Numblocks:{num_blocks}")
        keys = list(range(-1, -num_blocks-1, -1))
        self.block_number = layer_number
        count =0
        for path, i, block in iter_transformer_blocks(self.model):
            if((i!=num_blocks-1) and i<layer_number):
                self.paramset[keys[i]] = []
                for param in block.parameters():
                    self.paramset[keys[i]].append(param)
                    count+=1
        self.num_params = count ## this is to be used to access gradient.

        self.lower_bound = parameters.get("lower_bound", 1)
        self.exponent = parameters.get("exponent", 1)

    def __zero_params__(self):
        return {
            k: torch.zeros_like(params)
            for k, params in self.paramset.items()
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
        self.model.train()
        batch_counter  = 0
        with tqdm(total=len(dataloader), desc="Calculating Importances") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                input_ids = batch['input_ids']
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                selected_hidden_state = hidden_states[self.layer_number]
                
                loss = loss_(selected_hidden_state)
                loss.backward()
                count =-1
                for (k, p), (ik, imp) in zip(self.model.named_parameters(), importances.items()):
                    count+=1
                    if(count<self.num_params):
                        if p.grad is None:
                            continue
                        imp += p.grad.detach().pow(2)

                        torch.cuda.empty_cache()
                    else:
                        break
                batch_counter += 1
                if batch_counter % accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                pbar.update(1)
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        return importances

    def modify_weight(self, original_importances, new_importances):
        """
        Modifies the model weights based on the difference between
        original_importances and new_importances, following your SSD scheme.
        """
        count =-1
        with torch.no_grad():
            for (name, p) in self.model.named_parameters():
                count+=1
                if(count>=self.num_params):
                    break
                oimp = original_importances[name]
                fimp = new_importances[name]
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.parameters["selection_weighting"])
                locations = torch.where(fimp > oimp_norm)

                # Dampening
                weight = (
                    (oimp.mul(self.parameters["dampening_constant"])).div(fimp)
                ).pow(self.exponent)
                update = weight[locations]

                # Bound by 1 to prevent parameter values to *increase*.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound

                p[locations] = p[locations].mul(update)





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

    forget_loss = forget_loss_class(concept_vector,reading_vector).to(device)
    retain_loss = retain_loss_class().to(device) if retain_loss_class is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    pdr = ParameterPerturber(model, device, optimizer, config.dict(),layer_number)
    forget_importances = pdr.calc_importance(dataloader,loss_=forget_loss,accum_steps= config.forget_threshold)
    if retain_loss is not None:
        retain_importances = pdr.calc_importance(dataloader,loss_=retain_loss,accum_steps= config.forget_threshold)
    else:
        retain_importances = pdr.__zero_params__()
    pdr.modify_weight(retain_importances,forget_importances)
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
    def unlearn(self, train_loader, test_loader,  eval_loaders=None):
        time_start = time.process_time()
        self.best_model = forget_retain_signal_tuning(
            self.model,
            train_loader,
            concept_vector=None,
            reading_vector=None,
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
    Replace or extend this with argparse as needed.
    For simplicity, we’ll assume environment variables or defaults are used.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Distributed SSD Example")
    # You can add arguments like:
    parser.add_argument("--local_rank", type=int, default=-1, help="Local process rank.")
    parser.add_argument("--world_size", type=int, default=1, help="Number of total processes.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend.")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for init.")
    parser.add_argument("--model_name", type=str, default="mistral7B",choices=["gpt2","mistral7B","vicuna7B"], help="Model name or path.")
    parser.add_argument("--concept ")
    args = parser.parse_args()
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

    ############################################################################
    # Model Setup
    ############################################################################
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
   
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ############################################################################
    # Data Setup (Distributed Samplers)
    ############################################################################


    train_pairs,test_pairs  = build_honesty_pairs(
        data_path=args.data_path,
        tokenizer=tokenizer,
        user_tag="USER:",
        assistant_tag = "ASSISTANT:",
        seed=args.seed,
        n_train_pairs=args.n_train_pairs,
        n_test_pairs=args.n_test_pairs,
        batch_size=args.batch_size)
    train_ds = PairsTextDataset(train_pairs)
    test_ds = PairsTextDataset(test_pairs)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler)  # Reduced from 32
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, sampler=test_sampler)  # Reduced from 32



    ############################################################################
    # Perform the "unlearn" step
    ############################################################################
    # Typically, you only need to do ssd_tuning once. If you want to coordinate
    # model updates across ranks, you can do so just by calling ssd_instance.unlearn
    # on each rank. DDP ensures the gradients and parameters are synchronized.
    SSSD_config = ParamConfig(
        lower_bound=1,
        exponent=1,
        min_layer=-1,
        max_layer=-1,
        forget_threshold=1,
        dampening_constant=0.1,
        selection_weighting=0.1
    )
    unlearner_instance = SSSD(
        optimizer=torch.optim.Adam(ddp_model.parameters(), lr=1e-4),
        model=ddp_model,
        config=SSSD_config,
        device=device
    )
    unlearner_instance.set_model(ddp_model)
    unlearned_model = unlearner_instance.unlearn(
        train_loader=train_loader,
        test_loader=test_loader,
    )
    
    print(f"Unlearning completed.") ## TODO: Add forget loss on test_loader as a signal here
    torch.save(obj=unlearned_model.state_dict(),f=f"unlearned_model_correct_.pth")
    dist.destroy_process_group()

def main():
    args = parse_args()
    local_rank = args.local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # world_size = int(os.environ.get("WORLD_SIZE", 1))

    main_worker(local_rank, world_size, args)

if __name__ == "__main__":
    main()
