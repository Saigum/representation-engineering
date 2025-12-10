from grad_parameterperturber import ConceptLoss,finetune_model
from function_utils import make_emotion_dataloaders,emotion_function_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry
import torch
import sys
import argparse
from peft import get_peft_model, LoraConfig, TaskType
    

import argparse
import yaml
import sys


repe_pipeline_registry()


def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)


    print("Model loaded.")

    ############################################################################
    # Data Setup (No distributed samplers)
    ############################################################################
    print("Setting up datasets...")
    train_loader,test_loader = make_emotion_dataloaders(data_path=args.data_path,
                                                        tokenizer=tokenizer,user_tag="USER:",
                                                        assistant_tag="ASSISTANT:",
                                                        emotion=args.concept,
                                                        seed=args.seed,n_train_pairs=args.n_train_pairs,    
                                                        n_test_pairs=args.n_test_pairs,batch_size=args.batch_size)
    
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    ############################################################################
    # Load Concept/Reading Vectors
    ############################################################################
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


    if concept_vector is None :
        print("ERROR: No concept_vector or reading_vector provided.")
        print("Please specify at least one in your config file.")
        sys.exit(1)

    ############################################################################
    # Perform the unlearn step
    ############################################################################
    ## reloading model ??? 
    del model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    ## Finetune the model to unlearn the concept
    ## adding peft lora config
    print("Adding LoRA adapters to the model...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    
    print("Starting finetuning...")
    finetuned_model = finetune_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        concept_vector=concept_vector,
        device=device,
        num_epochs=args.num_epochs,
        # learning_rate=args.learning_rate,
    )
    
    print("Finetuning completed.")
    

def parse_finetune_args():
    """
    Parses command-line arguments for finetuning, reading defaults from a YAML config file.
    Compatible with the SSSD config.yaml structure.
    """
    parser = argparse.ArgumentParser(
        description="Finetune Language Model to Unlearn a Concept using Concept Vectors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML configuration file.",
    )

    # First pass: only parse --config
    config_args, remaining_argv = parser.parse_known_args()

    # Load YAML config if provided
    config_data = {}
    if config_args.config:
        print(f"Loading configuration from: {config_args.config}")
        with open(config_args.config, "r") as f:
            try:
                config_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
    else:
        print("No --config file provided. Using command-line args or defaults.")

    # Sections from YAML
    distributed_args = config_data.get("distributed_args", {})
    model_args       = config_data.get("model_args", {})
    data_args        = config_data.get("data_args", {})
    vector_args      = config_data.get("vector_args", {})
    train_args       = config_data.get("train_args", {})  # optional, for finetune hyperparams

    # ---------------- Distributed args (optional, but matches your YAML) ----------------
    parser.add_argument(
        "--local_rank",
        type=int,
        default=distributed_args.get("local_rank", -1),
        help="Local rank for distributed training.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=distributed_args.get("world_size", 1),
        help="World size for distributed training.",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default=distributed_args.get("dist_backend", "nccl"),
        help="Distributed backend.",
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default=distributed_args.get("dist_url", "env://"),
        help="URL used to set up distributed training.",
    )

    # ---------------- Model args ----------------
    parser.add_argument(
        "--model_name",
        type=str,
        default=model_args.get("model_name", "gpt2"),
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=model_args.get("save_dir", "save_dir/default"),
        help="Base directory to save unlearned/finetuned artifacts.",
    )
    # Handle both 'finetuned_save_dir' (YAML) and 'finetune_save_dir' (older code) gracefully
    parser.add_argument(
        "--finetuned_save_dir",
        type=str,
        default=model_args.get(
            "finetuned_save_dir",
            model_args.get("finetune_save_dir", "finetune_dir/default"),
        ),
        help="Directory to save the finetuned model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=model_args.get("save_path", "unlearned_model.pth"),
        help="Path to save the final (optionally unlearned) model checkpoint.",
    )

    # ---------------- Data args ----------------
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_args.get("data_path", "./data/emotions/anger_train.json"),
        help="Path to the emotion dataset.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=data_args.get("concept", "anger"),
        help="Concept to unlearn (e.g., anger, sadness).",
    )
    parser.add_argument(
        "--n_train_pairs",
        type=int,
        default=data_args.get("n_train_pairs", 1000),
        help="Number of training pairs.",
    )
    parser.add_argument(
        "--n_test_pairs",
        type=int,
        default=data_args.get("n_test_pairs", 100),
        help="Number of testing pairs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=data_args.get("batch_size", 1),
        help="Batch size for training and testing (per GPU).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=data_args.get("seed", 42),
        help="Random seed.",
    )

    # ---------------- Vector args ----------------
    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default=vector_args.get("concept_vector_path", None),
        help="Path to the concept vector (.pt file).",
    )
    parser.add_argument(
        "--reading_vector_path",
        type=str,
        default=vector_args.get("reading_vector_path", None),
        help="Path to the reading vector (.pt file).",
    )
    parser.add_argument(
        "--coeff",
        type=float,
        default=vector_args.get("coeff", 1.0),
        help="Coefficient for scaling the concept/reading vector.",
    )

    # ---------------- Finetuning-specific args ----------------
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=train_args.get("num_epochs", 3),
        help="Number of epochs for finetuning.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=train_args.get("learning_rate", 5e-5),
        help="Learning rate for finetuning.",
    )

    # Final parse with remaining args
    args = parser.parse_args(remaining_argv)

    # Also attach param_config blob, same as in your SSSD script
    args.param_config = config_data.get("param_config", {})

    return args

if __name__ == "__main__":

    args = parse_finetune_args()
    main(args)
    