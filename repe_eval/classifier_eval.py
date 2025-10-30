import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from statistics import mean, fmean, stdev
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class TestSet(Dataset):
    def __init__(self, user_prompts):
            self.user_prompts = user_prompts

    def __len__(self):
        return len(self.user_prompts)

    def __getitem__(self, idx):
        # You can add preprocessing here if needed, e.g., tokenization
        sample = self.user_prompts[idx]
        return sample


def generate_response(model, tokenizer, user_prompts, system_message=None, max_length=256):
    """
    Generate model responses for one or multiple prompts.
    Supports batching for parallel inference.

    Args:
        model: The loaded model (e.g., from transformers)
        tokenizer: Corresponding tokenizer
        user_prompts: str or list[str] — can be a single prompt or a batch
        system_message: optional system message string
        max_length: maximum number of generated tokens
    """
    # Ensure user_prompts is a list
    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]

    # Prepare full message lists per prompt
    formatted_prompts = []
    for prompt in user_prompts:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        formatted_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # Tokenize all prompts together for batch inference
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate outputs in parallel
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode each output
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    # Clean responses
    responses = []
    for full_response in decoded:
        if '<|assistant|>' in full_response:
            text = full_response.split('<|assistant|>')[-1].replace('</s>', '').strip()
        else:
            text = full_response.strip()
        responses.append(text)

    return responses


def compare_models(base_model_path, unlearned_model_path, user_prompts, system_message=None, focus_label="joy", num_samples=None):
    """Compare base model and unlearned model using label-wise sentiment classifier"""

    # Limit samples if specified
    if num_samples is not None and num_samples < len(user_prompts):
        user_prompts = user_prompts[:num_samples]
        print(f"Using {num_samples} samples out of total available prompts")

    testset = TestSet(user_prompts)
    testloader = DataLoader(testset, batch_size=2, shuffle=True)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')

    print("Loading fine-tuned model...")
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        unlearned_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    unlearned_tokenizer = AutoTokenizer.from_pretrained(unlearned_model_path, padding_side='left')

    # Sentiment classifier
    classifier = pipeline(
        "sentiment-analysis",
        model="michellejieli/emotion_text_classifier",
        return_all_scores=True
    )

    base_scores = defaultdict(list)
    unlearned_scores = defaultdict(list)
    
    # Store detailed outputs
    base_outputs = []
    unlearned_outputs = []

    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80)


    for i, prompts in enumerate(tqdm(testloader, desc="Processing prompts", unit="prompt"), 1):
        # print(f"\n{'='*80}")
        # print(f"PROMPT {i}: {user_prompt}")
        # print(f"{'='*80}")

        base_response = generate_response(base_model, base_tokenizer, prompts, system_message)
        unlearned_response = generate_response(unlearned_model, unlearned_tokenizer, prompts, system_message)

        base_pred = classifier(base_response)  # list of dicts [{label, score}, ...]
        unlearned_pred = classifier(unlearned_response)

        for i, (prompt_text, base_resp, unlearned_resp, b_pred, u_pred) in enumerate(
            zip(prompts, base_response, unlearned_response, base_pred, unlearned_pred)):  

            # Store per-label scores
            for entry in b_pred:
                base_scores[entry["label"]].append(entry["score"])
            for entry in u_pred:
                unlearned_scores[entry["label"]].append(entry["score"])

            # Save detailed outputs
            base_outputs.append({
                "prompt_id": i,
                "prompt": prompt_text,
                "response": base_resp,
                "scores": {entry["label"]: entry["score"] for entry in b_pred}
            })
            
            unlearned_outputs.append({
                "prompt_id": i,
                "prompt": prompt_text,
                "response": unlearned_resp,
                "scores": {entry["label"]: entry["score"] for entry in u_pred}
            })

            # Print focus label comparison (joy)
            base_focus = next(d["score"] for d in b_pred if d["label"] == focus_label)
            unlearned_focus = next(d["score"] for d in u_pred if d["label"] == focus_label)
            better = "base" if base_focus > unlearned_focus else "unlearned" if unlearned_focus > base_focus else "tie"

            # print(f"\nBase {focus_label}={base_focus:.4f} | unlearned {focus_label}={unlearned_focus:.4f} → {better.upper()}")

    # Save individual model outputs
    os.makedirs("results", exist_ok=True)
    
    with open("results/base_model_outputs.json", "w") as f:
        json.dump(base_outputs, f, indent=4)
    print("\nSaved base model outputs to results/base_model_outputs.json")
    
    with open("results/unlearned_model_outputs.json", "w") as f:
        json.dump(unlearned_outputs, f, indent=4)
    print("Saved unlearned model outputs to results/unlearned_model_outputs.json")

    # --- Comprehensive Statistics Summary ---
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - COMPREHENSIVE STATISTICS")
    print("="*80)

    all_labels = sorted(set(base_scores.keys()) | set(unlearned_scores.keys()))

    summary = {
        "per_label_stats": {},
        "focus_label": focus_label,
        "num_prompts": len(user_prompts)
    }
    
    print(f"\n{'Label':<15} {'Model':<8} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    for label in all_labels:
        base_vals = base_scores[label]
        unlearned_vals = unlearned_scores[label]
        
        # Calculate statistics
        base_stats = {
            "mean": fmean(base_vals) if base_vals else 0.0,
            "std": stdev(base_vals) if len(base_vals) > 1 else 0.0,
            "min": min(base_vals) if base_vals else 0.0,
            "max": max(base_vals) if base_vals else 0.0,
            "count": len(base_vals)
        }
        
        unlearned_stats = {
            "mean": fmean(unlearned_vals) if unlearned_vals else 0.0,
            "std": stdev(unlearned_vals) if len(unlearned_vals) > 1 else 0.0,
            "min": min(unlearned_vals) if unlearned_vals else 0.0,
            "max": max(unlearned_vals) if unlearned_vals else 0.0,
            "count": len(unlearned_vals)
        }
        
        # Calculate improvement
        improvement = unlearned_stats["mean"] - base_stats["mean"]
        improvement_pct = (improvement / base_stats["mean"] * 100) if base_stats["mean"] != 0 else 0.0
        
        summary["per_label_stats"][label] = {
            "base": base_stats,
            "unlearned": unlearned_stats,
            "improvement": improvement,
            "improvement_percent": improvement_pct
        }
        
        # Print base stats
        print(f"{label:<15} {'Base':<8} {base_stats['mean']:>9.4f} {base_stats['std']:>9.4f} "
              f"{base_stats['min']:>9.4f} {base_stats['max']:>9.4f}")
        
        # Print unlearned stats
        print(f"{label:<15} {'unlearned':<8} {unlearned_stats['mean']:>9.4f} {unlearned_stats['std']:>9.4f} "
              f"{unlearned_stats['min']:>9.4f} {unlearned_stats['max']:>9.4f}")
        
        # Print comparison
        print(f"{'':<15} {'Δ':<8} {improvement:>+9.4f} ({improvement_pct:>+6.2f}%)")
        print()

    # Highlight focus label
    if focus_label in summary["per_label_stats"]:
        print("\n" + "="*80)
        print(f"FOCUS LABEL: {focus_label.upper()}")
        print("="*80)
        focus_stats = summary["per_label_stats"][focus_label]
        print(f"Base Model:  Mean={focus_stats['base']['mean']:.4f}, StdDev={focus_stats['base']['std']:.4f}")
        print(f"unlearned Model: Mean={focus_stats['unlearned']['mean']:.4f}, StdDev={focus_stats['unlearned']['std']:.4f}")
        print(f"Improvement: {focus_stats['improvement']:+.4f} ({focus_stats['improvement_percent']:+.2f}%)")
        
        # Determine winner
        if focus_stats['improvement'] > 0:
            print(f"\n✓ unlearned model shows improvement on {focus_label}")
        elif focus_stats['improvement'] < 0:
            print(f"\n✗ Base model performs better on {focus_label}")
        else:
            print(f"\n= Models perform equally on {focus_label}")

    # Save comprehensive summary
    with open("results/comprehensive_statistics.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*80)
    print("Saved comprehensive statistics to results/comprehensive_statistics.json")
    print("="*80)


if __name__ == "__main__":
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    unlearned_model_path = "../lorra_finetune/lorra_tqa_1b"

    with open("../data/emotions/happiness.json", "r") as f:
        user_prompts = json.load(f)

    # Configure number of samples to use (None = use all)
    NUM_SAMPLES = 10  # Change this to desired number, or set to None for all samples

    # classes: anger, disgust, fear, joy, neutrality, sadness, and surprise.
    compare_models(base_model_path, unlearned_model_path, user_prompts, focus_label="joy", num_samples=NUM_SAMPLES)
