import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
import open_clip
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from open_clip import create_model_and_transforms, get_tokenizer

# Custom imports
from datasets.mvtec_dataset import mvtec_dataset
from binary_focal_loss import BinaryFocalLoss
import argparse
import logging
import toml
import torch.nn.functional as F

def generate_text_prompts(obj_name, config):
    """Generate text prompts for normal and anomaly states based on object name."""
    normal_prompts = [t.format(state) for state in [s.format(obj_name) for s in config['state_level']['normal']] for t in config['template_level']]
    anomaly_prompts = [t.format(state) for state in [s.format(obj_name) for s in config['state_level']['anomaly']] for t in config['template_level']]
    return normal_prompts, anomaly_prompts

def initialize_model(config, device):
    """Load model, tokenizer, and preprocessing transforms based on config."""
    tokenizer = get_tokenizer('ViT-B-16-plus-240')
    _, _, preprocess = create_model_and_transforms('ViT-B-16-plus-240')

    # Load model configuration
    with open(config['model_cfg_path'], 'r') as f:
        model_cfg = json.load(f)

    # Initialize and load model
    model = open_clip.model.WinCLIP(
        embed_dim=model_cfg["embed_dim"],
        vision_cfg=model_cfg["vision_cfg"],
        text_cfg=model_cfg["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype('fp32')
    ).to(device)

    # Load model weights
    with pathmgr.open(config['checkpoint_path'], "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer, preprocess

def compute_text_features(model, tokenizer, normal_texts, anomaly_texts, device):
    """Encode and normalize text features for normal and anomaly texts."""
    pos_features = F.normalize(model.encode_text(tokenizer(normal_texts).to(device)), dim=-1)
    neg_features = F.normalize(model.encode_text(tokenizer(anomaly_texts).to(device)), dim=-1)
    return torch.cat([pos_features.mean(dim=0, keepdim=True), neg_features.mean(dim=0, keepdim=True)], dim=0)

def compute_image_score(model, text_features, image):
    """Compute anomaly score for a single image based on image and text features."""
    _, _, image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)
    score = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return score[0, 1].item()

def prepare_few_shot_images(image, ref_images, device):
    """Prepare image tensors for few-shot learning."""
    all_images = [image] + [ref for ref in ref_images]
    return all_images

def calculate_metrics(scores, ground_truths):
    """Calculate AUROC, AUPR, and F1-Max metrics."""
    auroc = roc_auc_score(ground_truths, scores)
    precision, recall, _ = precision_recall_curve(ground_truths, scores)
    aupr = auc(recall, precision)

    f1_max = max(
        f1_score(ground_truths, (np.array(scores) > threshold).astype(int))
        for threshold in np.arange(0, 1, 0.01)
    )

    return auroc, aupr, f1_max

def evaluate_model(model, tokenizer, preprocess, config, device):
    """Run evaluation on the dataset and calculate metrics."""
    dataset = mvtec_dataset(config, config["data_dir"], mode='test', shot=config["shot"], preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)

    normal_texts, anomaly_texts = generate_text_prompts(config['obj_type'].replace('_', " "), config)
    text_features = compute_text_features(model, tokenizer, normal_texts, anomaly_texts, device)

    scores, ground_truths = [], []

    # Process each image in the dataset
    for data in tqdm(dataloader, desc="Eval"):
        image, ref_images, mask, has_anomaly, _ = data
        #print(f'reference images length: {len(ref_images)}')
        if isinstance(image, list):
            score = sum(compute_image_score(model, text_features, img.to(device)) for img in image) / len(image)
        else:
            image = image.to(device)
            score = compute_image_score(model, text_features, image)

        if config["shot"] == 0:
            # Zero-shot classification
            scores.append(score)
            ground_truths.append(has_anomaly[0].item())
        else:
            # Few-shot classification
            combined_image = prepare_few_shot_images(image, ref_images, device)
            # Check shape of combined_image before passing to model
            #print(f'combined image length: {len(combined_image)}')
            anomaly_map, vis_probs = model.forward(image=combined_image, shot=config['shot'])
            avg_score = (vis_probs + score) / 2 if not math.isinf(score) else 0
            scores.append(avg_score)
            ground_truths.append(has_anomaly[0].item())

    return scores, ground_truths

def run_evaluation(config):
    """Execute the evaluation loop for each domain shift in config['obj_types'] and print aggregated results."""
    device = torch.cuda.current_device()
    model, tokenizer, preprocess = initialize_model(config, device)

    # Initialize dictionaries to collect domain-specific scores and ground truths
    domain_shift_scores = {domain: [] for domain in config['obj_types']}
    domain_shift_ground_truths = {domain: [] for domain in config['obj_types']}

    all_scores, all_ground_truths = [], []
    aurocs, auprs, f1_maxes = [], [], []

    for domain_shift in config['obj_types']:
        config['obj_type'] = domain_shift  # Update current domain in config
        scores, ground_truths = evaluate_model(model, tokenizer, preprocess, config, device)
        
        # Calculate and store metrics for each domain shift
        auroc, aupr, f1_max = calculate_metrics(scores, ground_truths)
        aurocs.append(auroc)
        auprs.append(aupr)
        f1_maxes.append(f1_max)

        # Append scores and ground truths for overall and domain-specific metrics
        all_scores.extend(scores)
        all_ground_truths.extend(ground_truths)
        domain_shift_scores[domain_shift].extend(scores)
        domain_shift_ground_truths[domain_shift].extend(ground_truths)

        # Print metrics for the individual domain shift
        print(f"Domain Shift: {domain_shift}, AUROC={auroc:.3f}, AUPR={aupr:.3f}, F1-Max={f1_max:.3f}")

    # Original average metrics across all domain shifts
    print(f"\nAverage Metrics Across All Domain Shifts:")
    print(f"Avg AUROC: {np.mean(aurocs):.3f}")
    print(f"Avg AUPR: {np.mean(auprs):.3f}")
    print(f"Avg F1-Max: {np.mean(f1_maxes):.3f}")

    # Combined metrics across all domain shifts
    auroc, aupr, f1_max = calculate_metrics(all_scores, all_ground_truths)
    print(f"\nCombined Metrics Across All Domains: AUROC={auroc:.3f}, AUPR={aupr:.3f}, F1-Max={f1_max:.3f}")

def load_config(dataset_name):
    config_file = 'config.toml'
    configs = toml.load(config_file)
    if dataset_name not in configs:
        raise ValueError(f"No configuration found for dataset '{dataset_name}' in {config_file}")
    
    # Load and add template and state levels from config
    configs[dataset_name]['template_level'] = configs.get('template_level', {}).get('templates', [])
    configs[dataset_name]['state_level'] = {
        'normal': configs.get('state_level', {}).get('normal', []),
        'anomaly': configs.get('state_level', {}).get('anomaly', [])
    }
    return configs[dataset_name]

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging initialized.")

def set_random_seeds(seed=10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info("Random seeds set.")

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run evaluation with specified configuration.")
    parser.add_argument('--dataset', type=str, choices=['visa', 'aebad'], default='aebad', help="Choose the dataset for evaluation.")
    args = parser.parse_args()

    config = load_config(args.dataset)
    set_random_seeds()

    try:
        logging.info(f"Starting evaluation for dataset '{args.dataset}'")
        with torch.no_grad(), torch.cuda.amp.autocast():
            run_evaluation(config)
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.exception("An error occurred during evaluation: %s", e)

if __name__ == "__main__":
    main()