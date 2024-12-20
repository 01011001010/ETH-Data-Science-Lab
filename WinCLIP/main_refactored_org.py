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
from datasets.mvtec_dataset import mvtec_dataset, OBJECT_TYPE
from binary_focal_loss import BinaryFocalLoss

# Configuration constants
STATE_LEVEL = {
    "normal": ["{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect", "{} without damage"],
    "anomaly": ["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
TEMPLATE_LEVEL = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def generate_text_prompts(obj_name):
    """Generate text prompts for normal and anomaly states based on object name."""
    normal_prompts = [t.format(state) for state in [s.format(obj_name) for s in STATE_LEVEL["normal"]] for t in TEMPLATE_LEVEL]
    anomaly_prompts = [t.format(state) for state in [s.format(obj_name) for s in STATE_LEVEL["anomaly"]] for t in TEMPLATE_LEVEL]
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
    dataloader = DataLoader(dataset, batch_size=4, num_workers=8, shuffle=False)

    normal_texts, anomaly_texts = generate_text_prompts(config['obj_type'].replace('_', " "))
    text_features = compute_text_features(model, tokenizer, normal_texts, anomaly_texts, device)

    scores, ground_truths = [], []

    # Process each image in the dataset
    for data in tqdm(dataloader, desc="Eval"):
        image, ref_images, _, has_anomaly, _ = data
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
            anomaly_map, vis_probs = model.forward(image=combined_image)
            avg_score = (vis_probs + score) / 2 if not math.isinf(score) else 0
            scores.append(avg_score)
            ground_truths.append(has_anomaly[0].item())

    return scores, ground_truths

def run_evaluation(config):
    """Execute the evaluation loop for each object type in OBJECT_TYPE and print aggregated results."""
    device = torch.cuda.current_device()
    model, tokenizer, preprocess = initialize_model(config, device)

    all_scores, all_ground_truths = [], []
    aurocs, auprs, f1_maxes = [], [], []

    for obj_type in OBJECT_TYPE:
        config['obj_type'] = obj_type
        scores, ground_truths = evaluate_model(model, tokenizer, preprocess, config, device)
        auroc, aupr, f1_max = calculate_metrics(scores, ground_truths)

        all_scores.extend(scores)
        all_ground_truths.extend(ground_truths)
        aurocs.append(auroc)
        auprs.append(aupr)
        f1_maxes.append(f1_max)

        print(f"Obj Type: {obj_type}, AUROC={auroc:.3f}, AUPR={aupr:.3f}, F1-Max={f1_max:.3f}")

    # Print average metrics across object types
    print(f"Avg AUROC: {np.mean(aurocs):.3f}")
    print(f"Avg AUPR: {np.mean(auprs):.3f}")
    print(f"Avg F1-Max: {np.mean(f1_maxes):.3f}")

    # Calculate metrics across all object types
    auroc, aupr, f1_max = calculate_metrics(all_scores, all_ground_truths)
    print(f"All Types: AUROC={auroc:.3f}, AUPR={aupr:.3f}, F1-Max={f1_max:.3f}")

if __name__ == "__main__":
    # Configuration setup
    config = {
        'datasetname': "visa",
        'dataset_root_dir': '/home/bearda/PycharmProjects/winclipbad/WinCLIP/visa_anomaly_detection',
        'data_dir': os.path.join('/home/bearda/PycharmProjects/winclipbad/WinCLIP/visa_anomaly_detection', "visa"),
        'model_cfg_path': './open_clip/model_configs/ViT-B-16-plus-240.json',
        'checkpoint_path': "./vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
        'shot': 1,
    }

    # Set random seeds for reproducibility
    np.random.seed(10)
    torch.manual_seed(10)

    # Run evaluation
    with torch.no_grad(), torch.cuda.amp.autocast():
        run_evaluation(config)
