import argparse
import logging
import os
import shutil
from pathlib import Path

def setup_logging(log_file_path="conversion.log"):
    logger = logging.getLogger("DatasetConverter")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

def setup_directory_structure(root_path, domain_classes):
    root_path = Path(root_path)
    for domain in domain_classes:
        (root_path / f"{domain}/train/good").mkdir(parents=True, exist_ok=True)
        (root_path / f"{domain}/test/good").mkdir(parents=True, exist_ok=True)
        (root_path / f"{domain}/test/defect").mkdir(parents=True, exist_ok=True)
        (root_path / f"{domain}/ground_truth/defect").mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory structure created for {domain}")

def copy_files(src_dir, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in src_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, dest_dir / file.name)
            logger.debug(f"Copied {file} to {dest_dir}")

def convert_train_set(train_input_dir, output_root, domain_classes):
    logger.info("Converting training data...")
    for domain in train_input_dir.iterdir():
        if domain.is_dir() and domain.name in domain_classes:
            dest_dir = output_root / f"{domain.name}/train/good"
            copy_files(domain, dest_dir)
    
    # Special case for "same" category: copy all training data into same/train/good
    same_train_dir = output_root / "same/train/good"
    for domain in domain_classes:
        if domain != "same":
            src_dir = output_root / f"{domain}/train/good"
            copy_files(src_dir, same_train_dir)
            logger.info(f"Added training data from {domain} to 'same' category.")
    logger.info("Training data conversion complete.")

def convert_test_set(test_input_dir, output_root, domain_classes):
    logger.info("Converting test data...")
    good_dir = test_input_dir / "good"
    if good_dir.exists():
        for domain in good_dir.iterdir():
            if domain.is_dir() and domain.name in domain_classes:
                dest_dir = output_root / f"{domain.name}/test/good"
                copy_files(domain, dest_dir)

    for defect_type in test_input_dir.iterdir():
        if defect_type.is_dir() and defect_type.name != "good":
            for domain in defect_type.iterdir():
                if domain.is_dir() and domain.name in domain_classes:
                    dest_dir = output_root / f"{domain.name}/test/defect"
                    copy_files(domain, dest_dir)
    logger.info("Test data conversion complete.")

def convert_ground_truth(ground_truth_input_dir, output_root, domain_classes):
    logger.info("Converting ground truth masks...")
    for defect_type in ground_truth_input_dir.iterdir():
        if defect_type.is_dir():
            for domain in defect_type.iterdir():
                if domain.is_dir() and domain.name in domain_classes:
                    dest_dir = output_root / f"{domain.name}/ground_truth/defect"
                    copy_files(domain, dest_dir)
    logger.info("Ground truth conversion complete.")

def count_files_in_directory(directory):
    """Helper function to count files in a given directory."""
    return len(list(directory.glob("*")))

def report_training_data_sizes(output_root, domain_classes):
    """Prints out the number of files in each train/good directory for each domain class."""
    logger.info("Training data sizes per domain shift type:")
    for domain in domain_classes:
        train_dir = output_root / f"{domain}/train/good"
        file_count = count_files_in_directory(train_dir)
        logger.info(f"Domain '{domain}': {file_count} training samples")

def convert_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    domain_classes = ["background", "illumination", "view", "same"]
    
    train_input_dir = input_root / "train" / "good"
    test_input_dir = input_root / "test"
    ground_truth_input_dir = input_root / "ground_truth"
    
    setup_directory_structure(output_root, domain_classes)
    convert_train_set(train_input_dir, output_root, domain_classes)
    convert_test_set(test_input_dir, output_root, domain_classes)
    convert_ground_truth(ground_truth_input_dir, output_root, domain_classes)
    
    # Report the resulting training data sizes
    report_training_data_sizes(output_root, domain_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AeBAD dataset to MVTec-like format with domain-specific classes.")
    parser.add_argument("--input_root", default='/home/bearda/PycharmProjects/winclipbad/AeBAD/AeBAD_S', type=str, help="Path to the AeBAD dataset root directory.")
    parser.add_argument("--output_root", default='/home/bearda/PycharmProjects/winclipbad/WinCLIP/AeBAD', type=str, help="Path to the output directory for MVTec format dataset.")
    args = parser.parse_args()

    try:
        convert_dataset(args.input_root, args.output_root)
        logger.info("Dataset conversion to MVTec format complete.")
    except Exception as e:
        logger.error(f"An error occurred during dataset conversion: {e}")
