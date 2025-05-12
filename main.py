import argparse
from utils import model_utils, data_utils, logging_utils, seed_utils
from utils.logging_utils import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to load")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to load")
    parser.add_argument('--device', type=str, required=False, default='cuda', help="Device to use for model inference")
    parser.add_argument('--seed', type=int, required=False, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    device = args.device
    seed = args.seed

    seed_utils.set_seed(42)

    model = model_utils.load_model_and_tokenizer(model_name, device=device)

    dataset = data_utils.load_dataset(dataset_name)
