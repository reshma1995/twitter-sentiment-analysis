import argparse
import os
from utils.helper_utils import set_seed
from operations.dataset_ops import create_and_load_dataset


def main():
    set_seed(2023)
    parser = argparse.ArgumentParser(description='Main Training Script for Sentiment Analysis')
    parser.add_argument('--model_name', type=str, help='Name of the model you want to load')
    parser.add_argument('--load_dataset', action='store_true', help='Load and clean dataset')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train-model', action='store_true', help='Train model')
    group.add_argument('--infer-model', action='store_true', help='Infer model')
    group.add_argument('--predict', action='store_true', help='Make Predictions')
    args = parser.parse_args()

    if args.load_dataset:
        create_and_load_dataset()

if __name__ == '__main__':
    main()
