import argparse
import torch
import torch.nn as nn
from utils.helper_utils import set_seed
from utils.model_utils import load_data,configure_model
from operations.dataset_ops.dataset_ops import create_and_load_dataset
from utils.constants import SPLIT_DATA_FILEPATH, BATCH_SIZE, DEVICE_TYPE, MODEL_SAVE_PATH, NUM_EPOCHS
from transformers import BertTokenizer
from operations.dataset_ops.dataset_loader_ops import DatasetLoader
from torch.utils.data import DataLoader
from operations.model_ops.train_model import EarlyStopping, train
from operations.inference_ops.inference import infer
from operations.inference_ops.predict import predict

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

    if args.train_model:
        set_seed(2023)
        data = load_data(SPLIT_DATA_FILEPATH)
        train_texts, train_labels = data["train_texts"], data["train_labels"]
        valid_texts, valid_labels = data["valid_texts"], data["valid_labels"]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = DatasetLoader(train_texts, train_labels, tokenizer)
        valid_dataset = DatasetLoader(valid_texts, valid_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        model, optimizer = configure_model(args, tokenizer.vocab_size)
        loss_fn = nn.CrossEntropyLoss()
        model.to(DEVICE_TYPE)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train(model, optimizer, train_loader, valid_loader, loss_fn, epochs=NUM_EPOCHS, model_save_path=f'{MODEL_SAVE_PATH}/{args.model_name}', early_stopping=early_stopping)

    elif args.infer_model:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model, _ = configure_model(args, tokenizer.vocab_size)
        try:
            model.load_state_dict(torch.load(f'models/{args.model_name}'))
            model.to(DEVICE_TYPE)
            loss_fn = nn.CrossEntropyLoss()
            infer(model, args.model_name, loss_fn, SPLIT_DATA_FILEPATH)
        except FileNotFoundError:
            raise Exception("Model file not found. Ensure the model has been trained and saved.")

    elif args.predict:
        input_text = input('Enter the tweet for sentiment analysis: ')
        model_name = input('Enter the model you want to use for prediction: ')
        predictions = predict(input_text, model_name)
        pred_dict = {0: 'Angry', 1:'Disappointed', 2:'Happy'}
        model_pred = pred_dict[int(predictions)]
        print("Prediction: ", model_pred)
    else:
        print("Unknown argument")

if __name__ == '__main__':
    main()
