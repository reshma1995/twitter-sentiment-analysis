"""
Constants
"""
import torch

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
else:
    DEVICE_TYPE = "cpu"

MAX_SEQ_LEN = 120
NUM_EPOCHS = 50
BATCH_SIZE = 128

CONFIG_PARAMS_FILEPATH = "configs/hyperparams.yaml"
DATSET_FILEPATH = "dataset/raw/dataset(clean).csv"
PREPROCESSED_DATA_PATH = 'dataset/processed/'
SPLIT_DATA_FILEPATH = PREPROCESSED_DATA_PATH + "split_data.json"
CLEANED_FILE_NAME = 'cleaned_tweets.csv'

MODEL_SAVE_PATH = "models"
INFERENCE_FILE_PATH = 'reports/figures/'