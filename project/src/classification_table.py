import json
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from transformers import BertTokenizer

from utils.constants import DEVICE_TYPE, MAX_SEQ_LEN, CONFIG_PARAMS_FILEPATH, MODEL_SAVE_PATH
from utils.helper_utils import read_config_file
from operations.model_ops.cnn import CNN_Model
from operations.model_ops.lstm_multihead import LSTM_Multi_Head_Attention
from operations.model_ops.lstm_text_classifier import LSTM_Text_Classifier
from operations.model_ops.mlp_classifier import MLP_Classifier
from operations.model_ops.bigru_attention_residuals import BiGRU_Attention_Residual
from operations.model_ops.rcnn_text_classifier import RCNN_Text_Classifier
from operations.dataset_ops.dataset_ops import CleanTweets

# Load dataset
with open("dataset/processed/split_data.json", 'r') as f:
    data = json.load(f)

test_texts = data["test_texts"]
test_labels = data["test_labels"]
y_true = [label.index(1.0) for label in test_labels]

# Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cleaned_texts = CleanTweets(test_texts).clean_tweet()

def encode_texts(texts):
    input_ids = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
    return torch.stack(input_ids)

X_test_tensor = encode_texts(cleaned_texts)

# Map model keys to classes
model_map = {
    "MLP_Classifier": MLP_Classifier,
    "CNN_Model": CNN_Model,
    "LSTM_Text_Classifier": LSTM_Text_Classifier,
    "LSTM_Multi_Head_Attention": LSTM_Multi_Head_Attention,
    "RCNN_Text_Classifier": RCNN_Text_Classifier,
    "BiGRU_Attention_Residual": BiGRU_Attention_Residual
}

results = []
target_names = ["angry", "disappointed", "happy"]

def evaluate_model(model_key):
    model_class = model_map[model_key]
    params = read_config_file(CONFIG_PARAMS_FILEPATH, model_key)
    params['vocab_size'] = tokenizer.vocab_size
    model = model_class(**params)
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/{model_key}", map_location=DEVICE_TYPE))
    model.to(DEVICE_TYPE)
    model.eval()
    print("Evaluating...", len(X_test_tensor))
    preds = []
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            print(f"Done {i} out of {len(X_test_tensor)} for model {model_key}")
            input_tensor = X_test_tensor[i].unsqueeze(0).to(DEVICE_TYPE)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            preds.append(pred)
    print("Loading report")
    report = classification_report(y_true, preds, target_names=target_names, output_dict=True)
    acc = accuracy_score(y_true, preds) * 100
    return [
        model_key.replace("_", "-"),
        round(report["angry"]["precision"] * 100, 2),
        round(report["angry"]["recall"] * 100, 2),
        round(report["angry"]["f1-score"] * 100, 2),
        round(report["disappointed"]["precision"] * 100, 2),
        round(report["disappointed"]["recall"] * 100, 2),
        round(report["disappointed"]["f1-score"] * 100, 2),
        round(report["happy"]["precision"] * 100, 2),
        round(report["happy"]["recall"] * 100, 2),
        round(report["happy"]["f1-score"] * 100, 2),
        round(acc, 2)
    ]

# Run evaluation
for model_key in model_map:
    print("Evaluating")
    results.append(evaluate_model(model_key))

# Display
columns = [
    "Model",
    "Angry P", "Angry R", "Angry F1",
    "Disappointed P", "Disappointed R", "Disappointed F1",
    "Happy P", "Happy R", "Happy F1",
    "Accuracy %"
]

df = pd.DataFrame(results, columns=columns)
print(df.to_string(index=False))
