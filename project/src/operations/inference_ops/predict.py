import torch
from transformers import BertTokenizer
from utils.constants import DEVICE_TYPE, MAX_SEQ_LEN, CONFIG_PARAMS_FILEPATH
from utils.helper_utils import read_config_file
from operations.model_ops.cnn import CNN_Model
from operations.model_ops.lstm_multihead import LSTM_Multi_Head_Attention
from operations.model_ops.lstm_text_classifier import LSTM_Text_Classifier
from operations.model_ops.mlp_classifier import MLP_Classifier
from operations.dataset_ops.dataset_ops import CleanTweets

def predict(input_text, model):
    input_text = [input_text]
    cleaned_tweet = CleanTweets(input_text).clean_tweet()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_text = tokenizer.encode_plus(
        cleaned_tweet,
        add_special_tokens=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    token_ids, atten_masks = encoded_text['input_ids'].squeeze(0), encoded_text['attention_mask'].squeeze(0)
    vocab_size = tokenizer.vocab_size
    if model == 'CNN':
        model_params = read_config_file(CONFIG_PARAMS_FILEPATH, 'CNN_Model')
        model_params['vocab_size'] = vocab_size
        model = CNN_Model(**model_params)
        model.load_state_dict(torch.load('models/CNN_Model'))
    elif model == 'LSTM':
        model_params = read_config_file(CONFIG_PARAMS_FILEPATH, 'LSTM_Text_Classifier')
        model_params['vocab_size'] = vocab_size
        model = LSTM_Text_Classifier(**model_params)
        model.load_state_dict(torch.load('models/LSTM_Text_Classifier'))
    elif model == 'MLP':
        model_params = read_config_file(CONFIG_PARAMS_FILEPATH, 'MLP_Classifier')
        model_params['vocab_size'] = vocab_size
        model = MLP_Classifier(**model_params)
        model.load_state_dict(torch.load('models/MLP_Classifier'))
    else:
        model_params = read_config_file(CONFIG_PARAMS_FILEPATH, 'LSTM_Multi_Head_Attention')
        model_params['vocab_size'] = vocab_size
        model = LSTM_Multi_Head_Attention(**model_params)
        model.load_state_dict(torch.load('models/LSTM_Multi_Head_Attention'))
    
    model.to(DEVICE_TYPE)
    token_ids = token_ids.to(DEVICE_TYPE)
    model.eval()
    with torch.no_grad():
        output = model(token_ids.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()
    return prediction

    