import json
from utils.helper_utils import read_config_file, set_seed, init_model
from utils.constants import CONFIG_PARAMS_FILEPATH
from operations.model_ops.cnn import CNN_Model
from operations.model_ops.lstm_text_classifier import LSTM_Text_Classifier
from operations.model_ops.lstm_multihead import LSTM_Multi_Head_Attention
from operations.model_ops.mlp_classifier import MLP_Classifier

def load_data(json_path):
    """
    Function to read json file
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise Exception(f"The file {json_path} was not found.")
    except json.JSONDecodeError:
        raise Exception(f"The file {json_path} could not be decoded.")
    
def configure_model(args, vocab_size):
    model_class = {
        'CNN_Model': CNN_Model,
        'LSTM_Text_Classifier': LSTM_Text_Classifier,
        'MLP_Classifier': MLP_Classifier,
        'LSTM_Multi_Head_Attention': LSTM_Multi_Head_Attention
    }.get(args.model_name, CNN_Model)
    
    model_params = read_config_file(CONFIG_PARAMS_FILEPATH, args.model_name)
    model_params['vocab_size'] = vocab_size
    set_seed(2023)
    model, optimizer = init_model(model_class, model_params)
    return model, optimizer