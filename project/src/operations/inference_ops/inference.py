import os
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torch.utils.data import  DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from operations.dataset_ops.dataset_loader_ops import DatasetLoader
from utils.constants import DEVICE_TYPE, BATCH_SIZE, INFERENCE_FILE_PATH


def inference(model, test_loader, loss_fn, device=DEVICE_TYPE):
    """
    Function to make inference of the trained model. The function will display the classification report and also 
    the accuracy of each of the model
    """
    model.eval()
    total_loss, total_correct = 0, 0
    total_samples = 0
    all_preds, all_labels = [], []
    heads_count = 8
    with torch.no_grad():
        for test_batch_input, test_batch_label in test_loader:
            input_ids, attention_masks = [t.to(device) for t in test_batch_input]
            test_batch_label = test_batch_label.to(device)
            logits = model(input_ids)
            loss = loss_fn(logits, test_batch_label)
            total_loss += loss.item() * test_batch_label.size(0)
            preds = torch.argmax(logits, dim=1)
            test_batch_labels = torch.argmax(test_batch_label, dim=1)
            total_correct += (preds == test_batch_labels).sum().item()
            total_samples += test_batch_label.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_batch_labels.cpu().numpy())  
    avg_loss = total_loss / total_samples
    avg_accuracy = (total_correct / total_samples) * 100
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4, target_names=['angry', 'disappointed', 'happy'])
    average_attention_weights = []
    return avg_loss, avg_accuracy, average_attention_weights, cm, cr


def infer(model, model_name, loss_fn, file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_texts, test_labels = data['test_texts'], data['test_labels']
        
    except FileNotFoundError:
        raise Exception(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        raise Exception(f"The file {file_name} could not be decoded.")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = DatasetLoader(test_texts, test_labels,tokenizer)
    print("Number of records in test set is: ", len(test_dataset))
    num_cpus = os.cpu_count()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_cpus)
    os.makedirs(os.path.dirname(INFERENCE_FILE_PATH), exist_ok=True)
    _, avg_accuracy, attention_weights, cm, cr = inference(model, test_loader, loss_fn, DEVICE_TYPE)
    print("Accuracy: ", avg_accuracy)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(INFERENCE_FILE_PATH,model_name+'confusion_matrix_cnn.png'))
    plt.show()
    print("Classification Report:\n", cr)
    


        
        
        
        
    
    