# Twitter Sentiment Anaylsis

This README provides clear instructions on how to install dependencies, how to run different components of the project, and how to contribute to the project, tailored for users familiar with GitHub projects.

- This project applies deep learning techniques to analyze sentiments expressed in tweets. 
- It uses six different models implemented in PyTorch: MLP, CNN, BiLSTM, BiLSTM with Multi-Head Attention, RCNN Text Classifier (LSTM+CNN) and BiGRU Attention Residual. 
- Each model aims to effectively capture the contextual nuances of Twitter data. 
- The source code of the project is present in the project/src directory.

## Models Overview:

1. MLP Classifier:
    - Input: (EMBEDDING_DIM*SEQUENCE_LENGTH, 2048)
    - First Hidden Layer: (2048, 1024)
    - Second Hidden Layer: (1024, 612)
    - Final Layer: (612, 3)
2. CNN Model:
    - filters:  Sizes of 3, 4, 5 with filter counts of 32, 64, 128 respectively
    - Pooling: Max pooling after each convolution
    - Output Layer: Sum of filters to 3 units

3. LSTM Text Classifier
    - Configuration: Bidirectional with 6 layers
    - Hidden State: 128 units
    - Output: Fully connected layer

4. LSTM Multi Head Attention:
    - Configuration: Bidirectional with 6 layers and 8 attention heads
    - Hidden State: 128 units
    - Output: Combined attention heads passed through a fully connected layer

5. RCNN Text Classifier:
    - Embedding Dimension: 128
    - Recurrent Layer: LSTM layer to capture sequential features
    - Convolutional Layer: Extracts local patterns from recurrent outputs
    - Dropout: 0.5
    - Output Layer: Dense layer mapping to 3 sentiment classes

6. BiGRU with Attention and Residual Connections:
    - Configuration: Bidirectional GRU with 2 layers
    - Embedding Dimension: 128
    - Hidden State: 128 units
    - Attention Mechanism: Applied to GRU outputs to focus on relevant tokens
    - Residual Connections: Enhance gradient flow and model depth
    - Dropout: 0.5
    - Output Layer: Dense layer projecting to 3 sentiment classes 

## How to run the project:

### Prerequisites:
- Before running the project, ensure you have Python installed on your system. 
- Install all required Python packages using the following command in the root directory of the project:
```bash
python -m pip install -r project/requirements.txt
```
OR (if you use anaconda)
```bash
conda create -n myenv python=3.x
conda activate myenv
conda install pip
pip install -r requirements.txt
```

- Navigate to the src directory before running any commands to test the project
```bash
cd project/src/
```

### Load and Clean the dataset:

- This is the first step to load the data and preparing it for modelling. 
- Run the following command first:
```bash
python -m main --load_dataset
```

### Training individual models:
- Type in the following command to train MLP model:
```bash
 python -m main --model_name MLP_Classifier --train-model
```
You can replace this with any model you want to train:
The following are the available model_name arguments:
1. MLP_Classifier
2. CNN_Model
3. LSTM_Text_Classifier
4. LSTM_Multi_Head_Attention
5. RCNN_Text_Classifier
6. BiGRU_Attention_Residual

- Training these models on a large dataset may take between 1 to 75 hours depending on the model complexity and the hardware used.

### Testing the model:

- Once the model has be trained, it's performance can be evaluated using the test set. 

- Run the following command from inside the src folder:
```bash
 python -m src.main --model_name MLP_Classifier --infer-model
```
- Replace the model name with the model inference you want to visualize. 
- The model_name will have the same name as the one you trained.

## Project Organization
------------

    ├── LICENSE
    ├── README.md   <- The top-level README for developers using this project.
    ├── project     <- All the source code/notebooks/figures
    │   ├── notebooks <- Jupyter notebooks
    │   │   ├── data_analysis.ipynb
    │   │   ├── data_preprocessing.ipynb
    │   │   ├── inference_and_predictions.ipynb
    │   │   ├── models  <- Models trained for 1 Epoch
    │   │   │   ├── CNN_Model_epoch_1
    │   │   │   ├── LSTM_Multi_Head_Attention_epoch_1
    │   │   │   ├── LSTM_Text_Classifier_epoch_1
    │   │   │   ├── MLP_Classifier_epoch_1
    │   │   │   └── RCNN_Text_Classifier_epoch_1
    │   │   └── training.ipynb
    │   ├── requirements.txt
    │   ├── setup.py
    │   └── src
    │       ├── __init__.py
    │       ├── classification_table.py
    │       ├── configs
    │       │   └── hyperparams.yaml
    │       ├── dataset
    │       │   ├── processed  <- The final, canonical data sets for modeling.
    │       │   └── raw        <- The original, immutable data dump.
    │       ├── main.py
    │       ├── models         <- Models trained for 50 Epochs or Early Stopping
    │       │   ├── BiGRU_Attention_Residual
    │       │   ├── CNN_Model
    │       │   ├── LSTM_Multi_Head_Attention
    │       │   ├── LSTM_Text_Classifier
    │       │   ├── MLP_Classifier
    │       │   └── RCNN_Text_Classifier
    │       ├── models_5_epochs <- Models trained for 5 Epochs or Early Stopping     
    │       │   ├── CNN_Model
    │       │   ├── LSTM_Multi_Head_Attention
    │       │   ├── LSTM_Text_Classifier
    │       │   ├── MLP_Classifier
    │       │   └── RCNN_Text_Classifier
    │       ├── operations
    │       │   ├── __init__.py
    │       │   ├── dataset_ops     <- Dataset operations
    │       │   │   ├── __init__.py
    │       │   │   ├── dataset_loader_ops.py
    │       │   │   └── dataset_ops.py 
    │       │   ├── inference_ops
    │       │   │   ├── __init__.py
    │       │   │   ├── inference.py  <- Scripts to infer models and gen. confusion matrix
    │       │   │   └── predict.py    <- Scripts to predict using trained models
    │       │   └── model_ops
    │       │       ├── __init__.py
    │       │       ├── bigru_attention_residuals.py<- BiGRU Attention Residual Model
    │       │       ├── cnn.py                      <- CNN Model
    │       │       ├── lstm_multihead.py           <- BiLSTM Multi Head Attention Model
    │       │       ├── lstm_text_classifier.py     <- BiLSTM Model
    │       │       ├── mlp_classifier.py           <- MLP Classifier
    │       │       ├── rcnn_text_classifier.py     <- RCNN (LSTM + CNN)
    │       │       └── train_model.py  <- Scripts to train models
    │       ├── reports <- Confusion Matrices and Loss Curve Images
    │       │   └── figures
    │       ├── training_logs   <- Logs for different EPOCH trainings
    │       ├── utils           <- Helper functions and classes
    │       │   ├── constants.py  <- Constant Declarations
    │       │   ├── helper_utils.py
    │       │   └── model_utils.py
    │       └── visualizations  <- Scripts to create loss curve visualizations
    │           ├── bigru_attention_residual_vis.py
    │           ├── cnn_vis.py
    │           ├── lstm_multihead_attention_vis.py
    │           ├── lstm_text_classifier_vis.py
    │           ├── mlp_classifier_vis.py
    │           └── rcnn_classifier_vis.py
    └── report_latex  <- Latex Tools for report
    └── Report.pdf    <- Report
--------

