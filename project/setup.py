import gdown
import os

# The models are available in a public drive link
google_drive_links = [
    {
        "link": 'https://drive.google.com/uc?id=1mCzvcKPYFVHK2BKQgQNdZCD_pXOkJ58v',
        "model": "BiGRU_Attention_Residual"
    },
    {
        "link": 'https://drive.google.com/uc?id=1GXEvoJB7bRtRhPMT76oKJc8Dbpt9V7Nd',
        "model": "RCNN_Text_Classifier"
    },
    {
        "link": 'https://drive.google.com/uc?id=1PMOUv7vTZbLrXJRlkw2h1X7RTSrs-2rM',
        "model": "LSTM_Multi_Head_Attention"
    },
    {
        "link": 'https://drive.google.com/uc?id=1EuDnAeisrSmUQWtXY4KTssKZNl4cqsTu',
        "model": "MLP_Classifier"
    },
    {
        "link": 'https://drive.google.com/uc?id=14F5iax7OLhKcdoIOykbmp7OBtsoJbgTX',
        "model": "LSTM_Text_Classifier"
    },
    {
        "link": "https://drive.google.com/uc?id=1R0tNVOwNvQsELw0wYs1DnF41mR1SNyzJ",
        "model": "CNN_Model"
    }
]

base_dir = "src/models/"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for i in google_drive_links:
    
    output_file = base_dir + i["model"]
    if os.path.exists(output_file):
        print(output_file, " already exists")
    else:
        gdown.download(i["link"], output_file, quiet=False)


