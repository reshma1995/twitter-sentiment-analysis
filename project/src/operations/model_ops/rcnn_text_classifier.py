import torch
import torch.nn.functional as F
import torch.nn as nn

class RCNN_Text_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, dropout=0.5):
        """
        RCNN model combining LSTM and CNN.

        Params:
        @vocab_size: Size of vocabulary.
        @embedding_dim: Dimension of input embeddings.
        @hidden_dim: Hidden state size for LSTM.
        @output_dim: Number of output classes.
        @dropout: Dropout rate.
        """
        super(RCNN_Text_Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(in_channels=embedding_dim + 2 * hidden_dim, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        combined = torch.cat([embeddings, lstm_out], dim=2)
        combined = combined.permute(0, 2, 1)
        conv_out = F.relu(self.conv(combined))
        pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits
