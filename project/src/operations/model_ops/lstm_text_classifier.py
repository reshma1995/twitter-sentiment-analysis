import torch
import torch.nn as nn


class LSTM_Text_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        """
        Initialize the LSTMTextClassifier model.

        Parameters:
        @vocab_size: Size of the vocabulary.
        @embedding_dim: Dimension of the input embeddings.
        @hidden_dim: Dimension of the hidden state in the LSTM.
        @output_dim: Number of classes in the output layer.
        @n_layers: Number of layers in the LSTM.
        @bidirectional: If True, initializes a bidirectional LSTM.
        @dropout: Dropout rate for regularization.
        """
        super(LSTM_Text_Classifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the model.
        Params:
        @x: input_ids
        Returns: The logits for each class.
        """
        text_embeddings = self.embedding(x)
        lstm_out, (hidden,cell) = self.lstm(text_embeddings)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits