import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRU_Attention_Residual(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        """
        Stacked BiGRU with Attention and Residual Connections for text classification.

        Params:
        @vocab_size: Size of vocabulary.
        @embedding_dim: Dimension of the input embeddings.
        @hidden_dim: Hidden state size of GRU.
        @output_dim: Number of output classes.
        @n_layers: Number of stacked GRU layers.
        @dropout: Dropout rate.
        """
        super(BiGRU_Attention_Residual, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.residual_proj = nn.Linear(embedding_dim, hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass for the model.

        @x: Input token ids tensor of shape [batch_size, seq_len]
        Returns: Logits tensor of shape [batch_size, output_dim]
        """
        embedded = self.embedding(x)
        gru_out, _ = self.bigru(embedded) 

        # Residual connection (project input embedding to match GRU output dim)
        residual = self.residual_proj(embedded) 
        gru_out += residual

        # Attention weights
        attention_scores = self.attention(gru_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1) 

        # Context vector
        context_vector = torch.bmm(attention_weights, gru_out).squeeze(1) 

        output = self.dropout(context_vector)
        logits = self.fc(output) 

        return logits
