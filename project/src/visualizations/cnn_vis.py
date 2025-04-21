import matplotlib.pyplot as plt

"""
Training Logs for CNN Model

=========Starting Training==========
 Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
------------------------------------------------------------
Validation loss decreased (inf --> 0.537215).  Saving model ...
   1    |   0.646957   |  0.537215  |   75.79   |  256.57  
Validation loss decreased (0.537215 --> 0.445877).  Saving model ...
   2    |   0.496096   |  0.445877  |   81.26   |  258.59  
Validation loss decreased (0.445877 --> 0.370125).  Saving model ...
   3    |   0.407540   |  0.370125  |   84.55   |  258.30  
Validation loss decreased (0.370125 --> 0.323019).  Saving model ...
   4    |   0.341714   |  0.323019  |   86.64   |  258.71  
Validation loss decreased (0.323019 --> 0.299926).  Saving model ...
   5    |   0.299586   |  0.299926  |   87.44   |  258.53  
Validation loss decreased (0.299926 --> 0.284685).  Saving model ...
   6    |   0.273239   |  0.284685  |   88.00   |  259.55  
Validation loss decreased (0.284685 --> 0.275670).  Saving model ...
   7    |   0.255203   |  0.275670  |   88.35   |  259.20  
Validation loss decreased (0.275670 --> 0.273548).  Saving model ...
   8    |   0.240994   |  0.273548  |   88.39   |  259.27  
Validation loss decreased (0.273548 --> 0.270748).  Saving model ...
   9    |   0.229654   |  0.270748  |   88.56   |  259.52  
EarlyStopping counter: 1 out of 3
  10    |   0.219828   |  0.271374  |   88.67   |  259.28  
EarlyStopping counter: 2 out of 3
  11    |   0.211008   |  0.273041  |   88.79   |  259.24  
EarlyStopping counter: 3 out of 3
Early stopping


==========Best Accuracy After training================ 88.79215240478516

"""

epochs = list(range(1, 12))
train_loss = [
    0.646957, 0.496096, 0.407540, 0.341714, 0.299586, 0.273239,
    0.255203, 0.240994, 0.229654, 0.219828, 0.211008
]
val_loss = [
    0.537215, 0.445877, 0.370125, 0.323019, 0.299926, 0.284685,
    0.275670, 0.273548, 0.270748, 0.271374, 0.273041
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss (CNN Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

