import matplotlib.pyplot as plt

"""
Training Log for LSTM_Multi_Head_Attention Model:

=========Starting Training==========
 Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
------------------------------------------------------------
Validation loss decreased (inf --> 1.098420).  Saving model ...
   1    |   1.098408   |  1.098420  |   34.19   |  867.84  
Validation loss decreased (1.098420 --> 1.098364).  Saving model ...
   2    |   1.098351   |  1.098364  |   34.19   |  879.14  
Validation loss decreased (1.098364 --> 1.098209).  Saving model ...
   3    |   1.098257   |  1.098209  |   34.19   |  937.09  
Validation loss decreased (1.098209 --> 1.096982).  Saving model ...
   4    |   1.097760   |  1.096982  |   34.61   |  954.73  
Validation loss decreased (1.096982 --> 1.074597).  Saving model ...
   5    |   1.090541   |  1.074597  |   41.52   |  985.99  
Validation loss decreased (1.074597 --> 0.983233).  Saving model ...
   6    |   1.046662   |  0.983233  |   47.62   |  4302.17 
Validation loss decreased (0.983233 --> 0.869567).  Saving model ...
   7    |   0.928236   |  0.869567  |   52.25   |  5055.43 
Validation loss decreased (0.869567 --> 0.809683).  Saving model ...
   8    |   0.834445   |  0.809683  |   60.22   |  874.41  
Validation loss decreased (0.809683 --> 0.687233).  Saving model ...
   9    |   0.741532   |  0.687233  |   69.94   |  922.28  
Validation loss decreased (0.687233 --> 0.593651).  Saving model ...
  10    |   0.636822   |  0.593651  |   74.98   |  936.22  
Validation loss decreased (0.593651 --> 0.546111).  Saving model ...
  11    |   0.565876   |  0.546111  |   77.46   |  5401.95 
Validation loss decreased (0.546111 --> 0.540135).  Saving model ...
  12    |   0.521171   |  0.540135  |   77.93   |  868.67  
Validation loss decreased (0.540135 --> 0.481859).  Saving model ...
  13    |   0.488976   |  0.481859  |   81.17   |  940.37  
Validation loss decreased (0.481859 --> 0.463855).  Saving model ...
  14    |   0.467914   |  0.463855  |   82.37   |  971.01  
 Validation loss decreased (0.463855 --> 0.462761).  Saving model ...
  15    |   0.451802   |  0.462761  |   82.37   |  939.84  
Validation loss decreased (0.462761 --> 0.447131).  Saving model ...
  16    |   0.437500   |  0.447131  |   83.77   |  943.20  
Validation loss decreased (0.447131 --> 0.430636).  Saving model ...
  17    |   0.424334   |  0.430636  |   84.11   |  946.13  
Validation loss decreased (0.430636 --> 0.416366).  Saving model ...
  18    |   0.410540   |  0.416366  |   84.93   |  960.41  
Validation loss decreased (0.416366 --> 0.411131).  Saving model ...
  19    |   0.394694   |  0.411131  |   84.88   |  925.60  
Validation loss decreased (0.411131 --> 0.383591).  Saving model ...
  20    |   0.373611   |  0.383591  |   84.91   |  1009.03 
Validation loss decreased (0.383591 --> 0.353435).  Saving model ...
  21    |   0.351265   |  0.353435  |   86.15   |  932.33  
Validation loss decreased (0.353435 --> 0.347053).  Saving model ...
  22    |   0.327921   |  0.347053  |   86.38   |  870.82  
Validation loss decreased (0.347053 --> 0.315123).  Saving model ...
  23    |   0.310769   |  0.315123  |   87.23   |  931.72  
Validation loss decreased (0.315123 --> 0.305082).  Saving model ...
  24    |   0.298941   |  0.305082  |   87.44   |  976.61  
Validation loss decreased (0.305082 --> 0.297693).  Saving model ...
  25    |   0.290193   |  0.297693  |   87.80   |  951.55  
Validation loss decreased (0.297693 --> 0.292942).  Saving model ...
  26    |   0.282936   |  0.292942  |   87.93   |  997.24  
Validation loss decreased (0.292942 --> 0.288946).  Saving model ...
  27    |   0.277160   |  0.288946  |   88.01   |  1023.48 
EarlyStopping counter: 1 out of 3
  28    |   0.271761   |  0.295369  |   87.47   |  958.10  
Validation loss decreased (0.288946 --> 0.277413).  Saving model ...
  29    |   0.267654   |  0.277413  |   88.48   |  953.82  
EarlyStopping counter: 1 out of 3
  30    |   0.263436   |  0.280371  |   88.45   |  953.40  
Validation loss decreased (0.277413 --> 0.274593).  Saving model ...
  31    |   0.259903   |  0.274593  |   88.71   |  953.60  
EarlyStopping counter: 1 out of 3
  32    |   0.256757   |  0.276220  |   88.50   |  1001.38 
Validation loss decreased (0.274593 --> 0.268554).  Saving model ...
  33    |   0.253832   |  0.268554  |   88.81   |  1006.90 
EarlyStopping counter: 1 out of 3
  34    |   0.251724   |  0.275192  |   88.60   |  929.11  
EarlyStopping counter: 2 out of 3
  35    |   0.249342   |  0.277226  |   88.38   |  931.80  
Validation loss decreased (0.268554 --> 0.266194).  Saving model ...
  36    |   0.247420   |  0.266194  |   89.05   |  953.53  
Validation loss decreased (0.266194 --> 0.264713).  Saving model ...
  37    |   0.245492   |  0.264713  |   89.10   |  962.36  
EarlyStopping counter: 1 out of 3
  38    |   0.243684   |  0.268147  |   88.85   |  873.98  
EarlyStopping counter: 2 out of 3
  39    |   0.241843   |  0.265277  |   89.02   |  868.29  
EarlyStopping counter: 3 out of 3
Early stopping
"""

# Extracted values from the training log
epochs = list(range(1, 40))
train_loss = [
    1.098408, 1.098351, 1.098257, 1.097760, 1.090541, 1.046662, 0.928236, 0.834445, 0.741532,
    0.636822, 0.565876, 0.521171, 0.488976, 0.467914, 0.451802, 0.437500, 0.424334, 0.410540,
    0.394694, 0.373611, 0.351265, 0.327921, 0.310769, 0.298941, 0.290193, 0.282936, 0.277160,
    0.271761, 0.267654, 0.263436, 0.259903, 0.256757, 0.253832, 0.251724, 0.249342, 0.247420,
    0.245492, 0.243684, 0.241843
]
val_loss = [
    1.098420, 1.098364, 1.098209, 1.096982, 1.074597, 0.983233, 0.869567, 0.809683, 0.687233,
    0.593651, 0.546111, 0.540135, 0.481859, 0.463855, 0.462761, 0.447131, 0.430636, 0.416366,
    0.411131, 0.383591, 0.353435, 0.347053, 0.315123, 0.305082, 0.297693, 0.292942, 0.288946,
    0.295369, 0.277413, 0.280371, 0.274593, 0.276220, 0.268554, 0.275192, 0.277226, 0.266194,
    0.264713, 0.268147, 0.265277
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss (LSTM Multi Head Attention)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
