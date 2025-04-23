import matplotlib.pyplot as plt

"""
Training Logs:

=========Starting Training==========
 Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
------------------------------------------------------------
Validation loss decreased (inf --> 0.596894).  Saving model ...
   1    |   0.700583   |  0.596894  |   71.33   |  2573.30 
Validation loss decreased (0.596894 --> 0.564709).  Saving model ...
   2    |   0.588294   |  0.564709  |   73.75   |  2554.22 
Validation loss decreased (0.564709 --> 0.533575).  Saving model ...
   3    |   0.558299   |  0.533575  |   75.71   |  2462.80 
Validation loss decreased (0.533575 --> 0.503008).  Saving model ...
   4    |   0.529903   |  0.503008  |   77.43   |  2512.18 
Validation loss decreased (0.503008 --> 0.467562).  Saving model ...
   5    |   0.497885   |  0.467562  |   79.40   |  2531.43 
Validation loss decreased (0.467562 --> 0.421487).  Saving model ...
   6    |   0.459612   |  0.421487  |   81.58   |  2517.35 
Validation loss decreased (0.421487 --> 0.395031).  Saving model ...
   7    |   0.421157   |  0.395031  |   82.61   |  2483.92 
Validation loss decreased (0.395031 --> 0.370974).  Saving model ...
   8    |   0.391543   |  0.370974  |   83.94   |  2470.24 
Validation loss decreased (0.370974 --> 0.354762).  Saving model ...
   9    |   0.371066   |  0.354762  |   84.52   |  2510.26 
Validation loss decreased (0.354762 --> 0.340224).  Saving model ...
  10    |   0.354933   |  0.340224  |   85.53   |  2496.20 
Validation loss decreased (0.340224 --> 0.326807).  Saving model ...
  11    |   0.342084   |  0.326807  |   86.36   |  2451.51 
Validation loss decreased (0.326807 --> 0.316775).  Saving model ...
  12    |   0.331022   |  0.316775  |   86.85   |  2514.13 
Validation loss decreased (0.316775 --> 0.305672).  Saving model ...
  13    |   0.320353   |  0.305672  |   87.26   |  2508.51 
Validation loss decreased (0.305672 --> 0.298092).  Saving model ...
  14    |   0.311521   |  0.298092  |   87.56   |  7459.79 
Validation loss decreased (0.298092 --> 0.292167).  Saving model ...
  15    |   0.303113   |  0.292167  |   87.94   |  2561.33 
Validation loss decreased (0.292167 --> 0.291552).  Saving model ...
  16    |   0.295431   |  0.291552  |   87.84   |  2530.92 
Validation loss decreased (0.291552 --> 0.282584).  Saving model ...
  17    |   0.289125   |  0.282584  |   88.38   |  2524.01 
Validation loss decreased (0.282584 --> 0.275524).  Saving model ...
  18    |   0.283301   |  0.275524  |   88.54   |  2538.96 
Validation loss decreased (0.275524 --> 0.270303).  Saving model ...
  19    |   0.278111   |  0.270303  |   88.69   |  2478.53 
Validation loss decreased (0.270303 --> 0.270010).  Saving model ...
  20    |   0.273459   |  0.270010  |   88.63   |  3544.76 
Validation loss decreased (0.270010 --> 0.265370).  Saving model ...
  21    |   0.269308   |  0.265370  |   88.79   | 17880.92 
Validation loss decreased (0.265370 --> 0.261870).  Saving model ...
  22    |   0.265524   |  0.261870  |   89.04   |  5233.95 
Validation loss decreased (0.261870 --> 0.259126).  Saving model ...
  23    |   0.261935   |  0.259126  |   89.07   |  9761.28 
Validation loss decreased (0.259126 --> 0.257713).  Saving model ...
  24    |   0.259210   |  0.257713  |   89.24   |  3266.67 
Validation loss decreased (0.257713 --> 0.256730).  Saving model ...
  25    |   0.256556   |  0.256730  |   89.27   |  3501.73 
Validation loss decreased (0.256730 --> 0.253718).  Saving model ...
  26    |   0.253835   |  0.253718  |   89.26   |  4258.97 
Validation loss decreased (0.253718 --> 0.253171).  Saving model ...
  27    |   0.251684   |  0.253171  |   89.33   |  2821.59 
EarlyStopping counter: 1 out of 3
  28    |   0.249535   |  0.263590  |   88.46   |  5218.78 
Validation loss decreased (0.253171 --> 0.251313).  Saving model ...
  29    |   0.247034   |  0.251313  |   89.43   |  5083.44 
Validation loss decreased (0.251313 --> 0.248370).  Saving model ...
  30    |   0.245241   |  0.248370  |   89.55   |  4099.10 
EarlyStopping counter: 1 out of 3
  31    |   0.243448   |  0.250311  |   89.57   |  5272.52 
Validation loss decreased (0.248370 --> 0.247109).  Saving model ...
  32    |   0.241810   |  0.247109  |   89.64   |  7577.77 
Validation loss decreased (0.247109 --> 0.246605).  Saving model ...
  33    |   0.240338   |  0.246605  |   89.71   |  3511.63 
EarlyStopping counter: 1 out of 3
  34    |   0.238660   |  0.248492  |   89.49   |  4575.22 
EarlyStopping counter: 2 out of 3
  35    |   0.237187   |  0.251706  |   89.46   |  2583.77 
Validation loss decreased (0.246605 --> 0.244515).  Saving model ...
  36    |   0.235729   |  0.244515  |   89.79   |  2637.67 
Validation loss decreased (0.244515 --> 0.243931).  Saving model ...
  37    |   0.234221   |  0.243931  |   89.84   |  2596.08 
EarlyStopping counter: 1 out of 3
  38    |   0.233094   |  0.246824  |   89.49   |  2583.67 
EarlyStopping counter: 2 out of 3
  39    |   0.231512   |  0.244789  |   89.86   |  2580.76 
EarlyStopping counter: 3 out of 3
Early stopping


==========Best Accuracy After training================ 89.8590087890625
"""

# Data for the sixth training log
epochs = list(range(1, 40))
train_loss = [
    0.700583, 0.588294, 0.558299, 0.529903, 0.497885, 0.459612, 0.421157, 0.391543, 0.371066,
    0.354933, 0.342084, 0.331022, 0.320353, 0.311521, 0.303113, 0.295431, 0.289125, 0.283301,
    0.278111, 0.273459, 0.269308, 0.265524, 0.261935, 0.259210, 0.256556, 0.253835, 0.251684,
    0.249535, 0.247034, 0.245241, 0.243448, 0.241810, 0.240338, 0.238660, 0.237187, 0.235729,
    0.234221, 0.233094, 0.231512
]
val_loss = [
    0.596894, 0.564709, 0.533575, 0.503008, 0.467562, 0.421487, 0.395031, 0.370974, 0.354762,
    0.340224, 0.326807, 0.316775, 0.305672, 0.298092, 0.292167, 0.291552, 0.282584, 0.275524,
    0.270303, 0.270010, 0.265370, 0.261870, 0.259126, 0.257713, 0.256730, 0.253718, 0.253171,
    0.263590, 0.251313, 0.248370, 0.250311, 0.247109, 0.246605, 0.248492, 0.251706, 0.244515,
    0.243931, 0.246824, 0.244789
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss (Sixth Training Run)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
