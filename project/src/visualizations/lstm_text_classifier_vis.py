import matplotlib.pyplot as plt

"""
Training Logs of LSTM Text Classifier Model:

=========Starting Training==========
 Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
------------------------------------------------------------
Validation loss decreased (inf --> 1.098439).  Saving model ...
   1    |   1.098575   |  1.098439  |   34.19   |  639.21  
Validation loss decreased (1.098439 --> 1.098383).  Saving model ...
   2    |   1.098510   |  1.098383  |   34.19   |  599.90  
Validation loss decreased (1.098383 --> 1.098209).  Saving model ...
   3    |   1.098404   |  1.098209  |   34.19   |  583.06  
Validation loss decreased (1.098209 --> 1.096271).  Saving model ...
   4    |   1.097755   |  1.096271  |   34.61   |  571.91  
Validation loss decreased (1.096271 --> 0.887419).  Saving model ...
   5    |   1.029901   |  0.887419  |   53.81   |  574.39  
Validation loss decreased (0.887419 --> 0.755923).  Saving model ...
   6    |   0.819687   |  0.755923  |   65.08   |  578.94  
Validation loss decreased (0.755923 --> 0.661666).  Saving model ...
   7    |   0.712001   |  0.661666  |   70.88   |  578.20  
Validation loss decreased (0.661666 --> 0.563350).  Saving model ...
   8    |   0.613239   |  0.563350  |   76.32   |  574.32  
Validation loss decreased (0.563350 --> 0.494737).  Saving model ...
   9    |   0.528067   |  0.494737  |   78.92   |  573.39  
Validation loss decreased (0.494737 --> 0.469499).  Saving model ...
  10    |   0.484306   |  0.469499  |   80.41   |  598.95  
Validation loss decreased (0.469499 --> 0.446962).  Saving model ...
  11    |   0.461654   |  0.446962  |   81.93   |  577.20  
Validation loss decreased (0.446962 --> 0.425843).  Saving model ...
  12    |   0.437605   |  0.425843  |   82.69   |  569.23  
Validation loss decreased (0.425843 --> 0.408439).  Saving model ...
  13    |   0.416049   |  0.408439  |   83.47   |  573.63  
Validation loss decreased (0.408439 --> 0.390492).  Saving model ...
  14    |   0.398495   |  0.390492  |   84.23   |  577.58  
Validation loss decreased (0.390492 --> 0.380516).  Saving model ...
  15    |   0.384425   |  0.380516  |   84.50   |  577.40  
Validation loss decreased (0.380516 --> 0.371920).  Saving model ...
  16    |   0.372062   |  0.371920  |   84.93   |  577.51  
Validation loss decreased (0.371920 --> 0.357032).  Saving model ...
  17    |   0.360317   |  0.357032  |   85.57   |  574.97  
Validation loss decreased (0.357032 --> 0.347759).  Saving model ...
  18    |   0.351489   |  0.347759  |   85.91   |  572.79  
Validation loss decreased (0.347759 --> 0.341368).  Saving model ...
  19    |   0.343217   |  0.341368  |   86.16   |  572.98  
Validation loss decreased (0.341368 --> 0.337906).  Saving model ...
  20    |   0.336104   |  0.337906  |   86.35   |  574.94  
Validation loss decreased (0.337906 --> 0.329790).  Saving model ...
  21    |   0.328781   |  0.329790  |   86.60   |  578.25  
Validation loss decreased (0.329790 --> 0.325249).  Saving model ...
  22    |   0.323049   |  0.325249  |   86.86   |  573.42  
Validation loss decreased (0.325249 --> 0.319597).  Saving model ...
  23    |   0.318964   |  0.319597  |   87.10   |  577.14  
Validation loss decreased (0.319597 --> 0.315914).  Saving model ...
  24    |   0.313975   |  0.315914  |   87.25   |  573.17  
Validation loss decreased (0.315914 --> 0.312718).  Saving model ...
  25    |   0.309854   |  0.312718  |   87.43   |  574.12  
Validation loss decreased (0.312718 --> 0.310650).  Saving model ...
  26    |   0.305774   |  0.310650  |   87.44   |  573.48  
Validation loss decreased (0.310650 --> 0.304164).  Saving model ...
  27    |   0.301547   |  0.304164  |   87.74   |  576.33  
EarlyStopping counter: 1 out of 3
  28    |   0.299023   |  0.306234  |   87.65   |  577.43  
Validation loss decreased (0.304164 --> 0.299984).  Saving model ...
  29    |   0.295527   |  0.299984  |   87.86   |  577.64  
Validation loss decreased (0.299984 --> 0.298119).  Saving model ...
  30    |   0.292448   |  0.298119  |   87.88   |  578.30  
Validation loss decreased (0.298119 --> 0.296045).  Saving model ...
  31    |   0.289175   |  0.296045  |   87.98   |  569.03  
Validation loss decreased (0.296045 --> 0.291246).  Saving model ...
  32    |   0.286482   |  0.291246  |   88.14   |  568.90  
Validation loss decreased (0.291246 --> 0.287574).  Saving model ...
  33    |   0.283945   |  0.287574  |   88.30   |  568.97  
EarlyStopping counter: 1 out of 3
  34    |   0.281046   |  0.289649  |   88.19   |  569.21  
Validation loss decreased (0.287574 --> 0.285981).  Saving model ...
  35    |   0.278731   |  0.285981  |   88.34   |  569.09  
Validation loss decreased (0.285981 --> 0.282947).  Saving model ...
  36    |   0.277225   |  0.282947  |   88.52   |  569.22  
EarlyStopping counter: 1 out of 3
  37    |   0.275269   |  0.283351  |   88.47   |  570.02  
Validation loss decreased (0.282947 --> 0.279935).  Saving model ...
  38    |   0.272160   |  0.279935  |   88.62   |  571.90  
Validation loss decreased (0.279935 --> 0.278331).  Saving model ...
  39    |   0.270574   |  0.278331  |   88.61   |  572.26  
Validation loss decreased (0.278331 --> 0.276415).  Saving model ...
  40    |   0.268300   |  0.276415  |   88.74   |  571.92  
EarlyStopping counter: 1 out of 3
  41    |   0.266757   |  0.276797  |   88.70   |  572.07  
Validation loss decreased (0.276415 --> 0.273313).  Saving model ...
  42    |   0.265443   |  0.273313  |   88.86   |  571.75  
Validation loss decreased (0.273313 --> 0.272697).  Saving model ...
  43    |   0.263102   |  0.272697  |   88.88   |  571.68  
Validation loss decreased (0.272697 --> 0.271695).  Saving model ...
  44    |   0.261606   |  0.271695  |   88.90   |  571.91  
EarlyStopping counter: 1 out of 3
  45    |   0.259659   |  0.278070  |   88.65   |  576.78  
Validation loss decreased (0.271695 --> 0.270396).  Saving model ...
  46    |   0.258620   |  0.270396  |   88.95   |  578.55  
Validation loss decreased (0.270396 --> 0.268737).  Saving model ...
  47    |   0.257705   |  0.268737  |   88.97   |  579.18  
Validation loss decreased (0.268737 --> 0.267453).  Saving model ...
  48    |   0.255952   |  0.267453  |   89.05   |  571.57  
EarlyStopping counter: 1 out of 3
  49    |   0.254752   |  0.267708  |   89.07   |  571.62  
Validation loss decreased (0.267453 --> 0.266456).  Saving model ...
  50    |   0.253033   |  0.266456  |   89.14   |  571.74  


==========Best Accuracy After training================ 89.1438980102539
"""
import os

save_dir = "reports/figures"
os.makedirs(save_dir, exist_ok=True)
plot_path = os.path.join(save_dir, "lstm_tc_loss_curve.png")

epochs = list(range(1, 51))
train_loss = [
    1.098575, 1.098510, 1.098404, 1.097755, 1.029901, 0.819687, 0.712001, 0.613239, 0.528067,
    0.484306, 0.461654, 0.437605, 0.416049, 0.398495, 0.384425, 0.372062, 0.360317, 0.351489,
    0.343217, 0.336104, 0.328781, 0.323049, 0.318964, 0.313975, 0.309854, 0.305774, 0.301547,
    0.299023, 0.295527, 0.292448, 0.289175, 0.286482, 0.283945, 0.281046, 0.278731, 0.277225,
    0.275269, 0.272160, 0.270574, 0.268300, 0.266757, 0.265443, 0.263102, 0.261606, 0.259659,
    0.258620, 0.257705, 0.255952, 0.254752, 0.253033
]
val_loss = [
    1.098439, 1.098383, 1.098209, 1.096271, 0.887419, 0.755923, 0.661666, 0.563350, 0.494737,
    0.469499, 0.446962, 0.425843, 0.408439, 0.390492, 0.380516, 0.371920, 0.357032, 0.347759,
    0.341368, 0.337906, 0.329790, 0.325249, 0.319597, 0.315914, 0.312718, 0.310650, 0.304164,
    0.306234, 0.299984, 0.298119, 0.296045, 0.291246, 0.287574, 0.289649, 0.285981, 0.282947,
    0.283351, 0.279935, 0.278331, 0.276415, 0.276797, 0.273313, 0.272697, 0.271695, 0.278070,
    0.270396, 0.268737, 0.267453, 0.267708, 0.266456
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss (Third Training Run)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=300) 

