import matplotlib.pyplot as plt

"""
Training Log of MLP Classifier Model:

=========Starting Training==========
 Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
------------------------------------------------------------
Validation loss decreased (inf --> 0.957696).  Saving model ...
   1    |   0.914106   |  0.957696  |   53.73   |  228.17  
Validation loss decreased (0.957696 --> 0.807112).  Saving model ...
   2    |   0.777409   |  0.807112  |   62.15   |  227.08  
Validation loss decreased (0.807112 --> 0.684891).  Saving model ...
   3    |   0.730325   |  0.684891  |   68.07   |  227.57  
EarlyStopping counter: 1 out of 3
   4    |   0.703493   |  0.688247  |   67.62   |  227.42  
Validation loss decreased (0.684891 --> 0.657206).  Saving model ...
   5    |   0.683936   |  0.657206  |   69.09   |  256.61  
Validation loss decreased (0.657206 --> 0.652045).  Saving model ...
   6    |   0.669068   |  0.652045  |   69.29   |  255.60  
Validation loss decreased (0.652045 --> 0.648623).  Saving model ...
   7    |   0.656966   |  0.648623  |   69.40   |  298.22  
EarlyStopping counter: 1 out of 3
   8    |   0.645865   |  0.649977  |   68.81   |  232.81  
Validation loss decreased (0.648623 --> 0.633457).  Saving model ...
   9    |   0.636981   |  0.633457  |   70.61   |  231.28  
Validation loss decreased (0.633457 --> 0.618557).  Saving model ...
  10    |   0.627996   |  0.618557  |   70.78   |  231.54  
EarlyStopping counter: 1 out of 3
  11    |   0.619865   |  0.636255  |   70.55   |  229.44  
Validation loss decreased (0.618557 --> 0.610227).  Saving model ...
  12    |   0.613128   |  0.610227  |   71.30   |  226.76  
Validation loss decreased (0.610227 --> 0.603842).  Saving model ...
  13    |   0.605366   |  0.603842  |   71.86   |  233.23  
Validation loss decreased (0.603842 --> 0.603684).  Saving model ...
  14    |   0.599294   |  0.603684  |   72.18   |  237.52  
Validation loss decreased (0.603684 --> 0.596631).  Saving model ...
  15    |   0.592551   |  0.596631  |   72.21   |  239.58  
Validation loss decreased (0.596631 --> 0.593863).  Saving model ...
  16    |   0.586597   |  0.593863  |   72.38   |  237.35  
Validation loss decreased (0.593863 --> 0.589226).  Saving model ...
  17    |   0.580864   |  0.589226  |   72.70   |  240.04  
Validation loss decreased (0.589226 --> 0.586195).  Saving model ...
  18    |   0.575006   |  0.586195  |   73.12   |  238.09  
EarlyStopping counter: 1 out of 3
  19    |   0.569894   |  0.598964  |   71.39   |  237.96  
Validation loss decreased (0.586195 --> 0.577007).  Saving model ...
  20    |   0.563838   |  0.577007  |   73.42   |  229.53  
EarlyStopping counter: 1 out of 3
  21    |   0.559010   |  0.583967  |   72.89   |  229.07  
EarlyStopping counter: 2 out of 3
  22    |   0.553521   |  0.577256  |   73.40   |  227.86  
Validation loss decreased (0.577007 --> 0.570995).  Saving model ...
  23    |   0.549324   |  0.570995  |   73.96   |  226.37  
EarlyStopping counter: 1 out of 3
  24    |   0.543175   |  0.581694  |   72.99   |  226.41  
EarlyStopping counter: 2 out of 3
  25    |   0.538623   |  0.573070  |   73.72   |  232.37  
EarlyStopping counter: 3 out of 3
Early stopping


==========Best Accuracy After training================ 73.95825958251953
"""
import os

save_dir = "reports/figures"
os.makedirs(save_dir, exist_ok=True)
plot_path = os.path.join(save_dir, "mlp_loss_curve.png")
epochs = list(range(1, 26))
train_loss = [
    0.914106, 0.777409, 0.730325, 0.703493, 0.683936, 0.669068, 0.656966, 0.645865, 0.636981,
    0.627996, 0.619865, 0.613128, 0.605366, 0.599294, 0.592551, 0.586597, 0.580864, 0.575006,
    0.569894, 0.563838, 0.559010, 0.553521, 0.549324, 0.543175, 0.538623
]
val_loss = [
    0.957696, 0.807112, 0.684891, 0.688247, 0.657206, 0.652045, 0.648623, 0.649977, 0.633457,
    0.618557, 0.636255, 0.610227, 0.603842, 0.603684, 0.596631, 0.593863, 0.589226, 0.586195,
    0.598964, 0.577007, 0.583967, 0.577256, 0.570995, 0.581694, 0.573070
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss (MLP Classifier)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=300) 