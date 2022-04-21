import matplotlib.pyplot as plt
import pandas as pd
loss_df = pd.read_csv('/media/teera/SSD250GB/model/belief/chessboard_mono_6_stage_lr_0.00003/loss_train.csv')
print(loss_df.iloc[:, 2])
loss_df.plot(kind='scatter',x=0,y=2,color='red')
plt.show()