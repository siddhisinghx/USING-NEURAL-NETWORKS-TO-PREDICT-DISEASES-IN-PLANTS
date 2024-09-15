
import pandas as pd
import matplotlib.pyplot as plt


filename1 = './logs/GAP/Alexnet_GAP_Adadelta.csv'
filename2 = './logs/GAP/Alexnet_GAP_Adagrad.csv'
filename3 = './logs/GAP/Alexnet_GAP_Adam.csv'
filename4 = './logs/GAP/Alexnet_GAP_Adamax.csv'
filename5 = './logs/GAP/Alexnet_GAP_RMSprop.csv'
filename6 = './logs/GAP/Alexnet_GAP_SGD.csv'


df1 = pd.read_csv(filename1).dropna().reset_index()
df2 = pd.read_csv(filename2).dropna().reset_index()
df3 = pd.read_csv(filename3).dropna().reset_index()
df4 = pd.read_csv(filename4).dropna().reset_index()
df5 = pd.read_csv(filename5).dropna().reset_index()
df6 = pd.read_csv(filename6).dropna().reset_index()


plt.plot(df1['val_acc'])
plt.plot(df2['val_acc'])
plt.plot(df3['val_acc'])
plt.plot(df4['val_acc'])
plt.plot(df5['val_acc'])
plt.plot(df6['val_acc'])
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Adadelta','Adagrad','Adam','Adamax','RMSProp','SGD'])
plt.savefig('Accuracy Comparison.png',dpi=150)
plt.show()


plt.plot(df1['val_loss'])
plt.plot(df2['val_loss'])
plt.plot(df3['val_loss'])
plt.plot(df4['val_loss'])
plt.plot(df5['val_loss'])
plt.plot(df6['val_loss'])
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Adadelta','Adagrad','Adam','Adamax','RMSProp','SGD'])
plt.savefig('Loss Comparison.png',dpi=150)
plt.show()
