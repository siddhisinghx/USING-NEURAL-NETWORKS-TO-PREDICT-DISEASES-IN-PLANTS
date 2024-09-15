
from keras.models import load_model
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

key = {'Raspberry Healthy': 0,
 'Apple Rust': 1,
 'Apple Frogeye Spot': 2,
 'Grape Healthy': 3,
 'Tomato Yellow Leaf Curl Virus': 4,
 'Corn Maize Common Rust_': 5,
 'Apple Scab': 6,
 'Pepper Healthy': 7,
 'Strawberry Scorch': 8,
 'Cherry Healthy': 9,
 'Grape Blight Leaf Spot': 10,
 'Tomato Healthy': 11,
 'Corn Maize Healthy': 12,
 'Potato Healthy': 13,
 'Potato Late Blight': 14,
 'Tomato Mosaic Virus': 15,
 'Grape Black Rot': 16,
 'Cherry Powdery Mildew': 17,
 'Tomato Late Blight': 18,
 'Tomato Leaf Mold': 19,
 'Tomato Septoria Leaf Spot': 20,
 'Squash Powdery Mildew': 21,
 'Potato Early Blight': 22,
 'Tomato Target Spot': 23,
 'Apple Healthy': 24,
 'Tomato Spider Mite': 25,
 'Grape Black Measles': 26,
 'Pepper Bacterial Spot': 27,
 'Peach Bacterial Spot': 28,
 'Corn Maize Leaf Blight': 29,
 'Soybean Healthy': 30,
 'Tomato Early Blight': 31,
 'Corn Maize Gray Leaf Spot': 32,
 'Orange Citrus Greening': 33,
 'Tomato Bacterial Spot': 34,
 'Blueberry Healthy': 35,
 'Peach Healthy': 36,
 'Strawberry Healthy': 37}

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20,
        }

np.set_printoptions(precision=2)
import os
 

model = load_model('./models/GAP/Alexnet_GAP_Adamax.hdf5')


alexnet_test_data = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_data.npy')
alexnet_test_label = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_label.npy')
pred=model.predict(alexnet_test_data)
real=alexnet_test_label
pr=[]
re=[]
for i in pred:
    pr.append(i.argmax())
pr=np.array(pr)
for i in real:
    re.append(i.argmax())
re=np.array(re)
mat=confusion_matrix(re,pr)


plt.subplots(figsize=(40,40))
ax = sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=key.keys(),yticklabels=key.keys(),cmap="Greys",annot_kws={'size':25},linewidths=2, linecolor='black')
for _, spine in ax.spines.items():
    spine.set_visible(True)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('True Labels', fontdict=font,labelpad=4)
plt.ylabel('Predicted Labels', fontdict=font,labelpad=4)
plt.savefig('cnfmatrix_val.png')
plt.show()



