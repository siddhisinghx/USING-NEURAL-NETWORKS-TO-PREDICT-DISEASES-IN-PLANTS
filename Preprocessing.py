# Importing libraries
import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Initialising the variables
train_labels = []
train_images = []

dic={'Raspberry___healthy': 0,
 'Apple___Cedar_apple_rust': 1,
 'Apple_Frogeye_Spot': 2,
 'Grape___healthy': 3,
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 4,
 'Corn_maize___Common_rust_': 5,
 'Apple___Apple_scab': 6,
 'Pepper_bell___healthy': 7,
 'Strawberry___Leaf_scorch': 8,
 'Cherry_including_sour___healthy': 9,
 'Grape___Leaf_blight_Isariopsis_Leaf_Spot': 10,
 'Tomato___healthy': 11,
 'Corn_maize___healthy': 12,
 'Potato___healthy': 13,
 'Potato___Late_blight': 14,
 'Tomato___Tomato_mosaic_virus': 15,
 'Grape___Black_rot': 16,
 'Cherry_including_sour___Powdery_mildew': 17,
 'Tomato___Late_blight': 18,
 'Tomato___Leaf_Mold': 19,
 'Tomato___Septoria_leaf_spot': 20,
 'Squash___Powdery_mildew': 21,
 'Potato___Early_blight': 22,
 'Tomato___Target_Spot': 23,
 'Apple___healthy': 24,
 'Tomato___Spider_mites_Two-spotted_spider_mite': 25,
 'Grape___Esca_Black_Measles': 26,
 'Pepper_bell___Bacterial_spot': 27,
 'Peach___Bacterial_spot': 28,
 'Corn_maize___Northern_Leaf_Blight': 29,
 'Soybean___healthy': 30,
 'Tomato___Early_blight': 31,
 'Corn_maize___Cercospora_leaf_spot Gray_leaf_spot': 32,
 'Orange___Haunglongbing_Citrus_greening': 33,
 'Tomato___Bacterial_spot': 34,
 'Blueberry___healthy': 35,
 'Peach___healthy': 36,
 'Strawberry___healthy': 37}

for folder in os.listdir('E:\\LeafDisease\\dataset'):
    print(folder)
    for file in os.listdir("E:\\LeafDisease\\dataset\\"+folder):
    	print(file)
        try:
            temp=Image.open('E:\\LeafDisease\\dataset\\'+folder+'\\'+file)
            temp=temp.resize((227,227))
            temp=np.array(temp)
            print(temp.shape)
            if temp.shape[-1]==4:
                print("WITH ALPHA 4")
                temp = cv2.cvtColor(temp, cv2.COLOR_RGBA2RGB)
            train_images.append(temp)
            train_labels.append(dic[folder])
        except:
        	continue
#train_labels = np.array(train_labels)
train_labels = np_utils.to_categorical(train_labels, num_classes=38, dtype='float32')
#print(train_images.shape)
print(train_labels.shape)

alexnet_train_data, alexnet_test_data, alexnet_train_label, alexnet_test_label = train_test_split(train_images,train_labels, train_size=0.8, random_state=21)
#print(train_labels)
# np.save('./data/train_images_38class_227X227.npy',np.array(train_images))
# np.save('./data/train_labels_38class_227X227.npy',np.array(train_labels))
np.save('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_data.npy',np.array(alexnet_train_data))
np.save('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_label.npy',np.array(alexnet_train_label))
np.save('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_data.npy',np.array(alexnet_test_data))
np.save('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_label.npy',np.array(alexnet_test_label))
