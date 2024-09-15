import os

# Do other imports now...

from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Conv2D,BatchNormalization,GlobalAveragePooling2D,Dropout
import numpy as np
from keras.utils import np_utils
from  keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint

alexnet_train_data = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_data.npy')
alexnet_train_label = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_label.npy')
alexnet_test_data = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_data.npy')
alexnet_test_label = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_label.npy')

print(alexnet_train_label.shape)
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# # Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(38))
model.add(Activation('softmax'))

model.summary()


lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('Alexnet_Adamax.csv')
early_stopper = EarlyStopping(min_delta=0.001,patience=7)
model_checkpoint = ModelCheckpoint('Alexnet_Adamax.hdf5',monitor = 'val_loss', verbose = 1,save_best_only=True)


model.compile(loss='categorical_crossentropy',
        optimizer="Adamax",
        metrics=['accuracy'])

model.fit(alexnet_train_data, alexnet_train_label,
              batch_size=32,
              epochs=30,
              validation_data=(alexnet_test_data,alexnet_test_label),
              shuffle=True,callbacks=[lr_reducer,csv_logger,early_stopper,model_checkpoint])

