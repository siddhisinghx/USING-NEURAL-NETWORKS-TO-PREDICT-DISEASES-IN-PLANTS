from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import os
from keras.models import load_model
import argparse

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument("-m",'--model', help="Type of model. Possible values are: {FC, GAP}.")
parser.add_argument("-o", '--optimizer', help="Optimizer used with the model. Possible values are: {SGD, RMS, ADAM, ADAMAX, ADAGRAD, ADADELTA}.")

mapper = {'SGD':'SGD', 'RMS': 'RMSprop', 'ADAM':'Adam', 'ADAMAX': 'Adamax', 'ADAGRAD':'Adagrad', 'ADADELTA':'Adadelta'}


# alexnet_train_data = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_data.npy')
# alexnet_train_label = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_train_class_label.npy')
alexnet_test_data = np.load('E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_data.npy')
alexnet_test_label = np.load(''E:\\LeafDisease\\Numpy_dataset\\alexnet_test_class_label.npy')


def model_predict(model_path):
    model = load_model(model_path)
    model.load_weights(model_path)
    pred = model.predict(alexnet_test_data)
    y_pred=[]
    y_real=[]
    for i in pred:
        y_pred.append(i.argmax())
    y_pred=np.array(y_pred)
    for i in alexnet_test_label:
        y_real.append(i.argmax())
    y_real=np.array(y_real)
    print(y_real.shape)
    print(y_pred.shape)
    acc = accuracy_score(y_real, y_pred)
    print(acc)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.model != None and args.optimizer != None:
        if (str(args.model).upper() == 'FC'):
            model_predict('../models/'+str(args.model).upper()+'/Alexnet_'+str(args.model)+'_'+mapper[str(args.optimizer).upper()]+'.hdf5')
        else:
            print("Please enter the arguments correctly!")
    else:
        print("Please enter the model type with -m and optimizer with -o.")
