#CIFAR10


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10 
(train_images,train_labels),(test_images,test_labels)=cifar10.load_dataset()
#print(train_image.shape)

#Norm
train_images,test_images = train_images/255, test_images/255

classname = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']

def show():
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xtricks([])
        plt.ytricks([])
        plt.grid(False)
        plt.imshow(train_images[i],camp=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
    
show()

#MODEL
model = keras.model.Sequential() 
model.add(layers.Conv2D(32,(3,3),strides=(1,1),padding="valid",activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))


#loss und Optimizer
loss = keras.losses.SparseCategorialCrossentropy(from_logits=True)
opti = keras.optimizers.Adam(lr=0.001)
metrics =["accuracy"]

model.compile(loss=loss,optimizer = opti, metrics=metrics)

#train
batch_size=64
epochs=5

model.fit(train_images,train_labels,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2)

#evaluate
model.evaluate(test_images,test_labels,batch_size=batch_size,verbose=2) 

