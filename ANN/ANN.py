#ANN
#environemnet
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

#data
mnist = keras.datasets.mnist 

#split
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)

#Norm 
x_train,x_test = x_train/255,x_test/255

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap='gray')
plt.show()

#model
model = Model.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)                    #output classes
])

#loss und Optimizer
loss = keras.losses.SparseCategorialCrossentropy(from_logits=True)
opti = keras.optimizers.Adam(lr=0.001)
metrics =["accuracy"]

model.compile(loss=loss,optimizer = opti, metrics=metrics)

#training
batch_size=64
epochs=5

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2)

#evaluate
model.evaluate(x_test,y_test,batch_size=batch_size,verbose=2) 

#pred_value
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)