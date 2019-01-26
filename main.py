import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test)= cifar10.load_data()
print('x_train shape:'+ str(x_train.shape))
print(str(x_train.shape[0])+'train samples')
 
print('y_train shape'+str(y_train.shape))
print(str(x_test.shape[0])+'test samples')
 
print('x_test shape'+str(x_test.shape))
print(str(y_test.shape[0])+'test samples')
batch_size = 32 
epochs = 100
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_images(x, y, number_of_images=2):
    fig, axes1 = plt.subplots(number_of_images,number_of_images,figsize=(10,10))
    for j in range(number_of_images):
        for k in range(number_of_images):
            i = np.random.choice(range(len(x)))
            title = class_names[y[i:i+1][0][0]]
            axes1[j][k].title.set_text(title)
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(x[i:i+1][0])        

model = tf.keras.Sequential()
 
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
 
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
 

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
 
# Train the model
model.fit(x_train / 255.0, tf.keras.utils.to_categorical(y_train),
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          validation_data=(x_test / 255.0, tf.keras.utils.to_categorical(y_test))
          )
 
# Evaluate the model
scores = model.evaluate(x_test / 255.0, tf.keras.utils.to_categorical(y_test))
 
print('Loss: %.3f'+str(scores[0]))

print('Accuracy: '+ str(scores[1]))
model.save('cifar-model.h5')