#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image


# In[51]:


image_width, image_height = 150,150
validation_data_dir = 'E:\\dogs-vs-cats\\validation'
train_data_dir = 'E:\\dogs-vs-cats\\train'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 20
batch_size = 20


# In[52]:


if K.image_data_format() == 'channels_first':
    input_shape = (3,image_width,image_height)
else:
    input_shape = (image_width,image_height,3)


# In[53]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)


# In[54]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[56]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width,image_height),
    batch_size=batch_size,
    class_mode='binary')


# In[57]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_width,image_height),
    batch_size=batch_size,
    class_mode='binary')


# In[58]:


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()


# In[59]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


model.fit_generator(train_generator,
                   steps_per_epoch=nb_train_samples // batch_size,
                   epochs = epochs,
                   validation_data=validation_generator,
                   validation_steps=nb_validation_samples // batch_size)

model.save('new1.h5')


# In[ ]:


image_pred = image.load_img('E:\\dogs-vs-cats\\test1\\test1\\4.jpg',target_size=(150,150))
image_pred = image.img_to_array(image_pred)
image_pred = np.expand_dims(image_pred,axis=0)


# In[ ]:


rslt = model.predict(image_pred)
print(rslt)
if rslt[0][0] == 1:
    pred = "dog"
else:
    pred = "cat"

print(pred)

