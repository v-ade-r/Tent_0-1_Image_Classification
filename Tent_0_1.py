import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import keras
from keras import layers
import gradio as gr
import os
import random
import cv2

# ----------------------------------------------------------------------------------------------------------------------
# -1. Data augmentation.
# ----------------------------------------------------------------------------------------------------------------------
# I only had 48 images of my tent, and I was wondering if that would be enough to fine-tune EfficientNetV2B0, after
# augmenting the data slightly. I decided to do it using cv2 for fun. This function effectively gives me 7 images (6 new)
# from each single image I had before. I manually inspected it, and divided into appropriate sets, trying to maintain
# as much image variation as possible in each set. Nontent images have been augmented, in order to prevent the model
# from learning that, for example, rotated images are always a Tent.
# ----------------------------------------------------------------------------------------------------------------------
def load_and_aug(filename,folder):
    if folder == folder1:
        img = cv2.imread('NonTent/' + filename)
    else:
        img = cv2.imread('Tent/' + filename)

    for i in range(7):
        if i == 0:
            img = cv2.flip(img,-1)
        elif i == 1:
            img = cv2.flip(img, 1)
        elif i == 2:
            img = cv2.flip(img, 0)
        elif i == 3:
            rows, cols, channels = img.shape
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), (random.randint(1,35) * 10), 1)
            img = cv2.warpAffine(img, M, (cols, rows))
        elif i == 4:
            rows, cols, channels = img.shape
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), (random.randint(1,35) * 10), 1)
            img = cv2.warpAffine(img, M, (cols, rows))
        elif i == 5:
            rows, cols, channels = img.shape
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), (random.randint(1,35) * 10), 1)
            img = cv2.warpAffine(img, M, (cols, rows))
        elif i == 6:
            rows, cols, channels = img.shape
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), (random.randint(1,35) * 10), 1)
            img = cv2.warpAffine(img, M, (cols, rows))


        n = random.randint(12475393, 988765382312)
        print(n)
        print(str(n)+'.jpg')
        if folder == folder1:
            cv2.imwrite('NonTent_aug/' + str(n) + '.jpg', img)
        else:
            cv2.imwrite('Tent_aug/' + str(n) + '.jpg', img)


folder1 = os.listdir('NonTent/')
folder2 = os.listdir('Tent/')
folders = [folder1, folder2]
for folder in folders:
    for file in folder:
        load_and_aug(file, folder)

# ----------------------------------------------------------------------------------------------------------------------
# 0. Data settings
# ----------------------------------------------------------------------------------------------------------------------
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32
input_shape = (224, 224, 3)

# ----------------------------------------------------------------------------------------------------------------------
# 1. Creating Dataset objects.
# ----------------------------------------------------------------------------------------------------------------------
# There are 2 labels: Tent, NonTent. Their ratio is almost 50/50.
# Train data size: 504 files
# Valid data size: 112 files
# Test data size: 12 files
# ----------------------------------------------------------------------------------------------------------------------
train_dir = r'C:\...\train'
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                 labels = 'inferred',
                                                                 image_size = IMAGE_SHAPE,
                                                                 batch_size = BATCH_SIZE)

valid_dir = r'C:\...\valid'
valid_data = tf.keras.preprocessing.image_dataset_from_directory(valid_dir,
                                                                 labels='inferred',
                                                                 image_size=IMAGE_SHAPE,
                                                                 batch_size=BATCH_SIZE)
test_dir = r'C:\...\test'
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                labels='inferred',
                                                                image_size=IMAGE_SHAPE,
                                                                batch_size=BATCH_SIZE)

# ----------------------------------------------------------------------------------------------------------------------
# 2. Creating Model, based on EfficientNetV2B0
# ----------------------------------------------------------------------------------------------------------------------
# After some tests, I discovered that there's no benefit in training only the newly added layers. I achieved the best
# results when all layers were unfrozen from the start.
# ----------------------------------------------------------------------------------------------------------------------
base_model = tf.keras.applications.EfficientNetV2B0(include_top=False)
inputs = layers.Input(input_shape)
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs, outputs)

#base_model.trainable = False
# model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
#                 optimizer=Adam(),
#                 metrics='accuracy')
#
# model_5.fit(train_data, epochs=4, steps_per_epoch=len(train_data), validation_data=valid_data,
#             validation_steps=len(valid_data))
# -------------------------------------------------------------------------------------------------------------
base_model.trainable = True#
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=Adam(learning_rate=0.0005),
              metrics='accuracy')#
model.fit(train_data, epochs=4, steps_per_epoch=len(train_data), validation_data=valid_data,
          validation_steps=len(valid_data))

# ----------------------------------------------------------------------------------------------------------------------
# 3. Saving/Loading Model and evaluating
# ----------------------------------------------------------------------------------------------------------------------
# model.save("model_4ep.h5")
# model = load_model("model_4ep.h5")
#print(model.evaluate(valid_data))
#print(model.evaluate(test_data))
# 4/4 [==============================] - 4s 42ms/step - loss: 0.0257 - accuracy: 0.9911
# Results: on valid data: [loss, acc]
# [0.025685233995318413, 0.9910714030265808]
# 1/1 [==============================] - 0s 179ms/step - loss: 0.0225 - accuracy: 1.0000
# Results: on test data: [loss, acc]
# [0.02247760072350502, 1.0]

# ----------------------------------------------------------------------------------------------------------------------
# 4. Manual predictions
# ----------------------------------------------------------------------------------------------------------------------
# Stunningly I was unable to make predictions for single images. Could it be that a model trained on Dataset objects
# also needs to be fed a Dataset object for predictions?


# img_path = r'C:\...\.jpg'

# img = tf.io.read_file(img_path)
# img = tf.image.decode_image(img , channels=3)
# img = tf.image.resize(img, size=IMAGE_SHAPE)
# img = img / 255
# pred = model_5.predict(tf.expand_dims(img, axis=0))
# print(pred)

# ----------------------------------------------------------------------------------------------------------------------
# 5. User predictions with Gradio
# ----------------------------------------------------------------------------------------------------------------------
# Continuing the line of thought from the previous chapter, I implemented in classifying function Dataset object
# containing user image and added Gradio API.

class_names = train_data.class_names


def classify_image(inp):
    plt.imsave(r'C:\...\user_image.jpg', inp)
    user_dir = r'C:\...\user'
    user_data = tf.keras.preprocessing.image_dataset_from_directory(user_dir,
                                                                    image_size=IMAGE_SHAPE,
                                                                    batch_size=32)
    for image, _ in user_data:
        pred = model.predict(tf.expand_dims(image[0], axis=0))
        print(pred)
        cl = class_names[int(np.round(pred))]
        if pred < 0.5:
            pred = 1 - pred
        results = {cl: pred}
        os.remove(r'C:\...\user_image.jpg')
    return results


gr.Interface(fn=classify_image,
             inputs=gr.Image(),
             outputs=gr.Label(num_top_classes=1)).launch()



