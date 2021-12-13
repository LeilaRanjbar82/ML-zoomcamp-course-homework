import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

# Parmeters
learning_rate = 0.001
size_inner = 500
drop_rate = 0.2


# Train_validation data
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    './yoga-posture-cleaned/train',
    target_size=(150, 150),
    batch_size=20,
    shuffle=False
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './yoga-posture-cleaned/validation',
    target_size=(150, 150),
    batch_size=20,
    shuffle=False
)


# Model function
def make_model(learning_rate=0.01, size_inner=100, drop_rate=0.5):
    base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################
    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    flatvectors = keras.layers.Flatten()(vectors)
    inner = keras.layers.Dense(size_inner, activation='relu')(flatvectors)
    drop = keras.layers.Dropout(drop_rate)(inner)
    outputs = keras.layers.Dense(6,activation='softmax')(drop)
    model = keras.Model(inputs, outputs)

    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    #########################################

    return model


#Create and save the best models
checkpoint = keras.callbacks.ModelCheckpoint(
    'yoga_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model = make_model(
        learning_rate=learning_rate,
        size_inner=size_inner,
        drop_rate=drop_rate
    )

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint]
)