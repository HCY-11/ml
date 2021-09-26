import tensorflow as tf
from tensorflow.keras import layers, models, datasets, losses

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

(train_X, train_y), (test_X, test_y) = datasets.cifar10.load_data()

train_X = train_X / 255.
test_X = test_X / 255.

train_y = tf.squeeze(tf.one_hot(train_y, 10))
test_y = tf.squeeze(tf.one_hot(test_y, 10))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D())

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss=losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_X, train_y, epochs=50, validation_data=(test_X, test_y))

_, test_acc = model.evaluate(test_X, test_y)
print(test_acc)