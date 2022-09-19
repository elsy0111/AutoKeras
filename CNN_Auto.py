import autokeras as ak
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

"""
import os
import tensorflow as tf

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=120000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""

images = np.load("../Use_CNN_Dataset/02/Dataset_1000_8in2/images.npy")

labels = np.load("../Use_CNN_Dataset/02/Dataset_1000_8in2/labels.npy")


# データセットを分ける
train_images = []
train_labels = []
test_images = []
test_labels = []

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.1)


# """
# 訓練用データ、テストデータに取り込んだデータを格納する
train_images = (np.array(train_images) + 80) / 80
train_labels =  np.array(train_labels)
test_images  = (np.array(test_images)  + 80) / 80
test_labels  =  np.array(test_labels)
# """

# clf = ak.ImageClassifier()

# clf は model のような扱いで大丈夫です。

# Initialize the image regressor.
clf = ak.ImageRegressor(overwrite=True, max_trials=10)

# Reshape the images to have the channel dimension.
train_images = train_images.reshape(train_images.shape + (1,))
train_labels = train_labels.reshape(train_labels.shape + (1,))
test_images  = test_images.reshape(test_images.shape + (1,))
test_labels  = test_labels.reshape(test_labels.shape + (1,))

print(train_images.shape)
print(train_labels.shape)

clf.fit(
    train_images,train_labels,
    batch_size = 16,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=30,
)

clf.final_fit(
    train_images,train_labels,
    test_images,test_labels,
    retrain=False)

# 予測
predictions = clf.predict(test_images)
print(predictions)

print(clf.evaluate(test_images, test_labels))

model = clf.export_model()
model.save('model.h5')
plot_model(model, show_shapes=True, show_layer_names=True)