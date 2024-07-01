import numpy as np
import struct
import random
import matplotlib.pyplot as plt
from array import array
from os.path import join

# load data
input_path = './data/mnist' # download and save Kaggle files to directory
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')

with open(training_labels_filepath, 'rb') as lbpath:
    magic, num = struct.unpack(">II", lbpath.read(8))
    labels = np.array(array("B", lbpath.read()))

with open(training_images_filepath, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
    images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows, cols)

# reshape data
images_reshaped = images.reshape(-1, 28, 28, 1) / 255.
labels_reshaped = np.eye(10, dtype=int)[labels].T

# check counts
class_counts = np.sum(labels_reshaped, axis=1)
print("Class Counts:", class_counts)

# save data
output_path = './data'
np.save(join(output_path, 'X_mnist.npy'), images_reshaped)
np.save(join(output_path, 'y_mnist.npy'), labels_reshaped)

# load data
loaded_images = np.load(join(output_path, 'X_mnist.npy'))
loaded_labels = np.load(join(output_path, 'y_mnist.npy'))
# print(loaded_images[0], loaded_labels[0])
print(loaded_images.shape, loaded_labels.shape)

# plot random data
idx = random.randint(0, len(loaded_images) - 1)
image = loaded_images[idx]
label = loaded_labels[:, idx]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
