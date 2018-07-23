import glob
import numpy as np
import os.path as path
import matplotlib.pyplot as plot
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

def load_images(image_path):
	normal_file_paths = image_path + 'normal/*.png'
	broken_file_paths = image_path + 'broken/*.png'

	for filename in glob.glob(normal_file_paths):
		img=Image.open(filename)
		imgCrop = img.crop([0,0,153,343])
		imgCrop.save(filename, 'PNG', quality=100)

	for filename in glob.glob(broken_file_paths):
		img=Image.open(filename)
		imgCrop = img.crop([0,0,153,343])
		imgCrop.save(filename, 'PNG', quality=100)

	# Load the images
	normal_images = []
	for filename in glob.glob(normal_file_paths):
		normal_images.append(cv2.imread(filename))
	broken_images = []
	for filename in glob.glob(broken_file_paths):
		broken_images.append(cv2.imread(filename))
	images = np.asarray(normal_images + broken_images)

	# Get image size
	image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])

	# Scale
	images = images / 255

	# Read the labels from the filenames
	n_images = len(normal_images)
	b_images = len(broken_images)
	labels = np.append(np.zeros(n_images),np.ones(b_images))

	return images, labels, image_size

def visualize_data(positive_images, negative_images):
    # INPUTS
    # positive_images - Images where the label = 1 (True)
    # negative_images - Images where the label = 0 (False)

    figure = plt.figure()
    count = 0
    for i in range(positive_images.shape[0]):
        count += 1
        figure.add_subplot(2, positive_images.shape[0], count)
        plt.imshow(positive_images[i, :, :])
        plt.axis('off')
        plt.title("1")

        figure.add_subplot(1, negative_images.shape[0], count)
        plt.imshow(negative_images[i, :, :])
        plt.axis('off')
        plt.title("0")
    plt.show()

def visualize_incorrect_labels(x_data, y_real, y_predicted):
    # INPUTS
    # x_data      - images
    # y_data      - ground truth labels
    # y_predicted - predicted label
    count = 0
    figure = plt.figure()
    incorrect_label_indices = (y_real != y_predicted)
    y_real = y_real[incorrect_label_indices]
    y_predicted = y_predicted[incorrect_label_indices]
    x_data = x_data[incorrect_label_indices, :, :, :]

    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)

    plt.show()

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

# Split the images and the labels
training_images, training_labels, training_image_size = load_images('TruckSpring/training/')

print("GOT IMAGES")

# Hyperparamater
N_LAYERS = 4

# Instantiate the model
model = cnn(size=training_image_size, n_layers=N_LAYERS)

print("MODEL INSTANTIATED")

# Training hyperparamters
EPOCHS = 150
BATCH_SIZE = 200

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = ''
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping]

print("CALLBACKS INSTALLED")

# Train the model
model.fit(training_images, training_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)

print("MODEL TRAINED")

# Number of positive and negative examples to show
N_TO_VISUALIZE = 10

# Select the first N positive examples
positive_example_indices = (training_labels == 1)
positive_examples = training_images[positive_example_indices, :, :]
positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]

# Select the first N negative examples
negative_example_indices = (training_labels == 0)
negative_examples = training_images[negative_example_indices, :, :]
negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]

# Call the visualization function
visualize_data(positive_examples, negative_examples)

# Make a prediction on the test set
test_predictions = model.predict(n_training_images)
test_predictions = np.round(test_predictions)

# Make a prediction on the test set
test_predictions = model.predict(n_training_images)
test_predictions = np.round(test_predictions)

print("PREDICTIONS MADE")

# Report the accuracy
accuracy = accuracy_score(n_training_images, test_predictions)
print("Accuracy: " + str(accuracy))

# Accuracy visualization
visualize_incorrect_labels(n_training_images, b_training_images, np.asarray(test_predictions).ravel())
