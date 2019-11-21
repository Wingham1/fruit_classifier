from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
import tensorflow.keras as keras
import os
import cv2
import numpy as np

def data_prep(path_to_training_images, img_rows, img_cols, color):
    """
    A function to get the input image data and the label (as a one hot pixel array) when given a path to a directory containing all the images.
    This function requires that the images for each class are in a seperate directory and the directory is named after class.
    
    param:
           - path_to_training_images, a string of the path to the directory containing the images
    return:
           - images, an numpy array of images with pixel values normalised to be between 0 and 1.
             numpy array dimensions are [number of images, number of rows, number of columns, number of chanels]
           - labels, a list of labels associated with each image (labels are a one hot pixel numpy array [1, 0, 0, ...] or [0, 1, 0, ...], etc)
    """
    
    images = []
    labels = []
    for image_class in os.listdir(path_to_training_images):
        print('image_class =', image_class)
        path_to_class_directory = os.path.join(path_to_training_images, image_class)
        for img_name in os.listdir(path_to_class_directory):
            true_path = os.path.join(path_to_class_directory, img_name)
            if color:
                images.append(cv2.imread(true_path, 1)/255.0)
            else:
                images.append(cv2.imread(true_path, 0)/255.0) # greyscale
            labels.append(os.listdir(path_to_training_images).index(image_class))
    data = list(zip(images, labels))
    np.random.shuffle(data)
    images, labels = zip(*data)
    if color:
        images = np.array(images).reshape(len(images), img_rows, img_cols, 3)
    else:
        images = np.array(images).reshape(len(images), img_rows, img_cols, 1)
    labels = keras.utils.to_categorical(labels, num_classes=len(os.listdir(path_to_training_images)))
    return images, labels

def build_CNN(color=False):
    model = Sequential()
    if color:
        model.add(Conv2D(20, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(img_rows, img_cols, 3)))
    else:
        model.add(Conv2D(20, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(20, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    #model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

def decode_preditions(predictions, class_names):
    """
    A funtion to get the name of the predicted class
    """
    
    return np.array(class_names)[[np.argmax(example) for example in predictions]]

def calc_accuracy(pred, real):
        return sum(pred==real) / len(pred)

if __name__ == '__main__':

    path = os.path.join('.', 'fruits-360_shrunk')
    path_to_training_images = os.path.join(path, 'Training')
    img_rows = 100
    img_cols = 100
    is_color = True
    model_filename = 'fruits-360_shrunk_cnn'

    print('\nloading training data\n')
    num_classes = len(os.listdir(path_to_training_images))
    x, y = data_prep(path_to_training_images, img_rows, img_cols, color=is_color)

    print('\nbuilding model\n')
    cnn = build_CNN(color=is_color)

    print('\ntraining model\n')
    cnn.fit(x, y, batch_size=50, epochs=1, validation_split=0.2)
    
    print('\nsaving model\n')
    if is_color:
        model_filename = model_filename + '_RGB' + '.h5'
    else:
        model_filename = model_filename + '_grey' + '.h5'
    cnn.save(model_filename)
    print('\nsaved model to file {}\n'.format(model_filename))
    
    print('\nloading model\n')
    loaded_cnn = keras.models.load_model(model_filename)

    print('\nloading test images\n')
    path_to_test_images = os.path.join(path, 'Test')
    x_test, y_test = data_prep(path_to_test_images, img_rows, img_cols, color=is_color)

    print('\ngenerating predictions\n')
    predictions = loaded_cnn.predict(x_test)
    dec_preds = decode_preditions(predictions, os.listdir(path_to_training_images))
    dec_ytest = decode_preditions(y_test, os.listdir(path_to_training_images))

    print('\naccuracy =', calc_accuracy(dec_preds, dec_ytest))