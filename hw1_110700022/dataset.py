import os
import cv2
import numpy as np


def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    dataset = []
    label =  []
    for filename in os.listdir(data_path + '/car'):
        image = cv2.imread(os.path.join(data_path + '/car', filename))
        image = cv2.resize(image, (36, 16), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dataset.append(image)
        label.append(1)
    for filename in os.listdir(data_path + '/non-car'):
        image = cv2.imread(os.path.join(data_path + '/non-car', filename))
        image = cv2.resize(image, (36, 16), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dataset.append(image)
        label.append(0)

    label = np.asarray(label)
    dataset = np.asarray(dataset)
    dataset = list(zip(dataset, label))
    dataset = (*dataset,)



    # End your code (Part 1)
    return dataset
