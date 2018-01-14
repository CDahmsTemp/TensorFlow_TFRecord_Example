# mnist_to_images.py

import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnistData
import numpy as np

# module level variables ##############################################################################################
MNIST_LOCAL_DIR = "C:\\mnist"
MNIST_IMAGE_WIDTH = 28
MNIST_IMAGE_HEIGHT = 28

SCALAR_GREEN = (0.0, 210.0, 0.0)        # for writing the classification on the image

#######################################################################################################################
def main():
    # get the MNIST data
    # if the MNIST data is not already present in the specified directory, this function will download and save the MNIST data
    # if the MNIST data is already present in the specified directory, this function will read the MNIST data from file
    mnistDataSets = mnistData.read_data_sets(MNIST_LOCAL_DIR, one_hot=True)

    # declare a string array to refer to the train, test, and validation data sets
    trainTestValStringArray = ["train", "test", "validation"]

    # for each index in the train, test, and validation string array . . .
    for i in range(len(trainTestValStringArray)):

        # get the current data set (train, test, or validation, depending on i)
        mnistDataSet = mnistDataSets[i]

        for j in range(mnistDataSet.images.shape[0]):

            # get the current label
            mnistOneHotLabel = mnistDataSet.labels[j]

            # show the current label to std out, note this is in "one_hot" format, so for example,
            # the digit 0 would be 1, 0, 0, 0, 0, 0, 0, 0, 0, 0  (1st element is a 1, all others zeros)
            # the digit 3 would be 0, 0, 0, 1, 0, 0, 0, 0, 0, 0  (4th element is a 1, all others zeros)
            print(mnistOneHotLabel)

            # convert the one hot array style label to a regular digit
            classification = convertOneHotLabelToRegularDigit(mnistOneHotLabel)

            # get the current image as a NumPy array
            npaFlatImage = mnistDataSet.images[j]

            # restore the image from 1d to 2d
            npaImage = npaFlatImage.reshape((MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT))

            # resize the image so it's big enough we can actually see it
            npaImageBigger = cv2.resize(npaImage, None, fx=12, fy=12)

            # if the image is currently 1 channel, change it to 3 channels so we can write on the image in color
            if getNumChannelsInImage(npaImageBigger) == 1:
                npaImageBigger = cv2.cvtColor(npaImageBigger, cv2.COLOR_GRAY2RGB)
            # end if

            # write the classification on the image
            writeClassificationOnImage(npaImageBigger, classification)

            # show each image
            cv2.imshow("npaImageBigger", npaImageBigger)
            cv2.waitKey()
        # end for

    # end for

# end main

#######################################################################################################################
def convertOneHotLabelToRegularDigit(mnistOneHotLabel):
    if np.array_equal(mnistOneHotLabel, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        return 0
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        return 1
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        return 2
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        return 3
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        return 4
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])):
        return 5
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])):
        return 6
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])):
        return 7
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])):
        return 8
    elif np.array_equal(mnistOneHotLabel, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])):
        return 9
    else:
        return -1
    # end if
# end function

#######################################################################################################################
def writeClassificationOnImage(image, classification):
    lowerLeftTextOrigin = (12, 55)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(1.4)
    intFontThickness = int(round(fltFontScale * 1.5))

    cv2.putText(image, str(classification), lowerLeftTextOrigin, fontFace, fltFontScale, SCALAR_GREEN, intFontThickness)
# end function

#######################################################################################################################
def getNumChannelsInImage(image):
    if len(image.shape) == 2:
        return 1
    elif len(image.shape) == 3:
        return 3
    else:
        print("error in function getNumChannelsInImage()")
        return -1
    # end if
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()