# images_to_tfrecord.py

import os
import cv2
import numpy as np
import tensorflow as tf

# module level variables ##############################################################################################
TF_RECORD_FILE_NAME = "bikes.tfrecord"

#######################################################################################################################
def main():

    # build up the path and file name that we will save to
    tfRecordsPathAndFileName = os.path.join(os.getcwd(), TF_RECORD_FILE_NAME)

    # instantiate a TFRecordWriter object using the path and file name to save to
    tfRecordWriter = tf.python_io.TFRecordWriter(tfRecordsPathAndFileName)

    imageWriteCount = 0
    for fileName in os.listdir():
        if not stringContainsSubstring(fileName, "road") or not fileName.endswith('.png'):
            continue
        # end if

        npaImage = cv2.imread(fileName)

        if npaImage is not None:
            # un-comment these two lines to see each image
            # cv2.imshow(fileName, npaImage)
            # cv2.waitKey()

            if getNumChannelsInImage(npaImage) != 3:
                print("error, image " + fileName + " is not a 3 channel image")
                continue
            # end if

            # convert the current image from a NumPy array to a bite string
            bsImage = npaImage.tostring()

            # see TensorFlowExample10/learning_tensor_flow_ch8_ex1.py, fix this

            tfTrainExample = tf.train.Example(
                features=tf.train.Features(feature={'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[720])),
                                                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[960])),
                                                    'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
                                                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[7])),
                                                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bsImage]))}))
            tfRecordWriter.write(tfTrainExample.SerializeToString())
            imageWriteCount += 1
        # end for
    # end for
    tfRecordWriter.close()
    print("done, wrote " + str(imageWriteCount) + " images to " + tfRecordsPathAndFileName)
# end main

#######################################################################################################################
def getNumChannelsInImage(image):
    if len(image.shape) == 2:
        return 1
    elif len(image.shape) == 3:
        dummy1, dummy2, numChannels = image.shape
        return numChannels
    else:
        print("error in function getNumChannelsInImage()")
        return -1
    # end if
# end function

#######################################################################################################################
def stringContainsSubstring(string, substring):
    if substring in string:
        return True
    else:
        return False
    # end if
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()