# tfrecord_to_images.py

import os
import cv2
import numpy as np
import tensorflow as tf

# module level variables ##############################################################################################
TF_RECORD_FILE_NAME = "bikes.tfrecord"

#######################################################################################################################
def main():
    # get the file name including path of the train images/data
    trainRecordsFileName = os.path.join(os.getcwd(), TF_RECORD_FILE_NAME)

    # instantiate a tf_record_iterator object using the path and file name to load from
    tfRecordIterator = tf.python_io.tf_record_iterator(trainRecordsFileName)

    # instantiate a tf.train.Example() object
    tfTrainExample = tf.train.Example()

    # for each record in the tfRecordIterator . . .
    for recordAsByteString in tfRecordIterator:
        # break out the record into the tf.train.Example object
        tfTrainExample.ParseFromString(recordAsByteString)

        # get the image, label, width, and height from the record
        bsImage = tfTrainExample.features.feature['image_raw'].bytes_list.value
        label = tfTrainExample.features.feature['label'].int64_list.value[0]
        width = tfTrainExample.features.feature['width'].int64_list.value[0]
        height = tfTrainExample.features.feature['height'].int64_list.value[0]
        depth = tfTrainExample.features.feature['depth'].int64_list.value[0]

        # convert the byte string image to a flattened NumPy array, then to a 2d NumPy array
        npaFlatImage = np.fromstring(bsImage[0], dtype=np.uint8)
        npaRestoredImage = npaFlatImage.reshape((height, width, -1))

        # un-comment to see the restored image
        cv2.imshow("npaRestoredImage", npaRestoredImage)
        print("restored image label = " + str(label))
        cv2.waitKey()

    # end for

    print("done !!")

# end main

#######################################################################################################################
if __name__ == "__main__":
    main()