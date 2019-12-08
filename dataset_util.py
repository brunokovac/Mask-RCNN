import os
import numpy as np
import math

class Dataset():

    images = "Image"
    localization = "groundtruth_localization"
    recognition = "groundtruth_recognition"

    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.data_names = np.array(os.listdir(self.path + self.images))
        self.new_epoch = True
        self.current_batch = 0
        self.total_batches = math.ceil(len(self.data_names) // self.batch_size)
        return

    def next_batch(self):
        if self.new_epoch:
            np.random.shuffle(self.data_names)

        batch = self.data_names[self.current_batch * self.batch_size :
                                min((self.current_batch + 1) * self.batch_size, len(self.data_names))]

        if (self.current_batch + 1) * self.batch_size >= len(self.data_names):
            self.new_epoch = True
            self.current_batch = 0
        else:
            self.new_epoch = False
            self.current_batch += 1

        return batch

if __name__ ==  "__main__":
    d = Dataset("dataset/AOLP/AOLP/Subset_LE/Subset_LE/Subset_LE/", 100)

    for i in range(8):
        b = d.next_batch()
        print(b, len(b))

