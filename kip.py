import numpy as np
from skimage.feature import blob_log


class ImageSet:
    def __init__(self, npz_filename):
        self.npz_filename = npz_filename
        self.npz_file = np.load(npz_filename)
        self.array_dict = dict()
        for key in self.npz_file.keys():
            self.array_dict.update({key: self.npz_file[key]})

    @staticmethod
    def find_blobs(np_array, channel_num):
        blob_list = []
        for i in range(np_array.shape[0]):
            plane = np_array[i, :, :, channel_num]
            blob = blob_log(plane, 1, 10, 100, 1, 0, False, exclude_border=3)
            blob_list.append(blob)
        return blob_list