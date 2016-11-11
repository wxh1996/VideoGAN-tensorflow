import os
import cv2
import numpy as np

class DataLoader(object):

    def __init__(self):
        #reading data list
        self.data_root = "/home/peng/data/Videogan/frames-stable-many"
        self.batch_size = 64
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128
        self.data_list_path = os.path.join(self.data_root, 'golf.txt')
        print 'reading data list...'
        with open(self.data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readlines()]
        self.size = len(self.video_index)
        self.cursor = 0

    def get_batch(self):

        if self.cursor +self.batch_size > self.size:
            self.cursor = 0
        out = np.zeros((self.batch_size, self.frame_size, self.crop_size, self.crop_size, 3))
        # tinyvideo = np.zeros((self.frame_size, self.crop_size, self.crop_size, 3))
        for idx in xrange(self.batch_size):
            video_path = os.path.join(self.data_root, self.video_index[self.cursor])
            self.cursor += 1
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self.image_size
            for j in xrange(self.frame_size):
                if j < count:
                    cut = j * self.image_size
                else:
                    cut = (count - 1) * self.image_size
                crop = inputimage[cut : cut + self.image_size, :]
                # tinyvideo[j, :, :, :] = cv2.resize(crop, self.crop_size, self.crop_size)
                out[idx, j, :, :, :] = cv2.resize(crop, (self.crop_size, self.crop_size))

        out = out / 255.0 * 2 - 1

        return out
