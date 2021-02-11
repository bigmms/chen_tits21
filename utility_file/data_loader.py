from glob import glob
import numpy as np
from skimage.io import imread

class DataLoader():
    def __init__(self, dataset_name=None, img_res=(80, 176)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.train_sum = 0
        self.n_batches = 1

    def load_batch(self, batch_size=1):
        path_I = glob('./%s/train/I/*.png' % self.dataset_name)
        self.n_batches = len(path_I) // batch_size
        indexes = np.random.permutation(len(path_I))
        for i in range(self.n_batches):
            idx = indexes[(batch_size * i):(batch_size * (i + 1))]
            Is = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
            Ls = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
            Bs = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
            GTs = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
            for j in range(batch_size):
                Is[j, :, :, :] = (imread(path_I[idx[j]]) / 127.5 - 1).astype(np.float32)
                Ls[j, :, :, :] = (imread(path_I[idx[j]].replace("I", "L"))).astype(np.float32)
                Bs[j, :, :, :] = (imread(path_I[idx[j]].replace("I", "B"))).astype(np.float32)
                GTs[j, :, :, :] = (imread(path_I[idx[j]].replace("I", "G"))).astype(np.float32)
            yield Is, Ls, Bs, GTs

    def load_data(self):
        path_I = glob('./dataset/train/I/*.png')
        path_G = glob('./dataset/train/G/*.png')
        idx = np.random.choice(len(path_I), 1)[0]
        Is = np.zeros((1, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
        GTs = np.zeros((1, self.img_res[0], self.img_res[1], 3)).astype(np.float32)
        Is[0, :, :, :] = (imread(path_I[idx])).astype(np.float32)
        GTs[0, :, :, :] = (imread(path_G[idx])).astype(np.float32)
        return Is, GTs

    def load_test_data(self):
        path_I = glob('./dataset/train/I/*.png')
        Is = []
        for img_path in path_I:
            I = imread(img_path)
            Is.append(I)
        Is = np.array(Is).astype(np.float32)
        return Is