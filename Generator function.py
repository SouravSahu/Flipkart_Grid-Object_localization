
'''



'''

#** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

from keras.utils import Sequence


class Generator(Sequence):

    def __init__(self, X, Y, batch):
        self.x = X
        self.y = Y
        self.batch = batch
        self.cnt = 0

    def __len__(self):
        return int(np.floor(np.array(self.x).shape[0]) / self.batch)

    def __getitem__(self, index):
        path_x = self.x[index * self.batch:(index + 1) * self.batch]
        path_y = self.y[index * (self.batch):(index + 1) * self.batch]

        X = []
        Y = []
        for x_path, y_path in zip(path_x, path_y):

            img = cv2.imread(x_path)

            bbox = [y_path[0], y_path[2], y_path[1], y_path[3]]
            bboxes = np.array([bbox])
            img_, bboxes_ = HorizontalFlip()(img, bboxes)
            flip_y_path = [bboxes_[0][0], bboxes_[0][2], bboxes_[0][1], bboxes_[0][3]]

            img_list = []
            for x in range(1):
                # img_list.append(img)
                img_list.append(img_)

            aug = iaa.SomeOf(1, [
                iaa.Sharpen(alpha=0.5),
                iaa.GaussianBlur(sigma=1.0),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
            ], random_order=True)
            img_aug_list = aug.augment_images(img_list)

            X.append(img)
            X.append(img_aug_list[0])
            Y.append(y_path)
            Y.append(flip_y_path)
            self.cnt = self.cnt + 1

        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        X = X / 255.0
        return X, Y

#*************************************



class ValGenerator(Sequence):

    def __init__(self, X, Y, batch):
        self.x = X
        self.y = Y
        self.batch = batch
        self.cnt = 0

    def __len__(self):
        return int(np.floor(np.array(self.x).shape[0]) / self.batch)

    def __getitem__(self, index):
        path_x = self.x[index * self.batch:(index + 1) * self.batch]
        path_y = self.y[index * (self.batch):(index + 1) * self.batch]

        X = []
        Y = path_y
        # print(path_y)

        for x_path in path_x:
            img = cv2.imread(x_path)
            temp_x = np.array(img, dtype="float32")
            temp_x = temp_x / 255

            self.cnt = self.cnt + 1
            X.append(temp_x)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y
