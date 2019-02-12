'''the input data was normalised to get it into the range of 0-1 from 0-255.
(library used to store= shutil, os)

and the target labels were divided by 2.

Image was resized to the shape (240,320,3) from  (480,340,3)
in (H, W, C) format.

then
for data augmentation, the images were flipped holrizontally and were apllied with some filters to double the data size.
we tried to increase the data size to 6 times first but that turned out to be giving a validation iou of not more than 0.87.
so we decided to double the datas size.'''




#** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

class HorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes

** ** ** ** ** ** ** ** ** *
then
the
following
augmentation
code
was
inserted
inside
the
generator
function.
#** ** ** ** ** ** ** ** ** *

img = cv2.imread(x_path)

bbox = [y_path[0], y_path[2], y_path[1], y_path[3]]
bboxes = np.array([bbox])
img_, bboxes_ = HorizontalFlip()(img, bboxes)
flip_y_path = [bboxes_[0][0], bboxes_[0][2], bboxes_[0][1], bboxes_[0][3]]

img_list = []
img_list.append(img_)

aug = iaa.SomeOf(1, [
    iaa.Sharpen(alpha=0.5),
    iaa.GaussianBlur(sigma=1.0),
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
], random_order=True)
img_aug_list = aug.augment_images(img_list)

'''''''** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **