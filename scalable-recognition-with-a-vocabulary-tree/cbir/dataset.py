import cv2
import os
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters


class Dataset():
    def __init__(self, folder="data/jpg"):
        """Dataset initialization
        Args:
            folder (str, optional): Path of the folder where images are. Images
                must be in jpg format
        """
        self.path = folder
        self.image_paths = [f for f in sorted(listdir(
            self.path)) if isfile(join(self.path, f))]
        self.subset = Subset(self)

    def __str__(self):
        images = []
        for i in range(len(self.image_paths)):
            images.append(self.image_paths[i])
            if i == 5:
                break
        return str(images)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, read=False):
        items = self.image_paths[idx]
        if read:
            return list(map(self.read_image, items))
        return items

    def read_image(self, image_path, scale=1.):
        """Reads an image from the image folder

        Args:
            image_path (TYPE): Image path
            scale (float, optional): Scale factor for image resizing. Default is 1, which means no scaling.

        Returns:
            Image: as an np.array

        Raises:
            FileNotFoundError: If the image is not found or can't be read
        """
        if not self.is_image(image_path):
            image_path = image_path + ".jpg"

        if not (isfile(image_path)):
            image_path = os.path.abspath(join(self.path, image_path))

        if not (isfile(image_path)):
            raise FileNotFoundError(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_random_image(self):
        """Returns a random image from the dataset

        Returns:
            Image: as an np.array
        """
        return self.read_image(random.choice(self.image_paths))

    def show_image(self, image, gray=False, **kwargs):
        """Displays an image

        Args:
            img (np.array): The image to show
            gray (bool, optional): Wether to use grayscale colormap
            **kwargs: Extra options to the plt.imshow() function
        """
        if isinstance(image, str):
            image = self.read_image(image)
        if not gray:
            plt.imshow(image, aspect="equal", **kwargs)
        else:
            plt.imshow(image, aspect="equal", cmap="gray", **kwargs)

    def is_image(self, path):
        allowed_extensions = [
            ".jpeg", ".jpg", ".jp2",
            ".png",
            ".bmp",
            ".tif", ".tiff",
            "pbm", ".pgm", "ppm"]
        return os.path.splitext(path)[-1] in allowed_extensions

    def extract_color_hist(self, img_bgr):
        b_hist = cv2.calcHist([img_bgr], [0], None, [20], [0, 255],).flatten()
        g_hist = cv2.calcHist([img_bgr], [1], None, [20], [0, 255]).flatten()
        r_hist = cv2.calcHist([img_bgr], [2], None, [20], [0, 255]).flatten()

        final_list=b_hist.tolist()
        final_list.extend(g_hist.tolist())
        final_list.extend(r_hist.tolist())
        tmean = np.mean(final_list)  # 求均值
        tstd = np.std(final_list)  # 求方差
        newfea = (final_list - tmean) / tstd  # 数值归一化
        # if show:
        #     '''显示三个通道的颜色直方图'''
        #     plt.hist(b_hist, label='B', color='blue')
        #     plt.hist(g_hist, label='G', color='green')
        #     plt.hist(r_hist, label='R', color='red')
        #     plt.legend(loc='best')
        #     # plt.xlim([0, 256])
        #     plt.show()
        return newfea.tolist()

    def extract_gabor(self, img,w=16,h=16):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
        # gabor变换
        real, imag = filters.gabor(img_gray, frequency=0.6, theta=45, n_stds=5)
        # 取模
        img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)
        # 图像缩放（下采样）
        newimg = cv2.resize(img_mod, (w, h), interpolation=cv2.INTER_AREA)
        tempfea = newimg.flatten()  # 矩阵展平
        tmean = np.mean(tempfea)  # 求均值
        tstd = np.std(tempfea)  # 求方差
        newfea = (tempfea - tmean) / tstd  # 数值归一化
        # if show:
        #     # 图像显示
        #     plt.figure()
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(img_gray, cmap='gray')
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(img_mod, cmap='gray')
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(real, cmap='gray')
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(imag, cmap='gray')
        #     plt.show()
        return newfea.tolist()

    def extract_other_features(self):
        # 读取图片
        other_features = {}
        for i, imgName in enumerate(self.image_paths):
            pic_file = join(self.path, imgName)
            img = cv2.imread(pic_file)  # 读图像

            color_feature = self.extract_color_hist(img)
            gabor_feature = self.extract_gabor(img)
            color_feature.extend(gabor_feature)
            other_features[imgName]= color_feature
        return other_features


class Subset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        subset = self.dataset
        subset.image_paths = subset.image_paths[idx]
        return subset
