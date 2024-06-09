import torch
from numpy import linalg
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')


def pca_color_augmention(image_array):
    '''
    image augmention: PCA jitter
    :param image_array: 图像array
    :return img2: 经过PCA-jitter增强的图像array
    '''
    assert image_array.dtype == 'uint8'
    assert image_array.ndim == 3
    # 输入的图像应该是 (w, h, 3)这样的三通道分布

    # 除以255.0，将像素值范围从0到255映射到0到1之间
    img1 = image_array.astype('float32') / 255.0
    # 分别计算R，G，B三个通道的方差和均值
    mean = img1.mean(axis=0).mean(axis=0)
    std = img1.reshape((-1, 3)).std()  # 不可以使用img1.std(axis = 0).std(axis = 0)

    # 将图像标按channel标准化（均值为0，方差为1）
    img1 = (img1 - mean) / (std)

    # 将图像按照三个通道展成三个长条
    img1 = img1.reshape((-1, 3))

    # 对矩阵进行PCA操作
    # 求矩阵的协方差矩阵
    cov = np.cov(img1, rowvar=False)
    # 求协方差矩阵的特征值和向量
    eigValue, eigVector = linalg.eig(cov)

    # 抖动系数（均值为0，方差为0.1的标准分布）
    rand = np.array([random.normalvariate(0, 0.2) for i in range(3)])

    # 这个点积的结果是一个与特征向量具有相同维度的数组，
    # 它表示了一个在特征向量方向上缩放的随机变化
    jitter = np.dot(eigVector, eigValue * rand)
    jitter_ret = jitter*255

    jitter = (jitter * 255).astype(np.int32)[np.newaxis, np.newaxis, :]

    # 数值饱和处理，即大于255则视为255，小于0则视为0
    img2 = np.clip(image_array + jitter, 0, 255)

    return img2, jitter_ret


def show_image(image_array):
    RGB = ["R", "G", "B"]
    for _ in range(8):
        ax = plt.subplot(2, 4, 1 + _)
        img, jitter = pca_color_augmention(image_array)
        [plt.text(0, image_array.shape[1]+i*300, (RGB[i], round(item, 2))) for i, item in enumerate(jitter)]
        print(_, " jitter:", jitter)
        ax.imshow(img)
        ax.axis('off')
    plt.show()


img_tensor = np.array(Image.open('./images/leaf.jpg'))
show_image(img_tensor)

