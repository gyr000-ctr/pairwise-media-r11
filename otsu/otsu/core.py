"""otsu 核心 API（初始未实现）。"""


class Otsu:
    def threshold(self, image):
        """最优阈值"""
        raise NotImplementedError

    def binarize(self, image, t):
        """二值化"""
        raise NotImplementedError

    def histogram(self, image):
        """归一化直方图"""
        raise NotImplementedError
