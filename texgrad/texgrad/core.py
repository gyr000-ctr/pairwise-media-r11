"""texgrad 核心 API（初始未实现）。"""


class MipChain:
    def __init__(self, width, height):
        """构造"""
        raise NotImplementedError

    def sizes(self):
        """各级尺寸"""
        raise NotImplementedError

    def downsample(self, level, image):
        """生成某级"""
        raise NotImplementedError
