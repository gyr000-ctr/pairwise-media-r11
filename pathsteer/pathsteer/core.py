"""pathsteer 核心 API（初始未实现）。"""


class Steering:
    def __init__(self, radius, vmax):
        """构造"""
        raise NotImplementedError

    def step(self, agents, dt):
        """单步"""
        raise NotImplementedError

    def weights(self):
        """三力权重"""
        raise NotImplementedError
