"""karaoke 核心 API（初始未实现）。"""


class Karaoke:
    def __init__(self, strength):
        """构造"""
        raise NotImplementedError

    def process(self, left, right):
        """返回 (l, r)"""
        raise NotImplementedError

    def center_db(self, left, right):
        """中置衰减量"""
        raise NotImplementedError
