class InterChars:
    def __init__(self, chars):
        self.chars = chars
    def __repr__(self):
        return self.chars

class Null:
    def __repr__(self):
        return ''
    def __add__(self, obj):
        return obj
    __radd__ = __add__
    __iadd__ = __add__
    def __lt__(self, obj):
        return False
    __le__ = __lt__
    __eq__ = __lt__
    __ge__ = __lt__
    __gt__ = __lt__
