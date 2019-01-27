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

null = Null()
