# Multiply layer

class Multiply:
    def __init__(self):     # 초기화
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout

        return dy, dx

class Add:
    def __init__(self):     # 초기화
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x + y
        return out

    def backward(self, dout):
        dx = dout - self.y
        dy = dout - self.x

        return dx, dy