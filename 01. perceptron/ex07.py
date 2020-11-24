# 계단함수
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Fount')


x = np.arange(-5.0, 5.0, 0.1)
y=step(x)
plt.plot(x, y)
plt.show()