# sigmoid function & graph

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
except ImportError:
    print('Library Module Can Not Fount')

x = np.arange(-10, 10, 0.1)
y = sigmoid(x)

plt.subplots(x, y)
plt.show()