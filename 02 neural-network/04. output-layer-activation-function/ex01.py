#출력함수(출력층 활성함수) sigma() -
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity

except ImportError:
    print('Library Module Can Not Fount')


x = np.array