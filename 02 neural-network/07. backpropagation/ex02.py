# Multiply & Add Layer Test

import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Multiply, Add
except ImportError:
    print('Library Module Can Not Found')



# data

apple = 100
applecount = 3
orange =200
orangecount = 5
discount = 0.9

# layers

layer1_1 = Multiply()       # 병렬
layer1_2 = Multiply()
layer2 = Add()
layer3 = Multiply()


# forward
appleprice = 0
orangeprice = 0
appleorangeprice = 0
totalprice = 0

appleprice = layer1_1.forward(apple, applecount)
print(f'appleprice = {appleprice}')

orangeprice = layer1_2.forward(orange, orangecount)
print(f'orangeprice = {orangeprice}')

appleorangeprice = layer2.forward(appleprice, orangeprice)
print(f'appleorangeprice = {appleorangeprice}')

totalprice = layer3.forward(appleorangeprice, discount)
print(f'totalprice = {totalprice}')

print('====================================')


# backward propagation
dtotalprice = 1

dappleorangeprice = 0
ddiscount = 0

dappleprice = 0
dorangeprice = 0

dapple = 100
dapplecount = 3

dorange =200
dorangecount = 5

dappleorangeprice, ddiscount = layer3.backward(1)
print(f'dappleorangeprice = {dappleorangeprice}, ddiscount = {ddiscount}')

dappleprice, dorangeprice = layer2.backward(dappleorangeprice)
print(f'dappleprice = {dappleprice}, dorangeprice = {dorangeprice}')

dorange, dorangecount = layer1_2.backward(dorangeprice)
print(f'dorange = {dorange}, dorangecount : {dorangecount}')

dapple, dapplecount = layer1_1.backward(dappleprice)
print(f'dapple = {dapple}, dapplecount : {dapplecount}')


















