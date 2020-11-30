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

layer1 = Multiply()
layer2 = Multiply()
layer3 = Add()
layer4 = Multiply()


# forward
appleprice = 0
orangeprice = 0
appleorangeprice = 0
totalprice = 0

appleprice = layer1.forward(apple, applecount)
print(f'appleprice = {appleprice}')

orangeprice = layer2.forward(orange, orangecount)
print(f'orangeprice = {orangeprice}')

appleorangeprice = layer3.forward(appleprice, orangeprice)
print(f'appleorangeprice = {appleorangeprice}')

totalprice = layer4.forward(appleorangeprice, discount)
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

dappleorangeprice, ddiscount = layer4.backward(1)
print(f'dappleorangeprice = {dappleorangeprice}, ddiscount = {ddiscount}')

dappleprice, dorangeprice = layer3.backward(dappleorangeprice)
print(f'dappleprice = {dappleprice}, dorangeprice = {dorangeprice}')

dorange, dorangecount = layer2.backward(1)
print(f'dorange = {dorange}, dorangecount : {dorangecount}')

dapple, dapplecount = layer1.backward(1)
print(f'dapple = {dapple}, dapplecount : {dapplecount}')


















