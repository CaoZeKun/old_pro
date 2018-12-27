# class Test(object):
#     def __enter__(self):
#         print( "In __enter__()")
#         return "test_with"
#     def __exit__(self, type, value, trace):
#         print( "In __exit__()")
# def get_example():
#     return Test()
# with get_example() as example:
#     print( "example:", example)
#     print("1")

from torch.autograd import Variable
import torch
import numpy as np

x = Variable(torch.Tensor([[0.5],[0.3]]))
w1 = Variable(torch.ones(1,1),requires_grad =True)
w2 = Variable(torch.ones(1,2),requires_grad =True)

y1 = w1*0.5
y2 = w2.mm(x)

t = torch.cat((y1,y2),dim=1)
print(t)
y_y = t.mm(x)

print(y_y)
y_y.backward()
print(w1.grad)
print(w2.grad)
# print(w1)
# print(w2)
# print(x)
a = np.array([1,2,3])
b = np.array([2,3,4])
print(a*b)