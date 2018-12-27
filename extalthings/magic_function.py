
class DataTest:
    def __init__(self, id, address):
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1"
                  }
    #  可以用于索引
    def __getitem__(self, key):
        return self.address[key]


data = DataTest(1, "192.168.2.11")
print(data[1])

'''
    desc：尝试定义一种新的数据类型
          等差数列
    author：pythontab.com
'''


class ArithemeticSequence(object):
    def __init__(self, start=0, step=1):
        print('Call function __init__')
        self.start = start
        self.step = step
        self.myData = {}

    # 定义获取值的方法
    def __getitem__(self, key):
        print('Call function __getitem__')
        try:
            return self.myData[key]
        except KeyError:
            return self.start + key * self.step

    # 定义赋值方法
    def __setitem__(self, key, value):
        print('Call function __setitem__')
        self.myData[key] = value

    # 定义获取长度的方法
    def __len__(self):
        print('Call function __len__')
        # 这里为了可以看出__len__的作用， 我们故意把length增加1
        return len(self.myData) + 1

    # 定义删除元素的方法
    def __delitem__(self, key):
        print('Call function __delitem__')
        del self.myData[key]


s = ArithemeticSequence(1, 2)
print(s[3])  # 这里应该执行self.start+key*self.step，因为没有3这个key
s[3] = 100  # 进行赋值
print(s[3])  # 前面进行了赋值，那么直接输出赋的值100
print(len(s))  # 我们故意多加了1，应该返回2
del s[3]  # 删除3这个key
