from __future__ import print_function
import torch


def sample_1():
    # x = torch.Tensor(5, 3)    # 构造一个未初始化的5*3的矩阵
    x = torch.rand(5, 3)        # 构造一个随机初始化的矩阵
    print(x)                    # 此处在notebook中输出x的值来查看具体的x内容
    print(x.size())

    # NOTE: torch.Size 事实上是一个tuple, 所以其支持相关的操作*
    y = torch.rand(5, 3)

    print(x + y)                # 语法一
    print(torch.add(x, y))      # 语法二

    # 另外输出tensor也有两种写法
    result = torch.Tensor(5, 3) # 语法一
    torch.add(x, y, out=result) # 语法二
    y.add_(x) # 将y与x相加

    # 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
    # 例如：x.copy_(y), x.t_(), 这俩都会改变x的值。

    # 另外python中的切片操作也是资次的。
    print(x[:, 1])              # 这一操作会输出x矩阵的第二列的所有值


def sample_2():
    # 此处演示tensor和numpy数据结构的相互转换
    a = torch.ones(5)
    b = a.numpy()

    # 此处演示当修改numpy数组之后,与之相关联的tensor也会相应的被修改
    a.add_(1)
    print(a)
    print(b)


if __name__ == '__main__':
    # sample_1()
    sample_2()
