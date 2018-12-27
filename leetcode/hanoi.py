def move(from1,to):  # 将盘子从from移动到to，动画效果需要脑补
    print(from1,'->',to)


def hanoi(n,src,tmp,dst):  # 将n个盘子从src搬到dst
    if n == 1:  # 只有一个盘子的情况
        move(src,dst)
    else:  # 有一个以上盘子的情况
        hanoi(n-1,src,dst,tmp)  # 将上方的n-1个盘子从src搬到tmp
        move(src,dst)  # 将第n个盘子从src轻松愉快地移动到dst
        hanoi(n-1,tmp,src,dst)  # 将tmp上的n-1个盘子搬到dst上


hanoi(3,'A','B','C')