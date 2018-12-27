class LNode:
    def __init__(self,x):
        self.data = x  # 数据域
        self.next = None  # 下一个结点的引用


def Reverse1(head):
    if head is None or head.next is None or head.next.next is None:
        return
    # if head == None or head.next == None or head.next.next == None:
    #     return
    pre = None
    cur = None
    next_node = None

    cur = head.next
    next_node = cur.next
    cur.next = None
    pre = cur
    cur = next_node

    while cur.next is not None:
        next_node = cur.next
        cur.next= pre
        pre = cur
        cur = next_node

    cur.next = pre
    head.next = cur


def RecursiveRevse2(head):
    if head is None or head.next is None:
        return head
    else:
        newhead = RecursiveRevse2(head.next)
        head.next.next = head
        head.next = None
        return newhead




def Reverse2(head):
    if head is None:
        return
    firstNode = head.next
    newhead = RecursiveRevse2(firstNode)
    head.next = newhead
    return head





if __name__=="__main__":

    i = 1
    head = LNode(None)
    head.next = None
    tmp = None
    cur = head

    while i <8:
        tmp = LNode(i)
        # tmp.data = i
        tmp.next =None
        cur.next = tmp
        cur = tmp
        i += 1
    print("逆序前")
    cur = head.next
    while cur != None:
        print(cur.data)
        cur = cur.next
    print("逆序后")
    # Reverse1(head)
    head = Reverse2(head)
    cur = head.next
    while cur != None:
        print(cur.data)
        cur = cur.next


