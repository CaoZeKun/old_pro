# Add Two Numbers
# You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit.
# Add the two numbers and return it as a linked list.
# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummyHead = ListNode(0)
        p = l1
        q = l2
        curr = dummyHead
        carry = 0
        while(p is not None or q is not None):
            x = p.val if p is not None else 0
            y = q.val if q is not None else 0

            s_sum = x + y + carry
            carry = s_sum // 10
            curr.next = ListNode(s_sum % 10)
            curr = curr.next
            if(p is not None):
                p = p.next
            if(q is not None):
                q = q.next
        if(carry>0):
            curr.next = ListNode(carry)
        return dummyHead.next








l11 = ListNode(2)
l12 = ListNode(4)
l13 = ListNode(9)

l11.next = l12
l12.next = l13

l21 = ListNode(5)
l22 = ListNode(6)
l23 = ListNode(9)

l21.next = l22
l22.next = l23

s = Solution()
l_3 = s.addTwoNumbers(l11,l21)

while l_3:
    print(l_3.val)
    l_3 = l_3.next



