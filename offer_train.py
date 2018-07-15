# ！/user/bin/env python3
# -*- coding:utf-8 -*-
# @Author   :Neways
# @Time     :2018/7/15 0:54
# @Contact  :NewaysNju@gmail.com

# 3_1. 找出长度为n的数组中，数字都在1~n-1之间。寻找任一重复数字
def duplicate2(numbers):
    for i in range(len(numbers)):
        while numbers[i] != i:
            if numbers[i] == numbers[numbers[i]]:
                return numbers[i]
            else:
                m = numbers[i]
                numbers[i], numbers[m] = numbers[m], numbers[i]
    return False


# 3_2. 不修改数组找出任意重复的数字
def countnum(numbers, subnum):
    l = 0
    for i in numbers:
        if i in subnum:
            l += 1
    return l


def duplicate3(numbers):
    # 空间效率优先，此方法空间复杂度为O(1),时间复杂度为O(nlogn)
    start = 0
    end = len(numbers) - 1
    while start <= end:
        mid = (start + end) // 2
        lenn = mid - start + 1  # 计算长度
        coun = countnum(numbers, numbers[start:(mid + 1)])  # 选择分段区间内的计数
        if end == start:  # 找到了
            if coun > 1:
                return start  # 的确有
            else:
                break  # 没有，接下来返回-1
        elif coun > lenn:  # 判别情况
            end = mid
        else:
            start = mid + 1
    return -1


# 二维数组中的查找
def Find(target, array):
    # write code here
    rown = 0
    coln = len(array[0]) - 1
    while (rown < len(array)) and (coln >= 0):
        if array[rown][coln] == target:
            return True
        elif array[rown][coln] > target:
            coln = coln - 1
        else:
            rown = rown + 1
    return False


# 替换空格
def replaceSpace(s):
    s = list(s)
    n = 0
    inds = len(s)
    for i in s:
        if i == ' ':
            n += 1
    s.extend(' ' * 2 * n)
    k = inds - 1 + 2 * n
    for j in s[:inds][::-1]:
        if j != ' ':
            s[k] = j
            k -= 1
        else:
            s[(k - 2):(k + 1)] = '%20'
            # s[k - 1] = '2'
            # s[k - 2] = '%'
            k -= 3
    return ''.join(s)


# 从头到尾打印链表
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        L = []
        head = listNode
        while head:
            L.insert(0, head.val)
            head = head.next
        return L


# 重建二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # pre是前序遍历序列,tin是中序遍历序列
        if len(pre) == 0:
            # 如果输入是空
            return None
        if len(pre) == 1:
            # 返回叶子节点
            return TreeNode(pre[0])
        else:
            tree = TreeNode(pre[0])
            tree.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0]) + 1],
                                                   tin[:tin.index(pre[0])])
            tree.right = self.reConstructBinaryTree(pre[(tin.index(pre[0]) + 1):],
                                                    tin[(tin.index(pre[0]) + 1):])
            return tree


# 二叉树的下一个节点
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None  # 指向父节点


class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return None
        if pNode.right:
            # 节点有右子树
            rightree = pNode.right
            while rightree.left:
                rightree = rightree.left
            return rightree
        else:
            if not pNode.next:
                return None
            if pNode == pNode.next.left:
                # 节点无右子树，且节点是其父节点的左子节点
                return pNode.next
            else:
                # 节点无右子树，且节点是其父节点的右子节点
                nextree = pNode
                if not nextree.next:
                    return None
                while nextree != nextree.next.left:
                    nextree = nextree.next
                    if not nextree.next:
                        return None
                return nextree.next


# 用两个栈实现队列
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, node):
        # 加只往stack1里添加
        self.stack1.append(node)

    def pop(self):
        if self.stack2 == []:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        # 如果stack2不是空的，那么直接去掉stack2里最后一个就行
        return self.stack2.pop()


# 斐波那契数列问题
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        a, b = 0, 1
        if n == 0:
            return a
        if n == 1:
            return b
        for i in range(2, n + 1):
            a, b = b, a + b
        return b


# 青蛙跳台阶问题
class Solution:
    def jumpFloor(self, number):
        a, b = 1, 2
        if number == 1:
            return a
        if number == 2:
            return b
        for i in range(3, number + 1):
            a, b = b, a + b
        return b


# 青蛙跳变态版
class Solution:
    def jumpFloorII(self, number):
        return (2 ** (number - 1))


# 矩形覆盖
class Solution:
    def rectCover(self, number):
        if number < 3:
            return number
        else:
            a, b = 1, 2
            for i in range(3, number):
                a, b = b, a + b
            return b


# 旋转数组的最小数字
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # 第一个指针始终在前面递增数组上，第二个指针始终在后面递增的数组上，结束条件，第一个指针与第二个指针相邻
        if len(rotateArray) == 0:
            return []
        left = 0
        right = len(rotateArray) - 1
        if rotateArray[left] < rotateArray[right]:
            return rotateArray[left]
        if rotateArray[left] == rotateArray[right]:
            for i in range(right):
                if rotateArray[i] > rotateArray[i + 1]:
                    return rotateArray[i + 1]
        mid = (left + right) // 2
        while right - left > 1:
            if rotateArray[mid] >= rotateArray[left]:
                left = mid
                mid = (left + right) // 2
            else:
                if rotateArray[mid] <= rotateArray[right]:
                    right = mid
                    mid = (left + right) // 2
        return rotateArray[right]


# 矩阵中的路径
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        if not matrix or rows < 1 or cols < 1 or not path:
            return False  # 排除异常情况1）矩阵是空2）行列错误3）字符串是空
        pathmatrix = [False] * rows * cols
        for i in range(rows):
            for j in range(cols):
                if self.dfs(matrix, rows, cols, i, j, 0, path, pathmatrix):
                    return True
        return False

    def dfs(self, matrix, rows, cols, x, y, index, path, pathmatrix):
        if x < 0 or x >= rows or y < 0 or y >= cols or matrix[x * cols + y] != path[index] or pathmatrix[x * cols + y]:
            return False  # 排除1）撞边2）值不匹配3）重复访问的问题
        if index == len(path) - 1:
            return True  # happy ending
        index += 1
        pathmatrix[x * cols + y] = True
        for [p, q] in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
            if self.dfs(matrix, rows, cols, p, q, index, path, pathmatrix):
                return True

        index -= 1
        pathmatrix[x * cols + y] = False
        return False


# 机器人的运动范围
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        if threshold < 0 or rows < 1 or cols < 1:
            return False
        pathmatrix = [False] * rows * cols
        self.dfs(threshold, rows, cols, 0, 0, pathmatrix)
        return sum(pathmatrix)

    def dfs(self, threshold, rows, cols, x, y, pathmatrix):
        if x >= 0 and x < rows and y >= 0 and y < cols and \
                        sum(map(lambda i: int(i), list(str(x) + str(y)))) <= threshold and not pathmatrix[x * cols + y]:
            pathmatrix[x * cols + y] = True
            for [p, q] in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                self.dfs(threshold, rows, cols, p, q, pathmatrix)


# 剪绳子
class Solution:
    def maxProductAfterCutting_solution1(self, n):
        if n < 2:
            return 0
        elif n == 2:
            return 1
        elif n == 3:
            return 2
        else:
            product = list()
            product[0:3] = range(4)
            for i in range(4, n + 1):
                max = 0
                for j in range(1, i):
                    if max < product[j] * product[i - j]:
                        max = product[j] * product[i - j]
            product.append(max)
        return product[-1]

    def maxProductAfterCutting_solution2(self, n):
        if n < 2:
            return 0
        elif n == 2:
            return 1
        elif n == 3:
            return 2
        else:
            odd3 = n // 3
            if n - odd3 * 3 == 1:
                return 3 ** (odd3 - 1) * 4
            elif n == odd3 * 3:
                return 3 ** odd3
            else:
                return 3 ** odd3 * 2


# 数值的整数次方
class Solution:
    def Power(self, base, exponent):
        if exponent > 0:
            ss = 1
            for i in range(exponent):
                ss = ss * base
            return ss
        elif exponent == 0:
            if base == 0:
                return '0的0次方没有意义'
            else:
                return 1
        else:
            try:
                base = 1 / base
                ss = 1
                for i in range(-exponent):
                    ss = ss * base
                return ss
            except ZeroDivisionError:
                return '0的负整数次方不存在!'


class Solution:
    # -2147483648
    def NumberOf1(self, n):
        count = 0
        while (n):
            count += 1
            n = n & (n - 1)
        return count

    def NumberOf1_2(self, n):
        count = 0
        for i in range(len(bin(n)) - 2):
            if (n & 1 << i) >> i:
                count += 1
        return count

    def NumberOf1_3(self, n):
        count = 0
        for i in range(len(bin(n)) - 2):
            if n >> i & 1:
                count += 1
        return count


# 打印从1到最大的n位数
class Solution():
    def Print1ToMaxOfNDigits_1(self, n):
        pass


# 删除链表中的节点--在O(1)时间内删除链表节点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deletNode(self, pHead, ptoBeDelete):
        if pHead == ptoBeDelete:
            # 如果链表只有一个节点
            pHead.val = None
            return None
        elif ptoBeDelete.next == None:
            # 如果待删除的节点在链表最后一位
            pnode = pHead
            while pnode.next != ptoBeDelete:
                pnode = pnode.next
            pnode.next = None
            return None
        else:
            # 如果待删除的节点在链表中
            ptoBeDelete = ptoBeDelete.next
            return None


# 删除链表中重复的节点
# 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针
class Solution:
    def deleteDuplication(self, pHead):
        if pHead.next == None or pHead == None:
            # 如果这个链表是空，或者只有一个结点
            return pHead
        elif pHead.val != pHead.next.val:
            # 开头不等于
            pHead.next = self.deleteDuplication(pHead.next)
        else:
            # 如果开头节点就是重复节点（需要删除）
            pnext = pHead.next
            while pHead.val == pnext.val and pnext.next != None:
                # 移动next节点，实际上就是删除重复节点
                pnext = pnext.next
            if pnext.val != pHead.val:
                # 如果已经不重复了，递归来考虑下一个节点是否重复
                pHead = self.deleteDuplication(pnext)
            else:
                # 这是整个链表都是重复的节点情况下
                return None
        return pHead


# 正则表达式匹配
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if len(s) == 0 and len(pattern) == 0:
            return True
        if len(s) != 0 and len(pattern) == 0:
            return False
        if len(pattern) > 1 and pattern[1] == '*':
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
                # 表示或者s与pattern元素相等，或者pattern元素为全通配符'.'
                # 下面分别表示三种状态，1)pattern中元素多次与s匹配；2)s与pattern匹配一次；3) s与pattern形式上匹配，但是应该跳过
                return self.match(s[1:], pattern[2:]) or self.match(s[1:], pattern) or self.match(s, pattern[2:])
            else:
                # 这里表示pattern与s不匹配
                return self.match(s, pattern[2:])
        if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
            return self.match(s[1:], pattern[1:])
        return False


# 表示数值的字符串
class Solution:
    # s字符串
    # 可识别模式： A[.[B]][e|EC] or .B[e|EC]
    def isNumeric(self, s):
        if len(s) == 0:
            return False
        # A部分
        s = self.scaninteger(s)
        # B 部分
        if s[0] == '.' and len(s) > 1:
            s = self.scanuninteger(s[1:])
        # C 部分
        if (s[0] == 'E' or s[0] == 'e') and len(s) > 1:
            s = self.scaninteger(s[1:])
        if len(s) == 1 and s[0] >= '0' and s[0] <= '9':
            return True
        else:
            return False

    def scaninteger(self, s):
        if len(s) > 1 and (s[0] == '+' or s[0] == '-'):
            s = s[1:]
        return self.scanuninteger(s)
    def scanuninteger(self, s):
        while s[0] >= '0' and s[0] <= '9' and len(s) > 1:
            s = s[1:]
        return s


# 调整数组顺序使奇数位于偶数前面
class Solution:
    # 奇数位于前半段，偶数位于后半段，并保证奇数与奇数，偶数与偶数相对位置不变
    def reOrderArray(self, array):
        L1 = []
        L2 = []
        for i in array:
            if i % 2 == 1:
                L1.append(i)
            else:
                L2.append(i)
        L1.extend(L2)
        return L1
