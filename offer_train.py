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

# 链表中倒数第k个结点

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        if k < 1:
            return None
        if not head:
            # 如果链表是空的，哪来的val和next呢
            return None
        p = head
        for i in range(k-1):
            if head.next != None:
                head = head.next
            else:
                return None
        while head.next != None:
            head = head.next
            p = p.next
        return p.val


# 求链表中间点
## 如果总节点是奇数，那么返回中间节点，如果是偶数，则返回中间两个节点的任意一个
class Solution:
    def FindhalfToTail(self, head):
        if not head:
            # 如果链表是空的，哪来的val和next呢
            return None
        p = head
        while head.next != None and head.next.next != None:
            head = head.next.next
            p = p.next
        return p.val

# 链表中环的入口节点
# 我的代码
class Solution:
    def EntryNodeOfLoop(self, pHead):
        if not pHead or pHead.next == None or pHead.next.next == None:
            return None
        findnode = self.FindNode(pHead)
        if not findnode:
            return None
        else:
            n = self.FindNodeN(findnode)
            pa = pHead
            for i in range(n):
                pa = pa.next
            k = 1
            while pa != pHead:
                pHead = pHead.next
                pa = pa.next
                k += 1
            return k

    def FindNode(self, pHead):
        pa = pHead.next.next
        pb = pHead.next
        while pa != pb:
            if pHead.next != None and pHead.next.next != None:
                pa = pa.next.next
                pb = pb.next
            else:
                return None
        return pa

    def FindNodeN(self, pHead):
        pn = pHead.next
        n = 1
        while pn != pHead:
            pn = pn.next
            n += 1
        return n


# 正确运营的代码
class Solution:
    def EntryNodeOfLoop(self, pHead):
        if pHead == None or pHead.next == None or pHead.next.next == None:
            return None
        low = pHead.next
        fast = pHead.next.next
        while low != fast:
            if fast.next == None or fast.next.next == None:
                return None
            low = low.next
            fast = fast.next.next
        fast = pHead
        while low != fast:
            low = low.next
            fast = fast.next
        return fast


# 反转链表
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            # 当链表为空或者链表只有一个节点时
            return pHead
        last = None  # 这个点实际上是反转链表最后一个点，为None
        while pHead:
            # 实际上一次性处理三个点，last->pHead->tmp
            # 首先将pHead赋予tmp
            # 其次斩断pHead与tmp关联，反转：last <- pHead
            # 再次，递进这个关系，pHead左移,变成last
            # 最后，tmp左移，变成pHead
            tmp = pHead.next
            pHead.next = last
            last = pHead
            pHead = tmp
        return last

    def reverse_recursion(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        # 整个链表右移1位
        new_head = self.reverse_recursion(pHead.next)
        pHead.next.next = pHead
        pHead.next = None
        return new_head


# 合并两个排序的链表
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if pHead1 == None:
            return pHead2
        elif pHead2 == None:
            return pHead1
        pHead = ListNode(0)
        if pHead1.val < pHead2.val:
            pHead = pHead1
            pHead.next = self.Merge(pHead1.next, pHead2)
        else:
            pHead = pHead2
            pHead.next = self.Merge(pHead1, pHead2.next)
        return pHead

# 树的子结构
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        result = False
        if pRoot1 and pRoot2:
            if pRoot1.val == pRoot2.val:
                result = self.DoseTree1HaveTree2(pRoot1, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
        return result
    def DoseTree1HaveTree2(self, pRoot1, pRoot2):
        if not pRoot2:
            # 顺序也是很讲逻辑
            return True
        if not pRoot1:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoseTree1HaveTree2(pRoot1.left, pRoot2.left) and self.DoseTree1HaveTree2(pRoot1.right, pRoot2.right)

# 二叉树的镜像
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        if not root:
            return root
        if (not root.left) and (not root.right):
            return root
        root.left, root.right = root.right, root.left
        if root.left:
            self.Mirror(root.left)
        if root.right:
            self.Mirror(root.right)
    def Mirror_while(self, root):
        pass





# 例子
# atree = TreeNode(8)
# atree.left = TreeNode(6)
# atree.right = TreeNode(10)
# atree.left.left = TreeNode(5)
# atree.left.right = TreeNode(7)
# atree.right.left = TreeNode(9)
# atree.right.right = TreeNode(11)
#
# tt = Solution()
# tt.Mirror(atree)
# print(atree.left.val)

# 对称的二叉树
class Solution:
    def isSymmetrical(self, pRoot):
        def is_same(p1, p2):
            if not p1 and not p2:
                return True
            if (p1 and p2) and p1.val == p2.val:
                return is_same(p1.left, p2.right) and is_same(p1.right, p2.left)
            return False
        if not pRoot:
            return True
        if pRoot.left and pRoot.right:
            return False
        if not pRoot.left and pRoot.right:
            return False
        return is_same(pRoot.left, pRoot.right)

# 顺时针打印矩阵
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        if not matrix:
            return None
        row = len(matrix)
        col = len(matrix[0])
        start = 0
        L = []
        while col > start *2 and row > start * 2:
            L.extend(self.printmatrixincircle(matrix, row, col, start))
            start += 1
        return L
    def printmatrixincircle(self, matrix, row, col, start):
        circleL = []
        endrow = row - start
        endcol = col - start
        for i in range(start, endcol):
            circleL.append(matrix[start][i])
        if start < endrow - 1:
            # 第二步的前提条件：起始行要小于终止行
            for i in range(start+1, endrow):
                circleL.append(matrix[i][endcol - 1])
        if start < endrow - 1 and start < endcol - 1:
            # 第三步的前提：起始行要小于终止行，起始列要小于终止列
            for i in range(endcol - 2, start - 1, -1):
                circleL.append(matrix[endrow - 1][i])
        if start < endcol - 1 and start < endrow - 2:
            # 第四步的前提：起始列要小于终止列，行数最少有3行
            for i in range(endrow - 2, start, -1):
                circleL.append(matrix[i][start])
        return circleL

# 包含min函数的栈
class Solution:
    def __init__(self):
        self.min_stack = []
        self.stack = []

    def push(self, node):
        self.stack.append(node)
        if not self.min_stack or node < self.min_stack[-1]:
            self.min_stack.append(node)
        else:
            self.min_stack.append(self.min_stack[-1])

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def min(self):
        return self.min_stack[-1]

# 栈的压入、弹出序列
class Solution:
    def IsPopOrder(self, pushV, popV):
        if not pushV or len(pushV) != len(popV):
            # 如果进栈是空，或者进栈与出栈的长度一样
            return False
        stack = []
        for i in pushV:
            stack.append(i)
            while len(stack) and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
        if len(stack):
            # 就是出现不等的情况
            return False
        return True

# 不分行从上到下打印二叉树
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        l = []
        if not root:
            return l
        L = [root]
        while len(L) != 0:
            proot = L.pop(0)
            l.append(proot.val)
            if proot.left:
                L.append(proot.left)
            if proot.right:
                L.append(proot.right)
        return l

# 分行从上到下打印二叉树
class Solution:
    def PrintFromTopToBottom2(self, root):
        l = []
        if not root:
            return l
        L = [root]
        i = 1
        while len(L) != 0:
            proot = L.pop(0)
            l.append(proot.val)
            i -= 1
            if proot.left:
                L.append(proot.left)
            if proot.right:
                L.append(proot.right)
            if i == 0:
                print(l)
                l = []
                i = len(L)
        return ''

# 之字形打印二叉树
class Solution:
    def PrintFromTopToBottom3(self, root):
        l = []
        if not root:
            return l
        L = [root]
        i = 1
        r = 1
        while len(L) != 0:
            proot = L.pop(0)
            l.append(proot.val)
            i -= 1
            if proot.left:
                L.append(proot.left)
            if proot.right:
                L.append(proot.right)
            if i == 0:
                if r % 2 == 1:
                    print(l)
                else:
                    print(self.reverse(l))
                r += 1
                l = []
                i = len(L)
        return ''

    def reverse(self, l):
        d = []
        while l:
            d.append(l.pop())
        return d
        
# 二叉搜索树的后序遍历序列
class Solution:
    def VerifySquenceOfBST(self, sequence):
        if not sequence:
            return False
        root = sequence[-1]
        j = 0
        for i in range(len(sequence)):
            j = i
            if sequence[i] > root:
                break
        for k in range(j, len(sequence)):
            if sequence[k] < root:
                return False
        left = True
        if j > 0:
            left = self.VerifySquenceOfBST(sequence[:j])
        right = True
        if j < len(sequence) - 1:
            right = self.VerifySquenceOfBST(sequence[j:-1])
        return left and right

# 二叉树中和为某一值的路径
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # 不满足
        if not root:
            return []
        # 叶子节点正好满足
        if root and not root.left and not root.right and root.val == expectNumber:
            return [[root.val]]
        res = []
        left = self.FindPath(root.left, expectNumber-root.val)
        right = self.FindPath(root.right, expectNumber-root.val)
        # 把该父节点下的所有满足的路径都返回
        for i in left + right:
            res.append([root.val]+i)
        return res

# 复杂链表的复制
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        pNode = pHead
        while pNode != None:
            pClone = RandomListNode(pNode.label)
            pClone.next = pNode.next
            pNode.next = pClone
            pNode = pClone.next
        self.ConnectSiblingNodes(pHead)
        return self.ReconnectNodes(pHead)

    def ConnectSiblingNodes(self, pHead):
        pNode = pHead
        while pNode != None:
            pClone = pNode.next
            if pNode.random != None:
                pClone.random = pNode.random.next
            pNode = pClone.next

    def ReconnectNodes(self, pHead):
        pNode = pHead
        pCloneHead = None
        pCloneNode = None

        if pNode != None:
            pCloneHead = pCloneNode = pNode.next
            pNode.next = pCloneNode.next
            pNode = pNode.next
        while pNode != None:
            pCloneNode.next = pNode.next
            pCloneNode = pCloneNode.next
            pNode.next = pCloneNode.next
            pNode = pNode.next
        return pCloneHead

# 二叉搜索树与双向链表
class Solution:
    def Convert(self, pRootOfTree):
        if not pRootOfTree:
            return pRootOfTree
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree
        # 处理左子树
        self.Convert(pRootOfTree.left)
        left = pRootOfTree.left

        # 连接根与左子树最大结点
        if left:
            while (left.right):
                left = left.right
            pRootOfTree.left, left.right = left, pRootOfTree

        # 处理右子树
        self.Convert(pRootOfTree.right)
        right = pRootOfTree.right

        # 连接根与右子树最小结点
        if right:
            while (right.left):
                right = right.left
            pRootOfTree.right, right.left = right, pRootOfTree

        while (pRootOfTree.left):
            pRootOfTree = pRootOfTree.left
        return pRootOfTree

# No.37 序列化二叉树
class Solution:
    def __init__(self):
        self.s = ''

    def Serialize(self, root):
        if not root:
            return '#'
        return str(root.val) + ',' + self.Serialize(root.left) + ',' + self.Serialize(root.right)

    def Deserialize(self, s):
        try:
            s = s.split(',')
        except:
            pass
        if len(s) == 0:
            return None
        if len(s) == 1 and s[0] != '#':
            return TreeNode(int(s[0]))
        elif len(s) == 1 and s == '#':
            return None
        else:
            self.s = s[1:]
            tree = None
            if s[0] != '#':
                tree = TreeNode(int(s[0]))
                tree.left = self.Deserialize(self.s)
                tree.right = self.Deserialize(self.s)
            return tree

# No.38 字符串的排列
class Solution:
    def Permutation(self, ss):
        if ss == '':
            return ss
        L = []
        self.J = []
        ss = list(ss)
        self.permutation_cursive(ss, L)
        return sorted(list(set(self.J)))

    def permutation_cursive(self, ssl, L):
        if len(ssl) == 0:
            self.J.append(''.join(L))
        for i in range(len(ssl)):
            L1 = L[:]
            L1.extend(ssl[i])
            ssll = ssl[:]
            _ = ssll.pop(i)
            self.permutation_cursive(ssll, L1)

# No.39 数组中出现次数超过一半的数字
class Solution:
    def MoreThanHalfNum_Solution1(self, numbers):
        n = 1
        if not numbers:
            return None
        before = numbers[0]
        if len(numbers) == 1:
            return before
        for i in numbers:
            if n == 0:
                before = i
                n = 1
            if i == before:
                n += 1
            else:
                n -= 1
        if self.checkmoretime(numbers, before):
            return before
        else:
            return 0

    def partition(self, lyst, left, right):
        # 定基准
        middle = (left + right) // 2  # 随机找了一个点，为了防止最坏情况：即顺序情况下每次都取第一个，我们每次选中点
        lyst[middle], lyst[right] = lyst[right], lyst[middle]
        pivot = lyst[right]  # 注意：这里的pivot实际上就是我们选的中点，最后我们要返回的实际上是这个中点在排好序的位子
        # 定边界
        boundary = left
        # 异位
        for index in range(left, right):
            if lyst[index] < pivot:
                lyst[index], lyst[boundary] = lyst[boundary], lyst[index]
                boundary += 1
        # 归位
        lyst[right], lyst[boundary] = lyst[boundary], lyst[right]
        return boundary

    def MoreThanHalfNum_Solution2(self, numbers):
        start = 0
        end = len(numbers) - 1
        mid = len(numbers) >> 1
        ind = self.partition(numbers, start, end)
        while ind != mid:
            if ind > mid:
                end = ind - 1
                ind = self.partition(numbers, start, end)
            else:
                start = ind + 1
                ind = self.partition(numbers, start, end)
        result = numbers[mid]
        if self.checkmoretime(numbers, result):
            return(result)
        else:
            return(0)


    def checkmoretime(self, lyst, val):
        n = 0
        for i in lyst:
            if i == val:
                n += 1
        if n > (len(lyst) >> 1):
            return True
        else:
            return False

# No.40 最小的K个数
# 用快排解决
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # 实际上就是先用快排，找到第K个大的位子，然后一一比较
        if k > len(tinput):
            return None
        start = 0
        end = len(tinput) - 1
        ind = self.partition(tinput, start, end)
        while ind != k - 1:
            if ind < k - 1:
                start = ind + 1
            else:
                end = ind - 1
            ind = self.partition(tinput, start, end)
        return sorted(tinput[:k])



    def partition(self, lyst, left, right):
        mid = (left + right) // 2
        lyst[mid], lyst[right] =  lyst[right], lyst[mid]
        pivot = lyst[right]

        boundary = left
        for i in range(left, right):
            if lyst[i] < pivot:
                lyst[i], lyst[boundary] = lyst[boundary], lyst[i]
                boundary += 1
        lyst[boundary], lyst[right] = lyst[right], lyst[boundary]
        return boundary

# 用堆排序解决
class Solution:
    def HeapSort(self, lyst):
        # 首先，我们做个大顶堆
        for i in range(len(lyst) // 2 - 1, -1, -1):
            self.HeapAdjust(lyst, i, len(lyst) - 1)
        # 每次把根和最后一个元素交换位置，去掉最后一个元素，剩下元素继续编程大顶堆，最后得出一个升序序列
        for i in range(len(lyst) - 1, 0, -1):
            lyst[0], lyst[i] = lyst[i], lyst[0]
            self.HeapAdjust(lyst, 0, i - 1)

    def HeapAdjust(self, lyst, s, m):
        # 堆调整，就是将顺序逐层排列的第s个节点变成以那个节点为根节点的子数的最大值
        # m是这个序列最后一项
        temp = lyst[s]
        for j in range(2 * s, m, 2):
            if lyst[j] < lyst[j + 1]:
                j += 1
            if temp < lyst[j]:
                lyst[s] = lyst[j]
                s = j
        lyst[s] = temp
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or k > len(tinput):
            return []
        self.HeapSort(tinput)
        return tinput[:k]

# 用改进后的堆解决：O(nlogk)
class Solution:
    def GetLeastNumbers_Solution2(self, tinput, k):
        stl = tinput[:k]
        for i in range(k,len(tinput)):
            if tinput[i] < self.HeapAdjust(stl):
                _ = stl.pop(0)
                stl.append(tinput[i])
        return stl

    def HeapAdjust(self, lyst):
        # 堆调整，返回堆里最大元素
        temp = lyst[0]
        s = 0
        m = len(lyst) - 1
        for j in range(2 * s, m, 2):
            if lyst[j] < lyst[j + 1]:
                j += 1
            if temp < lyst[j]:
                lyst[s] = lyst[j]
                s = j
        lyst[s] = temp
        return lyst[0]

# 调用现有的库
# https://github.com/qiwsir/algorithm/blob/master/heapq.md
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or k > len(tinput):
            return []
        heapq.heapify(tinput)  # 将list转为堆
        return [heapq.heappop(tinput) for _ in range(k)]  # heappop是堆中取出最小的数

# No.42 连续子数组的最大和
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if not array:
            return None
        cusumL = []
        cusum = 0
        for i in array:
            cusumL.append(i)
            if sum(cusumL) < 0:
                cusumL = []
            else:
                cusum = max(cusum, sum(cusumL))
        if len(cusumL) == 0:
            return max(array)
        return cusum

# 1 应用动态规划解题
class Solution:
    def FindGreatestSumOfSubArray(self, array):
            # DP: 状态：f[i]是以第i个数字结尾的子数组的最大和
            #     状态转移公式： f[i] = max(array[i],f[i-1]+array[i])
            #                        = array[i] if f[i-1] <= 0 else f[i-1] + array[i]
        if not array:
            return None
        temp = 0
        res = max(array)
        for i in array:
            temp = max(i, temp + i)
            res = max(res, temp)
        return res

# 2 应用动态规划解题
class Solution:
    def FindGreatestSumOfSubArray(self, array):
            # DP: 状态：f[i]是以第i个数字结尾的子数组的最大和
            #     状态转移公式： f[i] = max(array[i],f[i-1]+array[i])
            #                        = array[i] if f[i-1] <= 0 else f[i-1] + array[i]
            # 最后选择所有状态里最大的一个
        if not array:
            return None
        temp = 0
        L = []
        for i in array:
            temp = max(i, temp + i)
            L.append(temp)
        return max(L)

# No.44 数字序列中某一位的数字
class Solution:
    def digitAtIndex(self, n):
        if n < 10:
            return n + 1
        count = 10
        i = 1
        while n > count:
            n = n - count
            count = 9 * 10 ** i * (i + 1)
            i += 1
        strn = n // i
        dot = n % i
        return str(strn)[dot]

# 把数组排成最小的数
class Solution:
    def PrintMinNumber(self, numbers):
        if not numbers:
            return ""
        numl = list(map(str, numbers))
        numL = self.compare(numl)
        return int(''.join(numL))
    def compare(self, strL):
        if len(strL) < 2:
            return strL
        temp = strL[0]
        aL = []
        bL = []
        for i in strL[1:]:
            if int(temp + i) > int(i + temp):
                aL.append(i)
            else:
                bL.append(i)
        return self.compare(aL) + [temp] + self.compare(bL)

# 把数字翻译成字符串
class Solution:
    def GetTranslation(self, num):
        if not num:
            return 0
        string = str(num)
        return self.translate_cursive(string)


    def translate_cursive(self, string):
        if len(string) < 2:
            return 1
        res = self.translate_cursive(string[1:])
        if int(string[:2]) < 26:
            res += self.translate_cursive(string[2:])
        return res
    def GetTranslation_dp(self, numbers):
        str1 = str(numbers)
        if str1 == None or str1 == "":  # 第一个惯例异常处理
            return 0
        cur = 1 if str1[-1] != '0' else 0  # 第二个异常之处
        nex = 1
        for i in range(len(str1) - 2, -1, -1):
            tmp = cur
            if int(str1[i:i + 2]) < 26:
                cur += nex  # nex是前者，cur是现在，一旦发生可以两种翻译，即前面所有遍历的字母都可以用在两种不同字串中
            nex = tmp
        return cur

# No.47 礼物的最大价值
class Solution:
    def getMaxValue_solution1(self, values, row, col):
        # 二维数组存储
        giftMatrix = [[0]*col] * row
        for i in range(col):
            for j in range(row):
                if i == 0 and j == 0:
                    giftMatrix[i][j] = values[0]
                elif i == 0:
                    giftMatrix[i][j] = giftMatrix[i][j-1] + values[i*col + j]
                elif j == 0:
                    giftMatrix[i][j] = giftMatrix[i-1][j] + values[i*col + j]
                else:
                    giftMatrix[i][j] = max(giftMatrix[i-1][j], giftMatrix[i][j-1]) + values[i*col + j]
        return giftMatrix[-1][-1]

    def getMaxValue_solution2(self, values, row, col):
        # 一维数组存储
        if not values or row <= 0 or col <= 0:
            return 0
        giftArray = [0] * col
        for i in range(col):
            for j in range(row):
                left = 0
                up = 0
                if i > 0:
                    up = giftArray[j]
                if j > 0:
                    left = giftArray[j-1]
                giftArray[j] = max(up, left) + values[i * col + j]
        return giftArray[-1]

# No.48 最长不含重复字符的子字符串
class Solution:
    def longestSubstringWithoutDuplication(self, string):
        if not string:
            return None
        tempL = [-1]*len(string)
        tempL[0] = 1
        temps = string[0]
        for i in range(1, len(string)):
            if string[i] not in temps:
                # 如果第i个字符之前没有出现过
                tempL[i] = tempL[i - 1] + 1
                temps = temps + string[i]
            else:
                # 如果第i个字符之前出现过
                ind = string[:i].rfind(string[i])
                temps = temps + string[i]
                temps = temps[(ind + 1): (i + 1)]
                d = i - ind
                if tempL[i-1] >= d:
                    tempL[i] = d
                else:
                    tempL[i] = tempL[i-1] + 1
        return max(tempL)


# No.49 丑数
class Solution:
    def GetUglyNumber_Solution(self, index):
        if not index:
            return 0
        if index < 0:
            return None
        uglyL = [1]
        T2 = 0
        T3 = 0
        T5 = 0
        for i in range(1, index):
            ugly = min(uglyL[T2] * 2, uglyL[T3] * 3, uglyL[T5] * 5)
            uglyL.append(ugly)
            if ugly % 2 == 0:
                T2 += 1
            if ugly % 3 == 0:
                T3 += 1
            if ugly % 5 == 0:
                T5 += 1
        return uglyL[-1]


# No.50 第一个只出现一次的字符
class Solution:
    def FirstNotRepeatingChar(self, s):
        if not s:
            return None
        sdict = dict()
        for i in s:
            if i in sdict.keys():
                sdict[i] += 1
            else:
                sdict[i] = 1
        for key, value in sdict.items():
            if value == 1:
                return s.find(key)