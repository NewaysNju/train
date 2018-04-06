# ！/user/bin/env python3
# -*- coding:utf-8 -*-
# @Author   :Neways
# @Time     :2018/4/7 0:45
# @Contact  :NewaysNju@gmail.com

# 387. First Unique Character in a String
class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = dict()
        L = []
        for i in range(len(s)):
            if s[i] not in L:
                d[str(s[i])] = i
                L.append(s[i])
            else:
                try:
                    del d[str(s[i])]
                except:
                    pass
        return sorted(d.items(), key=lambda v: v[1])


# 771. Jewels and Stones
class Solution:
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        L = list(J)
        num = 0
        for i in S:
            if i in L:
                num += 1
        return num


# 804. Unique Morse Code Words
class Solution:
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        table = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
                 "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        al = []
        for word in words:
            L = []
            for j in word:
                L.append(str(table[ord(j) - 97]))
            al.append(''.join(L))
        uniqueal = set(al)
        return len(uniqueal)


# 661. Image Smoother
import copy


class Solution:
    def imageSmoother(self, M):
        """
        :type M: List[List[int]]
        :rtype: List[List[int]]
        """
        N = copy.deepcopy(M)
        indmax = len(M)
        colmax = len(M[0])
        for i in range(indmax):
            for j in range(colmax):
                L = []
                location = {(max(i - 1, 0), max(j - 1, 0)), (max(i - 1, 0), j), (max(i - 1, 0), min(j + 1, colmax - 1)),
                            (i, max(j - 1, 0)), (i, j), (i, min(j + 1, colmax - 1)),
                            (min(i + 1, indmax - 1), max(j - 1, 0)), (min(i + 1, indmax - 1), j),
                            (min(i + 1, indmax - 1), min(j + 1, colmax - 1))}
                for n in location:
                    L.append(N[n[0]][n[1]])
                M[i][j] = int(sum(L) / len(L))

        return M


if __name__ == '__main__':
    pass