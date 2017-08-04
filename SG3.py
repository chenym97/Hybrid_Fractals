import numpy as np
from numpy import linalg as LA
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import collections

level0Code = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53, 61, 62, 63]

level0CodeMapWithoutBoundary = {12: 0, 13: 1, 21: 2, 23: 3, 31: 4, 32: 5, 41: 6, 42: 7, 43: 8,
                                51: 9, 52: 10, 53: 11, 61: 12, 62: 13, 63: 14}

level0nbCodePartial = {12: 41, 13: 51, 21: 42, 23: 62, 31: 53, 32: 63, 41: 12, 42: 21, 51: 13, 53: 31, 62: 23, 63: 32,
                       43: [52, 61], 52: [43, 61], 61: [43, 52]}

level1CodeMapWithoutBoundaryPartial = {211: 17, 212: 18, 213: 19, 221: 20, 223: 21, 231: 22, 232: 23, 233: 24, 241: 25,
                                       242: 26, 243: 27, 251: 28, 252: 29, 253: 30, 261: 31, 262: 32, 263: 33, 311: 34,
                                       312: 35, 313: 36, 321: 37, 322: 38, 323: 39, 331: 40, 332: 41, 341: 42, 342: 43,
                                       343: 44, 351: 45, 352: 46, 353: 47, 361: 48, 362: 49, 363: 50}


codeInSG = [[], [712, 713, 723], [7112, 7113, 7122, 7123, 7133, 7212, 7213, 7223, 7233, 7312, 7313, 7323],
            [71112, 71113, 71122, 71123, 71133, 71212, 71213, 71222, 71223, 71233, 71312, 71313, 71323, 71333,
             72112, 72113, 72122, 72123, 72133, 72212, 72213, 72223, 72233, 72312, 72313, 72323, 72333, 73112,
             73113, 73122, 73123, 73133, 73212, 73213, 73223, 73233, 73312, 73313, 73323]]

codeMapInSG = [{}, {712: 0, 713: 1, 723: 2}, {7112: 0, 7113: 1, 7122: 2, 7123: 3, 7133: 4, 7212: 5,
                                              7213: 6, 7223: 7, 7233: 8, 7312: 9, 7313: 10, 7323: 11},
               {71112: 0, 71113: 1, 71122: 2, 71123: 3, 71133: 4, 71212: 5, 71213: 6, 71222: 7, 71223: 8, 71233: 9,
                71312: 10, 71313: 11, 71323: 12, 71333: 13, 72112: 14, 72113: 15, 72122: 16, 72123: 17, 72133: 18,
                72212: 19, 72213: 20, 72223: 21, 72233: 22, 72312: 23, 72313: 24, 72323: 25, 72333: 26, 73112: 27,
                73113: 28, 73122: 29, 73123: 30, 73133: 31, 73212: 32, 73213: 33, 73223: 34, 73233: 35, 73312: 36,
                73313: 37, 73323: 38}]

'''
a = 1.0
b = 1.0
c = 1.0
r = 1.0
rou = 1.0
'''

# '''
a = 1.0 / 12
b = 1.0 / 13
c = 1 - 6 * a - 6 * b
r = a
rou = (np.sqrt(61 * r * r - 138 * r + 81) - 31 * r + 9) / 30
# '''

# number of points in SG (WITHOUT 3 boundary points)
def nOfSG(level):
    return (pow(3, level + 1) - 3) / 2


def n(level):
    if level == 0:
        return 18
    return 6 * n(level - 1) + nOfSG(level) + 6 * (pow(2, level) - 1)


def intToCode(num):
    return [int(i) for i in str(num)]


def codeToInt(code):
    return int(''.join(str(e) for e in code))


def numToEightNine(num, length):
    if length == 0:
        return []
    return numToEightNine(num / 2, length - 1) + [num % 2 + 8]


def eightNineToNum(eightNines):
    length = len(eightNines)
    if length == 0:
        return 0
    return (eightNines[length - 1] - 8) + 2 * eightNineToNum(eightNines[0: length - 1])


def nToC(index, level):
    if level == 0:
        return intToCode(level0Code[index])
    if index >= 6 * n(level - 1) + nOfSG(level) + 5 * (pow(2, level) - 1):
        return [3, 2] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level) + 5 * (pow(2, level) - 1)), level)
    if index >= 6 * n(level - 1) + nOfSG(level) + 4 * (pow(2, level) - 1):
        return [3, 1] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level) + 4 * (pow(2, level) - 1)), level)
    if index >= 6 * n(level - 1) + nOfSG(level) + 3 * (pow(2, level) - 1):
        return [2, 3] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level) + 3 * (pow(2, level) - 1)), level)
    if index >= 6 * n(level - 1) + nOfSG(level) + 2 * (pow(2, level) - 1):
        return [2, 1] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level) + 2 * (pow(2, level) - 1)), level)
    if index >= 6 * n(level - 1) + nOfSG(level) + pow(2, level) - 1:
        return [1, 3] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level) + pow(2, level) - 1), level)
    if index >= 6 * n(level - 1) + nOfSG(level):
        return [1, 2] + numToEightNine(index - (6 * n(level - 1) + nOfSG(level)), level)
    if index >= 6 * n(level - 1):
        return intToCode(codeInSG[level][index - 6 * n(level - 1)])
    return [index / n(level - 1) + 1] + nToC(index % n(level - 1), level - 1)


def isBoundary(code):
    if len(code) == 1:
        return True
    if code[-1] != code[-2]:
        return False
    else:
        return isBoundary(code[:-1])


def generator(level):
    temp = []
    for i in range(0, n(level)):
        if not isBoundary(nToC(i, level)):
            temp += [nToC(i, level)]
    return temp


def cToN(code):
    length = len(code)
    level = length - 2
    if length == 2:
        return 3 * code[0] + code[1] - 4
    if code[0] == 1 and code[1] == 2 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + eightNineToNum(code[2:])
    if code[0] == 1 and code[1] == 3 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + (pow(2, level) - 1) + eightNineToNum(code[2:])
    if code[0] == 2 and code[1] == 1 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + 2 * (pow(2, level) - 1) + eightNineToNum(code[2:])
    if code[0] == 2 and code[1] == 3 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + 3 * (pow(2, level) - 1) + eightNineToNum(code[2:])
    if code[0] == 3 and code[1] == 1 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + 4 * (pow(2, level) - 1) + eightNineToNum(code[2:])
    if code[0] == 3 and code[1] == 2 and (code[2] == 8 or code[2] == 9):
        return 6 * n(level - 1) + nOfSG(level) + 5 * (pow(2, level) - 1) + eightNineToNum(code[2:])
    if code[0] == 7:
        return 6 * n(level - 1) + codeMapInSG[level][codeToInt(code)]
    return (code[0] - 1) * n(level - 1) + cToN(code[1:])


def cToNWithoutBoundary(code):
    length = len(code)
    if length == 2:
        return level0CodeMapWithoutBoundary[codeToInt(code)]
    if code[0] in [4, 5, 6, 7] or code[2] in [8, 9]:
        return cToN(code) - 3
    if code[0] == 1:
        return cToN(code) - 1
    if length == 3:
        return level1CodeMapWithoutBoundaryPartial[codeToInt(code)]
    if code[0] == 2:
        if code[1] in [3, 4, 5, 6, 7] or code[3] in [8, 9]:
            return cToN(code) - 2
        if code[1] == 1:
            return cToN(code) - 1
        return n(length - 3) + cToNWithoutBoundary(code[1:])
    if code[0] == 3:
        if code[1] in [4, 5, 6, 7] or code[3] in [8, 9]:
            return cToN(code) - 3
        if code[1] == 1 or code[1] == 2:
            return cToN(code) - 2
        return 2 * n(length - 3) + cToNWithoutBoundary(code[1:])


def exceptFor(i):
    if i == 1:
        return [2, 3]
    if i == 2:
        return [1, 3]
    if i == 3:
        return [1, 2]


def findAnotherAddressInSG(code):
    if code[-2] != code[-1]:
        return code[:-2] + [code[-1], code[-2]]
    tempCode = findAnotherAddressInSG(code[:-1])
    return tempCode + [tempCode[-1]]


def isInvalidCode(code):
    if isBoundary(code[1:]):
        return False
    if code[-1] < code[-2]:
        return True
    if code[-1] > code[-2]:
        return False
    return isInvalidCode(code[:-1])


def nbCodeOnBoundaryPtsOfTriangle(code):
    length = len(code)
    nbCode = [code[:-1] + [i] for i in exceptFor(code[-1])]
    if length == 2:
        if code[0] + code[1] == 7:
            return nbCode + [intToCode(someIntCode) for someIntCode in level0nbCodePartial[codeToInt(code)]]
        return nbCode + [intToCode(level0nbCodePartial[codeToInt(code)])]
    if code[0] in [1, 2, 3]:
        return nbCode + [code[:2] + (length - 2) * [8]]
    if code[0] + code[1] != 7:
        return nbCode + [intToCode(level0nbCodePartial[codeToInt(code[:2])]) + (length - 3) * [9] + [8]]
    tempList = [[7] + (length - 2) * [code[1]] + [i] for i in exceptFor(code[1])]
    for someCode in tempList:
        if isInvalidCode(someCode):
            nbCode += [findAnotherAddressInSG(someCode)]
        else:
            nbCode += [someCode]
    return nbCode


def nbCodeOnInterval(code):
    length = len(code)
    num = eightNineToNum(code[2:])
    if length == 3:
        tempCode = intToCode(level0nbCodePartial[codeToInt(code[:2])])
        return [code[:2] + [code[1]], tempCode + [tempCode[1]]]
    if num == 0:
        return [code[:2] + (length - 2) * [code[1]], code[:2] + numToEightNine(num + 1, length - 2)]
    if num == pow(2, length - 2) - 2:
        tempCode = intToCode(level0nbCodePartial[codeToInt(code[:2])])
        return [tempCode + (length - 2) * [tempCode[1]], code[:2] + numToEightNine(num - 1, length - 2)]
    return [code[:2] + numToEightNine(i, length - 2) for i in [num - 1, num + 1]]



def nbCodeInSGIncomplete(code):
    nbCode = []
    tempNbCode = [code[:-1] + [i] for i in exceptFor(code[-1])]
    for someCode in tempNbCode:
        if isInvalidCode(someCode):
            nbCode += [findAnotherAddressInSG(someCode)]
        else:
            nbCode += [someCode]
    if isBoundary(code[1:]):
        return nbCode
    anotherAddress = [7] + findAnotherAddressInSG(code[1:])
    tempList = [anotherAddress[:-1] + [i] for i in exceptFor(anotherAddress[-1])]
    for someCode in tempList:
        if isInvalidCode(someCode):
            nbCode += [[7] + findAnotherAddressInSG(someCode[1:])]
        else:
            nbCode += [someCode]
    return nbCode


def nbCodeInSG(code):
    nbCode = []
    for someCode in nbCodeInSGIncomplete(code):
        if isBoundary(someCode[1:]):
            nbCode += [[7 - someCode[1]] + someCode[1:]]
        else:
            nbCode += [someCode]
    return nbCode


def nbCode(code):
    if isBoundary(code[1:]) and code[1] in [1, 2, 3]:
        return nbCodeOnBoundaryPtsOfTriangle(code)
    if code[2] in [8, 9]:
        return nbCodeOnInterval(code)
    if code[0] == 7:
        return nbCodeInSG(code)
    return [[code[0]] + someCode for someCode in nbCode(code[1:])]


def integral(code):
    length = len(code)
    level = length - 2
    if isBoundary(code[1:]) and code[1] in [1, 2, 3]:
        if code[:2] in [[4, 3], [5, 2], [6, 1]]:
            return 1 / 3.0 * pow(a, level + 1) + c / pow(3, level + 1)
        return 1 / 3.0 * pow(a, level + 1) + b / pow(2, level + 1)
    if code[2] in [8, 9]:
        return b / pow(2, level)
    if code[0] == 7:
        return 2 * c / pow(3, level + 1)
    return a * integral(code[1:])


def matrixBuilder(code):
    length = len(code)
    level = length - 2
    if isBoundary(code[1:]) and code[1] in [1, 2, 3]:
        nbCodeResistPairs = []
        if code[:2] in [[4, 3], [5, 2], [6, 1]]:
            for someNbCode in nbCode(code):
                if someNbCode[0] != 7:
                    nbCodeResistPairs += [(someNbCode, pow(r, level + 1))]
                else:
                    nbCodeResistPairs += [(someNbCode, rou * pow(0.6, level))]
        else:
            for someNbCode in nbCode(code):
                if length == 2 and someNbCode[0] != code[0]:
                    nbCodeResistPairs += [(someNbCode, rou / pow(2, level))]
                elif length > 2 and someNbCode[2] in [8, 9]:
                    nbCodeResistPairs += [(someNbCode, rou / pow(2, level))]
                else:
                    nbCodeResistPairs += [(someNbCode, pow(r, level + 1))]
        return nbCodeResistPairs
    if code[2] in [8, 9]:
        return [(someNbCode, rou / pow(2, level)) for someNbCode in nbCode(code)]
    if code[0] == 7:
        return [(someNbCode, rou * pow(0.6, level)) for someNbCode in nbCode(code)]
    return [([code[0]] + pair[0], r * pair[1]) for pair in matrixBuilder(code[1:])]


def matrix(level):
    size = n(level) - 3
    codes = generator(level)
    laplacian = [[0 for i in range(size)] for j in range(size)]
    for code in codes:
        coeffOfX = 0
        indexOfX = cToNWithoutBoundary(code)
        integralAtX = integral(code)
        for pair in matrixBuilder(code):
            if not isBoundary(pair[0]):
                indexOfY = cToNWithoutBoundary(pair[0])
                laplacian[indexOfX][indexOfY] = - 1 / (pair[1] * integralAtX)
            coeffOfX += 1 / (pair[1] * integralAtX)
        laplacian[indexOfX][indexOfX] = coeffOfX
    return laplacian


def linearComb(coord1, coord2, weight):
    return [(1 - weight) * coord1[0] + weight * coord2[0], (1 - weight) * coord1[1] + weight * coord2[1]]


def cToCoord(code):
    length = len(code)
    level = length - 2
    boundaryPts = ['todayisagoodday', (0.5, 0.5 * np.sqrt(3)), (0.0, 0.0), (1.0, 0.0), (0.25, 0.25 * np.sqrt(3)),
                   (0.75, 0.25 * np.sqrt(3)), (0.5, 0)]
    if isBoundary(code[1:]) and code[1] in [1, 2, 3]:
        if code[0] in [1, 2, 3]:
            return linearComb(boundaryPts[code[0]], boundaryPts[code[1]], 0.2)
        if code[0] + code[1] != 7:
            return linearComb(boundaryPts[code[1]], boundaryPts[code[0] - code[1] - 1], 0.4)
        tempPoint = linearComb(boundaryPts[exceptFor(code[1])[0]], boundaryPts[exceptFor(code[1])[1]], 0.5)
        return linearComb(tempPoint, boundaryPts[code[1]], 0.2)
    if code[2] in [8, 9]:
        tempCode = intToCode(level0nbCodePartial[codeToInt(code[:2])])
        endpoints = [cToCoord(code[:2] + (length - 2) * [code[1]]), cToCoord(tempCode[:2] + (length - 2) * [tempCode[1]])]
        index = eightNineToNum(code[2:])
        return linearComb(endpoints[0], endpoints[1], (index + 1.0) / pow(2, level))
    if code[0] == 7:
        boundaryPtsOfSG = ['istomorrowagoodday?'] + [cToCoord([7 - i] + (length - 1) * [i]) for i in [1, 2, 3]]
        currentPoint = boundaryPtsOfSG[code[-1]]
        for i in range(length - 2, 0, -1):
            currentPoint = linearComb(currentPoint, boundaryPtsOfSG[code[i]], 0.5)
        return currentPoint
    return linearComb(boundaryPts[code[0]], cToCoord(code[1:]), 0.2)


def getOrderedEigPairs(level):
    laplace = matrix(level)
    w, v = LA.eig(laplace)
    w = w.real
    v = v.real
    v = np.transpose(v)
    eigPairs = []
    for i in range(n(level) - 3):
        if v[i][0] < 0:
            eigPairs += [(w[i], (-1) * v[i])]
        else:
            eigPairs += [(w[i], v[i])]
    sortedEigPairs = sorted(eigPairs, key=lambda pair: pair[0])
    return sortedEigPairs


def getSortedEigenvalue(level, someList):
    sortedEigPairs = getOrderedEigPairs(level)
    return [sortedEigPairs[index][0] for index in someList]


def getSortedEigenfunction(level, someList):
    sortedEigPairs = getOrderedEigPairs(level)
    return [sortedEigPairs[index][1].tolist() for index in someList]


def plotSortedEigenfunction(level, someList):
    laplace = matrix(level)
    pts = generator(level)
    coords = []
    for code in pts:
        coords.append(cToCoord(code))
    sortedEigPairs = getOrderedEigPairs(level)
    plot = np.transpose(coords)
    for index in someList:
        fig = plt.figure()
        fig.suptitle('a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
        ax = fig.add_subplot(111, projection='3d')
        zs = np.array(sortedEigPairs[index][1].tolist())
        ax.set_title('eigenvalue[' + str(index) + '] = ' + str(sortedEigPairs[index][0]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(plot[0], plot[1], zs * 1000, s=0.5)
        l = np.matrix(laplace)
        print(zs)
        for i in range(len(plot[0])):
            for j in range(i):
                if l.item((i, j)) < 0 - 0.01:
                    ax.plot([plot[0][i], plot[0][j]], [plot[1][i], plot[1][j]],
                            zs=[zs.item(i) * 1000, zs.item(j) * 1000])
        # fig.savefig('SG3Images/a=' + str(round(a, 2)) + ' b=' + str(round(r, 2)) + ' r=' + str(round(r, 2)) + ' level=' + str(level) + ' index=' + str(index) + '.png')  # need to create a folder 'SG3Images' first


def cToCoordRestrictedToSG(code):
    length = len(code)
    boundaryPts = ['tomorrowisdefinitelyagoodday!', (0.5, 0.5 * np.sqrt(3)), (0.0, 0.0), (1.0, 0.0)]
    currentPoint = boundaryPts[code[-1]]
    for i in range(length - 2, 0, -1):
        currentPoint = linearComb(currentPoint, boundaryPts[code[i]], 0.5)
    return currentPoint


def plotSortedEigenfunctioninSG(level, someList):
    laplace = matrix(level)
    pts = generator(level)
    sortedEigPairs = getOrderedEigPairs(level)
    coords = []
    indexes = []
    boundaryIndexes = []
    for code in pts:
        if code[:2] in [[4, 3], [5, 2], [6, 1]] and isBoundary(code[1:]):
            coords.append(cToCoordRestrictedToSG(code))
            boundaryIndexes += [len(indexes)]
            indexes += [cToNWithoutBoundary(code)]
        if code[0] == 7:
            coords.append(cToCoordRestrictedToSG(code))
            indexes += [cToNWithoutBoundary(code)]
    plot = np.transpose(coords)
    for index in someList:
        eigenfunction = sortedEigPairs[index][1].tolist()
        zs = []
        for code in pts:
            if (code[:2] in [[4, 3], [5, 2], [6, 1]] and isBoundary(code[1:])) or code[0] == 7:
                zs += [1000 * eigenfunction[cToNWithoutBoundary(code)]]
        fig = plt.figure()
        fig.suptitle('a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('eigenvalue[' + str(index) + '] = ' + str(sortedEigPairs[index][0]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(plot[0], plot[1], zs, s=0.5)
        l = np.matrix(laplace)
        print(zs)
        for i in range(len(plot[0])):
            for j in range(i):
                if l.item((indexes[i], indexes[j])) < 0 - 0.01:
                    ax.plot([plot[0][i], plot[0][j]], [plot[1][i], plot[1][j]],
                            zs=[zs[i], zs[j]])
        boundaryX = [plot[0][boundaryIndexes[i]] for i in [0, 1, 2]]
        boundaryY = [plot[1][boundaryIndexes[i]] for i in [0, 1, 2]]
        boundaryZ = [zs[boundaryIndexes[i]] for i in [0, 1, 2]]
        for x, y, z in zip(boundaryX, boundaryY, boundaryZ):
            text = '(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')'
            ax.text(x, y, z, text, zdir=(1, 1, 1))
        fig.savefig('SG3Images/a=' + str(round(a, 2)) + ' b=' + str(round(r, 2)) + ' r=' + str(round(r, 2)) + ' level=' + str(level) + ' index=' + str(index) + ' (SG).png')  # need to create a folder 'SG3Images' first


def plotSG3(level):
    laplace = matrix(level)
    pts = generator(level)
    coords = []
    for code in pts:
        coords.append(cToCoord(code))
    plot = np.transpose(coords)
    for index in [0]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        zs = [0 for i in range(n(level) - 3)]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(plot[0], plot[1], zs, s=0.5)
        l = np.matrix(laplace)
        for i in range(len(plot[0])):
            for j in range(i):
                if l.item((i, j)) < 0 - 0.01:
                    ax.plot([plot[0][i], plot[0][j]], [plot[1][i], plot[1][j]],
                            zs=[zs[i] * 1000, zs[j] * 1000])


def multi_eig(listOfEgvl):
    multi_dict = {}
    ##multi_dict = {listOfEgvl[0]:1}
    j = 1
    for i in range(0, n(m) - 3):
        if i != (n(m) - 4):
            if (listOfEgvl[i] - listOfEgvl[i + 1] < 1 * pow(10, -2)) and (listOfEgvl[i] - listOfEgvl[i + 1] > - 1 * pow(10, -2)):
                j += 1
            else:
                multi_dict[listOfEgvl[i]] = j
                j = 1
        else:
            multi_dict[listOfEgvl[i]] = j

    sorted_dict = collections.OrderedDict(sorted(multi_dict.items()))
    for k, v in sorted_dict.items():
        print(k, v)
    return


def contains1457(code):
    if len(code) == 1:
        return code[0] in [1, 4, 5, 7]
    if code[0] in [1, 4, 5, 7]:
        return True
    return contains1457(code[1:])


def plotOnOneSide(level, index):
    sortedEigPairs = getOrderedEigPairs(level)
    eigenfunction = sortedEigPairs[index][1].tolist()
    pts = generator(level)
    fig1 = plt.figure()
    fig1.suptitle('a = ' + str(a) + ', r = ' + str(r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax1 = fig1.add_subplot(111)
    ax1.set_title('eigenvalue[' + str(index) + '] = ' + str(sortedEigPairs[index][0]) + ', 1->2')
    points3 = [(0.0, 0.0), (1.0, 0.0)]
    for code in pts:
        if not contains1457(code):
            indexOfCode = cToNWithoutBoundary(code)
            points3 += [(cToCoord(code)[0], eigenfunction[indexOfCode])]
            print (cToCoord(code)[0], eigenfunction[indexOfCode])
    sortedPoints = sorted(points3, key=lambda pair: pair[0])
    x = [point[0] for point in sortedPoints]
    y = [point[1] for point in sortedPoints]
    ax1.plot(x, y)
    ax1.scatter(x, y, s=10)


def trySpectralDecimation(listOfPairsYouWantToTry):
    numOfPairs = len(listOfPairsYouWantToTry)
    valueAt1bla2 = 0.0
    fig = plt.figure()
    fig.suptitle('a = ' + str(a) + ', r = ' + str(r), fontsize=10, fontweight='bold')
    ax = fig.add_subplot(111)
    legend = []
    for i in range(numOfPairs):
        level = listOfPairsYouWantToTry[i][0]
        length = level + 2
        index = listOfPairsYouWantToTry[i][1]
        pts = generator(level)
        sortedEigPairs = getOrderedEigPairs(level)
        eigenfunction = sortedEigPairs[index][1].tolist()
        eigenvalue = sortedEigPairs[index][0]
        if i == 0:
            valueAt1bla2 += eigenfunction[cToNWithoutBoundary([2] + (length - 1) * [3])]
        points3 = [(0.0, 0.0), (1.0, 0.0)]
        for code in pts:
            if not contains1457(code):
                indexOfCode = cToNWithoutBoundary(code)
                points3 += [(cToCoord(code)[0], eigenfunction[indexOfCode])]
        sortedPoints = sorted(points3, key=lambda pair: pair[0])
        factor = valueAt1bla2 / eigenfunction[cToNWithoutBoundary([2] + (length - 1) * [3])]
        x = [point[0] for point in sortedPoints]
        y = [factor * point[1] for point in sortedPoints]
        ax.plot(x, y)
        ax.scatter(x, y, s=10)
        legend += ["level = " + str(level) + ", egvl[" + str(index) + "] = " + str(eigenvalue)]
    ax.legend(legend, loc='upper left', prop={'size':10})

'''
def linearSpline(listOfX, listOfY):
    def valueAtX(x):
        i = 0
        while (x < listOfX[i] or x > listOfY[i + 1]):
            i += 1
        fit = np.polyfit([listOfX[i], listOfX[i + 1]], [listOfY[i], listOfY[i + 1]])
        fitValue = np.poly1d(fit)
        return fitValue(x)
    return valueAtX
'''


def logLogPlot(level):
    listOfEgvls = getSortedEigenvalue(level, range(n(level) - 3))
    x = [np.log(eigenvalue) for eigenvalue in listOfEgvls]
    integers = np.arange(1., n(level) - 2, 1.0)
    y = [np.log(integer) for integer in integers]
    fit = np.polyfit(x[50:], y[50:], 1)
    fitValue = np.poly1d(fit)
    fig1 = plt.figure()
    fig1.suptitle('a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', rou = ' + str(round(rou, 5)) + ', r = ' + str(
        round(r, 5)) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('log(x)')
    ax1.set_ylabel('log(N(x))')
    ax1.set_title('log(N(x)) = ' + str(fit[0]) + ' * log(x) + ' + str(fit[1]))
    ax1.plot(x, y)
    ax1.plot(x, fitValue(x))
    # fig1.savefig('SG3Images/a=' + str(round(a, 3)) + ' b=' + str(round(b, 3)) + ' r=' + str(round(r, 3)) + ' level=' + str(level) + ' (egvlcounting_loglogplot).png')  # need to create a folder 'SG3Images' first
    fig2 = plt.figure()
    fig2.suptitle(
        'a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(
            r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.plot(listOfEgvls, integers)
    # fig2.savefig('SG3Images/a=' + str(round(a, 2)) + ' b=' + str(round(b, 2)) + ' r=' + str(round(r, 2)) + ' level=' + str(level) + ' (egvlcounting).png')  # need to create a folder 'SG3Images' first
    y1 = [integers[i] / pow(listOfEgvls[i], float(fit[0])) for i in range(n(level) - 3)]
    fig3 = plt.figure()
    fig3.suptitle(
        'a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(
            r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax3 = fig3.add_subplot(111)
    ax3.set_title('y = N(x) / x ^ ' + str(fit[0]) + ', y --- x --- plot')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.plot(listOfEgvls, y1)
    # fig3.savefig('SG3Images/a=' + str(round(a, 3)) + ' b=' + str(round(b, 3)) + ' r=' + str(round(r, 3)) + ' level=' + str(level) + ' (coefficient).png')  # need to create a folder 'SG3Images' first
    y1 = [integers[i] / pow(listOfEgvls[i], float(fit[0])) for i in range(n(level) - 3)]
    fig4 = plt.figure()
    fig4.suptitle(
        'a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(
            r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax4 = fig4.add_subplot(111)
    ax4.set_title('y = N(x) / x ^ ' + str(fit[0]) + ', y --- x --- plot --- zoomed')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    x2 = []
    y2 = []
    for i in range(n(level) - 3):
        if listOfEgvls[i] < int(listOfEgvls[-1]) / (7 * 7):
            x2.append(listOfEgvls[i])
            y2.append(y1[i])
        else:
            break
    ax4.plot(x2, y2)
    # fig4.savefig('SG3Images/a=' + str(round(a, 3)) + ' b=' + str(round(b, 3)) + ' r=' + str(round(r, 3)) + ' level=' + str(level) + ' (coefficient(p2)).png')  # need to create a folder 'SG3Images' first
    fig5 = plt.figure()
    fig5.suptitle(
        'a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(
            r) + ', level = ' + str(level), fontsize=10, fontweight='bold')
    ax5 = fig5.add_subplot(111)
    ax5.set_title('y = N(x) / x ^ ' + str(fit[0]) + ', y --- log(x) --- plot')
    ax5.set_xlabel('log(x)')
    ax5.set_ylabel('y')
    ax5.plot(x, y1)
    fig5.savefig('SG3Images/a=' + str(round(a, 3)) + ' b=' + str(round(b, 3)) + ' r=' + str(round(r, 3)) + ' level=' + str(level) + ' (coefficient-log-log).png')  # need to create a folder 'SG3Images' first


'''
m = 1
x = getSortedEigenvalue(m, range(n(m) - 3))
multi_eig(x)
y = np.arange(1., n(m) - 2, 1.0)
plt.suptitle('a = ' + str(round(a, 5)) + ', b = ' + str(round(b, 5)) + ', c = ' + str(round(c, 5)) + ', r = ' + str(r) + ', level = ' + str(m), fontsize=10, fontweight='bold')
plt.plot(x, y)
plt.show()
plt.savefig('SG3Images/a=' + str(round(a, 2)) + ' b=' + str(round(r, 2)) + ' r=' + str(round(r, 2)) + ' level=' + str(m) + ' (egvlcounting).png')  # need to create a folder 'SG3Images' first
'''

m = 1
multi_eig(getSortedEigenvalue(m, range(n(m) - 3)))

# trySpectralDecimation([(0, 0), (1, 0), (2, 0), (3, 0)])
# plotSortedEigenfunctioninSG(1, range(1))
# logLogPlot(3)
# plt.show()
# plt.show()