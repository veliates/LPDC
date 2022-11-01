# Decoder for a Binary Symmetric Channel

import numpy as np
import matplotlib.pyplot as plt
import math
import commpy

# Getting Parameters from the Encoding
img = np.load('image.npy')
lx, ly = len(img), len(img[0])
code = np.loadtxt("encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("mat_1.dat"), np.loadtxt("mat_2.dat")
n, k = G.shape[1], G.shape[0]
img = np.array(img).flatten()  # vectorising the matrix
size = len(img)

row = [[] for _ in range(len(H))]
col = [[] for _ in range(n)]

for i in range(len(H)):
    for j in range(n):
        if(H[i][j] == 1):
            row[i].append(j)
            col[j].append(i)


def BSC(data, p):
    b = np.array(np.random.rand(len(data)) < p, dtype=np.int)
    b ^= data.astype(int)
    return b.astype(float)

# utility functions


def xor(arr, n):
    res = 0
    for i in range(len(arr)):
        if i != n:
            res += arr[i]
    return res % 2


def findMax(arr, index, n):
    res = -1
    for i in range(len(col[index])):
        if(i != n):
            res = max(res, arr[col[index][i]])

    return res


def findMin(arr, index, n):
    res = 2
    for i in range(len(col[index])):
        if(i != n):
            res = min(res, arr[col[index][i]])

    return res


def computeTanh(arr, rowIndex):
    res = 1
    for i in row[rowIndex]:
        res *= np.tanh(arr[i]/2)

    return res


def intrinsic(y, p):
    if y:
        return np.log(p/(1 - p))
    return -np.log(p/(1 - p))

# decoders


def belief_prop(bits, p):  # decoding
    L = H.copy()
    decode = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):
        r = bits[n*i: n*(i+1)]
        for j in range(len(L)):
            # Analogous to adjacency matrix representation of the Tanner graph
            L[j] = H[j] * r

        decision = np.zeros(n)  # to store the current decision

        for _ in range(10):
            # row operations
            for j in range(len(L)):
                x = L[j]
                l_j = computeTanh(L[j], j)

                for l in row[j]:
                    temp = 0
                    if l_j != 0:
                        temp = l_j/np.tanh(x[l]/2)
                        if temp == 1: # to handle the "division by zero" error
                            temp = 0.999

                        if temp == -1:
                            temp = -0.999
                    
                    x[l] = np.log((1 + temp)/(1 - temp))

                L[j] = x

            # column operations
            for j in range(n):
                sum_j = intrinsic(r[j], p) + np.sum(L[:, j])
                for l in col[j]:
                    L[l][j] = sum_j - L[l][j]

                if sum_j < 0:
                    decision[j] = 1
                else:
                    decision[j] = 0

            decode[k*i: k*(i+1)] = decision[0:k]

    return decode


def gallagher(bits):
    L = H.copy()
    res = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):
        r = bits[n*i: n*(i+1)]
        c = r

        for j in range(len(L)):
            L[j] = H[j]*r

        for _ in range(10):  # no. of iterations
            for j in range(len(H)):
                for l in range(n):
                    # column operations
                    temp = findMax(L[:, l], l, j)
                    # if all entries of the column are equal.
                    if(temp == findMin(L[:, l], l, j)):
                        L[j][l] = temp
                        c[l] = temp
                    else:
                        L[j][l] = r[l]

                    # row operations
                    L[j][l] = xor(L[j], l)

            res[k*i: k*(i+1)] = c[0:k]

    return res


p = [0.1, 0.3, 0.5, 0.7] #probability of erasures
BER_g = []
BER_b = []

print("Belief Propagation: \n")
for i in p:
    print("For p = ", i)
    r = BSC(code.astype(int), i)  # Received Bits
    decode = belief_prop(r, i)
    incorrect_bit = (img != decode).sum()
    print("Bits decoded incorrectly:", incorrect_bit)
    BER_b.append(incorrect_bit/size)
    print("Bit Error rate:", incorrect_bit/size, "\n")

print("Gallagher-A: \n")
for i in p:
    print("For p = ", i)
    r = BSC(code.astype(int), i)  # Received Bits
    decode_ = gallagher(r)
    incorrect_bit = (img != decode_).sum()
    print("Bits decoded incorrectly:", incorrect_bit)
    BER_g.append(incorrect_bit/size)
    print("Bit Error rate:", incorrect_bit/size, "\n")

plt.plot(p, BER_b, marker='*', label="Belief Propagation")
plt.plot(p, BER_g, marker='*', label="Gallagher A")
plt.xlabel("p (erasure probability)")
plt.ylabel("BER(Bit Error Rate)")
plt.legend()
plt.savefig("BSC.png")
plt.show()

#veli#
