# Decoder for BEC (binary erasure channel)

import numpy as np 
import matplotlib.pyplot as plt
import math

img = np.load('image.npy')
lenx, leny = len(img), len(img[0])

code = np.loadtxt("encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("mat_1.dat"), np.loadtxt("mat_2.dat")
n, k = G.shape[1], G.shape[0]

img = np.array(img).flatten() #matrix vectorising
size = len(img)

rows = [[] for _ in range(len(H))]
columns = [[] for _ in range(n)]

for i in range(len(H)):
	for j in range(n):
		if(H[i][j] == 1):
			rows[i].append(j)
			columns[j].append(i)

wc, wr = len(columns[0]), len(rows[0])

# Utility functions
def count_e(arr): # function for counting the no. of erasure symbols in a code block
    return np.count_nonzero(arr == -1)

def xor(arr, n):
	res = 0
	for i in range(len(arr)):
		if i != n:
			res += arr[i]
	return res % 2

def assign(arr, colIndex):
    bit = 0
    for i in columns[colIndex]:
        if arr[i] != -1:
            bit = arr[i]
            break
    
    for i in columns[colIndex]:
        arr[i] = bit

    return arr

def BEC(data, p): #simulation
	_abc_ = np.array(np.random.rand(len(data)) < p, dtype = np.int)
	answer = np.zeros(len(data))
	for i in range(len(data)):
		if(_abc_[i]):
			answer[i] = -1
		else:
			answer[i] = data[i]
	return answer

def belief_prop(bits): # decoding
    copy_H = H.copy()
    res = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):
        Z = bits[n*i : n*(i+1)]

        for j in range(len(copy_H)):
            copy_H[j] = copy_H[j]*Z

        for _ in range(10): # iterations

            # row functions
            for j in range(len(copy_H)):
                index = 0
                cnt = 0
                for l in range(n):
                    if copy_H[j][l] == -1:
                        index = l
                        cnt += 1
                
                if cnt == 1: # in case one variable is erased from a row
                    copy_H[j][index] = xor(copy_H[j], index)

            # column functions
            for j in range(n):
                if count_e(copy_H[:, j]) != wc:
                    copy_H[:, j] = assign(copy_H[:, j], j)
                    for l in columns[j]:
                        if copy_H[l][j] != -1:
                            r[j] = copy_H[l][j]
                            break            

            res[k*i : k*(i+1)] = r[0:k]

    return res
p = [0.1, 0.3, 0.5, 0.7] # probability of erasures
BER = []

for i in p:
    print("For p =", i)
    r = BEC(code, i) # received bits. Here, '-1' is the erasure symbol.
    decode_ = belief_prop(r)
    incorrect_bit = (img != decode_).sum()
    print("Bits decoded incorrectly:", incorrect_bit)
    BER.append(incorrect_bit/size)
    print("Bit Error Rate:", incorrect_bit/size, "\n")

plt.plot(p, BER, marker = 'o')
plt.xlabel('p (erasure probability)')
plt.ylabel('BER')
plt.savefig('BEC.png')
plt.show() 