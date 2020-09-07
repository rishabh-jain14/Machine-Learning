import numpy as np
n , m = map(int, input().split())
X = []
Y = []
for i in range(m):
    Z = list(map(float, input().split()))
    Y.append(list(Z[n:n+1]))
    X.append(Z[:n])
    X[i].insert(0, 1)
X_arr = np.array(X)
Y_arr = np.array(Y)

B = (np.linalg.inv(np.transpose(X_arr).dot(X_arr))).dot(np.transpose(X_arr).dot(Y_arr))

q = int(input())
Q = []
for i in range(q):
    Z = list(map(float, input().split()))
    Q.append(Z[:n])
    Q[i].insert(0, 1)
Q_arr = np.array(Q)
R = Q_arr.dot(B)
R = np.around(R,2)
for row in R:
    print("{:.2f}".format(row[0]))
