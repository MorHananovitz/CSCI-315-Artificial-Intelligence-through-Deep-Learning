import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Part 1
x1, x2, y, z = np.loadtxt(fname = "assign1_data.txt",skiprows= 2, unpack = True)
yavg=np.mean(y)
x1avg=np.mean(x1)
m1=np.sum((x1-x1avg)*(y-yavg))/np.sum((x1-x1avg)**2)
b1=yavg-m1*x1avg

x2avg=np.mean(x2)
m2=np.sum((x2-x2avg)*(y-yavg))/np.sum((x2-x2avg)**2)
b2=yavg-m2*x2avg

print("Part 1")
print("Regression Parameters for y=f(x1): m = %s, b = %s" % (m1, b1))
print("Regression Parameters for y=f(x2): m = %s, b = %s" % (m2, b2))
print()

#Part 2
A = np.vstack([x1, x2, np.ones(len(x1))]).T
w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
print("Part 2")
print("w1: %s" % w1)
print("w2: %s" % w2)
print("b: %s" % b)
print()

#Part 3
regpar = np.array([w1, w2, b])
ycalc=np.matmul(A, regpar.T)
ycalc_classifier = ycalc > 0
model_classifier = ycalc_classifier == z
success = np.sum(model_classifier)*100/len(z)
print("Part 3")
print("Model Success is %s %% " % success)
print()

#Part 4
print("Part 4")
def reg_learn(x1, x2, y, z, n):
    x1_train=x1[0:n]
    x1_test=x1[n:]
    x2_train = x2[0:n]
    x2_test = x2[n:]
    y_train=y[0:n]
    y_test=y[n:]
    z_test=z[n:]

    A_train = np.vstack([x1_train, x2_train, np.ones(len(x1_train))]).T
    w1, w2, b = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
    regpar = np.array([w1, w2, b])

    A_test = np.vstack([x1_test, x2_test, np.ones(len(x1_test))]).T
    ycalc = np.matmul(A_test, regpar.T)
    ycalc_classifier = ycalc > 0
    model_classifier = ycalc_classifier == z_test
    success = np.sum(model_classifier) * 100 / len(z_test)

    #Baseline Test
    zero=np.array([0, 0, 0])
    y_zero = np.matmul(A_test, zero.T)
    y_zero_classifier = y_zero > 0
    zero_classifier = y_zero_classifier == z_test
    baseline = np.sum(zero_classifier) * 100 / len(z_test)

    print("Model Success for %d is %s %% " % (n, success))
    print("Model Baseline test for %d is %s %% " % (n, baseline))

reg_learn(x1, x2, y, z, 25)
reg_learn(x1, x2, y, z, 50)
reg_learn(x1, x2, y, z, 75)

#Part 5 - Extra Credit
xx, yy = np.meshgrid(range(2), range(2))
normal = np.array([0, 0, 1])
Z=(-normal[0]*xx-normal[1]*yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_low_i = np.where(y < 0)
x1_low = x1[y_low_i]
x2_low = x2[y_low_i]
y_low = y[y_low_i]

y_high_i = np.where(y > 0)
x1_high = x1[y_high_i]
x2_high = x2[y_high_i]
y_high = y[y_high_i]

ax.scatter(x1_low, x2_low, y_low, c = 'fuchsia', marker = 'o')
ax.scatter(x1_high, x2_high, y_high, c = 'aqua', marker = '^')
ax.plot_surface(xx, yy, Z, alpha=0.2)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.title("Part 5 Plot")
plt.show()