from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from backpropfast import Backprop
import numpy as np

def booltrain(target):
    pair = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    back = Backprop(n= 2, m= 1, h = 3, part = 1)
    eta, t, h, mew, RMSerr, WIH_00, WHO_00 = back.train(pair,target, t=5000)

    for i in range(4):
        print(str(pair[i]) + ' >>', back.test(pair[i], target[i]))
        print()
    print("XOR Function Results with eta=%.1f %d Iterations and %d Hidden Units" % (eta, t, h))
    return eta, t, h, mew, RMSerr, WIH_00, WHO_00

target_xor = np.array([1, 0, 0, 1]).reshape(-1,1)
eta, t, h, mew, RMSerr, WIH_00, WHO_00 = booltrain(target_xor)

fig = plt.figure()
ax = fig.gca(projection='3d')

    # Make data.
X = np.array(WIH_00)
Y = np.array(WHO_00)
X, Y = np.meshgrid(X, Y)
Z = np.array(RMSerr).reshape(-1, 1)

    # Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(Y), np.max(Y))
ax.set_zlim(np.min(Z), np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('WIH at (1,0)')
ax.set_ylabel('WHO at (1,0)')
ax.set_zlabel('RMS error')
plt.title("Part 4c Plot")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
