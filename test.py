from matplotlib import pyplot as plt
import numpy as np
a = np.array([[255,245.1,0],[0,0,0],[0,0,0]])
plt.imshow(a.reshape(3,3), cmap="binary")
plt.show()