import numpy as np
import matplotlib.pyplot as plt
plt.style.use('deeplearning.mplstyle')

x = np.arange(0,100,1)
y = x + x**2 + x**3 + x**4
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax[0].plot(x, y, linewidth =0.3 )
plt.show()