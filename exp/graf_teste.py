import matplotlib.pyplot as plt
import numpy as np
a=np.array([[0],[1],[2]], np.int32)
b=np.array([[3],[4],[5]], np.int32)

plt.plot(a, color = 'red', label = 'Historical data')
plt.plot(b, color = 'blue', label='Predicted data')
plt.legend()
plt.show()
print("kkk")
plt.savefig('foo.png')