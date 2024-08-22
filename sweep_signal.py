import numpy as np
import matplotlib.pyplot as plt

C1 = 4.0
C2 = 0.0187
Trec = 13.0
wmin = 0.4 # rad/s
wmax = 6.0 # rad/s
A = 0.2
t = np.linspace(0, 100, 1000)   
K = C2*(np.exp(C1*t)/Trec-1)
w = wmin + K*(wmax-wmin)
theta = np.trapz(w, t) # integration of w(t) with respect to t
delta_sweep = A*np.sin(theta)
plt.plot(t, delta_sweep)
plt.show()