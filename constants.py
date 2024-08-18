import math
import numpy as np

MAX_BUFFER = 1000000
MAX_EPISODES = 1000000
S_DIM = 15 # 37
A_DIM = 11
A_MAX = np.array([0.3, 0.8, 2*math.pi, 0.5*math.pi, 2*math.pi, 0.75*math.pi, 2*math.pi, 0.75*math.pi, 2*math.pi, 0.75*math.pi, 2*math.pi])