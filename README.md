# Python car simulator

usage:

'''
import pycar

pycar.env(visualize=False)

pycar.reset()

state, reward, terminate, info = pycar.step([steering,accel])
'''

state : dimesion (6 + 12)
