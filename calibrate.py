import time

from piper_utils import RealPiper

piper = RealPiper("can0")

input("press enter to disable torque")

piper.disable_torque()

time.sleep(1)

input("set joint 1 to zero position and enter")

piper.calibrate(1)

input("set joint 2 to zero position and enter")

piper.calibrate(2)

input("set joint 3 to zero position and enter")

piper.calibrate(3)

input("set joint 4 to zero position and enter")

piper.calibrate(4)

input("set joint 5 to zero position and enter")

piper.calibrate(5)

input("set joint 6 to zero position and enter")

piper.calibrate(6)

time.sleep(1)
