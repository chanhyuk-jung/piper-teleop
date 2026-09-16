from piper_utils import Piper

piper = Piper("can0")

# piper.bus.SearchAllMotorMaxAngleSpd()
msg = piper.bus.GetAllMotorAngleLimitMaxSpd()
print(msg.all_motor_angle_limit_max_spd)

piper.bus.SearchMotorMaxAngleSpdAccLimit(1, 2)
piper.bus.SearchMotorMaxAngleSpdAccLimit(2, 2)
piper.bus.SearchMotorMaxAngleSpdAccLimit(3, 2)
piper.bus.SearchMotorMaxAngleSpdAccLimit(4, 2)
piper.bus.SearchMotorMaxAngleSpdAccLimit(5, 2)
piper.bus.SearchMotorMaxAngleSpdAccLimit(6, 2)

msg = piper.bus.GetAllMotorMaxAccLimit()
print(msg.all_motor_max_acc_limit)

piper.set_max_vel(1, 3.0)
piper.set_max_vel(2, 3.0)
piper.set_max_vel(3, 3.0)
piper.set_max_vel(4, 3.0)
piper.set_max_vel(5, 3.0)
piper.set_max_vel(6, 3.0)

piper.set_max_acc(1, 5.0)
piper.set_max_acc(2, 5.0)
piper.set_max_acc(3, 5.0)
piper.set_max_acc(4, 5.0)
piper.set_max_acc(5, 5.0)
piper.set_max_acc(6, 5.0)
