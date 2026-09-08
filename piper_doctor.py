import time

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="can0"
)

arm = AgxArmFactory.create_arm(cfg)
end_effector = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

arm.connect()

time.sleep(0.05)

arm.reset()

time.sleep(0.05)

arm.connect()

while not arm.enable():
    time.sleep(0.05)

print("enabled robot arm...")

while not arm.disable():
    time.sleep(0.05)

print("disabled robot arm...")

arm.disconnect()
