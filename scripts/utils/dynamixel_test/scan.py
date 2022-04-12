from dynamixel_control import Dynamixel
motorHead = Dynamixel('COM11', 1000000)
motorHead.connect()
# motorHead.setDeviceMoving(41, 'Mx', 2048, 1023, 1023)
# motorHead.setDeviceMoving(42, 'Mx', 2048, 1023, 1023)
for i in range(20, 50):
    print(motorHead.getMotorPosition(i))