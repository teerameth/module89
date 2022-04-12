import argparse
import pygame
import serial
from dynamixel_control import Dynamixel
from lowlevel_control import Lowlevel
import time

parser = argparse.ArgumentParser(description='Enter Comports : con, head')
# parser.add_argument('-c','--con', dest='con_comport',help='Controller Comport',required=True)
# parser.add_argument('-h','--head', dest='head_comport',help='Head Comport',required=True)


parser.add_argument('--headType', help='motor type on head Mx , Rx ', default='Mx')
parser.add_argument('-c', '--con', help='Controller Comport', default='/dev/ttyACM0')
parser.add_argument('--head', help='Head Comport', default='/dev/ttyUSB0')

args = parser.parse_args()

robot_head_type = args.headType
print("HEAD TYPE =", robot_head_type)
# robot_head_type = 'Rx'
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

screen_width = 300
screen_height = 200

step_flag = False

motorHead = Dynamixel(args.head, 115200)
motorHead.connect()

lowLevel_Control = Lowlevel(args.con, 115200)
lowLevel_Control.connect()

if robot_head_type == 'Mx':
    motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
    motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
else:
    motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
    motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)

pygame.display.set_mode([screen_width, screen_height])
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()  # sys.exit() if sys is imported
        if event.type == 9: #walk control
            if event.value == (0, 1): # forward
                print("Forward ")
                lowLevel_Control.forward_walk(step_flag)
            if event.value == (0, -1): # backward
                print("Backward")
                lowLevel_Control.backward_walk(step_flag)
            if event.value == (1, 0): #left slide
                print("Turn Left")
                lowLevel_Control.turn_left(step_flag)
            if event.value == (-1, 0):  # right slide
                print("Turn Right")
                lowLevel_Control.turn_right(step_flag)
            if event.value == (0, 0): # stop action (Auto activated when release button)
                pass
        if event.type == 10: # action (Joy Button Down)
            if event.button == 4: # LB
                print("Left Slide")
                lowLevel_Control.left_walk(step_flag)
            if event.button == 5: # RB
                print("Right Slide")
                lowLevel_Control.right_walk(step_flag)
            if event.button == 6: # Scan head (Back Button)
                if event.key == pygame.K_RCTRL:
                    if robot_head_type == 'Mx':
                        motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
                        motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
                        time.sleep(0.5)
                        motorHead.setDeviceMoving(41, robot_head_type, (2048 - 1200), 200, 1023)
                        time.sleep(0.8)
                        motorHead.setDeviceMoving(42, robot_head_type, (2048 + 400), 1023, 1023)
                        motorHead.setDeviceMoving(41, robot_head_type, (2048 + 1200), 200, 1023)
                        time.sleep(1.6)
                        motorHead.setDeviceMoving(42, robot_head_type, (2048), 1023, 1023)
                        motorHead.setDeviceMoving(41, robot_head_type, (2048), 200, 1023)
                    else:
                        motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
                        motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)
                        time.sleep(0.5)
                        motorHead.setDeviceMoving(41, robot_head_type, (512 - 300), 200, 1023)
                        time.sleep(0.8)
                        motorHead.setDeviceMoving(42, robot_head_type, (512 + 100), 1023, 1023)
                        motorHead.setDeviceMoving(41, robot_head_type, (512 + 300), 200, 1023)
                        time.sleep(1.6)
                        motorHead.setDeviceMoving(42, robot_head_type, (512), 1023, 1023)
                        motorHead.setDeviceMoving(41, robot_head_type, (512), 200, 1023)
            if event.button == 7: # Toggle between continuous walking mode and one step walking mode (Start Button)
                if step_flag == False:
                    step_flag = True
                elif step_flag == True:
                    step_flag = False
                print(step_flag)
            if event.button == 9: # Left Analog Press
                print("Head Center")
                if robot_head_type == 'Mx':
                    motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
                else:
                    motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)
            #######################################################
            if event.button == 0: # A
                print("Sit")
                lowLevel_Control.sit()
            if event.button == 1: # B
                print("Getup")
                lowLevel_Control.getup()
            if event.button == 2: # X (left)
                print("Stop")
                lowLevel_Control.stop_walk()
            if event.button == 3: # Y
                print("Stand")
                lowLevel_Control.stand()
        if event.type == 11: # activated when release action button (Joy Button Up)
            pass
        if event.type == 7:
            if event.axis == 2: # LT
                print("Left Kick")
                lowLevel_Control.left_kick()
            if event.axis == 5: # RT
                print("Right Kick")
                lowLevel_Control.right_kick()
            # if event.axis == 3 and event.value < -0.5: # Right Analog Horizontal
            #     print("Right Save")
            #     lowLevel_Control.right_save()
            # elif event.value > 0.5:
            #     print("Left Save")
            #     lowLevel_Control.left_save()
            # if event.axis == 0 and event.value < -0.1: # Left Analog Horizontal
            #     print("Head LEFT")
            #     if robot_head_type == 'Mx':
            #         position = motorHead.getMotorPosition(41)
            #         motorHead.setDeviceMoving(41, robot_head_type, position + 350, 300, 1023)
            #     else:
            #         position = motorHead.getMotorPosition(41)
            #         motorHead.setDeviceMoving(41, robot_head_type, position + 70, 300, 1023)
            # elif event.value > 0.1:
            #     print("Head RIGHT")
            #     if robot_head_type == 'Mx':
            #         position = motorHead.getMotorPosition(41)
            #         motorHead.setDeviceMoving(41, robot_head_type, position - 350, 300, 1023)
            #     else:
            #         position = motorHead.getMotorPosition(41)
            #         motorHead.setDeviceMoving(41, robot_head_type, position - 70, 300, 1023)
            # if event.axis == 1 and event.value < -0.1: # Left Analog Vertical
            #     print("Head Down")
            #     if robot_head_type == 'Mx':
            #         position = motorHead.getMotorPosition(42)
            #         motorHead.setDeviceMoving(42, robot_head_type, position + 350, 300, 1023)
            #     else:
            #         position = motorHead.getMotorPosition(42)
            #         motorHead.setDeviceMoving(42, robot_head_type, position + 70, 300, 1023)
            # elif event.value > 0.1:
            #     print("Head Up")
            #     if robot_head_type == 'Mx':
            #         position = motorHead.getMotorPosition(42)
            #         motorHead.setDeviceMoving(42, robot_head_type, position - 350, 300, 1023)
            #     else:
            #         position = motorHead.getMotorPosition(42)
            #         motorHead.setDeviceMoving(42, robot_head_type, position - 70, 300, 1023)
            #
