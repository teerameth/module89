import argparse
import pygame
import serial
from dynamixel_control import Dynamixel
from lowlevel_control import Lowlevel
import time


parser = argparse.ArgumentParser(description='Enter Comports : con, head' )
# parser.add_argument('-c','--con', dest='con_comport',help='Controller Comport',required=True)
# parser.add_argument('-h','--head', dest='head_comport',help='Head Comport',required=True)


parser.add_argument('--headType',help='motor type on head Mx , Rx ',default='Mx')
parser.add_argument('-c','--con',help='Controller Comport',default='/dev/ttyACM0')
parser.add_argument('--head',help='Head Comport',default='/dev/ttyUSB0')



args = parser.parse_args()


robot_head_type = args.headType
print("HEAD TYPE =", robot_head_type)
#robot_head_type = 'Rx'
pygame.init()

screen_width=300
screen_height=200

step_flag =False

motorHead = Dynamixel(args.head,115200)
motorHead.connect()

lowLevel_Control = Lowlevel(args.con,115200)
lowLevel_Control.connect()


if robot_head_type == 'Mx':
    motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
    motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
else:
    motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
    motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)



pygame.display.set_mode([screen_width,screen_height])
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                print("Forward ")
                lowLevel_Control.forward_walk(step_flag)
            if event.key == pygame.K_a:
                print("Left Slide")
                lowLevel_Control.left_walk(step_flag)
            if event.key == pygame.K_s:
                print("Backward")
                lowLevel_Control.backward_walk(step_flag)
            if event.key == pygame.K_d:
                print("Right Slide")
                lowLevel_Control.right_walk(step_flag)
            if event.key == pygame.K_q:
                print("Turn Left")
                lowLevel_Control.turn_left(step_flag)
            if event.key == pygame.K_e:
                print("Turn Right")
                lowLevel_Control.turn_right(step_flag)
            if event.key == pygame.K_SPACE:
                print("Stop")
                lowLevel_Control.stop_walk()
            if event.key == pygame.K_z:
                if step_flag == False:
                    step_flag = True
                   
                elif step_flag == True:
                    step_flag = False
                print(step_flag)

            #######################################################
            if event.key == pygame.K_KP1:
                print("Sit")
                lowLevel_Control.sit()
            if event.key == pygame.K_KP2:
                print("Stand")
                lowLevel_Control.stand()
            if event.key == pygame.K_KP3:
                print("Getup")
                lowLevel_Control.getup()
            if event.key == pygame.K_KP4:
                print("Left Kick")
                lowLevel_Control.left_kick()
            if event.key == pygame.K_KP6:
                print("Right Kick")
                lowLevel_Control.right_kick()
            if event.key == pygame.K_KP7:
                print("Left Save")
                lowLevel_Control.left_save()
            if event.key == pygame.K_KP9:
                print("Right Save")
                lowLevel_Control.right_save()


            #######################################################
            ######### HEAD CONTROL ###############
            if event.key == pygame.K_UP:
                print("Head Up")
                if robot_head_type == 'Mx':
                    position = motorHead.getMotorPosition(42)
                    motorHead.setDeviceMoving(42, robot_head_type, position - 350, 300, 1023)
                else:
                    position = motorHead.getMotorPosition(42)
                    motorHead.setDeviceMoving(42, robot_head_type, position - 70, 300, 1023)
            if event.key == pygame.K_DOWN:
                print("Head Down")
                if robot_head_type == 'Mx':
                    position = motorHead.getMotorPosition(42)
                    motorHead.setDeviceMoving(42, robot_head_type, position + 350, 300, 1023)
                else:
                    position = motorHead.getMotorPosition(42)
                    motorHead.setDeviceMoving(42, robot_head_type, position + 70, 300, 1023)
            if event.key == pygame.K_LEFT:
                print("Head LEFT")
                if robot_head_type == 'Mx':
                    position = motorHead.getMotorPosition(41)
                    motorHead.setDeviceMoving(41, robot_head_type, position + 350, 300, 1023)
                else:
                    position = motorHead.getMotorPosition(41)
                    motorHead.setDeviceMoving(41, robot_head_type, position + 70, 300, 1023)
            if event.key == pygame.K_RIGHT:
                print("Head RIGHT")
                if robot_head_type == 'Mx':
                    position = motorHead.getMotorPosition(41)
                    motorHead.setDeviceMoving(41, robot_head_type, position - 350, 300, 1023)
                else:
                    position = motorHead.getMotorPosition(41)
                    motorHead.setDeviceMoving(41, robot_head_type, position - 70, 300, 1023)
            if event.key == pygame.K_RSHIFT :
                print("Head Center")
                if robot_head_type == 'Mx':
                    motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
                else:
                    motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)
            if event.key == pygame.K_RCTRL :
                if robot_head_type == 'Mx':
                    motorHead.setDeviceMoving(41, robot_head_type, 2048, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 2048, 1023, 1023)
                    time.sleep(0.5)
                    motorHead.setDeviceMoving(41, robot_head_type, (2048-1200), 200, 1023)
                    time.sleep(0.8)
                    motorHead.setDeviceMoving(42, robot_head_type, (2048+400), 1023, 1023)
                    motorHead.setDeviceMoving(41, robot_head_type, (2048+1200), 200, 1023)
                    time.sleep(1.6)
                    motorHead.setDeviceMoving(42, robot_head_type, (2048), 1023, 1023)
                    motorHead.setDeviceMoving(41, robot_head_type, (2048), 200, 1023)

                else:

                    motorHead.setDeviceMoving(41, robot_head_type, 512, 1023, 1023)
                    motorHead.setDeviceMoving(42, robot_head_type, 512, 1023, 1023)
                    time.sleep(0.5)
                    motorHead.setDeviceMoving(41, robot_head_type, (512-300), 200, 1023)
                    time.sleep(0.8)
                    motorHead.setDeviceMoving(42, robot_head_type, (512+100), 1023, 1023)
                    motorHead.setDeviceMoving(41, robot_head_type, (512+300), 200, 1023)
                    time.sleep(1.6)
                    motorHead.setDeviceMoving(42, robot_head_type, (512), 1023, 1023)
                    motorHead.setDeviceMoving(41, robot_head_type, (512), 200, 1023)


            

            