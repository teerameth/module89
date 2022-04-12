import argparse
import pygame
import serial
from dynamixel_control import Dynamixel
from lowlevel_control import Lowlevel
import time
import os, struct, array
from fcntl import ioctl
for fn in os.listdir('/dev/input'):
    if fn.startswith('js'):
        print('  /dev/input/%s' % (fn))

fn = '/dev/input/js0'
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

screen_width = 300
screen_height = 200

step_flag = False
pygame.display.set_mode([screen_width, screen_height])
while True:
    for event in pygame.event.get():
        print(event)
        if event.type == pygame.QUIT:
            pygame.quit()  # sys.exit() if sys is imported
        if event.type == 9: #walk control
            if event.value == (0, 1): # forward
                pass
        if event.type == 10: # action
            pass