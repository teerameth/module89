import serial
import math
# from dynamixel_control import Dynamixel
from lowlevel_control import Lowlevel
import time

class Dynamixel:
    def __init__(self, com, baud):
        self.str_comport = com
        self.str_baudrate = baud
    def connect(self):
        self.serialDevice = serial.Serial(port = self.str_comport, baudrate = self.str_baudrate, timeout=0)

    def setReadMotorPacket(self,deviceID,Offset,Length):
        readPacket = [0xFF, 0xFF, deviceID, 0x04, 0x02, Offset, Length]
        checkSumOrdList = readPacket[2:]
        checkSumOrdListSum = sum(checkSumOrdList)
        computedCheckSum = ( ~(checkSumOrdListSum%256) ) % 256
        readPacket.append(computedCheckSum)
        self.serialDevice.write(readPacket)
        #print(readPacket)

    def getMotorQueryResponse( self, deviceID, Length ):

        queryData = 0
        responsePacketSize = Length + 6
        # responsePacket = readAllData(serialDevice)
        responsePacket = self.serialDevice.read(self.serialDevice.inWaiting())

        if len(responsePacket) == responsePacketSize:

            # print("responsePacket=", responsePacket)

            responseID = responsePacket[2]
            errorByte = responsePacket[4]

            ### python 3
            if responseID == deviceID and errorByte == 0:
                if Length == 2:
                    queryData = responsePacket[5] + 256 * responsePacket[6]
                elif Length == 1:
                    queryData = responsePacket[5]
                    # print "Return data:", queryData
            else:
                print("Error response:", responseID, errorByte)

            responsePacketStatue = True

        else:
            responsePacketStatue = False

        # print("queryData=", queryData)
        return queryData, responsePacketStatue
    def get(self,deviceID, address, Length):

            for i in range(0,5):
                self.setReadMotorPacket(deviceID, address, Length)
                time.sleep(0.02)
                data, status = self.getMotorQueryResponse(deviceID, Length)

                if status == True:
                    break
                else:
                    print("motor ID " + str(deviceID) + "  no response " + str(i))

            return data

    def getMotorPosition(self,id):
            data = self.get(id,36,2)
            return data
    def rxPacketConversion( self,value ):
            if value < 1024 and value >= 0:
                    hiByte = int(value/256)
                    loByte = value%256
            else:
                    print("rxPacketConversion: value out of range", value)
            return loByte, hiByte
    def setMotorSpeed(self, deviceID, speed):
        (speedL, speedH) = self.rxPacketConversion(speed)
        Packet = [0xFF, 0xFF, deviceID, 0x05, 0x03, 0x20, speedL, speedH]
        checkSumOrdList = Packet[2:]
        checkSumOrdListSum = sum(checkSumOrdList)
        computedCheckSum = (~(checkSumOrdListSum % 256)) % 256
        Packet.append(computedCheckSum)
        self.serialDevice.write(Packet)

motor_id = 1
baudrate = 57600
device_name = "/dev/ttyUSB0"
rpm = 1

motor = Dynamixel(device_name, baudrate)
motor.connect()
motor.setMotorSpeed(1, 3)
time.sleep(1)

while True:
    pos = motor.getMotorPosition(1)
    pos = float(pos)/4096
    print(pos)
    time.sleep(0.1)