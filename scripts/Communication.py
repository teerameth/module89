from numpy import int16
import serial
import time



def CRC8(Data, wLength):
    wCRCTable = [
        0x00, 0x1D, 0x3A, 0x27, 0x74, 0x69, 0x4E, 0x53, 0xE8, 0xF5, 0xD2, 0xCF, 0x9C, 0x81, 0xA6, 0xBB,
        0xCD, 0xD0, 0xF7, 0xEA, 0xB9, 0xA4, 0x83, 0x9E, 0x25, 0x38, 0x1F, 0x02, 0x51, 0x4C, 0x6B, 0x76,
        0x87, 0x9A, 0xBD, 0xA0, 0xF3, 0xEE, 0xC9, 0xD4, 0x6F, 0x72, 0x55, 0x48, 0x1B, 0x06, 0x21, 0x3C,
        0x4A, 0x57, 0x70, 0x6D, 0x3E, 0x23, 0x04, 0x19, 0xA2, 0xBF, 0x98, 0x85, 0xD6, 0xCB, 0xEC, 0xF1,
        0x13, 0x0E, 0x29, 0x34, 0x67, 0x7A, 0x5D, 0x40, 0xFB, 0xE6, 0xC1, 0xDC, 0x8F, 0x92, 0xB5, 0xA8,
        0xDE, 0xC3, 0xE4, 0xF9, 0xAA, 0xB7, 0x90, 0x8D, 0x36, 0x2B, 0x0C, 0x11, 0x42, 0x5F, 0x78, 0x65,
        0x94, 0x89, 0xAE, 0xB3, 0xE0, 0xFD, 0xDA, 0xC7, 0x7C, 0x61, 0x46, 0x5B, 0x08, 0x15, 0x32, 0x2F,
        0x59, 0x44, 0x63, 0x7E, 0x2D, 0x30, 0x17, 0x0A, 0xB1, 0xAC, 0x8B, 0x96, 0xC5, 0xD8, 0xFF, 0xE2,
        0x26, 0x3B, 0x1C, 0x01, 0x52, 0x4F, 0x68, 0x75, 0xCE, 0xD3, 0xF4, 0xE9, 0xBA, 0xA7, 0x80, 0x9D,
        0xEB, 0xF6, 0xD1, 0xCC, 0x9F, 0x82, 0xA5, 0xB8, 0x03, 0x1E, 0x39, 0x24, 0x77, 0x6A, 0x4D, 0x50,
        0xA1, 0xBC, 0x9B, 0x86, 0xD5, 0xC8, 0xEF, 0xF2, 0x49, 0x54, 0x73, 0x6E, 0x3D, 0x20, 0x07, 0x1A,
        0x6C, 0x71, 0x56, 0x4B, 0x18, 0x05, 0x22, 0x3F, 0x84, 0x99, 0xBE, 0xA3, 0xF0, 0xED, 0xCA, 0xD7,
        0x35, 0x28, 0x0F, 0x12, 0x41, 0x5C, 0x7B, 0x66, 0xDD, 0xC0, 0xE7, 0xFA, 0xA9, 0xB4, 0x93, 0x8E,
        0xF8, 0xE5, 0xC2, 0xDF, 0x8C, 0x91, 0xB6, 0xAB, 0x10, 0x0D, 0x2A, 0x37, 0x64, 0x79, 0x5E, 0x43,
        0xB2, 0xAF, 0x88, 0x95, 0xC6, 0xDB, 0xFC, 0xE1, 0x5A, 0x47, 0x60, 0x7D, 0x2E, 0x33, 0x14, 0x09,
        0x7F, 0x62, 0x45, 0x58, 0x0B, 0x16, 0x31, 0x2C, 0x97, 0x8A, 0xAD, 0xB0, 0xE3, 0xFE, 0xD9, 0xC4]
    cCRC_val = 0xC4
    for i in range(wLength):
        index = cCRC_val ^ Data[i]
        cCRC_val = (cCRC_val >> 8) ^ wCRCTable[index]
    return cCRC_val ^ 0xFF

class Communication:
    """
    Narwhal Protocol
    =========================================================
    Function List
    ==============
    Write() \n
    """

    def __init__(self, port="com3", baudrate=1000000):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate)
            self.status = 1
            print("Comport is Open")
            # while(True):
            #     if(self.Readline() == "start"):
            #         print("connect to dsPIC success")
            #         break
        except:
            print("\nConnection Error !\nComport = " +
                  str(port)+" ?\n")
            self.status = 0

    def Write(self, Command, Data):
        Buffer = []
        Buffer.append(0xFF)             # Header
        Buffer.append(Command)          # Command
        if (len(Data) == 11):
            for i in Data:
                Buffer.append(i)
        Buffer.append(CRC8(Buffer, 13))  # Add CRC Check Sum
        for _ in range(5):
            DataSent = bytes(Buffer)
            self.ser.write(DataSent)
            Rx = self.ser.read(4)
            if Rx[0] == 0xFF:
                if (Rx[-1] == CRC8(Rx[0:-1], 3)):
                    if Rx[1] == 0xFF:
                        return [True, Rx[2]]
                    elif Rx[1] == 0xCC:
                        print("W Sent CRC Fail")
                    elif Rx[1] == 0xAA:
                        print("W Sent Header Fail")
                else:
                    # int_val = int.from_bytes(byte_val, "big")
                    print("W Read CRC Fail")
            else:
                print("W Read Header Fail")
            TimeBefore = time.time()
            while(time.time() - TimeBefore <= 0.02):
                pass
        else:
            print("W Comunication Error")
        return [False, []]

    def Read(self, Command, Data_Count):
        Buffer = [0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Buffer[1] = Command             # Command
        Buffer[13] = CRC8(Buffer, 13)    # Add CRC Check Sum
        for _ in range(5):
            DataSent = bytes(Buffer)
            self.ser.write(DataSent)
            Rx = self.ser.read(Data_Count)
            if Rx[0] == 0xFF:
                if (Rx[-1] == CRC8(Rx[0:-1], Data_Count-1)):
                    if Rx[1] == 0xEE:
                        return True, Rx[2:-1]
                    elif Rx[1] == 0xFF:
                        print("R Command not found")
                    elif Rx[1] == 0xCC:
                        print("R Sent CRC Fail")
                    elif Rx[1] == 0xAA:
                        print("R Sent Header Fail")
                else:
                    print("R Read CRC Fail")
            else:
                print("R Read Header Fail")
            TimeBefore = time.time()
            while(time.time() - TimeBefore <= 0.02):
                pass
        else:
            print("R Comunication Error")
        return False, []

    def SetHome(self):
        Stat, Feedback = self.Write(0xF5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if Stat:
            if Feedback == 0:
                return True
            elif Feedback == 1:
                print("Trajectory not Finish")
        return False

    def JogJoint(self, q1, q2, q3, q4):
        """
        JogJoint
        ==============
        Input Delta Position [rad,rad,rad,rad]
        In list format
        """
        q1 = int(q1*1000)
        q2 = int(q2*1000)
        q3 = int(q3*1000)
        q4 = int(q4*1000)
        Buffer = [(q1 & 0xFF00) >> 8, q1 & 0xFF, (q2 & 0xFF00) >> 8, q2 & 0xFF,
                  (q3 & 0xFF00) >> 8, q3 & 0xFF, (q4 & 0xFF00) >> 8, q4 & 0xFF, 0, 0, 0]
        Stat, Feedback = self.Write(0xFA, Buffer)
        return Stat

    def SetJoint(self, q1, q2, q3, q4):
        """
        JogJoint
        ==============
        Input Delta Position [rad,rad,rad,rad]
        In list format
        """
        q1 = int(float(q1)*1000)
        q2 = int(float(q2)*1000)
        q3 = int(float(q3)*1000)
        q4 = int(float(q4)*1000)

        Buffer = [(q1 & 0xFF00) >> 8, q1 & 0xFF, (q2 & 0xFF00) >> 8, q2 & 0xFF,
                  (q3 & 0xFF00) >> 8, q3 & 0xFF, (q4 & 0xFF00) >> 8, q4 & 0xFF, 0, 0, 0]
        Stat, Feedback = self.Write(0xFD, Buffer)
        return Stat

    def SetTask(self, q1, q2, q3, q4):
        """
        JogJoint
        ==============
        Input Delta Position [rad,rad,rad,rad]
        In list format
        """
        q1 = int(q1*1000)
        q2 = int(q2*1000)
        q3 = int(q3*1000)
        q4 = int(q4*1000)

        Buffer = [(q1 & 0xFF00) >> 8, q1 & 0xFF, (q2 & 0xFF00) >> 8, q2 & 0xFF,
                  (q3 & 0xFF00) >> 8, q3 & 0xFF, (q4 & 0xFF00) >> 8, q4 & 0xFF, 0, 0, 0]
        Stat, Feedback = self.Write(0xFE, Buffer)
        return Stat

    def JogTask(self, x, y, z):
        """
        Cartesian Joint
        ==============
        Input Delta Position [mm,mm,mm,rad]
        In list format
        """
        x = int(x*50)
        y = int(y*50)
        z = int(z*50)
        Buffer = [(x & 0xFF00) >> 8, x & 0xFF, (y & 0xFF00) >> 8, y &
                  0xFF, (z & 0xFF00) >> 8, z & 0xFF, 0, 0, 0, 0, 0]
        Stat, Feedback = self.Write(0xFB, Buffer)
        return Stat

    def SystemStat_Read(self):
        Stat, Buf = self.Read(0xA0, 5)
        if Stat:
            return ((Buf[0] << 8) | Buf[1])/1000.0
        return False

    def Base_Enc_Read(self):
        Stat, Buf = self.Read(0xA1, 5)
        if Stat:
            return ((Buf[0] << 8) | Buf[1])
        return False

    def Enc_Raw_Pos(self):
        Stat, Buf = self.Read(0xA2, 13)
        if Stat:
            return [((Buf[0] << 8) | Buf[1]), ((Buf[2] << 8) | Buf[3]), ((Buf[4] << 8) | Buf[5]), ((Buf[6] << 8) | Buf[7]), ((Buf[8] << 8) | Buf[9])]
        return False

    def Joint_Position(self):
        Stat, Buf = self.Read(0xAA, 13)
        if Stat:
            return [(int16((Buf[0] << 8) | Buf[1]))/1000.0, (int16((Buf[2] << 8) | Buf[3]))/1000.0, (int16((Buf[4] << 8) | Buf[5]))/1000.0, (int16((Buf[6] << 8) | Buf[7]))/1000.0, (int16((Buf[8] << 8) | Buf[9]))/1000.0]
        return False

    def TaskSpace_Position(self):
        Stat, Buf = self.Read(0xAB, 9)
        if Stat:
            return [int16((Buf[0] << 8) | Buf[1])/10.0, int16((Buf[2] << 8) | Buf[3])/10.0, int16((Buf[4] << 8) | Buf[5])/10.0]
        return False

    def ReadAll(self):
        Stat, Buf = self.Read(0xAF, 35)
        if Stat:
            return [(Buf[0] << 8) | Buf[1], ((Buf[2] << 8) | Buf[3])/1000.0, ((Buf[4] << 8) | Buf[5])/1000.0, (int16((Buf[6] << 8) | Buf[7]))/1000.0, (int16((Buf[8] << 8) | Buf[9]))/1000.0, (int16((Buf[10] << 8) | Buf[11]))/1000.0, (int16((Buf[12] << 8) | Buf[13]))/1000.0, (int16((Buf[14] << 8) | Buf[15]))/1000.0, (int16((Buf[16] << 8) | Buf[17]))/1000.0, (int16((Buf[18] << 8) | Buf[19]))/1000.0, (int16((Buf[20] << 8) | Buf[21]))/1000.0, (int16((Buf[22] << 8) | Buf[23]))/1000.0, (int16((Buf[24] << 8) | Buf[25]))/1000.0, int16((Buf[26] << 8) | Buf[27])/10.0, int16((Buf[28] << 8) | Buf[29])/10.0, int16((Buf[30] << 8) | Buf[31])/10.0]
        return False

    def Connection_Test(self):
        """
        Connection Test
        ==============
        return --> bool
            True in case can init uart
            False in case can't init uart
        """
        if(self.status):
            stat, data = self.Write(0xF0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            return stat
        return False

    def Close(self):
        self.ser.close()

