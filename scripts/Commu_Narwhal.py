import time
import UdpComms as U
import cv2
from Communication import Communication,CRC8

ip = "127.0.0.1"

def CallbackImageSubscribe1():
    image_bytes = cv2.imencode('.jpg', cv2.imread("2.jpg"))[1].tobytes()
    while True:
        # sock.SendData("start")
        # sock.SendData(image_bytes,True)
        # time.sleep(0.)
        # print("Send Image")
        pass

if __name__ == '__main__':


    # data_sock = U.UdpComms(udpIP=ip, portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True) #Main Communication
    img_sock1 = U.UdpComms(udpIP=ip, portTX=8002,portRX=8001, suppressWarnings=True) # Image1 Socket-
    img_sock2 = U.UdpComms(udpIP=ip, portTX=8003,portRX=8001, suppressWarnings=True) # Image2 Socket
    image_bytes = cv2.imencode('.jpg', cv2.imread("2.jpg"))[1].tobytes()
    # while True:
    #     img_sock1.SendData(image_bytes,True)
    #     img_sock1.SendData(image_bytes, True)
    #     # data_sock.SendData("GGGG")
    #     # time.sleep(0.001)
    try:
        Narwhal = Communication(port="com5", baudrate=1000000)
        print("Connection : " + str(Narwhal.Connection_Test()))
        # print(Narwhal.SetHome())
        # print(Narwhal.JogJoint(0,0.1,0,0))
        # print(Narwhal.JogTask(0,-30,0))
        TimeBefore = time.time()
        while(True):
            #Send with 20HZ
            if (time.time() - TimeBefore >= 0.05):
                TimeBefore = time.time()
                # print(Narwhal.SystemStat_Read())
                # print(Narwhal.Base_Enc_Read())
                # print(Narwhal.Enc_Raw_Pos())
                # print(Narwhal.Joint_Position())
                # print(Narwhal.TaskSpace_Position())
                status = str(Narwhal.ReadAll())
                status = status.replace("[","")
                status = status.replace("]","")
                status = status.replace(",","")
                status = "status " + status
                data_sock.SendData(status)

            #Alway Recieve
            data = data_sock.ReadReceivedData()  # read data
            if data != None:  # if NEW data has been received since last ReadReceivedData function call
                print(data)  # print new received data
                while(time.time() - TimeBefore <= 0.005):
                    pass
                dl = data.split(" ")
                if dl[0] == "SETHOME":
                    pass
                elif dl[0] == "STOPPOSE":
                    pass
                elif dl[0] == "GRIP":
                    pass
                elif dl[0] == "FEN":
                    pass
                elif dl[0] == "REQUESTFEN":
                    pass
                elif dl[0] == "SET":
                    pass
                elif dl[0] == "JOG":
                    pass

                if len(dl) == 1:
                    if dl[0] == 'sethome':
                        print(Narwhal.SetHome())
                elif len(dl) == 2:
                    j1 = 0
                    j2 = 0
                    j3 = 0
                    j4 = 0
                    x = 0
                    y = 0
                    z = 0
                    if (dl[0] == 'q1'):
                        j1 = float(dl[1])
                    elif (dl[0] == 'q2'):
                        j2 = float(dl[1])
                    elif (dl[0] == 'q3'):
                        j3 = float(dl[1])
                    elif (dl[0] == 'q4'):
                        j4 = float(dl[1])
                    # elif (dl[0] == 'q5'):
                    #     y_in = float(dl[1])
                    elif (dl[0] == 'x'):
                        x = float(dl[1])
                    elif (dl[0] == 'y'):
                        y = float(dl[1])
                    elif (dl[0] == 'z'):
                        z = float(dl[1])

                    if abs(j1) >= 0 or abs(j2) >= 0 or abs(j3) >= 0 or abs(j4) >= 0:
                        Narwhal.JogJoint(j1,j2,j3,j4)
                    elif abs(x) >= 0 or abs(y) >= 0 or abs(z) >= 0:
                        Narwhal.JogTask(x,y,z)
                # print(Narwhal.JogTask(x_in,y_in,0))

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt!!!!\n\n\nShutdown ...\n\n\n\n")

# echo-server.py

# import socket
# from threading import *
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# host = "192.168.100.240"
# port = 8000
#
# s.bind((host, port))
# s.listen(5)
# try:
#     c, addr = s.accept()
#     print('Got connection from', addr)
#     while True:
#         # Establish connection with client.
#
#
#         # send a thank you message to the client. encoding to send byte type.
#         c.send('Thank you for connecting'.encode())
#
#         # Close the connection with the client
#
#
#         # Breaking once connection closed
# except KeyboardInterrupt:
#     c.close()
