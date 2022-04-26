#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from module89.srv import StringMessage
from module89.srv import ClusterLock
from module89.srv import PoseLock
from module89.srv import FindBestMove
from std_msgs.msg import String

#Communication
import time
import UdpComms as U
import cv2
from Communication import Communication,CRC8


class GameController(Node):

    def __init__(self):
        super().__init__('game_controller')
        self.pyserial_connected = False
        ip = "192.168.100.240"
        self.data_sock = U.UdpComms(udpIP=ip, portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True) #Main Communication
        self.q1 = float(0)
        self.q2= float(0)
        self.q3= float(0)
        self.q4= float(0)
        self.x= float(0)
        self.y= float(0)
        self.z= float(0)
        self.status = [0,0,0,0,0,0,0]
        self.name = ['q1', 'q2', 'q3', 'q4', 'x', 'y', 'z']
        # self.img_sock1 = U.UdpComms(udpIP=ip, portTX=8002,portRX=8001, suppressWarnings=True) # Image1 Socket-
        # self.img_sock2 = U.UdpComms(udpIP=ip, portTX=8003,portRX=8001, suppressWarnings=True) # Image2 Socket
        # self.image_bytes = cv2.imencode('.jpg', cv2.imread("2.jpg"))[1].tobytes()
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0
        #command_cluster_lock , command_pose_lock

        self.bestmove_cli = self.create_client(FindBestMove, 'find_best_move')
        # while not self.bestmove_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        self.bestmove_req = FindBestMove.Request()
        self.bestmove_get = False

        # self.fen_cli = self.create_client(StringMessage, 'fen')
        # while not self.fen_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.fen_req = StringMessage.Request()

        self.pose_lock_cli = self.create_client(PoseLock, 'command_pose_lock')
        # while not self.pose_lock_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        self.pose_lock_req = PoseLock.Request()
        self.pose_lock_get = False

        self.cluster_lock_cli = self.create_client(ClusterLock, 'command_cluster_lock')
        # while not self.cluster_lock_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        self.cluster_lock_req = ClusterLock.Request()
        self.cluster_lock_get = False

        self.get = False
        self.TimeBefore = time.time()
        self.connect_pyserial(com="/dev/ttyUSB0")
        timer_period = 1 / 100  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.run()

    def bestmove_request(self):
        bestmove = String()
        bestmove.data = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
        self.bestmove_req.fen = bestmove
        self.bestmove_future = self.bestmove_cli.call_async(self.bestmove_req)


    def fen_request(self):
        pass
        # self.req.input = "REQUEST"
        # self.future = self.cli.call_async(self.req)

    def cluster_lock_request(self,lock,flip):
        self.cluster_lock_req.lock = lock
        self.cluster_lock_req.flip = flip
        self.cluster_lock_future = self.cluster_lock_cli.call_async(self.cluster_lock_req)

    def pose_lock_request(self,mode,corners = None):
        self.pose_lock_req.mode = mode
        self.pose_lock_req.corners = corners
        self.pose_lock_future = self.pose_lock_cli.call_async(self.pose_lock_req)


    def connect_pyserial(self,com="/dev/ttyUSB0",baudrate=1000000):
        if not self.pyserial_connected:
            self.narwhal = Communication(port=com, baudrate=baudrate)
            print("Connection : " + str(self.narwhal.Connection_Test()))
        else:
            print("Already Connected!")
    def check_pyserial(self):
        return self.pyserial_connected

    def close_pyserial(self):
        self.narwhal.Close()

    def timer_callback(self):

        if self.bestmove_get:
            if self.bestmove_future.done():
                try:
                    response = self.bestmove_future.result()
                except Exception as e:
                    print(e)
                else:
                    self.bestmove_get = False
                    self.data_sock.SendData(response.bestmove.data)

        if self.cluster_lock_get:
            if self.cluster_lock_future.done():
                try:
                    response = self.cluster_lock_future.result()
                except Exception as e:
                    print(e)
                else:
                    self.cluster_lock_get = False
                    self.data_sock.SendData(response.acknowledge)

        if self.pose_lock_get:
            if self.pose_lock_future.done():
                try:
                    response = self.pose_lock_future.result()
                except Exception as e:
                    print(e)
                else:
                    self.pose_lock_get = False
                    self.data_sock.SendData(response.acknowledge)


        data = self.data_sock.ReadReceivedData()  # read data
        if (time.time() - self.TimeBefore >= 0.05):
            self.TimeBefore = time.time()
            if self.pyserial_connected:
                status = str(self.narwhal.ReadAll())
                status = status.replace("[", "")
                status = status.replace("]", "")
                status = status.replace(",", "")
                status_unpack = status.split(" ")
                self.status.clear()
                self.status = status_unpack[3:10]
                status = "status " + status
                self.data_sock.SendData(status)
        if data != None:  # if NEW data has been received since last ReadReceivedData function call
            self.get_logger().info('Publishing: "%s"' % data)
            dl = data.split(" ")
            if dl[0] == "SETHOME":
                print(self.narwhal.SetHome())
            elif dl[0] == "STOPPOSE":
                pass
            elif dl[0] == "GRIP":
                pass
            elif dl[0] == "FEN":
                pass
            elif dl[0] == "POSE":
                self.pose_lock_request(int(dl[1]))
            elif dl[0] == "CLUSTER":
                self.cluster_lock_request(int(dl[1]),int(dl[2]))
            elif dl[0] == "REQUESTFEN":
                self.bestmove_request()
                self.get = True
            elif dl[0] == "SET":
                if dl[1] == "q1":
                    self.q1 = float(dl[2])
                elif dl[1] == "q2":
                    self.q2 = float(dl[2])
                elif dl[1] == "q3":
                    self.q3 = float(dl[2])
                elif dl[1] == "q4":
                    self.q4 = float(dl[2])
                    print(type(self.q4))
                self.narwhal.SetJoint(self.q1,self.q2,self.q3,self.q4)
                # for i in range(4):
                #     if abs(all_jt[i]) >=0:
                #         self.narwhal.JogJoint(*all_jt[0:4])
                #         break
                # for i in range(4,7):
                #     if abs(all_jt[i]) >=0:
                #         self.narwhal.JogTask(*all_jt[4:7])
                #         break
            elif dl[0] == "JOG":
                name = ['q1', 'q2', 'q3', 'q4', 'x', 'y', 'z']
                all_jt = [0, 0, 0, 0, 0, 0, 0]  # j1j2j3j4xyz
                all_jt[name.index(dl[1])] = float(dl[2])
                for i in range(4):
                    if(abs(all_jt[i]) >=0):
                        self.narwhal.JogJoint(*all_jt[0:4])
                        break
                else:
                    for i in range(4,7):
                        if (abs(all_jt[i]) >= 0):
                            self.narwhal.JogTask(*all_jt[4:7])
                            break

                # if abs(all_jt[0]) >= 0 or abs(all_jt[1]) >= 0 or abs(all_jt[2]) >= 0 or abs(all_jt[3]) >= 0:
                #     self.narwhal.JogJoint(all_jt[0], all_jt[1], all_jt[2], all_jt[3])
                # elif abs(all_jt[4]) >= 0 or abs(all_jt[5]) >= 0 or abs(all_jt[6]) >= 0:
                #     self.narwhal.JogTask(all_jt[4], all_jt[5], all_jt[6])

def main(args=None):
    rclpy.init(args=args)

    game_controller = GameController()

    rclpy.spin(game_controller)



    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    game_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()