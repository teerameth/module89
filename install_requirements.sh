HERE=`pwd`
pip install pyclipper simplejson tensorflow-object-detection-api
# pip install tensorflow-gpu
# Install python-v4l2capture
sudo apt install libv4l-dev
cd ~
git clone https://github.com/jnohlgard/python-v4l2capture
cd ~/python-v4l2capture
python3 setup.py build
sudo python3 setup.py install
cd $HERE

cd /workspaces/isaac_ros-dev
rosdep install -i --from-path /workspaces/isaac_ros-dev/src/module89 --rosdistro foxy -y
colcon build --packages-select module89
. install/setup.bash
# Fix permission (executable 'chessboard_encoder_fake.py' not found on the libexec directory ... )
sudo chmod 777 /workspaces/isaac_ros-dev/install/module89/lib/module89/*.py
# ros2 launch module89 test.launch.py