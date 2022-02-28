#!/bin/bash

# Build ROS dependency
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
echo "alias res=\"cd /workspaces/isaac_ros-dev && rosdep install -i --from-path /workspaces/isaac_ros-dev/src/module89 --rosdistro foxy -y\"" >> ~/.bashrc
echo "alias build=\"cd /workspaces/isaac_ros-dev && colcon build --packages-select module89 \"" >> ~/.bashrc
echo "alias ins=\"cd /workspaces/isaac_ros-dev && . /workspaces/isaac_ros-dev/install/setup.bash\"" >> ~/.bashrc
source ~/.bashrc

sudo apt-get update
rosdep update

### Install development tools and ROS tools ###
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  libbullet-dev \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-pip \
  python3-pytest-cov \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  wget

sudo apt-get install -y git-lfs	# Git for large file
cd ~ || exit 1
# Determine CPU architecture
PLATFORM="$(uname -m)"
# Install Github CLI
echo "Get Github CLI tool"
if [[ $PLATFORM == "x86_64" ]]; then
  wget https://github.com/cli/cli/releases/download/v2.5.1/gh_2.5.1_linux_amd64.deb
  sudo dpkg -i gh_2.5.1_linux_amd64.deb
fi

if [[ "$PLATFORM" == "aarch64" ]]; then
  wget https://github.com/cli/cli/releases/download/v2.5.1/gh_2.5.1_linux_arm64.deb
  sudo dpkg -i gh_2.5.1_linux_arm64.deb
fi
## Login with token ##
gh auth login --with-token "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"
## Create src folder if not exist ##
if [[ ! -d "/workspaces/isaac_ros-dev/src" ]]; then
  mkdir -p /workspaces/isaac_ros-dev/src
fi
## Clone repositories ##
echo "Cloning repositories of module89 dependencies ..."
#if [[ ! -d "/workspaces/isaac_ros-dev/src/module89" ]]; then
#  cd /workspaces/isaac_ros-dev/src/ || exit 1
#  git clone https://github.com/teerameth/module89
#  cd ..
#fi

if [[ ! -d "/workspaces/isaac_ros-dev/src/isaac_ros_common" ]]; then
  cd /workspaces/isaac_ros-dev/src/ || exit 1
  git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
  cd /workspaces/isaac_ros-dev/src/isaac_ros_common || exit 1
  git lfs pull && cd ..
fi

if [[ ! -d "/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference" ]]; then
  cd /workspaces/isaac_ros-dev/src/ || exit 1
  git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
  cd /workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference || exit 1
  git lfs pull && cd ..
fi

if [[ ! -d "/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation" ]]; then
  cd /workspaces/isaac_ros-dev/src/ || exit 1
  git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation
  cd /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation || exit 1
  git lfs pull && cd ..
fi

if [[ ! -d "/workspaces/isaac_ros-dev/src/image_pipeline" ]]; then
  cd /workspaces/isaac_ros-dev/src/ || exit 1
  git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
fi

echo "=== Install library for module89 ==="
pip install pyclipper simplejson tensorflow-object-detection-api
sudo apt install -y libv4l-dev
## Install python-v4l2 library from source
cd ~ || exit 1
sudo apt-get install -y libv4l-dev
git clone https://github.com/jnohlgard/python-v4l2capture
cd ~/python-v4l2capture || exit 1
python3 setup.py build
sudo python3 setup.py install

## Build & Install module89 ##
cd /workspaces/isaac_ros-dev || exit 1
echo "Install dependencies of module89 ..."
rosdep install -i --from-path /workspaces/isaac_ros-dev/src/module89 --rosdistro foxy -y
echo "Building module89 ..."
colcon build --packages-up-to module89
echo "Install module89 ..."
. /workspaces/isaac_ros-dev/install/setup.bash
# Fix permission (executable 'chessboard_encoder_fake.py' not found on the libexec directory ... )
echo "Fix executable permission of *.py"
sudo chmod 777 -R /workspaces/isaac_ros-dev/
# sudo chmod 777 /workspaces/isaac_ros-dev/install/module89/lib/module89/*.py

arch=$(uname -i)
if [[ $arch == aarch64 ]]; then
  # OpenCV (Build from source) [arrch64 only]
  echo "Build OpenCV from source"
  export VERSION=4.2.0
  cd ~ || exit 1
  wget -O opencv.zip https://github.com/opencv/opencv/archive/$VERSION.zip
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$VERSION.zip
  unzip opencv.zip
  unzip opencv_contrib.zip
  mv opencv-$VERSION opencv
  mv opencv_contrib-$VERSION opencv_contrib
  rm opencv.zip
  rm opencv_contrib.zip
  wget https://bootstrap.pypa.io/get-pip.py
  sudo python3 get-pip.py
  pip install numpy imutils
  cd opencv
  mkdir build && cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=OFF \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D PYTHON_EXECUTABLE=/usr/bin/python3 \
      -D BUILD_EXAMPLES=ON ..
  make -j$(nproc)
  sudo make install
  sudo ldconfig
fi

# Setup DOPE
echo "Build & Install DOPE"
cd /workspaces/isaac_ros-dev
colcon build --packages-up-to isaac_ros_dope
chmod +x /workspaces/isaac_ros-dev/install/setup.bash
. /workspaces/isaac_ros-dev/install/setup.bash
echo "Build & Install module89"
rosdep install -i --from-path /workspaces/isaac_ros-dev/src/module89 --rosdistro foxy -y
colcon build --packages-up-to module89
. /workspaces/isaac_ros-dev/install/setup.bash

$@
