### Use CUDA 11.4 ###
pip install pillow pandas

### Protobuf Installation/Compilation ###
sudo apt install protobuf-compiler
cd ~/tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.

### Install COCO API ###
pip install Cython
cd ~/tensorflow/models/research
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/tensorflow/models/research/

### Install the Object Detection API ###
cd ~/tensorflow/models/research
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
pip install imutils opencv-contrib-python

conda create -n dope python=3.8.10
### RTX 30xx will error with CUDA 10.x -> Use 11.x instead ###
## CUDA 11.3 (Stable) ##
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
## PIP install ##
pip install -r requirements.txt

## Upgrade torchvision to 0.11
pip install --upgrade torchvision

## Serial udev rule (bind serial port as /dev/Narwhal)
sudo nano /etc/udev/rules.d/99-usb-serial.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="ANZ25FFJ", SYMLINK+="Narwhal"