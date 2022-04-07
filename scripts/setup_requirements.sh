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