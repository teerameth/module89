HERE=`pwd`
pip install pyclipper simplejson
# Install python-v4l2capture
sudo apt install libv4l-dev
cd ~
git clone https://github.com/jnohlgard/python-v4l2capture
cd ~/python-v4l2capture
python3 setup.py build
sudo python3 setup.py install
cd $HERE