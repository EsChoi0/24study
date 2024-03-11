import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices() )

# Python 3.8.1
# cuda 11.8
# cudnn 8.6
# tensorflow 2.8.0