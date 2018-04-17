import numpy as np
import tensorflow as tf
import qdDataset as qd;
from PIL import Image

data = qd.QdDataset();
data.read_data();


x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,3]));
b = tf.Variable(tf.zeros([3]));

y = tf.nn.softmax(tf.matmul(x,W) + b);

y_


# ds = tf.data.Dataset.from_tensor_slices(testImages);
# print(ds);


# ind = 77;

# p1 = data.train.images[ind];
# lable1 = data.train.lables[ind];

# n = np.size(p1,0);
# s = int(np.sqrt(n));

# img = Image.new("L", (s,s));

# for i in range(n):
	# x = i % s;
	# y = i // s;
	# img.putpixel((x,y),255-int(p1[i]));
# img.show();
# print(data.getLable(lable1));