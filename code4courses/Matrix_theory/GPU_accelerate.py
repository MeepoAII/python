import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

#输入的图像矩阵：下面每行数据对应图像矩阵的行；每行中的每组值，分别对应0,1,2三个通道相应位置的值
input = tf.constant([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
 [[0,0,0],[0,0,1],[1,0,0],[2,1,1],[0,0,2],[1,0,2],[0,0,0]],
 [[0,0,0],[2,0,0],[0,1,1],[1,1,2],[0,2,0],[0,1,1],[0,0,0]],
 [[0,0,0],[0,0,0],[0,2,1],[2,1,0],[2,0,2],[1,0,2],[0,0,0]],
 [[0,0,0],[2,2,0],[2,1,1],[0,2,1],[1,1,2],[0,0,0],[0,0,0]],
 [[0,0,0],[1,2,2],[2,1,2],[0,1,2],[0,1,1],[0,2,1],[0,0,0]],
 [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],shape=[1,7,7,3],dtype=tf.float32)

bias = tf.constant([1,0],shape=[2],dtype=tf.float32)

filter = tf.constant([[[[-1,1],[0,0],[0,1]],
  [[0,1],[-1,-1],[-1,0]],
  [[1,0],[-1,0],[1,0]]],
 [[[0,-1],[-1,-1],[0,-1]],
  [[1,-1],[1,0],[0,-1]],
  [[0,0],[-1,-1],[-1,0]]],
 [[[0,1],[0,-1],[0,1]],
  [[-1,1],[-1,-1],[1,-1]],
  [[-1,-1],[1,1],[-1,-1]]]],  shape=[3,3,3,2],dtype=tf.float32)

# op1 = tf.nn.conv2d(input,filter,strides = [1,2,2,1],padding ='VALID')+bias
# op2 = tf.nn.conv2d(input,filter,strides = [1,2,2,1],padding = 'SAME')+bias
# with tf.Session() as sess:
#     result1 = sess.run(op1)
#     result2 = sess.run(op2)
#     print(sess.run(input))
#     print(sess.run(filter))
#     print(result1)
# print('###############')

image_raw_data = tf.gfile.FastGFile('/home/lab-1/Downloads/meepo.jpg', 'rb').read()
image = tf.image.decode_jpeg(image_raw_data, channels=3)
image = tf.to_float(image, name='ToFloat')
image = tf.image.resize_images(image, [512,512],method=0)
batch_shape = (1,512,512,3)
image = tf.reshape(image,batch_shape)

start = datetime.datetime.now()

for i in range(8):
    op1 = tf.nn.conv2d(image, filter, strides=[1, 3, 3, 3], padding='VALID') + bias

end = datetime.datetime.now()
print("Time used: ", end-start)

