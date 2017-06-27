import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
node1 = tf.constant([3.0])
node2 = tf.constant([4.0])
node3 = node1 + node2
node4 = tf.add(node1,node3)
sess = tf.Session()

"""print(50*"*-")
print (sess.run(node3*node4))
print (node1,node4)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b
mulp_node = a*b

print (sess.run(adder_node,{a:[3.0,3.0],b:[4.0,5.0]}))
print (sess.run(mulp_node,{a:3,b:5}))"""

M = tf.Variable([.3], dtype= tf.float32)
c = tf.Variable([-.3], dtype= tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = M * x + c



loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	sess.run(train,{x:x_train, y:y_train})

print (sess.run([M,c]))

##eval

print(sess.run(linear_model, {x:x_eval}))
print(sess.run(loss, {x:x_eval, y:y_eval}))
print (y_eval)






####using the built in logistic regressor modle with contrib.learn
"""
feat = [tf.contrib.layers.real_valued_columns("x", dimension = 1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns = feat)
x_tr = np.array([1., 2., 3., 4.])
y_tr = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x" :x_tr, y_tr,batch})


"""









