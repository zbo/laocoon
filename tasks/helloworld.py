import tensorflow as tf

hello = tf.constant('Hello, TF!')
sess = tf.Session()

print(sess.run(hello))
sess.close()