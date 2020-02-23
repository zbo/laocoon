import tensorflow as tf

hello = tf.constant('Hello, TF!')
mylist = tf.Variable([3.142,3.201,4.019],tf.float16)
mylist_rank = tf.rank(mylist)
myzeros = tf.zeros([5,5,4])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(myzeros))
print(sess.run(mylist))
sess.close()