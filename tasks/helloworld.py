import tensorflow as tf

hello = tf.constant('Hello, TF!')
mylist = tf.Variable([3.142,3.201,4.019],tf.float16)
mylist_rank = tf.rank(mylist)
myzeros = tf.zeros([2,2,3])
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(myzeros))
    print(sess.run(mylist))
    sess.run(tf.compat.v1.assign_add(mylist,[1,1,1]))
    print(sess.run(mylist))

with tf.compat.v1.Session() as sess:
    x = tf.compat.v1.placeholder(tf.int16, shape=(), name='x')
    y = tf.compat.v1.placeholder(tf.int16, shape=(), name='y')
    add = tf.add(x,y)
    mul = tf.multiply(x,y)
    print(sess.run(add, feed_dict={x:2,y:3}))
    print(sess.run(mul, feed_dict={x:2,y:3}))