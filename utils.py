+from constants import *
 +import tensorflow as tf
 +
 +#define different loss functions
 +#option1 mse
 +#option2 KL distance
 +#todo add discriminative objective for vocal + instrumental case
 +def loss_function(option, output, batchY):
 +
 +    if (option == 1):
 +        sub = tf.subtract(output, batchY)
 +        loss = tf.sqrt(tf.abs(sub))
 +    elif (option == 2):
 +        loss = d_measure(batchY, output)
 +
 +    return loss
 +
 +#compute KL distance of input tensors
 +def d_measure(A, B):
 +    c1 = tf.multiply(A, tf.log(tf.div(A,B)) )
 +    c2 = tf.subtract(B, A)
 +    s = tf.summary(c1 + c2)
 +
 +    return s
