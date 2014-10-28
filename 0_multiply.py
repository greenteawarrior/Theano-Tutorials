import theano
from theano import tensor as T # theano's equivalent of numpy

# theano symbolic variable initialization
a = T.scalar()
b = T.scalar()

y = a * b # our model

# can be run either on a CPU or a GPU
multiply = theano.function(inputs=[a, b], outputs=y)

# use it!
print multiply(1, 2) #2
print multiply(3, 3) #9

