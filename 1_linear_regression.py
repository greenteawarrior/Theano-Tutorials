import theano
from theano import tensor as T
import numpy as np

# underlying truth
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# symbolic variable initialization
X = T.scalar()
Y = T.scalar()

# KNN things?
def model(X, w):
    return X * w 

# initialize model parameters
# theano.shared is the way to have hybrid variables
# i.e. whenever you have a param for your model
w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X, w)

# metric to be optimized by model
cost = T.mean(T.sqr(y - Y))

# theano typically used for gradient-based learning
gradient = T.grad(cost=cost, wrt=w)

# how to change parameter based on learning signal
# we have w; on next timestep timestep should be w - gradient * .01
updates = [[w, w - gradient * 0.01]]

# so we don't have to worry about CPU (usually 64 bit things) vs GPU types (usually 32 bit things)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# iterate through things to train your model :)
for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)
        
print w.get_value() #something around 2, which is what we made Y to be

