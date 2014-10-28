import theano
from theano import tensor as T
import numpy as np
from load import mnist

# takes care of conversions to make your stuff theano-friendly: float32 or float64
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01)) #initialiaze to a some gaussian

# our model in matrix format
def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

# training matrices
trX, teX, trY, teY = mnist(onehot=True) #one hot encoding!

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((784, 10))

# probability of the labels, given the input
py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

# categorical cross entropy is basically telling us to maximize the value that's true
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) 
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

# training function tells us how well the model's doing given an input pair
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)

# allows us to get actual outputs from the model
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)): # train on mini-batches of 128
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))


# template matching won't work too well... b/c handwriting has flexibility and such....
# this model can't handle the complexity of the handwriting dataset
