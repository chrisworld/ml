from theano import *
import theano.tensor as T

# Scalar Addition
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
print "scalar addition\n", f(2, 3)

# Matrix Addition
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print "matrix addition\n", f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

# Exercise
a = theano.tensor.vector()      # declare variable
out = a + a ** 10               # build symbolic expression
f = theano.function([a], out)   # compile function
print(f([0, 1, 2]))
b = theano.tensor.vector()
out1 = a ** 2 + b ** 2 + 2 * a * b
f = theano.function([a, b], out1)   # compile function
print(f([0, 1, 2], [0, 1, 2]))

# Function
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
print "log\n", logistic([[0, 1], [-1, -2]])

# Multiple Outputs
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f_diff = theano.function([a, b], [diff, abs_diff, diff_squared])
print "dif outputs\n", f_diff([[1, 1], [1, 1]], [[0, 1],[2, 3]])
