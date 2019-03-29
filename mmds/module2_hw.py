# Question 2: Consider three Web pages with the following links:
# Suppose we compute PageRank with a beta of 0.7, and we introduce the additional constraint that the sum of the PageRanks of the three pages must be 3, to handle the problem that otherwise any multiple of a solution will also be a solution. Compute the PageRanks a, b, and c of the three pages A, B, and C, respectively. Then, identify from the list below, the true statement.

A = np.array((0, 0, 0, 0.5, 0, 0, .5, 1, 1)).reshape(3, 3)
v_original = np.ones(3).reshape(-1, 1) / 3
beta = 0.7

for i in range(5):
    v = beta * A.dot(v) + (1 - beta) * np.ones(3).reshape(-1, 1) / 3

(a, b, c) = v * 3
a + b
b + c
a + c
a + b

####
# Question 3: Suppose we compute PageRank with β=0.85. Write the equations for the PageRanks a, b, and c of the three pages A, B, and C, respectively. Then, identify in the list below, one of the equations.
A = np.array((0,0,1, 0.5, 0, 0, 0.5, 1, 0)).reshape(3, 3)
v_original = np.ones(3).reshape(-1, 1) / 3
v = np.copy(v_original)
beta = .85

for i in range(100):
    v = beta * A.dot(v) + (1 - beta) * np.ones(3).reshape(-1, 1) / 3

(a, b, c) = v

np.allclose(.95*c, .9*b + .475*a)
np.allclose(.85*a, c + .15*b)
np.allclose(c, .9*b + .475*a)
np.allclose(a, .9*c + .05*b)

#####
# Question 4: Assuming no "taxation," compute the PageRanks a, b, and c of the three pages A, B, and C, using iteration, starting with the "0th" iteration where all three pages have rank a = b = c = 1. Compute as far as the 5th iteration, and also determine what the PageRanks are in the limit. Then, identify the true statement from the list below. null
A = np.array((0,0,1, 0.5, 0, 0, 0.5, 1, 0)).reshape(3, 3)
v_original = np.ones(3)
v = np.copy(v_original)

for i in range(5):
    v = A.dot(v)

(a, b, c) = v

np.allclose(b, 9.0 / 16)
np.allclose(c, 9.0 / 8)
np.allclose(c, 1.0)

for i in range(100):
    v = A.dot(v)

(a, b, c) = v
np.allclose(b, .5)
