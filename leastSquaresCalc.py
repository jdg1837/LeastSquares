
''' Name: 	Juan Diego Gonzalez German
	ID:	 	1001401837
	Date: 	04/12/2018'''

import numpy as np


def leastSquaresStudent(a, b) :

    ones = np.ones([a.shape[0], 1])
    A = np.reshape(a,(-1, 1))
    A2 = np.square(A)

    linA = np.append(ones, A, 1)
    linAt = np.transpose(linA)

    x_lin = np.dot(linAt,linA)
    x_lin = np.linalg.inv(x_lin)
    x_lin = np.dot(x_lin, linAt)
    x_lin = np.dot(x_lin, b)

    norm_l = np.linalg.norm(b - np.dot(linA, x_lin))

    qA = np.append(linA, A2, 1)
    qAt = np.transpose(qA)

    x_q = np.dot(qAt,qA)
    x_q = np.linalg.inv(x_q)
    x_q = np.dot(x_q, qAt)
    x_q = np.dot(x_q, b)

    norm_q = np.linalg.norm(b - np.dot(qA, x_q))

    return x_lin, norm_l, x_q, norm_q


