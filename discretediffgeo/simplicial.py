import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, diags
import matplotlib.pyplot as plt

from math import sqrt, exp
from discretediffgeo import SimplexMeasure, random_pointcloud_sphere, alpha_shape_3D
from discretediffgeo.laplacians import *


def polydata_conn_list_to_array(facelist):
    """
    takes a pyvista polydata face list and returns a dict of the faces.
    """
    facedict = dict()
    facenum = 0
    while len(facelist) > 0:
        i = facelist[0]
        simplex = facelist[1:i+1]
        facelist = facelist[i+1:]
        facedict[frozenset(simplex)] = facenum
        facenum += 1
    return facedict


def complex_from_faces(facedict):
    from itertools import combinations
    """
    takes a dict of faces and iteratively adds the boundaries of the faces.
    """
    scomplex = [dict(), dict(), facedict]
    edgeset = set()
    for face in facedict.keys():
        for edge in combinations(face, 2):
            edge = frozenset(edge)
            edgeset.add(edge)
    index = 0
    for edge in sorted(edgeset):
        scomplex[1][edge] = index
        index += 1

    numnodes = max(frozenset.union(*list(facedict.keys())))
    for i in range(numnodes + 1):
        scomplex[0][frozenset([i])] = i
    return scomplex


def make_weight_matrix(simplices, points, function=None, oriented=False):
    '''
    For each dimension d, creates the weight matrix of the simplices of dimension d
    '''
    weights = []
    for nsimplices in simplices:
        num_nsimplices = len(nsimplices)
        nweights = np.zeros(num_nsimplices)
        for simplex in nsimplices.keys():
            i = nsimplices[simplex]
            measure = SimplexMeasure(points[list(simplex)])
            if oriented==False:
                measure = abs(measure)
            if function != None:
                measure = function(measure)
            nweights[i] = measure
        weights.append(diags(nweights))
    return weights


#TESTS
"test polydata_conn_list_to_array"
facelist = [3, 0, 1, 2, 3, 1, 2, 3]
facedict = polydata_conn_list_to_array(facelist)
assert facedict == dict([(frozenset([0, 1, 2]), 0), (frozenset([1, 2, 3]), 1)])

"test complex_from_faces"
faceddict = dict([(frozenset([0,1,2]),0)])
scomplex0 = [dict([(frozenset([0]),0),(frozenset([1]),1), (frozenset([2]),2)]), 
             dict([(frozenset([0,1]),0),(frozenset([0,2]),1),(frozenset([1,2]),2)]), 
             dict([(frozenset([0,1,2]),0)])]
scomplex1 = complex_from_faces(faceddict)
assert scomplex0 == scomplex1

"test for make_weight_matrix"
simplex={frozenset([0,1,2]):0}
points=np.array([[1,0,0],[0,1,0],[0,0,1]])
scomplex = complex_from_faces(simplex)
weights0 = [diags([1,1,1]), diags(sqrt(2)*np.array([1,1,1])),diags([sqrt(3)/2])]
weights1 = make_weight_matrix(scomplex, points)
i=0
for w0,w1 in zip(weights0,weights1):
    assert (w0 != w1).sum()==0

"test orient"
L = np.array([[1,1],[1,1]])
assert (orient(L) == np.array([[1,-1],[-1,1]])).all()
L = -L
assert (orient(L) == np.array([[1,-1],[-1,1]])).all()

