import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection, LineCollection
import pyvista as pv

from discretediffgeo.utils import *

pts = np.array([[1,0,0],[0,1,0],[0,0,1]])
assert abs(SimplexMeasure(pts))==sqrt(3)/2

pts = np.array([[0,0], [1,0], [0,1]])
assert (CayleyMengerMatrix(pts) - np.array([[0,1.,1.,1.],[1.,0.,2.,1.],[1.,2.,0.,1.],[1.,1.,1.,0.]]) < 10e-15).all()

simplices = ((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
weighted_simplices = SimplicesMeasureDict(simplices, pts)
assert weighted_simplices == {(0,):1,(1,):1,(2,):1,(0,1):1,(0,2):1,(1,2):sqrt(2),(0,1,2):0.5}

simplices = MakeComplex(((0,1,2),(0,2,3)))
pts = np.array([[0,0],[1,0],[1,1],[0,1]])
weighted_simplices = SimplicesMeasureDict(simplices, pts)
assert weighted_simplices == {(0,): 1, (0, 1): 1.0, (0, 1, 2): 0.5,
(0, 2): sqrt(2), (0, 2, 3): 0.5, (0, 3): 1.0, (1,): 1, (1, 2): 1.0, (2,): 1, (2, 3): 1.0, (3,): 1}