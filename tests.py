import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from discretediffgeo import *

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

 # plotter = pv.Plotter()
# sphere = pv.Sphere()
# triangles = sphere.faces.reshape(1680,4)[:,1:4]
# points = sphere.points
# hl = HodgeLaplacians(triangles)
# L = hl.getHodgeLaplacian(2).todense()
# M = L==1
# L = L-2*M
# eigfun = np.linalg.eigh(L)[1][:,0]
# plotter.add_mesh(mesh=sphere, scalars=eigfun)
# plotter.show()


fig = plt.figure()
ax = plt.axes(projection="3d")
plotter = pv.Plotter()

pts = random_pointcloud_sphere()
verts, edges, faces = alpha_shape_3D(pts, alpha=10)
weighted_simplices = PointCloudSphereToWeightedComplex(pts)
#print(weighted_simplices)
whl = WeightedHodgeLaplacians(weighted_simplices)
L = whl.getHodgeLaplacian(2).todense()
#M = L>0
#L = L-2*M*L
eigfun = np.linalg.eigh(L)[1][:,0]
faces_conn_list = np.insert(faces, 0, 3, axis=1)
num_faces = len(faces)
mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)
plotter.add_mesh(mesh, reset_camera=True, scalars=eigfun)
plotter.show()


#make_mesh_plot(random_pointcloud_sphere())
#laplacian_eigenfunction_3Dplot(random_pointcloud_sphere())
#laplacian_eigenfunction_3Dplot(regular_ptcloud_sphere())
'''Les eigenfunctions des laplaciens ont pas la bonne forme. ce serait probablement 
préférable d'utiliser le laplacien de WEIGHTED graph/complexe simplicial'''