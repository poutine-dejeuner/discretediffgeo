import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection, LineCollection
import pyvista as pv

from discretediffgeo import *

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

""" # dual complex test
dual_simplices, dual_pts = DualComplex(simplices, pts)
assert dual_simplices == [[0,1]]
assert dual_pts == np.array([[2/3,1/3],[1/3,2/3]], dtype=np.float64) """

"""tetrahèdre sans coloration"""
plotter = pv.Plotter()
pts = np.array([[-0.2,  0.8, -0.5],
                [-0.4,  0.6,  0.5],
                [ 0.4, -0.5,  0.6],
                [-0.9, -0.1,  0.1]])
faces_conn_list = np.array([[3, 0, 1, 2],
                            [3, 0, 1, 3],
                            [3, 1, 2, 3],
                            [3, 0, 2, 3]])
mesh = pv.PolyData(pts, faces_conn_list)
plotter.add_mesh(mesh, reset_camera=True)
plotter.show()

"""Sphère sans coloration"""
plotter = pv.Plotter()
pts = random_pointcloud_sphere()
verts, edges, faces = alpha_shape_3D(pts, alpha=10)
faces_conn_list = np.insert(faces, 0, 3, axis=1)
num_faces = faces.shape[0]
mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)
plotter.add_mesh(mesh, reset_camera=True)
plotter.show()

"""spher eigenvecteur"""
plotter = pv.Plotter()
sphere = pv.Sphere()
triangles = sphere.faces.reshape(1680,4)[:,1:4]
points = sphere.points
hl = HodgeLaplacians(triangles)
L = hl.getHodgeLaplacian(2).todense()
M = L==1
L = L-2*M
eigfun = np.linalg.eigh(L)[1][:,0]
plotter.add_mesh(mesh=sphere, scalars=eigfun)
plotter.show()


def SimplicialComplexEigFunPlots(simplices, nodescoord, oriented, laplacians=None,
                                 plotsindices=[0, 1, 2, 3]):
    """
    simplices: an iterable of triples of integer indices.
    nodescoord: a list of coordinates with the indices of the entries
    corresponding to the indices of the nodes in simplices.
    plotsnumber: the number of plots to print for each dimension 0,1,2

    Prints matplotlib plots of the first k = plotnumbers eigenfunction
    of the simplicial complex laplacian in dimension n = 0,1,2.
    """
    if laplacians == None:
        hl = HodgeLaplacians(simplices, oriented=oriented)
    else:
        hl = laplacians
    pts = nodescoord
    xmax = max(nodescoord[:, 0])
    xmin = min(nodescoord[:, 0])
    ymax = max(nodescoord[:, 1])
    ymin = min(nodescoord[:, 1])

    # triangles
    eigvec = hl.getHodgeSpectrum(2, dense=True)[1]
    eigvec = eigvec[:, plotsindices]
    vmax = eigvec.max()
    vmin = eigvec.min()
    plt.figure(figsize=(24, 3))
    for i in range(len(plotsindices)):
        ax = plt.subplot(1, len(plotsindices), i+1)
        ax.triplot(pts[:, 0], pts[:, 1], hl.n_faces(2))
        patches = []
        for triangle in hl.n_faces(2):
            triangle_nodes = pts[list(triangle)]
            polygon = matplotlib.patches.Polygon(triangle_nodes, True)
            patches.append(polygon)
        p = PatchCollection(patches)
        p.set_array(eigvec[:, i])
        p.set_clim(vmin=vmin, vmax=vmax)
        ax.add_collection(p)
        plt.colorbar(p)
        plt.box(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.axis('off')

    # lines
    if True:
        linescoord = list(pts[list(paire)] for paire in hl.n_faces(1))
        plt.figure(figsize=(24, 3))
        plt.suptitle('Upper Laplacian eigenfunctions')
        L = np.array(hl.getHodgeLaplacianUp(1).todense())
        eigval, eigvec = np.linalg.eigh(L)
        mask = eigval > 10e-10
        mask = np.nonzero(mask)[0]
        eigvec = eigvec[:, mask]
        vmax = eigvec[:, 0:4].max()
        vmin = eigvec[:, 0:4].min()
        for i in range(4):
            ax = plt.subplot(1, 4, i+1)
            lines = LineCollection(linescoord, linewidths=3)
            lines.set_array(eigvec[:, i].flatten())
            lines.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(lines)
            plt.colorbar(lines)
            plt.box(False)
            plt.axis('off')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        plt.figure(figsize=(24, 3))
        plt.suptitle('Lower Laplacian eigenfunctions')
        L = np.array(hl.getHodgeLaplacianDown(1).todense())
        eigval, eigvec = np.linalg.eigh(L)
        mask = eigval > 10e-10
        mask = np.nonzero(mask)[0]
        eigvec = eigvec[:, mask]
        vmax = eigvec[:, 0:4].max()
        vmin = eigvec[:, 0:4].min()
        for i in range(4):
            ax = plt.subplot(1, 4, i+1)
            lines = LineCollection(linescoord, linewidths=3)
            lines.set_array(eigvec[:, i].flatten())
            lines.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(lines)
            plt.colorbar(lines)
            plt.box(False)
            plt.axis('off')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    else:
        linescoord = list(pts[list(paire)] for paire in hl.n_faces(1))
        plt.figure(figsize=(24, 3))
        eigvec = hl.getHodgeSpectrum(1, dense=True)[1]
        eigvec = eigvec[:, plotsindices]
        vmax = eigvec.max()
        vmin = eigvec.min()
        for i in range(len(plotsindices)):
            ax = plt.subplot(1, len(plotsindices), i+1)
            lines = LineCollection(linescoord, linewidths=3)
            lines.set_array(eigvec[:, i])
            lines.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(lines)
            plt.colorbar(lines)
            plt.box(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            plt.axis('off')

    # nodes
    plt.figure(figsize=(24, 3))
    eigvec = hl.getHodgeSpectrum(0, dense=True)[1]
    eigvec = eigvec[:, plotsindices]
    vmax = eigvec.max()
    vmin = eigvec.min()
    for i in range(len(plotsindices)):
        ax = plt.subplot(1, len(plotsindices), i+1)
        im = ax.scatter(pts[:, 0], pts[:, 1], c=eigvec[:, i],
                        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        plt.colorbar(im)
        plt.box(False)
        plt.axis('off')

def maketriangulation(tailleX, tailleY):
    triangles = []
    for i in range(tailleX-1):
        for j in range(tailleY-1):
            # (i, j), (i+1, j), (i, j+1)
            triangles.append((i + tailleX*j, i+1 + tailleX*j, i + tailleX*(j+1)))
            # (i+1, j), (i+1, j+1), (i, j+1)
            triangles.append((i+1 + tailleX*j, i+1+tailleX*(j+1), i + tailleX*(j+1)))
    pts = []
    for j in range(tailleY):
        for i in range(tailleX):
            pts.append((i, j))
    pts = np.array(pts)
    return triangles, pts

def round(arr, decimals=2):
    output = arr.copy()
    output = np.ceil(output*10**decimals)/(10**decimals)
    return output


# Icosahedron expt
plotter = pv.Plotter()
fig = plt.figure()
ax = plt.axes(projection="3d")
PHI = (1+sqrt(5))/2
pts=[]
for i in range(2):
    for j in range(2):
        pts = pts + [ (0, (-1)**i, (-1)**j*PHI),((-1)**i,(-1)**j*PHI,0), ((-1)**i*PHI,0,(-1)**j) ]
pts = np.array(pts)
#make_mesh_plot(pts)
vertices, edges, triangles = alpha_shape_3D(pts)
hl = HodgeLaplacians(triangles)
L = np.array(hl.getHodgeLaplacian(2).todense())
M = np.array(L>0)
E = np.bool_(np.eye(M.shape[0]))
E = np.invert(E)
MM = M*E
LL = L - 2*L*MM
eig = np.linalg.eigh(LL)
eigfun = eig[1]

for i in range(2):
    plotter = pv.Plotter()
    faces_conn_list = np.insert(triangles, 0, 3, axis=1)
    num_faces = len(triangles)
    mesh = pv.PolyData(pts[vertices], faces_conn_list, n_faces=num_faces)
    plotter.add_mesh(mesh, reset_camera=True, scalars=eigfun[:,i])
    plotter.show()

weighted_simplices = PointCloudSphereToWeightedComplex(pts)
whl = WeightedHodgeLaplacians(weighted_simplices)
L = np.array(whl.getHodgeLaplacian(2).todense())
print(L)
M = np.array(L>10e-15)
E = np.bool_(np.eye(M.shape[0]))
E = np.invert(E)
MM = M*E
LL = L - 2*L*MM
print()
print(round(LL))
eigfun = np.linalg.eigh(LL)[1][:,0]
faces_conn_list = np.insert(triangles, 0, 3, axis=1)
num_faces = len(triangles)
mesh = pv.PolyData(pts[vertices], faces_conn_list, n_faces=num_faces)
plotter.add_mesh(mesh, reset_camera=True, scalars=eigfun)
plotter.show()

'''
pts = random_pointcloud_sphere(600)
verts, edges, faces = alpha_shape_3D(pts, alpha=10)
weighted_simplices = PointCloudSphereToWeightedComplex(pts)
#print(weighted_simplices)
whl = WeightedHodgeLaplacians(weighted_simplices)
L = np.array(whl.getHodgeLaplacian(2).todense())
LL = orient(L)
print(LL)
eigfun = np.linalg.eig(LL)[1]
faces_conn_list = np.insert(faces, 0, 3, axis=1)
num_faces = len(faces)
mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)

for i in range(10):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, reset_camera=True, scalars=eigfun[:,i])
    plotter.show()
'''


#make_mesh_plot(random_pointcloud_sphere())
#laplacian_eigenfunction_3Dplot(random_pointcloud_sphere())
#laplacian_eigenfunction_3Dplot(regular_ptcloud_sphere())
'''Les eigenfunctions des laplaciens ont pas la bonne forme. ce serait probablement 
préférable d'utiliser le laplacien de WEIGHTED graph/complexe simplicial'''

"""
je sais pas si les methodes que j'utilise produisent vraiment une sphere, à l'intérieur
il y a peut-être des arrêtes et des triangles...
"""