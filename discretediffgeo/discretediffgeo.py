from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from hodgelaplacians import HodgeLaplacians, WeightedHodgeLaplacians

from collections import defaultdict
from math import factorial, sqrt
from itertools import combinations

import numpy as np
from numpy.linalg import norm, det
from scipy.spatial import Delaunay

import pyvista as pv


def alpha_shape_3D(pos, alpha):
    """
    Compute concave hull of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    
    from https://stackoverflow.com/questions/69110871/alpha-shapes-in-3d-follow-up
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles

def random_pointcloud_sphere():
    N = 600
    dim = 3

    norm = np.random.normal
    normal_deviates = norm(size=(dim, N))

    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    points = normal_deviates/radius
    points = points.transpose()
    return points

def regular_ptcloud_sphere():
    r = 3
    phi = np.linspace(0, np.pi, 18)
    theta = np.linspace(0, 2 * np.pi, 36)

    PHI, THETA = np.meshgrid(phi, theta)

    x = r * np.sin(PHI) * np.cos(THETA)
    y = r * np.sin(PHI) * np.sin(THETA)
    z = r * np.cos(PHI)
    
    pts = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1) + 10

    return np.unique(pts, axis=0) - 10

def random_sphere_mesh_3Dplot():
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    plotter = pv.Plotter()

    pts = random_pointcloud_sphere()
    verts, edges, faces = alpha_shape_3D(pts, alpha=10)

    faces_conn_list = np.insert(faces, 0, 3, axis=1)
    num_faces = faces.shape[0]

    mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)

    plotter.add_mesh(mesh, reset_camera=True)
    plotter.show()

def make_orientation(triangles):
    '''TODO'''
    hl = HodgeLaplacians(triangles)
    L = hl.getHodgeLaplacian(2)
    faces = hl.n_faces(2)

def orient(L):
    M = L==1
    L = L - 2*M
    return L

def laplacian_eigenfunction_3Dplot(pts):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    plotter = pv.Plotter()

    verts, edges, faces = alpha_shape_3D(pts, alpha=10)

    hl = HodgeLaplacians(faces)
    faces = hl.n_faces(2)
    L = hl.getHodgeLaplacian(2).todense()
    L = orient(L)
    eigfun = np.linalg.eigh(L)
    eigfun = eigfun[1][:,0]

    faces_conn_list = np.insert(faces, 0, 3, axis=1)
    num_faces = len(faces)

    mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)

    plotter.add_mesh(mesh, reset_camera=True, scalars=eigfun)
    plotter.show()

def make_mesh_plot(pts):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #plt.xlim(0,2)
    plotter = pv.Plotter()
    verts, edges, faces = alpha_shape_3D(pts, alpha=10)
    faces_conn_list = np.insert(faces, 0, 3, axis=1)
    num_faces = faces.shape[0]
    mesh = pv.PolyData(pts[verts], faces_conn_list, n_faces=num_faces)
    plotter.add_mesh(mesh, reset_camera=True)
    plotter.show()

def PointCloudSphereToWeightedComplex(points):
    vertices, edges, faces = alpha_shape_3D(points, alpha=10)
    vertices = tuple((i,) for i in tuple(vertices))
    vertices = SimplicesMeasureDict(vertices, points)
    edges = SimplicesMeasureDict(tuple(edges), points)
    faces = SimplicesMeasureDict(tuple(faces), points)
    simplices = dict()
    simplices.update(faces)
    simplices.update(edges)
    simplices.update(vertices)
    return simplices

def SimplicesMeasureDict(simplices, point_coord):
    weighted_simplices = dict()
    for simplex in simplices:
        if len(simplex) == 1:
            weighted_simplices[simplex] = 1
        else:
            simplex_points = point_coord[list(simplex),:]
            weight = SimplexMeasure(simplex_points)
            weighted_simplices[tuple(simplex)] = weight
    return weighted_simplices
        
def CayleyMengerMatrix(points):
    n = points.shape[0]
    CM = np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(i+1, n):
            CM[i,j] = norm(points[i,:] - points[j,:])**2
    CM[0:n,n] = np.ones(n)
    CM = CM + CM.transpose()
    return CM

def SimplexMeasure(points):
    n = points.shape[0]-1
    CM = CayleyMengerMatrix(points)
    measuresquared = (-1)**(n+1)/(factorial(n)**2 * 2**n)*det(CM)
    measure = sqrt(measuresquared)
    return measure

def MakeComplex(simplices):
    faceset = set()
    for simplex in simplices:
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                    faceset.add(tuple(face))
    return tuple(sorted(faceset))