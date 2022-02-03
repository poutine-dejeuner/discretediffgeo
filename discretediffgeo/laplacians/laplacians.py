import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, diags
import matplotlib.pyplot as plt

from discretediffgeo import SimplexMeasure


def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.

    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension: laplacian of degree i is
       in the i-th position
    """
    laplacians = list()
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(down))
    return laplacians


def build_up_down_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.

    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension: laplacian of degree i is
       in the i-th position
    """
    laplacians = list()
    ups = []
    downs = [[]]
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    ups.append(up)
    laplacians.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
        ups.append(coo_matrix(up))
        downs.append(coo_matrix(down))
    down = boundaries[-1].T @ boundaries[-1]
    downs.append(down)
    laplacians.append(coo_matrix(down))
    return ups, downs, laplacians


def build_unweighted_boundaries(simplices):
    """Build unweighted boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                             dtype=np.float32,
                             shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)

    return boundaries


def build_weighted_boundaries(simplices, weights):
    """Build weighthd boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.
    weights: list of sparse matrices
        List of sparse matrices for each dimension which diagonal contains the wights
    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                     dtype = np.float32,
                     shape = (len(simplices[d-1]), len(simplices[d])))

        Wn1 = weights[d-1]
        w = weights[d].data[0]
        nz = np.nonzero(w)[0]
        inv = np.zeros(len(w))
        inv[nz] = (1/(w[nz]))
        inv_Wn = diags(inv) 
        boundary = Wn1@boundary@inv_Wn
        boundaries.append(boundary)

    return boundaries


def build_boundaries(simplices, weights=False):
    """Build weighted or unweighted boundary operators from a list of simplices.
        Default: unweighted
    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.
    weights:None or list of sparse matrices, default None
        List of sparse matrices for each dimension which diagonal contains the wights
    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    if weights is False:
        boundaries = build_unweighted_boundaries(simplices)

    else:
        boundaries = build_weighted_boundaries(simplices, weights=weights)

    return boundaries


def plot_eigen(laplacian, plot, n_eigenvectors=4):
    values, vectors = sparse.linalg.eigsh(laplacian, n_eigenvectors, which='SM')
    fix, axes = plt.subplots(1, n_eigenvectors+3, figsize=(20, 3))
    axes[0].plot(values, '.')
    axes[1].scatter(vectors[:, 0], vectors[:, 1])
    axes[2].scatter(vectors[:, 2], vectors[:, 3])
    for i in range(n_eigenvectors):
        plot(vectors[:, i], axes[i+3])
        axes[i+3].axis('off')


def orient(L):
    '''Takes a matrix and sets all nondiagonal elements to negative numbers
    with the same absolute value. This is how a Laplacian matrix with coherent
    orientations in the top dimension looks like.

    Hack, replace with something that actually orients the complex. '''

    eye = np.bool8(np.eye(L.shape[0]))
    L0 = L*eye
    L0 = np.abs(L0)
    L1 = L*np.invert(eye)
    L1 = np.abs(L1)
    LL = L0 - L1
    return LL

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