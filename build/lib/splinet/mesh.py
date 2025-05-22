import numpy as np

from .aux import frozen, freeze, abs_tuple, _return_sorted_boundary_vertices
from .template import MultiPatchTemplate
from .network import EdgeNetwork
from .index import as_edge

from functools import cached_property, lru_cache, wraps
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import orient
import pygmsh
from scipy.sparse import linalg as splinalg

from mapping.sol import make_matrix, transfinite_interpolation, forward_laplace
from mapping import pc

from numba import njit, prange
from numba.typed import List
from numba.core import types

i64 = types.int64
f64 = types.float64

unitpl = types.UniTuple(types.int64, 2)
int_array = types.int64[:]
int_dict = types.DictType(unitpl, int_array)
int_set = types.Set(i64)

_ = np.newaxis


""" Piecewise - linear jitted mesh evaluation """


@njit(cache=True)
def _batches_(triangles, points, nx, ny):
  """
    Divide the trinangles in `triangles` into batches { (i, j): list of triangles with x in (xvals[i], xvals[i+1]) and y in (yvals[i], yvals[i+1]) },
    where xvals = linspace(points.T[0].min(), points.T[0].max(), nx+1)
          yvals = linspace(points.T[1].min(), points.T[1].max(), ny+1)
    
    returns
    -------
    batches : :class: numba.typed.Dict containing the batches
    xvals, yvals : :class: np.ndarray, the breakpoints that divide the pointcloud / trianlges
  """
  x, y = points.T
  xvals = np.linspace(x.min(), x.max(), nx+1)
  yvals = np.linspace(y.min(), y.max(), ny+1)

  batches = {}
  for i, tri in enumerate(triangles):
    x, y = points[tri].T
    myxmin, myxmax, myymin, myymax = x.min(), x.max(), y.min(), y.max()
    imin, imax = find_interval(xvals, myxmin), find_interval(xvals, myxmax)
    jmin, jmax = find_interval(yvals, myymin), find_interval(yvals, myymax)
    for k in range(imin, imax+1):
      for L in range(jmin, jmax+1):
        if (k, L) not in batches:
          batches[(k, L)] = List((i,))
        else:
          batches[(k, L)].append(i)

  return batches, xvals, yvals


@njit(cache=True)
def find_interval(values, x):
  return min(max(np.searchsorted(values, x) - 1, 0), len(values) - 2)


@njit(cache=True)
def _batches(triangles, points, xvals, yvals):
  x, y = points.T

  batches = [List.empty_list(int_set), List.empty_list(int_set)]
  for _ in range(len(xvals)-1):
    myset = set([1])
    myset.pop()
    batches[0].append(myset)
  for _ in range(len(yvals)-1):
    myset = set([1])
    myset.pop()
    batches[1].append(myset)

  for i, tri in enumerate(triangles):
    x, y = points[tri].T
    myxmin, myxmax, myymin, myymax = x.min(), x.max(), y.min(), y.max()
    imin, imax = find_interval(xvals, myxmin), find_interval(xvals, myxmax)
    jmin, jmax = find_interval(yvals, myymin), find_interval(yvals, myymax)
    for k in range(imin, imax+1):
      batches[0][k].update({i})
    for k in range(jmin, jmax+1):
      batches[1][k].update({i})

  return batches


@njit(cache=True)
def _find_simplex(triangles: i64[:, :], points: f64[:, :], boundary_indices: i64[:], batches, xvals: f64[:], yvals: f64[:], xi: f64[:], tol: f64):
  """ Find the simplex index that contains the value `xi`.
      
      returns
      -------
      index : :class: `int64`
        If a simplex that contains `xi` is found, this is the index, else defaults to 0
      localweights : :class: `np.ndarray[f64]`
        The coordinates of `xi` in the barycentric coordinate system defined by the triangle's vertices. Defaults to [0, 0, 0]
        if no viable triangles is found.
  """
  i, j = find_interval(xvals, xi[0]), find_interval(yvals, xi[1])  # find the x and y intervals that contain xi
  candidates = batches[0][i] & batches[1][j]  # get the candidate triangles that may contain xi
  for candidate in candidates:  # find the exact triangle and return xi's barycentric coordinates
    a, b, c = points[triangles[candidate]]
    s, t = np.linalg.solve(np.stack((b - a, c - a), axis=1), xi - a)
    r = 1 - s - t
    if r >= -tol and s >= -tol and t >= -tol:
      return candidate, np.array([r, s, t], dtype=np.float64)
  else:  # not found => return the closest simplex
    bpoints = points[boundary_indices]
    distances = np.empty((len(bpoints),), dtype=np.float64)
    for i in range(len(distances)):
      distances[i] = ((bpoints[i] - xi)**2).sum()
    argmin = np.argmin(distances)
    xi = bpoints[argmin]
    i, j = find_interval(xvals, xi[0]), find_interval(yvals, xi[1])  # find the x and y intervals that contain xi
    candidates = batches[0][i] & batches[1][j]  # get the candidate triangles that may contain xi
    for candidate in candidates:  # find the exact triangle and return xi's barycentric coordinates
      a, b, c = points[triangles[candidate]]
      s, t = np.linalg.solve(np.stack((b - a, c - a), axis=1), xi - a)
      r = 1 - s - t
      if r >= -tol and s >= -tol and t >= -tol:
        return candidate, np.array([r, s, t], dtype=np.float64)
    else:
      return 0, np.array([0, 0, 0], dtype=np.float64)


@njit(cache=True, parallel=True)
def _find_simplices(triangles, points, boundary_indices, batches, xvals, yvals, Xi, tol):
  """ Vectorised version of `_find_simplex`. """
  ret = np.empty(Xi.shape[:1], dtype=i64)
  for i in prange(len(Xi)):
    xi = Xi[i]
    ret[i] = _find_simplex(triangles, points, boundary_indices, batches, xvals, yvals, xi, tol)[0]
  return ret


@njit(cache=True)
def _evaluate(triangles, points, boundary_indices, batches, xvals, yvals, tol, weights, Xi):
  """ Evaluate a function define by `weights` on the triangulation defined by
      `triangles` and `points`.

      Parameters
      ----------
      batches : :class: `numba.typed.Dict`
        The batch dictionary constructed using `_batches(...)`.
      tol : :class: `f64`
        The tolerance >= 0 << 1 for considering a point inside a simplex.
      weights : :class: `f64[:, :]`
        The weights of the vector-value function. If the function is scalar-valued, just take weight.shape[1:] == (1,)
      Xi : :class: `f64[:, :]`
        The evaluation abscissae. Of shape (npoints, 2)
  """

  # XXX: parallelise. Test carefully.

  ret = np.empty(Xi.shape[:1] + weights.shape[1:], dtype=np.float64)
  for i, xi in enumerate(Xi):
    itri, localweights = _find_simplex(triangles, points, boundary_indices, batches, xvals, yvals, xi, tol)
    for j, myweights in enumerate(weights.T):
      w = myweights[triangles[itri]]
      ret[i, j] = (localweights * w).sum()
  return ret


def _sample_network(network: EdgeNetwork, npoints):
  """
     Uniformly sample `npoints` from a one-faced
     network's edges.
  """
  assert len(network.face_indices) == 1
  indices, = network.face_indices.values()
  edges = network.edges

  if isinstance(npoints, int):
    npoints = [npoints] * len(indices)

  npoints = np.asarray(npoints, dtype=int)
  assert len(npoints) == len(indices)
  new_edges = [edge.toPointCloud().toInterpolatedUnivariateSpline(k=1)(np.linspace(0, 1, n)) for edge, n in zip(edges, npoints)]

  return network.edit(edges=new_edges)


def _from_template(template: MultiPatchTemplate, npoints, **kwargs):
  """ Create Triangulation from template.
      Parameters
      ----------
      template : :class: MultiPatchTemplate
        the template to be sampled
      npoints : :class: [int, Sequence[int]]
        number of points to sample from each edge.
        Must be :class: `int` or a :class: Sequence of :class: `int`.
        If of :class: `int`, repeat the value n_boundary_edges times.
      kwargs: forwarded to Triangulation.from_polygon

      Returns
      -------
      tri : :class: Triangulation
  """
  obe, = template.ordered_boundary_edges

  if isinstance(npoints, int):
    npoints = [npoints] * len(obe)

  npoints = np.asarray(npoints, dtype=int)
  assert len(npoints) == len(obe)

  patchverts = np.asarray(template.patcherts)
  X = np.concatenate([ pc.PointCloud(patchverts[list(edge)]).toInterpolatedUnivariateSpline(k=1)(np.linspace(0, 1, n)) for edge, n in zip(obe, npoints) ])

  return Triangulation.from_polygon(X, **kwargs)


@lru_cache
def _floater(tri, X, shape):
  X = np.frombuffer(X, dtype=float).reshape(shape)
  freezeindices = tri.ordered_boundary_vertex_indices
  dofindices = tri.interior_vertex_indices
  A, B = tri._A_B
  points = np.empty(tri.points.shape, dtype=np.float64)
  points[freezeindices] = X
  points[dofindices] = A.solve(B @ X.T.ravel()).reshape(2, -1).T
  return Triangulation(tri.triangles, points)


class Triangulation:

  @classmethod
  def from_gmsh(cls, mesh, **kwargs):
    return cls(mesh.cells_dict['triangle'], mesh.points[:, :2], **kwargs)

  @staticmethod
  def from_polygon(X, **kwargs):
    return make_mesh(X, **kwargs)

  @staticmethod
  def from_template(*args, **kwargs):
    """ See `_from_template`. """
    return _from_template(*args, **kwargs)

  @staticmethod
  def from_sampled_network(network, *args, **kwargs):
    return Triangulation.from_network(_sample_network(network, *args, **kwargs))

  @classmethod
  def from_network(cls, network: EdgeNetwork, **kwargs):
    """ This routine is cached so that the same face is not remeshed
        each time.
    """
    assert len(network.face_indices) == 1
    indices, = network.face_indices.values()
    edges = network.get_edges(indices)
    X = np.concatenate([edge.points[:-1] for edge in edges])
    return make_mesh(X, **kwargs)

  def __init__(self, triangles, points):
    self.triangles = frozen(triangles, dtype=np.int64)
    self.points = frozen(points, dtype=np.float64)
    assert self.triangles.shape[1:] == (3,)
    assert self.points.shape[1:] == (2,)
    assert set(np.unique(self.triangles)) == set(range(len(self.points)))

  @cached_property
  def batches(self):
    _xvals, _yvals = [np.sort(np.unique(x)) for x in self.points.T]
    # give a little bit of slack on either sides
    _xvals[0] -= 10.0
    _xvals[-1] += 10.0
    _yvals[0] -= 10.0
    _yvals[-1] += 10.0
    self._xvals, self._yvals = map(frozen, (_xvals, _yvals))
    batches = _batches(self.triangles, self.points, self._xvals, self._yvals)
    return batches

  @property
  def _args(self):
    batches = self.batches
    return (self.triangles, self.points, self.ordered_boundary_vertex_indices, batches, self._xvals, self._yvals)

  def find_simplex(self, xi, tol=1e-10):
    return _find_simplex(*self._args, xi, tol)

  def find_simplices(self, Xi, tol=1e-10):
    return _find_simplices(*self._args, Xi, tol)

  def evaluate(self, weights, Xi, tol=1e-10):
    weights, Xi = map(np.asarray, (weights, Xi))
    assert Xi.shape[1:] == (2,)
    assert weights.shape[0] == len(self.points)
    shape = Xi.shape[:1] + weights.shape[1:]
    if len(weights.shape) == 1:
      weights = weights[:, None]
    assert len(weights.shape) == 2
    ret = _evaluate(*self._args, tol, weights, Xi)
    return ret.reshape(shape)

  evalf = evaluate

  def eval_template(self, weights: np.ndarray, template: MultiPatchTemplate):
    """
      Convenience function for creating a new template by evaluating
      a function on self over the template vertices.
    """
    weights = np.asarray(weights)
    assert weights.shape == (len(self.points), 2)
    points = self.evaluate(weights, np.asarray(template.patchverts))
    return template.edit(patchverts=points)

  @cached_property
  def _hash(self):
    return hash((self.triangles.tobytes(), self.points.tobytes()))

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    if self.__class__ != other.__class__:
      return False
    if self.triangles.shape != other.triangles.shape:
      return False
    if self.points.shape != other.points.shape:
      return False
    return (self.triangles == other.triangles).all() and (self.points == other.points).all()

  @cached_property
  @freeze
  def BK(self):
    """
      Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BK[i, :, :] or, in short, mesh.BK[i] gives
      the Jacobi matrix corresponding to the i-th element.

      Example
      -------

      for i, BK in enumerate(mesh.BK):
        # do stuff with the Jacobi matrix BK corresponding to the i-th element.
    """
    a, b, c = self.points[self.triangles.T]
    # freeze the array to avoid accidentally overwriting stuff
    return np.stack([b - a, c - a], axis=2)

  @cached_property
  @freeze
  def detBK(self):
    """ Jacobian determinant (measure) per element. """
    # the np.linalg.det function returns of an array ``x`` of shape
    # (n, m, m) the determinant taken along the last two axes, i.e.,
    # in this case an array of shape (nelems,) where the i-th entry is the
    # determinant of self.BK[i]
    return np.abs(np.linalg.det(self.BK))

  @cached_property
  @freeze
  def BKinv(self):
    """
      Inverse of the Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BKinv[i, :, :] or, in short, mesh.BKinv[i] gives
      the nverse Jacobi matrix corresponding to the i-th element of shape (2, 2).
    """
    (a, b), (c, d) = self.BK.T
    return np.rollaxis(np.stack([[d, -b], [-c, a]], axis=1), -1) / self.detBK[:, _, _]

  @cached_property
  @freeze
  def ordered_boundary_vertex_indices(self):
    """ Sorted starting on the vertex with the lowest index. Counterclockwise. """
    slices = np.array([[0, 1], [1, 2], [2, 0]])
    all_edges = np.array(list(map(abs_tuple, self.triangles[:, slices].reshape(-1, 2))))
    unique_edges, counts = np.unique(all_edges, return_counts=True, axis=0)

    boundary_vertex_indices = _return_sorted_boundary_vertices(unique_edges[counts == 1])
    pol = Polygon(self.points[boundary_vertex_indices])
    if pol == orient(pol):
      return boundary_vertex_indices.copy()
    return np.roll(boundary_vertex_indices[::-1], 1).copy()

  @cached_property
  @freeze
  def interior_vertex_indices(self):
    Npoints = len(self.points)
    freezeindices = self.ordered_boundary_vertex_indices
    return np.sort(np.setdiff1d(np.arange(Npoints, dtype=int), freezeindices))

  def plot(self, **kwargs):
    return plot_mesh(self.triangles, self.points, **kwargs)

  def to_Delaunay(self, scipykwargs=None, **kwargs):
    scipykwargs = dict(scipykwargs or {})
    delau = Delaunay(self.points, **scipykwargs)
    return Triangulation(delau.simplices, self.points)

  def is_valid(self):
    return (self.detBK > 0).all()

  def floater(self, X, as_Triangulation=True):
    assert X.shape == (len(self.ordered_boundary_vertex_indices), 2)

    if not hasattr(self, '_A_B'):
      freezeindices = self.ordered_boundary_vertex_indices
      dofindices = self.interior_vertex_indices
      self._A_B = _floater_data(self, freezeindices, dofindices)

    tri = _floater(self, X.astype(float).tobytes(), X.shape)
    if as_Triangulation:
      return tri
    return tri.points

  def eval_network(self, weights: np.ndarray, network: EdgeNetwork, **evalkwargs) -> EdgeNetwork:
    edges = tuple(self.evaluate(weights, edge.points, **evalkwargs) for edge in network.edges)
    return network.edit(edges=edges)


def plot_mesh(triangles, points, show=True, ax=None):
  """ Plot a mesh of type ``Triangulation``. """

  if ax is None:
    fig, ax = plt.subplots()
  else:
    fig = ax.figure

  ax.set_aspect('equal')
  ax.triplot(*points.T, triangles, linewidth=.75)

  if show:
    plt.show()
  
  return fig, ax


def _floater_data(tri: Triangulation, freezeindices, dofindices):
  Npoints = len(tri.points)
  freezeindices = tri.ordered_boundary_vertex_indices
  dofindices = np.sort(np.setdiff1d(np.arange(Npoints), freezeindices))
  M = make_matrix(tri.triangles, tri.points).tocsr()
  ext_dofindices = np.concatenate([dofindices, dofindices + Npoints])
  M = M[ext_dofindices].tocsc()
  A = M[:, ext_dofindices]
  B = - M[:, np.concatenate([freezeindices, freezeindices + Npoints])]
  A = splinalg.splu(A)
  return A, B


@lru_cache
def _make_mesh(X, shape, mesh_size):
  X = np.frombuffer(X).reshape(shape)
  assert shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(X, mesh_size=mesh_size)
    tri = geom.generate_mesh(algorithm=5, dim=2, order=1)

  return Triangulation.from_gmsh(tri)


def make_mesh(X, mesh_size=None):
  """ From boundary vertices """
  X = np.asarray(X)
  assert X.shape[1:] == (2,)

  if mesh_size is None:
    X_ = np.concatenate([X, X[:1]])
    # make sure no additional boundary vertices are introduced
    mesh_size = 1.1 * np.linalg.norm(X_[1:] - X_[:-1], axis=1).max()

  return _make_mesh(X.tobytes(), X.shape, round(mesh_size, 10))


def test():
  xi = np.linspace(0, 2 * np.pi, 201)[:-1]
  X = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  tri = make_mesh(X)

  Xi = np.stack([np.cos(xi), 3 * np.sin(xi)], axis=1)
  tri.floater(Xi).plot()
