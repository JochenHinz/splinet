import numpy as np

from aux import frozen, freeze
from functools import cached_property
from splinet.network import EdgeNetwork

from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from numba import njit
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
def _find_simplex(triangles: i64[:, :], points: f64[:, :], batches, xvals: f64[:], yvals: f64[:], xi: f64[:], tol: f64):
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
  else:  # not found => return first simplex on first local vertex
    return 0, np.array([0, 0, 0], dtype=np.float64)


@njit(cache=True)
def _find_simplices(triangles, points, batches, xvals, yvals, Xi, tol):
  """ Vectorised version of `_find_simplex`. """
  ret = np.empty(Xi.shape[:1], dtype=i64)
  for i, xi in enumerate(Xi):
    ret[i] = _find_simplex(triangles, points, batches, xvals, yvals, xi, tol)[0]
  return ret


@njit(cache=True)
def _evaluate(triangles, points, batches, xvals, yvals, tol, weights, Xi):
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
    itri, localweights = _find_simplex(triangles, points, batches, xvals, yvals, xi, tol)
    for j, myweights in enumerate(weights.T):
      w = myweights[triangles[itri]]
      ret[i, j] = (localweights * w).sum()
  return ret


class Triangulation:

  @classmethod
  def from_gmsh(cls, mesh, **kwargs):
    return cls(mesh.cells_dict['triangle'], mesh.points[:, :2], **kwargs)

  def __init__(self, triangles, points, nbatches=None, maxbatches=None):
    self.triangles, self.points = map(frozen, (triangles, points))
    if isinstance(nbatches, int):
      nbatches = (nbatches,) * 2
    if nbatches is None:
      nx, ny = map(lambda x: len(np.unique(x)), self.points.T)
      nbatches = (max(nx, 2), max(ny, 2))
    if maxbatches is None:
      nx, ny = map(lambda x: len(np.unique(x)), self.points.T)
      maxbatches = (nx + ny) // 2
    assert len(nbatches) == 2
    self.nbatches = tuple(map(lambda x: int(min(x, maxbatches)), nbatches))
    assert all(n > 0 for n in self.nbatches)
    assert self.triangles.shape[1:] == (3,)
    assert self.points.shape[1:] == (2,)
    assert set(np.unique(self.triangles)) == set(range(len(self.points)))

  @cached_property
  def batches(self):
    self._xvals, self._yvals = map(frozen, [np.sort(np.unique(_x)) for _x, n in zip(self.points.T, self.nbatches)])
    batches = _batches(self.triangles, self.points, self._xvals, self._yvals)
    return batches

  @property
  def _args(self):
    batches = self.batches
    return (self.triangles, self.points, batches, self._xvals, self._yvals)

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

  def plot(self, **kwargs):
    return plot_mesh(self.triangles, self.points, **kwargs)

  def to_Delaunay(self, scipykwargs=None, **kwargs):
    scipykwargs = dict(scipykwargs or {})
    delau = Delaunay(self.points, **scipykwargs)
    return Triangulation(delau.simplices, self.points)

  def is_valid(self):
    return (self.detBK > 0).all()


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