from . import aux
from .aux import log

import numpy as np
from numbers import Number
from itertools import chain
from abc import abstractstaticmethod, abstractmethod, abstractproperty
from shapely import geometry, ops
from mapping import ko, go, gridop, std
from nutils import function
from functools import cached_property, lru_cache
from collections import defaultdict
from scipy import interpolate


def as_KnotObject(knotvector):
  if knotvector is None:
    knotvector = std.KnotObject
  if isinstance(knotvector, int):
    assert knotvector >= 1
    knotvector = ko.KnotObject(knotvalues=np.linspace(0, 1, knotvector+1))
  assert isinstance(knotvector, ko.KnotObject)
  return knotvector


def transform_index(index):
  if isinstance(index, (int, np.integer)):
    index = (abs(index), 0) if index < 0 else (abs(index), -1)
  assert isinstance(index, tuple)
  assert index[0] > 0
  return index


def as_edge(edge):
  if isinstance(edge, Edge):
    return edge
  return Edge(edge)


def transform_position(position, length):
  if isinstance(position, float):
    assert 0 <= position <= 1
    position = int(length * position)
  assert isinstance(position, (int, np.integer))
  return position % length


def unpack_list_of_lists(LoL, dtype):
  if np.iterable(LoL) and type(LoL[0]) == dtype:
    assert all(type(item) == dtype for item in LoL), 'Heterogeneous list detected.'
    return list(LoL)
  return list(chain(*[unpack_list_of_lists(item, dtype) for item in LoL]))


class Network:

  def __init__(self, indices, edges, face_indices, index_dtype, edge_dtype, undo=None):
    edges = [ {True: 1, False: -1}[abs(index) == index] * edge
                                   for index, edge in zip(indices, edges) ]
    indices = as_IndexSequence([ abs(index) for index in indices ], dtype=index_dtype)
    key = np.argsort(indices.indices)
    self._indices = indices[key]
    self._edges = as_IndexSequence(edges, dtype=edge_dtype)[key]
    assert len(self.indices) == len(self.edges)
    self._keys = dict(zip(self.indices, range(len(key))))
    # assert set(chain(*map(abs, self.face_indices.values()))).issubset(set(self.indices))
    self.index_dtype = self.indices.dtype
    self.edge_dtype = self.edges.dtype
    self._face_indices = FrozenDict({key: IndexSequence(list(value)) for key, value in face_indices.items()})
    assert set(map(abs, unpack_list_of_lists(as_IndexSequence(self.face_indices.values()), self.index_dtype))).issubset(set(self.indices))
    assert all( all(isinstance(i, int) for i in key) for key in self.face_indices.keys() )
    if undo is not None: assert issubclass(undo.__class__, Network)
    self._undo = undo

  @property
  def indices(self):
    return self._indices

  @property
  def edges(self):
    return self._edges

  @property
  def face_indices(self):
    return self._face_indices

  @property
  def faces(self):
    return list(self.face_indices.keys())

  def get_edges(self, indices):
    indices = list(map(self.index_dtype, indices))
    assert len(indices) == len(set(indices))
    return tuple( (index,) @ self.indices * self.edges.indices[self._keys[abs(index)]] for index in indices )

  @abstractmethod
  def _lib(self):
    return {'indices': self.indices,
            'edges': self.edges,
            'face_indices': self.face_indices.copy()}

  def edit(self, **kwargs):
    return self.__class__(**{**self._lib(), **kwargs})

  @cached_property
  def boundary_indices(self):
    index_counts = defaultdict(lambda: 0)
    for key, indices in self.face_indices.items():
      for index in indices:
        index_counts[abs(index)] += 1
    return as_IndexSequence(np.array([key for key, value in index_counts.items() if value == 1]))


class CannotSplitEdgeAtThisPositionError(Exception):
  pass


class CannotMergeMultiEdgesError(Exception):
  pass


class SaveLoadMixin:

  def _save(self):
    return self._lib()

  @classmethod
  def load(cls, **kwargs):
    return cls(**kwargs)


class EdgeBase:

  @abstractmethod
  def _lib(self):
    pass

  @abstractstaticmethod
  def new(cls, *args, **kwargs):
    if args[0].__class__ == cls:
      assert len(args) == 1 and len(**kwargs) == 0
      return args[0]
    return cls(*args, **kwargs)

  @abstractmethod
  def split(self):
    pass

  @abstractmethod
  def merge(self):
    pass

  def edit(self, **kwargs):
    return self.__class__(**{**self._lib(), **kwargs})

  @abstractproperty
  def length(self):
    pass

  @abstractmethod
  def flip(self):
    pass

  def __neg__(self):
    return -1 * self

  def __mul__(self, other):
    assert other in (1, -1), NotImplementedError
    if other == 1: return self
    return self.flip()

  __rmul__ = __mul__

  def pairing(self, other):
    if self == other: return 1
    if self == -other: return -1
    return 0

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __eq__(self, other):
    if not self.__class__ == other.__class__: return False
    if len(self) != len(other): return False
    if not hash(self) == hash(other): return False

  def __hash__(self):
    return hash(tuple(self._lib().values()))


def ordered_sublist_index(l0, l1):
  """ check if l0 is an ordered sublist of l1 """
  if len(l0) == 0 or len(l0) > len(l1):
    return -1
  dtype = l0.dtype
  l0 = np.array(list(l0), dtype=dtype)
  l1 = np.array(list(l1), dtype=dtype)
  n, m = len(l0), len(l1)
  l1 = np.concatenate([l1]*2)
  for i in range(m):
    if (l1[i: n + i] == l0).all(): return i
  return -1


def is_ordered_sublist(l0, l1):
  """ check if l0 is an ordered sublist of l1 """
  index = ordered_sublist_index(l0, l1)
  if index == -1:
    return False
  return True


def valid_verts(verts):
  if not len(verts.shape) == 1:
    return False
  if verts[0] != 0:
    return False
  if verts[-1] != 1:
    return False
  return (verts[1:] - verts[:-1] > 0).all()


def arclength_verts(points):
  verts = np.concatenate([[0], (((points[1:] - points[:-1])**2).sum(1)**0.5).cumsum()])
  return verts / verts[-1]


class Edge(EdgeBase, SaveLoadMixin):

  @staticmethod
  def new(*args, **kwargs):
    if args[0].__class__ == Edge:
      assert len(args) == 1 and len(kwargs) == 0
      return args[0]
    return Edge(*args, **kwargs)

  @staticmethod
  def merge_multiple(*edges):
    edge, *edges = edges
    while True:
      if len(edges) == 0: break
      edge0, *edges = edges
      edge = edge.merge(edge0)
    return edge

  @classmethod
  def between_points(cls, p0, p1, npoints=101):
    assert p0.shape == p1.shape == (2,)
    xi = np.linspace(0, 1, npoints)[:, None]
    return cls( p0[None] * (1 - xi) + p1[None] * xi )

  def _lib(self):
    return {'points': self.points,
            'verts': self._verts,
            'orientation': self.orientation}

  def __init__(self, points, vertices=None, verts=None, orientation=1):
    if vertices is None:
      vertices = (points[0], points[-1])
      points = points[1: -1]
    self._vertices = tuple(vertices)
    assert (len(self.vertices) == 2 and all( isinstance(vert, np.ndarray) and vert.shape == (2,) for vert in self.vertices ) )
    self._points = np.asarray(points)
    assert self.points.shape[1] == 2 and \
           len(self.points.shape) == 2 and \
           self.points.shape[0] >= 2
    self.line = geometry.LineString(self.points)
    self.orientation = int(orientation)
    self._verts = verts
    if self._verts is not None:
      self._verts = np.round(self._verts, 8)
      assert valid_verts(self._verts)

  @property
  def inner_points(self):
    return self._points

  @property
  def points(self):
    v0, v1 = self.vertices
    return np.concatenate([v0[None], self.inner_points, v1[None]])

  @property
  def verts(self):
    if self._verts is None:
      self._verts = np.round(arclength_verts(self.points), 8)
      assert valid_verts(self._verts)
    return self._verts

  @property
  def vertices(self):
    return self._vertices

  @property
  def dx(self):
    return ((self.points[1:] - self.points[:-1])**2).sum(1)**.5

  @property
  def length(self):
    return np.round(np.sqrt(((self.points[1:] - self.points[:-1])**2).sum(1)).sum(), 5)

  def is_arclength_parameterised(self, eps=1e-9):
    return (self.verts == np.round(arclength_verts(self.points), 8)).all()

  def merge(self, other, orientation=1):
    if not (self.points[-1] == other.points[0]).all():
      log.warning('WARNING: attempting to merge edges that do not attach.')
    dxi0, dxi1 = [edge.verts[1:] - edge.verts[:-1] for edge in (self, other)]
    length0, length1 = [edge.length for edge in (self, other)]
    length0, length1 = length0 / (length0 + length1), length1 / (length0 + length1)
    verts = np.concatenate([ np.concatenate([[0], dxi0]).cumsum() * length0, length0 + dxi1.cumsum() * length1 ])
    return self.__class__(np.concatenate([self.points[:-1], other.points]), orientation=orientation, verts=verts)

  def edit(self, **kwargs):
    return super().edit(**kwargs)

  def angle(self, other, forward=True):
    p1, p0 = other.points, self.points
    if forward:
      return aux.angle_between_vectors(p0[-1] - p0[-2], p1[1] - p1[0])
    return aux.angle_between_vectors(p1[1] - p1[0], p0[-2] - p0[-1])

  def copy(self):
    return self.__class__(self.points.copy(), orientation=self.orientation, verts=(self._verts if self._verts is not None else None))

  def transform(self, func):
    newpoints = func(self.point)
    return self.__class__(newpoints, verts=self._verts)

  def flip(self):
    dx = self.verts[1:] - self.verts[:-1]
    verts = np.concatenate([[0], dx[::-1]]).cumsum()
    return self.__class__(self.points[::-1], orientation=-self.orientation, verts=verts / verts[-1])

  def remove_duplicates(self, thresh=1e-13):
    points = self.points.copy()
    dvalue = points[1:] - points[:-1]
    mask = np.linalg.norm(dvalue, axis=1) > thresh
    if not mask[-1]:  # periodic, keep last point
      mask[-1] = True
    mask = np.concatenate([[True], mask])
    return self.__class__(points[mask], orientation=self.orientation, verts=self.verts[mask])

  def smooth(self, n=3, npoints=1001):
    from mapping import pc
    points = pc.PointCloud(np.concatenate([self.points[:1],
                                           self.points[1:-1][::n],
                                           self.points[-1:]      ])). \
                toInterpolatedUnivariateSpline(c0_thresh=100). \
                toPointCloud(np.linspace(0, 1, npoints)).points
    return self.__class__(points, orientation=self.orientation)

  def linearize(self):
    return self.__class__(self.points[[0, -1]], orientation=self.orientation)

  def __len__(self):
    return self.inner_points.shape[0] + 2

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.points.tobytes(), self.verts.tobytes()))
    return self._hash

  def __eq__(self, other):
    if super().__eq__(other) is None:
      if not (self.verts == other.verts).all():
        return False
      return (self.points == other.points).all()
    return False

  def __abs__(self):
    if self.orientation == 1: return self
    return -self

  def get_point(self, index):
    if isinstance(index, np.float):
      assert 0 <= index <= 1
      index = int(len(self) * index)
    return self.points[index]

  def change_vertices(self, v0=None, v1=None):
    if v0 is None and v1 is None: return self
    if v0 is None: v0 = self.vertices[0]
    if v1 is None: v1 = self.vertices[1]
    return self.__class__(self.inner_points, vertices=(v0, v1), orientation=self.orientation, verts=self._verts)

  def split(self, position):
    position = transform_position(position, len(self))
    if position in (0, len(self) - 1): raise CannotSplitEdgeAtThisPositionError
    p0, p1 = self.points[:position+1], self.points[position:]
    if self._verts is None:
      v0, v1 = None, None
    else:
      v0, v1 = self.verts[:position+1], self.verts[position:] - self.verts[position]
      v0, v1 = v0 / v0[-1], v1 / v1[-1]
    return self.__class__(p0, orientation=self.orientation, verts=v0), \
           self.__class__(p1, orientation=self.orientation, verts=v1)

  def split_arclength(self, percentage):
    assert 0 <= percentage <= 1
    index = np.argmin( np.abs(arclength_verts(self.points) - percentage) )
    return self.split(index)

  def split_length(self, length):
    assert 0 <= length <= self.length
    percentage = length / self.length
    return self.split_arclength(percentage)

  def normal(self, position):
    if isinstance(position, float):
      assert 0 <= position <= 1
      position = int(len(self) * position)
    assert isinstance(position, (int, np.integer))
    if position in (-1, len(self) - 1): position = len(self) - 2
    position = position % len(self)
    p1 = self.points[position + 1]
    p0 = self.points[position]
    d = p1 - p0
    return aux._normalized(np.array([d[1], -d[0]]))

  def normals(self):
    points = self.points
    dpoints = points[1:] - points[:-1]
    normals = aux._normalized( np.stack([ dpoints[:, 1], - dpoints[:, 0] ], axis=1 ) )
    return np.concatenate([normals, normals[-1:]], axis=0)

  def get_nearest_point(self, point, return_index=False):
    point = np.asarray(point)
    assert point.shape == (2,)
    argmin_index = np.argmin(np.linalg.norm(self.points - point[None], axis=1))
    if return_index:
      return self.points[argmin_index], argmin_index
    return self.points[argmin_index]

  def toPointCloud(self, **kwargs):
    from mapping import pc
    kwargs.setdefault('verts', self.verts)
    return pc.PointCloud(self.points, **kwargs)

  def refine(self, npoints=101, n=1, interpolargs=None):
    n = int(n)
    if n == 0: return self
    if interpolargs is None: interpolargs = {}
    from mapping import rep
    xi = np.linspace(0, 1, npoints)
    repfunc = rep.ReparameterizationFunction.fromverts(np.linspace(0, 1, len(self)), self.verts)
    points = self.toPointCloud().toInterpolatedUnivariateSpline(**interpolargs)(xi)
    ret = self.__class__(points, verts=repfunc(xi), orientation=self.orientation)
    return ret.refine(npoints=npoints, n=n-1, interpolargs=interpolargs)


def as_IndexSequence(sequence, format=None, **kwargs):
  if format is None:
    format = IndexSequence
  assert issubclass(format, IndexSequence)
  if isinstance(sequence, format):
    return sequence
  if not np.iterable(sequence):
    sequence = (sequence,)
  return format(tuple(sequence), **kwargs)


class FrozenVector(np.ndarray):

  def __new__(cls, vec, dtype=int):
    ret = np.empty( (len(vec),), dtype=dtype )
    if hasattr(dtype, 'new'):
      vec = list(map(dtype.new, vec))
    ret[:] = vec
    # ret = np.array(vec, dtype=dtype)
    ret.flags.writeable = False
    assert len(ret.shape) == 1
    return ret.view(cls)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(tuple(self))
    return self._hash


class FrozenDict(dict):

  def _immutable(self, *args, **kwargs):
    raise TypeError('Cannot change object - object is immutable')

  __setitem__ = _immutable
  __delitem__ = _immutable
  pop = _immutable
  popitem = _immutable
  clear = _immutable
  update = _immutable
  setdefault = _immutable

  del _immutable


as_sorted_tuple_dict = lambda x: {key: tuple(sorted(value)) for key, value in x.items()}
as_sorted_tuple_FrozenDict = lambda x: FrozenDict(as_sorted_tuple_dict(x))


class IndexSequence:

  @staticmethod
  def new(*args, **kwargs):
    return as_IndexSequence(*args, **kwargs)

  @staticmethod
  def stack(list_of_IndexSequences):
    assert all(isinstance(iseq, IndexSequence) for iseq in list_of_IndexSequences)
    list_of_IndexSequences = list(map(as_IndexSequence, list_of_IndexSequences))
    return sum(list_of_IndexSequences, ())

  def _lib(self):
    return {'indices': self.indices, 'dtype': self.dtype}

  def __init__(self, indices, dtype=None):
    if dtype is None:
      dtype = np.int64 if len(indices) == 0 else type(indices[0])
    self._indices = FrozenVector(indices, dtype=dtype)
    if len(self.indices) > 0 and (y := type(self.indices[0])) != dtype:
      log.warning("Data type {} silently converted to type {}".format(str(dtype), str(y)))
      dtype = y
    self._dtype = dtype

    if len(self.indices) > 0:
      assert hasattr(dtype, '__neg__'), 'Expected data type to be negateable.'
      assert hasattr(dtype, '__hash__')
      assert hasattr(dtype, '__abs__')
      if not issubclass(dtype, Number): assert hasattr(dtype, 'new')

  @property
  def indices(self):
    return self._indices

  @property
  def dtype(self):
    return self._dtype

  def index(self, ind, mod=False):
    # index found in self: return the position + 1 for the "head" of the vertex
    pairing = ind @ self
    if not pairing: raise ValueError
    if pairing == 1: ret = tuple(self.indices).index(ind) + 1
    else: ret = tuple(self.indices).index(-ind)
    if mod: ret = ret % len(self)
    return ret

  def split(self, i0, i1, newindex):
    myi0, myi1 = self.index(i0), self.index(i1)
    diff = myi1 - myi0
    vertices = np.roll(np.asarray(self.vertices), -myi0)
    ret0 = vertices[:diff] + (newindex,)
    ret1 = vertices[diff:] + (-newindex,)
    return ret0, ret1

  def replace(self, fromindices, toindices, strict=True):
    fromindices = as_IndexSequence(fromindices, self.__class__)
    toindices = as_IndexSequence(toindices, self.__class__)
    pairing = fromindices @ self
    if pairing:
      fromindices = pairing * fromindices
      toindices = pairing * toindices
      nelems_before_first_index = ordered_sublist_index(fromindices, self.indices)
      # sequence crosses boundaries => shift back by number of elements before the end
      if nelems_before_first_index + len(fromindices) > len(self):
        nelems_before_first_index = nelems_before_first_index - len(self)
      ret = toindices + self.roll_to(fromindices[0])[len(fromindices):]
      return ret.roll(nelems_before_first_index)
    if strict: raise ValueError
    return self

  def roll(self, amount):
    return self.__class__(np.roll(self.indices, amount), dtype=self.dtype)

  def roll_to(self, item):
    """ roll self such that item or -item becomes self[0] """
    pairing = item @ self
    assert pairing in (-1, 1)
    index0 = self.index(-pairing * item)
    return self[index0:] + self[:index0]

  def flip(self):
    return self.__class__([-ind for ind in self.indices][::-1], dtype=self.dtype)

  def index_neighbours(self, index):
    pairing = index @ self
    assert bool(pairing), 'Vertex not found.'
    if pairing == 1:
      n = len(self)
      index = tuple(self.indices).index(index)
      return (self.indices[(index-1) % n], -self.indices[(index+1) % n])
    return self.index_neighbours(-index)[::-1]

  def __add__(self, other):
    other = as_IndexSequence(other, self.__class__)
    assert self.__class__ == other.__class__ and self.dtype == other.dtype
    return self.__class__(tuple(self.indices) + tuple(other.indices), dtype=self.dtype)

  __radd__ = __add__

  def __len__(self):
    return len(self.indices)

  def __neg__(self):
    return (-1) * self

  def __mul__(self, item):
    if item not in (-1, 1): raise NotImplementedError
    if item == 1: return self
    return self.flip()

  __rmul__ = __mul__

  def __iter__(self):
    return iter(self.indices)

  def __contains__(self, other):
    if isinstance(other, self.dtype):
      other = (other,)
    other = as_IndexSequence(other, self.__class__)
    if is_ordered_sublist(other.indices, self.indices):
      return 1
    if is_ordered_sublist(other.indices, self.flip().indices):
      return -1
    return 0

  contains = __contains__

  def __matmul__(self, other):
    if isinstance(other, self.__class__):
      return other.contains(self)
    return any(item.contains(self) for item in other)

  def __rmatmul__(self, other):
    return self.contains(other)

  def __repr__(self):
    return self.indices.__repr__()

  def __getitem__(self, indices):
    ret = self.indices.__getitem__(indices)
    if np.isscalar(indices): return ret
    # if not np.iterable(ret): return ret
    return self.__class__(ret, dtype=self.dtype)

  def __hash__(self):
    return hash((self.indices, str(self.indices.dtype)))

  def __eq__(self, other):
    if other.__class__ != self.__class__: return False
    if self.dtype != other.dtype: return False
    if len(self) != len(other): return False
    return bool((self.indices == other.indices).all())

  def __abs__(self):
    return self.__class__(np.abs(self.indices), dtype=self.dtype)

  def merge(self, other):
    ret0, ret1 = self, other
    n, m = len(ret0), len(ret1)
    intersection = set(abs(ret0)) & set(abs(ret1))
    nitems = len(intersection)
    if nitems == 0: raise CannotMergeMultiEdgesError

    # roll until ret0 starts with the intersection indices
    for i in range(n):
      if set(abs(ret0[:nitems])) == intersection: break
      ret0 = ret0.roll(-1)
    else: raise CannotMergeMultiEdgesError

    for i in range(m):
      if set(abs(ret1[:nitems])) == intersection: break
      ret1 = ret1.roll(-1)
    else: raise CannotMergeMultiEdgesError

    if ret0[:nitems] != ret1[:nitems].flip(): raise CannotMergeMultiEdgesError

    return ret0[nitems:] + ret1[nitems:]

  def ravel(self):
    if not hasattr(self.dtype, '__iter__'):
      return self
    return IndexSequence(list(chain(*self))).ravel()


def as_multiedge(arg):
  if isinstance(arg, MultiEdge): return arg
  assert all( isinstance(_arg, Edge) for _arg in arg )
  return MultiEdge(arg)


class MultiEdge(IndexSequence):

  def __init__(self, edges, dtype=None):
    super().__init__(edges, dtype=Edge)
    self.polygon = geometry.Polygon(self.points)

  def check_vertices(self, thresh=1e-13):
    edges = tuple(self.edges) + (self.edges[0],)
    for e0, e1 in zip(edges, edges[1:]):
      if np.linalg.norm( e0.vertices[-1] - e1.vertices[0] ) > thresh:
        return False
    return True

  @property
  def edges(self):
    return self.indices

  @property
  def points(self):
    return np.concatenate([ edge.points[:-1] for edge in self ])

  @property
  def is_valid(self):
    return self.polygon.is_valid

  @property
  def is_oriented(self):
    return ops.orient(self.polygon) == self.polygon

  @property
  def length(self):
    return sum( edge.length for edge in self )

  @property
  def area(self):
    return self.polygon.area

  def orient(self):
    if not self.is_oriented: return self.flip()
    return self

  def repair_vertices(self):
    if self.check_vertices():
      return self
    newvertices = list(map(lambda x: list(x.vertices), self.edges))
    for v0, v1 in zip(newvertices, newvertices[1:]):
      v0[1] = v1[0] = (v0[1] + v1[0]) / 2
    newvertices[0][0] = (newvertices[0][0] + newvertices[-1][-1]) / 2
    newedges = [edge.change_vertices(**dict(zip(['v0', 'v1'], verts))) for edge, verts in zip(self.edges, newvertices)]
    return MultiEdge(newedges)

  def get_point(self, index):
    if isinstance(index, int):
      index = (index, 0) if index < 0 else (index, -1)
    return self.edges[index[0]].get_point(index[1])

  def angles(self, **kwargs):
    edges = (self.edges[-1],) + tuple(self.edges)
    ret = [ e0.angle(e1, **kwargs) for e0, e1 in zip(edges, edges[1:]) ]
    return np.asarray(ret)

  def average_normals(self):
    edges = (self[-1],) + tuple(self)
    ret = []
    for e0, e1 in zip(edges, edges[1:]):
      ret.append( (e0.normal(-1) + e1.normal(0)) / 2 )
    return ret

  def normal_jumps(self):
    edges = (self[-1],) + tuple(self)
    ret = []
    for e0, e1 in zip(edges, edges[1:]):
      ret.append( (e1.normal(0) - e0.normal(-1)) / 2 )
    return ret


# Orientables


class OrientableTuple(tuple):

  @staticmethod
  def new(*args, **kwargs):
    if args[0].__class__ == OrientableTuple:
      assert len(args) == 1 and len(kwargs) == 0
      return args[0]
    return OrientableTuple(*args, **kwargs)

  def __new__(cls, args):
    if isinstance(args, cls):
      return args
    assert len(args) == len(set(args)) == 2
    assert all(i >= 0 for i in args)
    return tuple.__new__(cls, args)

  def __abs__(self):
    x, y = self
    if x < y: return self
    return self.__class__(self[::-1])

  def __mul__(self, other):
    assert other in (-1, 1), NotImplementedError
    if other == 1: return self
    return self.__class__(self[::-1])

  __rmul__ = __mul__

  def __neg__(self):
    return (-1) * self


class OrientableFloat(float):

  @classmethod
  def new(cls, value):
    if isinstance(value, cls):
      return value
    return cls(value)

  def __new__(cls, value):
    ret = super(OrientableFloat, cls).__new__(cls, round(value, 10))
    assert 0 <= ret <= 1
    return ret

  @abstractproperty
  def orientation(self):
    return 1

  def __abs__(self):
    if self.__class__ == PositiveFloat: return self
    return PositiveFloat(1 - self)

  def __mul__(self, other):
    assert other in (-1, 1), NotImplementedError
    if other == 1: return self
    return {1: NegativeFloat, -1: PositiveFloat}[self.orientation](1 - self)

  __rmul__ = __mul__

  def __neg__(self):
    return (-1) * self


class PositiveFloat(OrientableFloat):

  @property
  def orientation(self):
    return 1


class NegativeFloat(OrientableFloat):

  @property
  def orientation(self):
    return - 1


def as_OrientableFloat(value):
  if isinstance(value, OrientableFloat): return value
  return PositiveFloat(value)


def as_OrientableTuple(tpl):
  if isinstance(tpl, OrientableTuple): return tpl
  return OrientableTuple(tpl)


class HashableVector(np.ndarray):

  def __new__(cls, ret):
    ret.flags.writeable = False
    return ret.view(cls)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(self.tobytes())
    return self._hash


class SplineEdge(EdgeBase, SaveLoadMixin):

  @staticmethod
  def merge_SplineEdges(*args, **kwargs):
    arg0, *args = args
    if len(args) == 0:
      return arg0
    return arg0.merge(args, **kwargs)

  @classmethod
  def from_knotvector_args(cls, x=None, **kwargs):
    assert x is not None
    return cls(ko.KnotObject(**kwargs), x)

  @classmethod
  def from_gridobject(cls, g, **kwargs):
    assert isinstance(g, go.TensorGridObject)
    assert len(g.knotvector) == 1
    return cls(g.knotvector[0], g.x.reshape([-1, g.targetspace]), **kwargs)

  def _lib(self):
    return { 'x': np.array(self.x),
             'knotvector': self.knotvector,
             'orientation': self.orientation }

  @staticmethod
  def new(*args, **kwargs):
    if args[0].__class__ == SplineEdge:
      assert len(args) == 1 and len(kwargs) == 0
      return args[0]
    return SplineEdge(*args, **kwargs)

  def __init__(self, knotvector, x, orientation=1):
    assert isinstance(knotvector, ko.KnotObject)
    assert knotvector.periodic is False, NotImplementedError
    assert orientation in (-1, 1)
    assert (knotvector.dx() > 1e-5).all()
    self.knotvector = knotvector
    x = np.asarray(x)
    if len(x.shape) == 1:
      x = x[:, None]
    self.targetspace, = x.shape[1:]
    assert self.targetspace == 2, NotImplementedError
    assert x.shape[0] == self.knotvector.dim
    self.x = HashableVector(x)
    self.orientation = int(orientation)

  @property
  def g(self):
    if not hasattr(self, '_g'):
      g = go.TensorGridObject(knotvector=self.knotvector, targetspace=self.targetspace)
      g.x = self.x.ravel()
      self._g = g
    return self._g

  def split(self, positions, lengths=None):
    if np.isscalar(positions):
      positions = positions,
    positions = tuple(positions)
    if lengths is None:
      lengths = [ self.length * (p1 - p0) for p0, p1 in zip((0,) + positions, positions + (1,)) ]
    assert len(lengths) == len(positions) + 1
    return [SplineEdge.from_gridobject(g, orientation=self.orientation, length=L) for g, L in zip(gridop.split(self.g, xs=positions), lengths)]

  def merge(self, others, breakpnt=None, stack_c=0):
    if not np.iterable(others):
      others = others,
    assert all(isinstance(other, self.__class__) for other in others)
    others = (self,) + tuple(others)
    if breakpnt is None:
      lengths = [other.length for other in others]
      total_length = sum(lengths)
      breakpnts = np.cumsum(lengths)[:-1] / total_length
    others = tuple(other.g for other in others)
    new_g = gridop.join(others, xs=breakpnts).to_c([stack_c])
    return SplineEdge.from_gridobject(new_g)

  def __len__(self):
    return self.knotvector.dim * self.targetspace

  def __abs__(self):
    if self.orientation == 1: return self
    return -self

  def __eq__(self, other):
    if super().__eq__(other) is None:
      if self.targetspace != other.targetspace: return False
      if self.knotvector != other.knotvector: return False
      return (self.x == other.x).all()
    return False

  def flip(self):
    return self.__class__(self.knotvector.flip(),
                          self.x[::-1],
                          orientation=-self.orientation)

  @cached_property
  def length(self):
    g = self.g
    return self.g.integrate( function.sqrt((g.mapping.grad(g.geom)**2).sum()) * function.J(g.geom) )

  @property
  def coord_length(self):
    return np.sqrt(((self.x[1:] - self.x[:-1])**2).sum(1)).sum()

  @lru_cache
  def as_Edge(self, n=1000):
    xi = np.linspace(0, 1, n)
    evals = [ interpolate.splev(xi, (self.knotvector.extend_knots(),
                                     x, self.knotvector.degree)) for x in self.x.T ]
    return Edge(np.stack(evals, axis=1), orientation=self.orientation)

  def __hash__(self):
    return hash((self.x, self.knotvector, self.orientation, self.length))


class ReflectedDictionary(dict):

  def __setitem__(self, item, value):
    # XXX: try to find a better solution here, possibly by creating abstract base class
    # for orientable dictionary keys.
    try: abs(item)
    except Exception as ex:
      raise Exception("Failed to add key to reflected dictionary because abs(key)"
                      "failed with exception '{}'.".format(ex))
    super().__setitem__(item, value)
    super().__setitem__(-item, -value)

  def positive_items(self):
    for key, value in self.items():
      if abs(key) == key:
        yield key, value


def reflect_dict(D):
  return {**D, **{-key: -value for key, value in D.items()}}
