import os
import numpy as np
import aux
from shapely import geometry, validation, ops
from collections import defaultdict
from itertools import cycle, count, chain
from functools import wraps, cached_property
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS
from curves import shortest_polygon_path
from mapping import go, gridop, pc, ko
from abc import abstractmethod
from index import transform_index, as_edge, transform_position, \
                  CannotSplitEdgeAtThisPositionError, CannotMergeMultiEdgesError, \
                  Edge, as_IndexSequence, IndexSequence, MultiEdge, OrientableTuple, arclength_verts, log
from template import FaceTemplate, face_template
from fit import univariate_fit, make_knotvectors
from aux import reverse_univariate_gridobject


UNDORECURRENCELENGTH = 5


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


class DictMatMulMixin:

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __matmul__(self, other):
    return other @ as_IndexSequence(self.keys())

  __rmatmul__ = __matmul__


class FitDict(DictMatMulMixin, dict):

  def __init__(self, indices, edges, *args, **kwargs):
    self.indices = frozenset(indices)
    self.edgedict = FrozenDict(zip(indices, edges))
    arg0, *args = args
    if isinstance(arg0, dict):
      assert all( -i not in arg0 for i in arg0 )
    super().__init__(arg0, *args, **kwargs)
    assert all( i in self.indices or -i in self._indices for i in self )

  def __setitem__(self, item, value):
    item = int(item)
    assert item in self.indices or -item in self.indices
    assert isinstance(value, go.TensorGridObject)
    if -item in self:
      log.warning("Warning, the key's reciprocal already exists and will be automatically deleted.")
      del self[-item]
    super().__setitem__(item, value)

  def __getitem__(self, item):
    if -item in self:
      return aux.reverse_univariate_gridobject(self[-item])
    return super().__getitem__(item)

  def stack(self, keys):
    if not np.iterable(keys): keys = keys,
    if len(keys) == 1: return self[keys[0]]
    assert len(keys) >= 2
    lengths = np.asarray([self.edgedict[abs(key)].length for key in keys]).cumsum()
    return gridop.join([self[key] for key in keys], (lengths / lengths[-1])[:-1])


class MapDict(DictMatMulMixin, dict):

  def __init__(self, face_indices, *args, **kwargs):
    self._face_indices = frozenset(face_indices)
    super().__init__(*args, **kwargs)
    assert all( face_index in self._face_indices for face_index in self )

  def __setitem__(self, item, value):
    assert isinstance(value, go.GridObject) and hasattr(value, 'break_apart')
    super().__setitem__(item, value)


class CannotMergeFaceMapsError(Exception):
  pass


def undo_operation(func):

  @wraps(func)
  def wrapper(self, *args, **kwargs):
    ret = func(self, *args, **kwargs)
    if isinstance(ret, tuple):
      ret, *rest = ret
    else:
      rest = None
    ret._undo = self
    undo = ret
    for i in count():
      undo = undo._undo
      if undo is None: break
      if i > UNDORECURRENCELENGTH:
        undo._undo = None
        break
    if not rest:
      return ret
    return ret, *rest

  return wrapper


# XXX: clean up EdgeNetwork by forwarding long routines to dedicated functions
#      for better readability.


def in_and_out_of_place(func):

  @wraps(func)
  def wrapper(self, *args, inplace=True, **kwargs):
    ret = self if inplace else self.copy()
    func(ret, *args, **kwargs)
    if not inplace:
      return ret

  return wrapper


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


class WithPolygonsMixin:

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._polygons = FrozenDict({key: MultiEdge([self.get_edges(index) for index in value]) for key, value in self.face_indices.items()})
    color_generator = cycle(sorted(set(BASE_COLORS.keys()) - set(['w', 'r'])))
    self._colors = FrozenDict(dict(zip(self.face_indices.keys(), color_generator)))

  def plot_polygons(self, ax=None, indices=True, show=True, linewidth=1, vertex_thickness=8, show_axis=True, fontsize=8):
    if ax is None: fig, ax = plt.subplots()
    else: fig = ax.figure
    ax.set_aspect('equal')
    if not show_axis:
      ax.set_axis_off()
    face_colors = self._colors
    for key, pol in self._polygons.items():
      # templated faces are red
      color = face_colors[key] if key not in self.templates else 'r'
      XY = pol.polygon.exterior.coords.xy
      ax.plot(*XY, color=color, linewidth=linewidth)
      ax.fill(*XY, color=color, alpha=0.1)
    if indices:
      for index, edge in zip(self.indices, self.edges):
        point = edge.points[y // 2] if (y := len(edge.points)) > 2 else edge.points.sum(0) / 2
        ax.text(*point, str(index), fontsize=fontsize, zorder=10, c='r')
      for face_index, pol in self._polygons.items():
        try:  # in case the polygon is invalid
          point = np.concatenate(ops.polylabel(pol.polygon).coords.xy)
          ax.text(*point, str(list(face_index)), fontsize=fontsize, color=face_colors[face_index])
        except Exception:
          pass
    for edge in self.edges:
      for point in edge.vertices:
        ax.scatter(*point, c='k', s=vertex_thickness, zorder=4)
    if not show:
      return fig, ax
    plt.show()

  qplot = plot_polygons


class EdgeNetwork(WithPolygonsMixin, Network):

  @classmethod
  def fromfolder(cls, foldername):
    import pickle
    if not foldername.endswith('/'):
      foldername = foldername + '/'
    with open(foldername + 'edges.pkl', 'rb') as file:
      (indices, edges) = zip(*pickle.load(file).items())
    edges = tuple(Edge(edge) for edge in edges)
    with open(foldername + 'face_indices.pkl', 'rb') as file:
      face_indices = pickle.load(file)

    maps = {}
    for name in os.listdir(foldername + 'maps/'):
      key = tuple(map(int, name[:-4].split('_')))
      with open(foldername + 'maps/' + name, 'rb') as file:
        maps[key] = pickle.load(file)

    try:
      with open(foldername + 'templates.pkl', 'rb') as file:
        templates = pickle.load(file)
    except Exception:
      templates = {}

    from mapping import xml
    fits = {}
    for name in os.listdir(foldername + 'fits/'):
      key = int(name[:-4])
      fits[key] = xml.load_xml(foldername + 'fits/' + name)

    ret = cls(indices, edges, face_indices, maps=maps, fits=fits)
    templates = {face_index: FaceTemplate(ret, face_index, **template) for face_index, template in templates.items()}
    ret.templates.update(templates)
    return ret

  @classmethod
  def from_edgedict(cls, edgedict, face_indices, **kwargs):
    assert isinstance(edgedict, dict)
    (indices, edges) = zip(*edgedict.items())
    return cls(indices, edges, face_indices, **kwargs)

  def tofolder(self, foldername):
    import pickle
    if not foldername.endswith('/'):
      foldername = foldername + '/'
    if not os.path.exists(foldername):
      os.mkdir(foldername)

    with open(foldername + '/edges.pkl', 'wb') as file:
      pickle.dump({index: edge.points for index, edge in zip(self.indices, self.edges)}, file)
    with open(foldername + '/face_indices.pkl', 'wb') as file:
      pickle.dump({face_index: tuple(indices) for face_index, indices in self.face_indices.items()}, file)

    if not os.path.exists(foldername + 'maps/'):
      os.mkdir(foldername + 'maps/')

    for key, _map in self.maps.items():
      name = '_'.join(list(map(str, key))) + '.pkl'
      with open(foldername + 'maps/' + name, 'wb') as file:
        pickle.dump(_map, file)

    if not os.path.exists(foldername + 'fits/'):
      os.mkdir(foldername + 'fits/')

    from mapping import xml
    for key, fit in self.fits.items():
      xml.save_xml(fit, foldername + 'fits/' + str(key))

    # of the FaceTemplates we only save the template and the sides
    with open(foldername + 'templates.pkl', 'wb') as file:
      pickle.dump({face_index: template.tolib() for face_index, template in self.templates.items()}, file)

  def copy(self, **kwargs):
    kwargs = {**self._lib(),
              **{'maps': self.maps,
                 'fits': self.fits}, **kwargs}
    return self.__class__(**kwargs)

  def _lib(self):
    return super()._lib()

  def __init__(self, indices, edges, face_indices, maps=None, fits=None, undo=None, templates=None):
    super().__init__(indices, [as_edge(edge) for edge in edges], face_indices, np.int64, Edge, undo=undo)
    assert all( val.dtype in (int, np.int64) for val in self.face_indices.values() )
    assert all(i > 0 for i in self.indices)
    if maps is None: maps = {}
    if fits is None: fits = {}
    self._maps = MapDict(self.face_indices.keys(), maps)
    self._fits = FitDict(self.indices, self.edges, fits)
    self.templates = dict(templates or {})

  def get_edges(self, indices):
    if np.isscalar(indices):
      return super().get_edges((indices,))[0]
    return super().get_edges(indices)

  def get_vertices(self, indices):
    edges = self.get_edges(indices)
    if np.isscalar(indices):
      return edges.vertices[-1]
    return tuple(edge.vertices[-1] for edge in edges)

  @property
  def undo(self):
    if self._undo is None:
      raise AssertionError
    return self._undo

  @property
  def maps(self):
    return self._maps

  @property
  def fits(self):
    return self._fits

  @property
  def polygons(self):
    return self._polygons

  @property
  def max_edge_index(self):
    return max(self.indices)

  @property
  def max_curve_index(self):
    return self.max_edge_index

  @property
  def max_face_index(self):
    return max(list(self.face_indices.keys()))

  @property
  def allmaps(self):
    return [self.maps[key] for key in sorted(self.maps.keys())]

  @property
  def is_oriented(self):
    return all( pol.is_oriented for pol in self._polygons.values() )

  @property
  def edgedict(self):
    edgedict = dict(zip(self.indices, self.edges))
    edgedict.update({ -key: -value for key, value in edgedict.items() })
    return edgedict

  def edge_neighbours(self, index, thresh=1e-8):
    assert index in self.indices
    edge = self.get_edges(index)
    ret = {'-': set(), '+': set()}
    for ind in self.indices:
      if ind == index: continue
      vertices = self.get_edges(ind).vertices
      for i, vert in enumerate(vertices):
        if np.linalg.norm(edge.vertices[0] - vert) < thresh:
          ret['-'].update({0: {-ind}, 1: {ind}}[i])
        if np.linalg.norm(edge.vertices[1] - vert) < thresh:
          ret['+'].update({0: {-ind}, 1: {ind}}[i])
    return {key: sorted(list(value)) for key, value in ret.items()}

  def smooth_edge(self, index, **kwargs):
    edge = self.get_edges(index)
    return self.edit(edges=self.edges.replace(edge, edge.smooth(**kwargs)))

  def test_corners(self, thresh=1e-10):
    keys = sorted(self.face_indices.keys())
    return {key: self._polygons[key].check_vertices(thresh=thresh) for key in keys}

  def check_vertices(self):
    return all(self.test_corners().values())

  def check_convex_corners(self, thresh=-1e-3):
    return {face_index: (pol.angles() >= thresh).all() for face_index, pol in self.polygons.items()}

  def concave_corners(self, thresh=1e-5):
    ret = {}
    for key, indices in self.face_indices.items():
      edges = list(self.get_edges(indices))
      edges = edges[-1:] + edges
      for i, (e0, e1) in enumerate(zip(edges, edges[1:])):
        t1, t0 = map(lambda x: (x[1] - x[0]) / np.linalg.norm(x[1] - x[0]), [e0.points[-2:][::-1], e1.points[:2]])
        cross = np.cross(t0, t1)
        if cross < 0 and np.abs(cross) > thresh:
          ret.setdefault(key, []).append(-indices[i])
    return ret

  @undo_operation
  def replace_edge(self, index, newedge, keep_templates=True):
    newedge = as_edge(newedge)

    if index < 0:
      index = -index
      newedge = -newedge

    assert index in self.indices

    edges = [edge if i != index else newedge for i, edge in zip(self.indices, self.edges)]
    ret = self.edit(edges=edges)

    if keep_templates:
      for face_index, template in self.templates.items():
        ret.set_template(face_index, template.template, sides=template.sides)

    return ret

  @undo_operation
  def edit(self, **kwargs):

    newargs = {**self._lib(), **kwargs}
    indices, edges, face_indices = newargs.pop('indices'), newargs.pop('edges'), newargs.pop('face_indices')
    edgedict = dict(zip(indices, edges))

    # merge existing (and still valid) fits with the (potentially) passed new fits
    fits = {**{edge_index: g for edge_index, g in self.fits.items() if edgedict.get((y := abs(edge_index)), None) == self.get_edges(y)},
            **dict(newargs.pop('fits', {}))}

    # merge existing (and still valid) maps with the newly passed ones
    # and existing map remains valid if the exact same face is still contained in the edited network
    # that means key in face_indices and face_indices[key] == self.face_indices[key]
    # and all edges are the same
    maps = {**{face_index: g for face_index, g in self.maps.items()
               if (y := face_indices.get(face_index, None)) == self.face_indices[face_index]
               and all(edgedict[(z := abs(_y))] == self.get_edges(z) for _y in y)},
            **dict(newargs.pop('maps', {})) }

    undo = newargs.pop('undo', None)
    assert len(newargs) == 0

    ret = self.from_edgedict(edgedict, face_indices, fits=fits,
                                                     maps=maps,
                                                     undo=undo)

    # keep all face_templates as long as they still correspond to the same face
    from template import FaceTemplate
    templates = {}
    for face_index, temp in self.templates.items():
      # face_index not in new face_indices anymore => ignore
      if set(face_indices.get(face_index, [])) == set(self.face_indices[face_index]):
        # all edges are still the same => add
        if all( self.get_edges(abs(side)) == edgedict.get(abs(side), None)
                               for side in as_IndexSequence(temp.sides).ravel() ):
          templates[face_index] = FaceTemplate(ret, face_index, temp.template, temp.sides)

    ret.templates.update(templates)
    return ret

  def renumber_face_indices_(self):
    face_indices = sorted(self.face_indices.keys())
    map_old_face_index_new = dict(zip(face_indices, map(lambda x: (x,), range(len(face_indices)))))
    face_indices = {map_old_face_index_new[face_index]: value for face_index, value in self.face_indices.items()}
    maps = {map_old_face_index_new[face_index]: value for face_index, value in self.maps.items()}
    ret = self.__class__(self.indices, self.edges, face_indices, maps=maps, fits=self.fits, undo=self)
    templates = {map_old_face_index_new[face_index]: FaceTemplate(ret, map_old_face_index_new[face_index], temp.template, temp.sides)
                 for face_index, temp in self.templates.items()}
    ret.templates.update(templates)
    return ret

  def renumber_face_indices(self):
    map_face_indices = defaultdict(list)
    for face_index in self.face_indices.keys():
      map_face_indices[face_index[0]].append(face_index)
    map_face_indices = {key: sorted(set(value)) for key, value in map_face_indices.items()}
    map_old_new = {}
    for key, values in map_face_indices.items():
      for i, value in enumerate(values):
        map_old_new[value] = (key, i)
    face_indices = {map_old_new[face_index]: value for face_index, value in self.face_indices.items()}
    maps = {map_old_new[face_index]: value for face_index, value in self.maps.items()}
    ret = self.__class__(self.indices, self.edges, face_indices, maps=maps, fits=self.fits, undo=self)
    templates = {map_old_new[face_index]: FaceTemplate(ret, map_old_new[face_index], temp.template, temp.sides)
                 for face_index, temp in self.templates.items()}
    ret.templates.update(templates)
    return ret

  def remove_duplicates(self, **kwargs):
    return self.edit(edges=[edge.remove_duplicates(**kwargs) for edge in self.edges])

  @undo_operation
  def repair_vertices(self):
    if self.check_vertices(): return self
    newedges = {ind: self.get_edges(ind) for ind in self.indices}
    newcorners = {}
    for key, indices in self.face_indices.items():
      indices_ext = indices + (indices[0],)
      for i0, i1 in zip(indices_ext, indices_ext[1:]):
        myindices = (i0, -1), (i1, 0)
        otherindices = (-i0, 0), (-i1, -1)
        for index in myindices + otherindices:
          if index in newcorners:
            corner = newcorners[index]
            break
        else:  # no break means KeyError
          corner = 0.5 * self.get_edges(i0).points[-1] + (1 - 0.5) * self.get_edges(i1).points[0]
          newcorners[myindices[0]] = newcorners[myindices[1]] = \
                                     newcorners[otherindices[0]] = \
                                     newcorners[otherindices[1]] = corner
        newedges[abs(i0)] = newedges[abs(i0)].change_vertices(*{False: [None, corner], True: [corner, None]}[i0 < 0])
        newedges[abs(i1)] = newedges[abs(i1)].change_vertices(*{False: [corner, None], True: [None, corner]}[i1 < 0])
    return self.__class__.from_edgedict(newedges, self.face_indices, maps=self.maps.copy())

  @undo_operation
  def repair_polygons(self):
    newedges = {key: edge.copy() for key, edge in zip(self.indices, self.edges)}
    for key, pol in self._polygons.items():
      if not pol.is_valid:
        print('Repairing the polynomial with key {}.'.format(key))
        mulpol = validation.make_valid(pol.polygon)
        newpol = mulpol.geoms[np.argmax([geom.area for geom in mulpol.geoms])]
        diff = pol.polygon.exterior.difference(newpol.exterior)
        for i, edge in zip(abs(self.face_indices[key]), abs(pol)):
          line = edge.line
          newline = line.difference(line.intersection(diff))
          if isinstance(newline, geometry.MultiLineString):
            newline = newline.geoms[ np.argmax([geom.length for geom in newline.geoms]) ]
          newedge = Edge(np.stack(newline.coords.xy, axis=1))
          newedges[i] = newedge
    return self.__class__.from_edgedict(newedges, self.face_indices, maps=self.maps.copy())

  def check_faces(self):
    for key, pol in self._polygons.items():
      if not pol.is_valid:
        string = validation.explain_validity(pol.polygon)
        # coord = np.array(string[string.find('[')+1: string.find(']')].split(' '), dtype=float)
        print("The face with index {} is invalid due to the following reason: {}.".format(key, string))
      else:
        print('The face with index {} is valid.'.format(key))

  def positively_orient(self):
    face_indices = self.face_indices.copy()
    for key, value in face_indices.items():
      if not self._polygons[key].is_oriented:
        face_indices[key] = -value
    return self.edit(face_indices=face_indices)

  @undo_operation
  def merge_edges(self, indices):
    assert len(indices) == 2, NotImplementedError
    assert all(i in self.indices for i in indices)
    indices = as_IndexSequence(indices)
    i0, i1 = sorted(abs(indices))
    neighbours_of_i0 = self.edge_neighbours(i0)
    pairings = (i1 @ as_IndexSequence(neighbours_of_i0['-']),
                i1 @ as_IndexSequence(neighbours_of_i0['+']))

    for prefac, pairing in zip([1, -1], pairings):
      if pairing:
        sequence = prefac * as_IndexSequence((pairing * i1, prefac * i0))
        I0, I1 = sequence
        newedge = self.get_edges(I0).merge(self.get_edges(I1))
        break
    else:
      raise AssertionError('Index {} is not a neighbour of index {}.'.format(i0, i1))

    newindices = tuple(index for index in self.indices if index != i1)
    newedges = tuple(self.get_edges(index) if index != i0 else newedge for index in newindices)

    face_indices = self.face_indices.copy()
    for key, indices in face_indices.items():
      if len(set([i0, i1]) & set(abs(indices))) == 1:  # this means that one index is contained but not the other
        raise AssertionError('Cannot merge curve indices {} and {} as this would lead to conflicts.'.format(i0, i1))
      if I0 not in indices: continue
      face_indices[key] = face_indices[key].replace(sequence, i0)

    if i0 @ self.fits or i1 @ self.fits:
      log.warning('Warning, spline fits are not preserved under edge merging.')

    kwargs = {}

    return self.edit(edges=newedges, indices=newindices, face_indices=face_indices, **kwargs)

  @undo_operation
  def split_edge(self, ei):
    edge_idx, position = ei
    edge = self.get_edges(edge_idx)
    position = transform_position(position, len(edge))

    try: edges = edge.split(position)
    except CannotSplitEdgeAtThisPositionError: return self

    kwargs = {}
    kwargs['edges'] = self.edges.replace(edge, edges)
    new_indices = as_IndexSequence((edge_idx, np.sign(edge_idx) * (max(self.indices) + 1)))
    kwargs['indices'] = self.indices.replace(edge_idx, new_indices)
    kwargs['face_indices'] = {key: indices.replace(edge_idx, new_indices, strict=False) for key, indices in self.face_indices.items()}

    pairing = edge_idx @ self.fits
    if pairing:
      pointcloud = self.get_edges(edge_idx).toPointCloud()
      gs = gridop.split(self.fits[edge_idx], xs=[pointcloud.verts[position]])
      fits = self.fits.copy()
      fits[new_indices[0]], fits[new_indices[1]] = gs
      kwargs['fits'] = fits

    ret = self.edit(**kwargs)

    return ret

  def refine_edge(self, index, **kwargs):
    return self.replace_edge(index, self.get_edges(index).refine(**kwargs))

  def split_at_nearest_point(self, point, edge_indices=None, return_point=False):
    if edge_indices is None:
      edge_indices = self.indices
    if not np.iterable(edge_indices):
      edge_indices = (edge_indices,)
    edge_indices = tuple(edge_indices)
    point = np.asarray(point)
    assert point.shape == (2,)
    edges = self.get_edges(edge_indices)
    # exclude first and last points
    mindists, positions = [], []
    for edge in edges:
      dist = np.linalg.norm(edge.points[1:-1] - point[None], axis=1)
      position = np.argmin(dist)
      mindists.append(dist[position])
      positions.append(position + 1)
    edge_idx = edge_indices[np.argmin(mindists)]
    position = positions[np.argmin(mindists)]
    ret = self.split_edge((edge_idx, position))
    if return_point:
      ret = ret, self.get_edges(edge_idx).points[position]
    return ret

  def split_edge_arclength(self, ei):
    edge_idx, percentage = ei
    assert 0 <= percentage <= 1
    edge = self.get_edges(edge_idx)
    pointcloud = pc.PointCloud(edge.points)
    index = np.argmin( np.abs(pointcloud.verts - percentage) )
    return self.split_edge((edge_idx, index))

  def get_point_on_edge(self, index):
    index = transform_index(index)
    edge_index, position = index
    points = self.get_edges(edge_index).points
    if not isinstance(position, (int, np.integer)):
      assert isinstance(position, float) and 0 <= position <= 1
      position = int(points.shape[0] * position)
    assert isinstance(position, (int, np.integer))
    return points[position]

  def face_angles(self, face_index, return_sorted_indices=False, **kwargs):
    indices = self._face_indices[face_index]
    indices = np.concatenate([[indices[0]], indices])
    edges = [self.get_edges(i) for i in indices]
    sorted_indices = []
    ret = {}
    for i, (e1, e0) in enumerate(zip(edges[1:], edges)):
      sorted_indices.append(-indices[i+1])
      ret[-indices[i+1]] = e0.angle(e1, **kwargs)
    if return_sorted_indices:
      return ret, sorted_indices
    return ret

  def discrete_unit_normals(self, face_index, return_sorted_indices=False, **kwargs):
    indices = self._face_indices[face_index]
    indices = np.concatenate([[indices[-1]], indices])
    edges = [self.get_edges(i) for i in indices]
    sorted_indices = []
    ret = {}
    for i, (e1, e0) in enumerate(zip(edges[1:], edges)):
      sorted_indices.append(-indices[i+1])
      pm1, p0, p1 = e0.points[-2], e0.points[-1], e1.points[1]
      t = (p0 - pm1) / np.linalg.norm(p0 - pm1) + (p1 - p0) / np.linalg.norm(p1 - p0)
      t = t / np.linalg.norm(t)
      ret[-indices[i+1]] = np.array([t[1], -t[0]])
    if return_sorted_indices:
      return ret, sorted_indices
    return ret

  @undo_operation
  def _split_face(self, face_index, v0, v1, connecting_edge=None):
    # XXX: docstring
    # XXX: add fail switch to detect when, in fact, v0 == v1
    p0, p1 = self.get_point_on_edge(v0), self.get_point_on_edge(v1)
    if connecting_edge is None:
      connecting_edge = Edge.between_points(p0, p1)
    if isinstance(connecting_edge, (Edge, np.ndarray)):
      connecting_edge = (connecting_edge,)
    connecting_edge = tuple(map(as_edge, connecting_edge))
    indices = self.face_indices[face_index]
    indices = indices.roll(-indices.index(v0))
    index1 = indices.index(v1)
    newcurveindices = tuple(range(self.max_curve_index+1, self.max_curve_index + 1 + len(connecting_edge)))
    indices0, indices1 = indices[:index1] + tuple(-i for i in newcurveindices)[::-1], \
                         indices[index1:] + newcurveindices
    face_indices = {key: value for key, value in self.face_indices.items() if key != face_index}
    edgedict = dict(zip(self.indices, self.edges))
    for index, edge in zip(newcurveindices, connecting_edge):
      edgedict[index] = edge
    face_indices[face_index + (1,)] = indices0
    face_indices[face_index + (2,)] = indices1
    indices = tuple(sorted(edgedict.keys()))
    edges = tuple( edgedict[key] for key in indices )
    return self.edit(indices=indices, edges=edges, face_indices=face_indices)

  def split_face(self, face_index, c0i0, c1i1, **kwargs):
    # XXX: docstring
    ret = self
    vs = []
    for ci in (c0i0, c1i1):
      if not isinstance(ci, (int, np.int64)):
        c, i = ci
        assert c >= 0 and c in self.indices
        ret = ret.split_edge(ci)
        vs.append(-ret.max_curve_index)
      else:
        vs.append(ci)
    return ret._split_face(face_index, *vs, **kwargs)

  def split_face_point(self, face_index, point, npoints=101):
    point = np.asarray(point)
    assert point.shape == (2,)

    indices = self.face_indices[face_index]
    assert len(indices) >= 3

    ret = self
    for index in abs(indices):
      ret = ret.split_edge((index, .5))

    polygon = self.polygons[face_index].polygon
    if not geometry.Point(point).within(polygon):
      log.warning('Warning, the provided point is not strictly contained within the face!')

    newindices = ret.face_indices[face_index]
    newedges = [Edge.between_points(point, edge.points[0], npoints=npoints)
                                   for edge in ret.get_edges(newindices)[1::2]]
    ret = ret.split_face(face_index, -newindices[1], -newindices[3], connecting_edge=(-newedges[0], newedges[1]))
    for v1, edge in zip(newindices[5::2], newedges[2:]):
      v0 = ret.max_curve_index
      ret = ret.split_face(ret.max_face_index, -v0, -v1, connecting_edge=edge)

    return ret

  def split_face_centroid(self, face_index, **kwargs):
    point = np.asarray( self.polygons[face_index].polygon.centroid.coords.xy ).ravel()
    return self.split_face_point(face_index, point, **kwargs)

  @undo_operation
  def merge_faces(self, fi0, fi1):
    face0, face1 = self.face_indices[fi0], self.face_indices[fi1]
    newface = face0.merge(face1)
    diff = set(abs(face0)) - set(abs(newface))
    face_indices = {key: value for key, value in self.face_indices.items() if key not in (fi0, fi1)}
    for indices in face_indices.values():
      if len(set(abs(indices)) & diff) != 0: raise CannotMergeMultiEdgesError
    face_indices[fi0] = newface
    indices = tuple(index for index in self.indices if index not in diff)
    edges = tuple(self.get_edges(index) for index in indices)
    return self.edit(indices=indices, edges=edges, face_indices=face_indices)

  def boundary_network(self):
    """ Return a new EdgeNetwork with only one face consisting of the boundary edges."""
    bindices = self.ordered_boundary_indices
    if len(bindices) > 1: raise NotImplementedError
    if len(self.face_indices) == 1:  # set the sole face to (0, 0) and return self
      return self.edit( face_indices={(0, 0): list(self.face_indices.values())[0] } )
    bindices, = bindices
    face_indices = {(0, 0): tuple(bindices)}
    bindices = list(map(abs, bindices))
    edges = tuple(self.get_edges(index) for index in bindices)
    return self.edit(indices=bindices, edges=edges, face_indices=face_indices)

  @cached_property
  def ordered_boundary_indices(self):
    boundary_indices = tuple(self.boundary_indices)
    ordered_boundary_indices = []
    while True:
      indices = [boundary_indices[0]]
      boundary_indices = boundary_indices[1:]
      while True:
        index = indices[-1]
        neighbours = [i for i in self.edge_neighbours(index)['+'] if abs(i) in boundary_indices]
        if len(neighbours) == 0: break
        elif len(neighbours) == 1: neighbour, = neighbours
        else: raise RuntimeError('Found more than one positive neighbour. This is a bug.')
        indices.append(-neighbour)
        neighbour_index = boundary_indices.index(abs(neighbour))
        boundary_indices = boundary_indices[:neighbour_index] + boundary_indices[neighbour_index+1:]
      indices = as_IndexSequence(indices)
      pol = MultiEdge([self.get_edges(index) for index in indices])
      ordered_boundary_indices.append( indices if pol == pol.orient() else -indices )
      if len(boundary_indices) == 0: break
    return tuple(ordered_boundary_indices)

  @property
  def genus(self):
    return len(self.ordered_boundary_indices) - 1

  @undo_operation
  def linearize(self):
    return self.edit(edges=tuple(edge.linearize() for edge in self.edges))

  def plot_grids(self, **kwargs):
    fig, ax = self.plot_polygons(show=False, **kwargs)
    for face_index, mg in self.maps.items():
      for g in mg.break_apart():
        plot(g, ax=ax, boundary=1)
    plt.show()

  def qplot(self, *args, **kwargs):
    return self.plot_polygons(*args, **kwargs)

  def transform(self, g):
    newedges = [Edge(g.call(*edge.points.T)) for edge in self.edges]
    return self.edit(edges=newedges)

  @undo_operation
  def take(self, list_of_faces):
    list_of_faces = set(map(tuple, list_of_faces))
    face_indices = {key: self.face_indices[key] for key in list_of_faces}
    all_indices = set.union(*map(set, face_indices.values()))
    indices = sorted([abs(key) for key in all_indices])
    edges = [self.get_edges(index) for index in indices]
    return self.edit(face_indices=face_indices, indices=indices, edges=edges)

  def remove_faces(self, list_of_faces):
    minus_faces, all_faces = set(map(tuple, list_of_faces)), set(self.face_indices.keys())
    assert minus_faces.issubset(all_faces)
    return self.take(list(all_faces - minus_faces))

  def inject(self, other):
    for face_index, mapping in other.maps.items():
      self.maps[face_index] = mapping

  def to_SplineNetwork(self):
    from spline import SplineNetwork
    return SplineNetwork.from_EdgeNetwork(self)

  def make_quick_grid(self, face_index, knotvectors=8, **kwargs):
    snetwork = self.take([face_index])
    assert len(snetwork.templates) == 1
    if not len(snetwork.fits) == len(snetwork.edges):
      snetwork.make_fits(knotvectors=knotvectors, **kwargs)
    snetwork = snetwork.to_SplineNetwork().make_maps()
    mg = snetwork.maps[(face_index)]
    from mapping import sol
    sol.forward_laplace(mg)
    sol.Blechschmidt(mg)
    return snetwork

  # CANVAS OPERATIONS

  # @undo_operation
  def click_vertex(self, face_index, **kwargs):
    kwargs.setdefault('title', 'Click on the desired position')
    from canvas import generate_vertex_from_click
    return generate_vertex_from_click(self, face_index, **kwargs)

  @undo_operation
  def connect_vertices(self, face_index, **kwargs):
    assert 'title' not in kwargs
    from canvas import draw_spline_from_vertex_to_vertex
    return draw_spline_from_vertex_to_vertex(self, face_index,
                                             title='Draw a spline by clicking the canvas',
                                             **kwargs)

  @undo_operation
  def connect_new_vertices(self, face_index, **kwargs):
    assert 'title' not in kwargs
    from canvas import connect_two_new_vertices_by_curve
    return connect_two_new_vertices_by_curve(self, face_index, **kwargs)

  @undo_operation
  def create_diamond(self, face_index, **kwargs):
    from canvas import create_diamond
    return create_diamond(self, face_index, **kwargs)

  @undo_operation
  def half_edge(self, face, **kwargs):
    from canvas import half_edge
    return half_edge(self, face, **kwargs)

  @undo_operation
  def half_face(self, face, **kwargs):
    from canvas import half_face
    return half_face(self, face, **kwargs)

  @undo_operation
  def create_full_diamond(self, face_index, **kwargs):
    from canvas import create_full_diamond
    return create_full_diamond(self, face_index, **kwargs)

  @undo_operation
  def draw_linears(self, face_index, **kwargs):
    from canvas import draw_linears
    return draw_linears(self, face_index, **kwargs)

  def select_edges(self, face_index, **kwargs):
    from canvas import select_edges
    return select_edges(self, face_index, **kwargs)

  def select_vertices(self, face_index):
    from canvas import select_vertices
    return select_vertices(self, face_index)

  @undo_operation
  def split_face_click(self, face_index, **kwargs):
    from canvas import GenerateClicks
    title = 'Click the canvas to select a centroid point'
    with GenerateClicks(self.take([face_index]), n=1, title=title) as handler:
      point, = handler.points
    return self.split_face_point(face_index, point, **kwargs)

  @undo_operation
  def split_face_hermite(self, face_index, eta=None, gamma=None, npoints=1001):
    from canvas import select_vertices
    vindices = select_vertices(self, face_index, nclicks=2)
    v0, v1 = self.get_vertices(vindices)
    if eta is None:
      eta = np.linalg.norm(v1 - v0)
    if gamma is None:
      gamma = eta

    n = self.discrete_unit_normals(face_index)
    n0, n1 = n[vindices[0]], n[vindices[1]]

    from aux import hermite_interpolation
    points = hermite_interpolation(v0, v1, -eta*n0, gamma*n1, npoints=npoints)

    return self._split_face(face_index, *vindices, connecting_edge=as_edge(points))

  def select_faces(self, **kwargs):
    from canvas import select_faces
    clicked_faces = select_faces(self, **kwargs)
    return self.take(tuple(clicked_faces))

  @undo_operation
  def divided_reparam(self, face_index, reparam_kwargs=None, **kwargs):
    reparam_kwargs = dict(reparam_kwargs or {})

    I0 = self.select_edges(face_index, title='Select the edges corresponding to the first side', **kwargs)
    I1 = self.select_edges(face_index, title='Select the edges corresponding to the second side', **kwargs)

    absI0, absI1 = abs(as_IndexSequence(I0)), abs(as_IndexSequence(I1))
    assert len(set(absI0) & set(absI1)) == 0, 'The two sides need to be mutually disjoint.'

    absindices = abs(self.face_indices[face_index])
    assert absI0 in absindices and absI1 in absindices, \
           'Both sides need to a sequence of neighbouring edges.'

    seqs = []
    for aII, II in zip((absI0, absI1), (I0, I1)):
      myindices = absindices.roll_to(aII[0])
      i1 = tuple(myindices).index(aII[-1])
      seqs.append(self.get_edges(self.face_indices[face_index].roll_to(II[0])[:i1 + 1]))

    seq0, seq1 = seqs

    e0, e1 = [ Edge.merge_multiple(*seq) for seq in seqs ]
    if len(seq1) == 1:
      if np.linalg.norm(e0.points[-1] - e1.points[0]) < np.linalg.norm(e0.points[0] - e1.points[0]):
        e1 = -e1
        seq1 = [-seq1[0]]
        I1 = [-I1[0]]
    elif len(seq0) == 0:
      if np.linalg.norm(e0.points[-1] - e1.points[0]) < np.linalg.norm(e0.points[0] - e1.points[0]):
        e0 = -e0
        seq0 = [-seq0[0]]
        I0 = [-I0[0]]

    breaks = [ list(map(lambda x: x.length, seq[:-1])) for seq in seqs ]

    from mapping import rep
    kwargs.setdefault('weights', (.5, .5))
    repfuncs = rep.divided_reparam(e0.toPointCloud(), e1.toPointCloud(), **kwargs)
    newedges = [e.edit(verts=repfunc(e.verts)) for e, repfunc in zip([e0, e1], repfuncs)]

    split_new_edges = []

    for mybreaks, newedge in zip(breaks, newedges):
      my_new_edges = (newedge,)
      while len(mybreaks):
        _break, *mybreaks = mybreaks
        my_new_edges = my_new_edges[:-1] + my_new_edges[-1].split_length(_break)
      split_new_edges.extend(list(my_new_edges))

    all_indices = I0 + I1

    edges = self.edges
    for index, edge in zip(all_indices, split_new_edges):
      edges = edges.replace(self.get_edges(abs(index)), edge if index > 0 else -edge)

    return self.edit(edges=edges)

  @undo_operation
  def relieve_tight_corner(self, face_index, distance_function=None, magnitude=1, keep_templates=True):
    from canvas import SelectVertices, GenerateVertexFromClick

    if distance_function is None:
      # the following function f(x) = x^3 - 2*x^2 + x satisfies:
      # f(0) = 0, f(1) = 0, f'(0) = 1, f'(1) = 0
      distance_function = lambda d: d**3 - 2*d**2 + d

    assert np.allclose(distance_function(1), 0)

    indices = self.face_indices[face_index]
    indices = indices + indices[:1]

    equivalent_corners = {}
    for i0, i1 in zip(indices, indices[1:]):
      equivalent_corners[i0] = -i1
      equivalent_corners[-i1] = i0

    snetwork = self.take([face_index])

    with SelectVertices(snetwork, nclicks=1, title="Select the corner's vertex.") as handler:
      vertex, = handler.clicked_vertices

    # select the two candidate edges that attach to this corner
    attaching_indices = abs(vertex), abs(equivalent_corners[vertex])

    title = 'Click a position on the edge till which the distance function applies.'
    with GenerateVertexFromClick(snetwork, title=title, indices=attaching_indices) as handler:
      (e, i) = handler.ei

    if abs(vertex) != abs(e): vertex = equivalent_corners[vertex]
    if not vertex == -e:
      e = -e
      i = len(self.get_edges(e)) - i

    edge = self.get_edges(e)
    assert i < len(edge) - 1

    points = edge.points
    distance = arclength_verts(points[:i])
    normals = edge.normals()

    points[1: i] += magnitude * distance_function(distance)[1:, None] * normals[1: i]

    newedge = edge.edit(points=points)
    newedges = self.edges.replace((edge,), (newedge,))

    ret = self.edit(edges=newedges)

    if keep_templates:
      templates = {}
      for face_index, temp in self.templates.items():
        if face_index in ret.templates: continue
        templates[face_index] = FaceTemplate(ret, face_index, temp.template, temp.sides)
      ret.templates.update(templates)

    return ret

  def create_corner(self, face_index):
    from operations import create_corner
    return create_corner(self, face_index)

  def create_triangle_corner(self, face_index):
    from operations import create_triangle_corner
    return create_triangle_corner(self, face_index)

  # TEMPLATES + FITTING

  def set_template(self, face_index, template, **kwargs):
    self.templates[face_index] = face_template(self, face_index, template, **kwargs)

  def set_knotvectors_from_templates(self, **kwargs):
    self.knotvectors.update(make_knotvectors(self, **kwargs))

  def make_fit(self, edge_index, kv, **fitkwargs):
    edge_index = abs(edge_index)

    pointcloud = self.get_edges(edge_index).toPointCloud()
    myknotvector, myweights = univariate_fit(pointcloud, knotvector=kv, **fitkwargs)
    self._fits[edge_index] = go.TensorGridObject(knotvector=myknotvector, targetspace=2)
    self._fits[edge_index].x = myweights

  def make_fits(self, knotvectors=8, overwrite=True, knotvector_factory=None, **fitkwargs):
    if knotvector_factory is not None:
      knotvectors = knotvector_factory(self)
    else:
      if isinstance(knotvectors, (int, np.int64)):
        knotvectors = ko.KnotObject(knotvalues=np.linspace(0, 1, knotvectors+1))
      if isinstance(knotvectors, ko.KnotObject):
        knotvectors = dict(zip(self.indices, [knotvectors]*len(self.indices)))

    assert set(knotvectors.keys()) == set(self.indices)
    assert all( isinstance(kv, ko.KnotObject) for kv in knotvectors.values() )

    for i, edge_index in enumerate(self.indices, 1):
      if not overwrite:
        if edge_index in self.fits or -edge_index in self.fits:
          log.warning('Edge with index {} has already been fit, skipping.'.format(edge_index))
          continue
      log.warning('Fitting spline curve {} out of {}.'.format(i, len(knotvectors)))
      self.make_fit(edge_index, knotvectors[edge_index], **fitkwargs)

  def make_boundary_fits(self, **kwargs):
    bnetwork = self.boundary_network()
    bnetwork.make_fits(**kwargs)
    for key, g in bnetwork.fits.items():
      self.fits[key] = g

  def make_map(self, face_index, **kwargs):
    face_template = self.templates.get(face_index, None)
    if face_template is None:
      raise AssertionError('The template for the face with index {} has not been set yet.'.format(face_index))
    self.maps[face_index] = face_template.make_MultipatchBSplineGridObject(**kwargs)

  def make_smooth_fits(self, knotvectors, angle_thresh=0.05, join_c=1, **fitkwargs):
    if isinstance(knotvectors, (int, np.int64)):
      knotvectors = ko.KnotObject(knotvalues=np.linspace(0, 1, knotvectors+1))
    if isinstance(knotvectors, ko.KnotObject):
      knotvectors = dict(zip(self.indices, [knotvectors]*len(self.indices)))
    assert set(knotvectors.keys()) == set(self.indices)
    assert all( isinstance(kv, ko.KnotObject) for kv in knotvectors.values() )
    all_neighbours = {**{i: self.edge_neighbours(i) for i in self.indices},
                      **{-i: self.edge_neighbours(-i) for i in self.indices} }
    matched_indices = set()
    sequences = []
    all_indices = set(self.indices)
    while True:
      if len(matched_indices) == len(all_indices): break
      i, *ignore = list(all_indices - matched_indices)
      mysequence = [i]
      matched_indices.update({i})
      while True:
        n = len(mysequence)
        if len((y := all_neighbours[mysequence[0]]['-'])) == 1:
          mneighbours = [j for j in y if abs(j) not in matched_indices]
          if mneighbours:
            edge = self.get_edges(mysequence[0])
            mangles = [abs(self.get_edges(j).angle(edge)) for j in mneighbours]
            if len(mangles) == 1 and min(mangles) < angle_thresh:
              mysequence = [mneighbours[np.argmin(mangles)]] + mysequence
              matched_indices.update({abs(mneighbours[np.argmin(mangles)])})
        if len((y := all_neighbours[mysequence[-1]]['+'])) == 1:
          pneighbours = [j for j in y if abs(j) not in matched_indices]
          if pneighbours:
            edge = self.get_edges(mysequence[-1])
            pangles = [abs(edge.angle(self.get_edges(-j))) for j in pneighbours]
            if len(pangles) == 1 and min(pangles) < angle_thresh:
              mysequence += [-pneighbours[np.argmin(pangles)]]
              matched_indices.update({abs(pneighbours[np.argmin(pangles)])})
        if len(mysequence) == n:
          break
      sequences.append(mysequence)

    for k, seq in enumerate(sequences):
      log.warning('Fitting spline curve {} out of {}.'.format(k, len(sequences)))
      if len(seq) == 1:
        edge = self.get_edges(seq[0])
        knotvector = knotvectors[seq[0]]
        myknotvector, myweights = univariate_fit(edge.toPointCloud(), knotvector=knotvector, **fitkwargs)
        self.fits[seq[0]] = go.TensorGridObject(knotvector=myknotvector, targetspace=2)
        self.fits[seq[0]].x = myweights
        continue
      myedges = self.get_edges(seq)
      mylengths = [edge.length for edge in myedges]
      mybreaks = np.round(np.cumsum(mylengths[:-1]) / sum(mylengths), 5)
      edge = Edge.merge_multiple(*myedges)
      myknotvectors = [knotvectors[i] if i in knotvectors else knotvectors[-i].flip() for i in seq]
      knotvector = gridop.join_KnotObjects(myknotvectors, xs=mybreaks).to_c(1)
      myknotvector, myweights = univariate_fit(edge.toPointCloud(), knotvector=knotvector, **fitkwargs)
      G = go.TensorGridObject(knotvector=myknotvector, targetspace=2)
      G.x = myweights
      gs = gridop.split(G, xs=mybreaks)
      for i, g in zip(seq, gs):
        if i < 0:
          i = -i
          g = reverse_univariate_gridobject(g)
        self.fits[i] = g

  @undo_operation
  def trefine(self, edge, position=.5, local_position=None, return_indexmap=False):
    """ Template refinement. Differs from the ordinary refinement in that
        edge refinements will also be mirrored on opposite template sides throughout the network."""
    assert all( all(len(side) == 1 for side in template.sides) for template in self.templates.values() ), NotImplementedError

    if local_position is None:
      local_position = position

    # turn to standard input
    if edge < 0:
      edge = -edge
      position = 1 - position
      local_position = 1 - local_position

    assert edge > 0 and 0 < position < 1 and 0 < local_position < 1
    assert edge in self.indices

    # refine_edges only contains positive indices
    refine_edges, new_refine_edges = {edge}, {edge}

    # positions maps the (global) edge to either 1 or -1. 1 means we refine it at
    # position, -1 means we refine it at (1 - position)
    positions = {edge: 1, -edge: -1}

    # collect all templates
    templates = {face_index: face_template.template for
                             face_index, face_template in self.templates.items()}

    # for now no templates with 'holes' in it.
    assert all(len(template.ordered_boundary_edges) == 1 for template in templates.values()), NotImplementedError

    # sides = {face_index: (face_template.sides, template.ordered_boundary_edges)}
    sides = {face_index: (IndexSequence(face_template.sides).ravel(), templates[face_index].ordered_boundary_edges[0]) for
                         face_index, face_template in self.templates.items()}

    while True:
      for face_index, (global_edges, local_boundary_edges) in sides.items():
        mytemplate = templates[face_index]
        abs_global_edges = abs(global_edges)

        # make reflected dictionary mapping global edge to local edge
        map_global_edge_local = dict(zip(global_edges, local_boundary_edges))
        map_global_edge_local.update(dict(zip([-i for i in global_edges], [edge[::-1] for edge in local_boundary_edges])))

        # invert
        map_local_edge_global = {value: key for key, value in map_global_edge_local.items()}

        # take intersection of the current refine-edges and the edges on the face
        intersect_edges = refine_edges & set(abs_global_edges)

        # add all edges in the intersection to the refine_edges
        for edge in intersect_edges:

          # get local edge
          local_edge = map_global_edge_local[edge]

          # get all neighbouring edges
          neigh_edges = IndexSequence(mytemplate.edge_neighbours(local_edge), OrientableTuple)

          # add the global counterpart of the neighbouring edges to the refine_edges
          new_global_edges = []
          for neigh in neigh_edges:
            new_global_edge = map_local_edge_global.get(neigh, None)
            if new_global_edge is not None:
              new_global_edges.append(new_global_edge)
              positions[new_global_edge] = positions[edge]
              positions[-new_global_edge] = -positions[edge]

          new_refine_edges.update(set(map(abs, new_global_edges)))

      if len(new_refine_edges) == len(refine_edges):
        break

      refine_edges = new_refine_edges.copy()

    # create mapping from old index to new
    map_old_index_new = {}
    ret = self

    # loop over all refine edges
    for edge in refine_edges:
      # myposition is simply given by position if positions[edge] == 1 else 1 - position
      myposition = position if positions[edge] == 1 else (1 - position)
      # split edge at myposition
      ret = ret.split_edge_arclength((edge, myposition))
      # we map edge to the current max_edge_index
      map_old_index_new[edge], map_old_index_new[-edge] = ret.max_edge_index, -ret.max_edge_index

    # explanation forthcoming
    for face_index, mysides in sides.items():
      mysides = sides[face_index]
      abssides, local_boundary_edges = abs(mysides[0]), \
                                       IndexSequence([edge if i > 0 else edge[::-1] for i, edge in zip(*mysides)], OrientableTuple)
      relevant_abssides = [side for side in abssides if side in refine_edges]
      my_refine_edges = [local_boundary_edges[i] for i, side in enumerate(abssides) if side in refine_edges]
      for edge, local_edge in zip(relevant_abssides, my_refine_edges):
        myposition = local_position if positions[edge] == 1 else (1 - local_position)
        try:
          templates[face_index] = templates[face_index].refine(local_edge, positions=myposition)
        except Exception:
          pass

    new_sides = {}
    for face_index, (mysides, my_local_edges) in sides.items():
      vertex0 = my_local_edges[0][0]
      mytemplate = templates[face_index]
      offset, = [i for i, edge in enumerate(mytemplate.ordered_boundary_edges[0]) if edge[0] == vertex0]
      newsides = []
      for side in mysides:
        new_side = map_old_index_new.get(side, None)
        if new_side is None:
          newsides.append(side)
        else:
          newsides.extend([new_side, side] if side < 0 else [side, new_side])

      new_sides[face_index] = tuple( (i,) for i in IndexSequence(newsides).roll(offset) )

    for face_index, template in templates.items():
      ret.set_template(face_index, template, sides=new_sides[face_index])

    if not return_indexmap:
      return ret
    return ret, map_old_index_new

  def click_trefine(self, face_index):
    from canvas import GenerateVertexFromClick

    with GenerateVertexFromClick(self.take([face_index]), title='Click on the template refinement position') as handler:
      ei = handler.ei

    arclength_position = self.get_edges(ei[0]).verts[ei[1]]
    return self.trefine(ei[0], position=arclength_position)


def split_at_large_angle(network: EdgeNetwork, angle_thresh):
  for index, edge in zip(network.indices, network.edges):
    pointcloud = edge.points
    if ((pointcloud[0] - pointcloud[-1])**2).sum() < 1e-13:  # periodic
      pointcloud = pointcloud[:-1]
    vs = pointcloud[1:] - pointcloud[:-1]
    vs = vs / np.linalg.norm(vs, ord=2, axis=1)[:, None]
    v0s, v1s = vs[:-1], vs[1:]
    dot = np.clip((v0s * v1s).sum(1), -1, 1)
    v1_min_proj = v1s - dot[:, None] * v0s
    cross = np.round(v0s[:, 0] * v1_min_proj[:, 1] - v0s[:, 1] * v1_min_proj[:, 0], 10)
    angles = np.sign(cross) * np.arccos(dot)
    break_indices = np.where( np.abs(angles) > angle_thresh )[0] + 1
    if len(break_indices):
      break_index, *break_indices = break_indices
      return split_at_large_angle(network.split_edge((index, int(break_index))), angle_thresh)
  else:
    return network


def make_face_splitting_curves(network: EdgeNetwork, face_index, **kwargs):
  paths = {}
  pol = network._polygons[face_index]
  face_indices = network._face_indices[face_index]
  corner_indices = [-i for i in face_indices]
  for v0 in corner_indices:
    for v1 in corner_indices:
      if abs(v0) < abs(v1):
        print('Computing shortest path between vertices {} and {}.'.format(v0, v1))
        paths[(v0, v1)] = shortest_polygon_path(pol,
                                                network.get_vertex(v0),
                                                network.get_vertex(v1), **kwargs)
  return paths


def seek_preimage(network, g):
  from mapping import jitBSpline
  Ys = {}
  for index, edge in zip(network.indices, network.edges):
    Ys[index] = edge.transform(lambda x: jitBSpline.seek_preimage(g, x))
  return network.__class__.from_edgedict(Ys, network.face_indices.copy())


def plot(g, npoints=1001, xi=None, eta=None, ax=None, boundary=None, linewidth=0.3, linecolor='k'):

  if xi is None:
    xi = g.knotvector[0].knots

  if eta is None:
    eta = g.knotvector[1].knots

  vals = np.linspace(0, 1, npoints)

  Xi, Eta = np.meshgrid(xi, vals)
  x_lines = g.call(Xi.ravel(), Eta.ravel()).reshape(Xi.shape + (2,))

  Xi, Eta = np.meshgrid(vals, eta)
  y_lines = g.call(Xi.ravel(), Eta.ravel()).reshape(Xi.shape + (2,))

  import matplotlib.collections as col
  if ax is None:
    X = np.concatenate([x_lines, y_lines], axis=1)
    fig, ax = plt.subplots()
    ax.set_aspect( 'equal' )
    ax.set_xlim(X[..., 0].min() - 0.05, X[..., 0].max() + 0.05)
    ax.set_ylim(X[..., 1].min() - 0.05, X[..., 1].max() + 0.05)
  line_segments = col.LineCollection( x_lines.swapaxes(0, 1),
                                      linewidth=linewidth,
                                      color=linecolor )
  ax.add_collection(line_segments)
  line_segments = col.LineCollection( y_lines,  # .swapaxes(0, 1),
                                      linewidth=linewidth,
                                      color=linecolor )
  ax.add_collection(line_segments)

  if boundary is not None:
    ax.plot(*x_lines[:, 0].T, linewidth=boundary, color=linecolor, zorder=10)
    ax.plot(*x_lines[:, -1].T, linewidth=boundary, color=linecolor, zorder=10)
    ax.plot(*y_lines[0].T, linewidth=boundary, color=linecolor, zorder=10)
    ax.plot(*y_lines[-1].T, linewidth=boundary, color=linecolor, zorder=10)


# Knotvector factories

def scaled_with_average(network, degree=3, base=2, baseline_exponent=3, minexponent=2):
  assert degree >= 1
  assert minexponent >= 1
  assert baseline_exponent >= minexponent

  import math

  lengths = np.asarray([x.length for x in network.edges])
  average = np.average(lengths)

  knotvectors = {}

  baseknotvector = ko.KnotObject(knotvalues=np.linspace(0, 1, base), degree=degree)

  for length, index in zip(lengths, network.indices):
    myexponent = max(minexponent, round(math.log(length, average)) * baseline_exponent)
    knotvectors[index] = baseknotvector.ref(myexponent - 1)

  return knotvectors


if __name__ == '__main__':
  network = EdgeNetwork.fromfolder('networks/templated_invnetwork/').take([(8,)])
  network.refine_template_edge(38, .1).qplot()


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=2:ts=2:sts=2:foldignore=#
