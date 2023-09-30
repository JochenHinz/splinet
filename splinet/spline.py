from .network import Network, plot, FrozenDict
from .index import MultiEdge, IndexSequence
from .index import SplineEdge
from .template import MultiPatchTemplate, FaceTemplate, singlepatch_template
from .multipatch import MultiPatchBSplineGridObject, singlepatch

from mapping import go, ko, sol
from functools import reduce, cached_property
from matplotlib import pyplot as plt
import numpy as np
import itertools
from collections import defaultdict
from nutils import log
from matplotlib.colors import BASE_COLORS
from shapely import ops


# SPLINE NETWORK


class SplineNetworkBase(Network):

  @classmethod
  def fromfolder(cls, foldername):
    import pickle
    import os
    if not foldername.endswith('/'):
      foldername = foldername + '/'
    with open(foldername + 'edges.pkl', 'rb') as file:
      (indices, edges) = zip(*pickle.load(file).items())
    edges = tuple(SplineEdge.load(**edge) for edge in edges)
    with open(foldername + 'sides.pkl', 'rb') as file:
      sides = pickle.load(file)

    maps = {}
    for name in os.listdir(foldername + 'maps/'):
      key = tuple(map(int, name[:-4].split('_')))
      with open(foldername + 'maps/' + name, 'rb') as file:
        try:
          maps[key] = MultiPatchBSplineGridObject.load(**pickle.load(file))
        except Exception as ex:
          print("Failed to load mapping {} with exception '{}'.".format(key, ex))

    with open(foldername + 'templates.pkl', 'rb') as file:
      templates = pickle.load(file)
    templates = {key: MultiPatchTemplate.load(**template) for key, template in templates.items()}

    return cls(indices, edges, sides, templates, maps=maps)

  def tofolder(self, foldername):
    import pickle
    import os
    if not foldername.endswith('/'):
      foldername = foldername + '/'
    if not os.path.exists(foldername):
      os.mkdir(foldername)

    with open(foldername + '/edges.pkl', 'wb') as file:
      pickle.dump({index: edge._save() for index, edge in zip(self.indices, self.edges)}, file)
    with open(foldername + '/sides.pkl', 'wb') as file:
      pickle.dump({face_index: tuple(map(tuple, side)) for face_index, side in self.sides.items()}, file)

    if not os.path.exists(foldername + 'maps/'):
      os.mkdir(foldername + 'maps/')

    for key, _map in self.maps.items():
      name = '_'.join(list(map(str, key))) + '.pkl'
      with open(foldername + 'maps/' + name, 'wb') as file:
        pickle.dump(_map._save(), file)

    templates = {}
    for face_index, template in self.templates.items():
      templates[face_index] = template.template._save()
    with open(foldername + 'templates.pkl', 'wb') as file:
      pickle.dump(templates, file)

  def _lib(self):
    return {'indices': self.indices,
            'edges': self.edges,
            'sides': self.sides,
            'templates': {key: value.template for key, value in self.templates.items()},
            'maps': self.maps.copy() }

  def __init__(self, indices, edges, sides, templates, maps=None, undo=None):
    self.sides = FrozenDict({key: IndexSequence([IndexSequence(val, dtype=np.int64) for val in value], dtype=IndexSequence) for key, value in sides.items()})

    face_indices = {key: value.ravel() for key, value in self.sides.items()}
    # the super init will make sure that indices edges and face_indices are compatible
    super().__init__(indices, edges, face_indices, np.int64, SplineEdge, undo=undo)

    maps = dict(maps or {})
    self.maps = FrozenDict({key: maps[key] for key in self.sides.keys() if key in maps})

    if any( len(side.ravel()) > len(side) for side in self.sides.values() ):
      raise NotImplementedError('Assigning multiple edges to one side is prohibited for now.')

    self.templates = FrozenDict({key: FaceTemplate(self, key, templates[key], self.sides[key]) for key in self.sides.keys()})
    self._polygons = FrozenDict({key: MultiEdge([self.get_edges(index).as_Edge() for index in value]) for key, value in self.face_indices.items()})

    color_generator = itertools.cycle(sorted(set(BASE_COLORS.keys()) - set(['w'])))
    all_face_numbers = sorted(list(set([face_index[0] for face_index in self.face_indices.keys()])))
    map_face_number_color = dict(zip(all_face_numbers, color_generator))
    self._colors = FrozenDict({face_index: map_face_number_color[face_index[0]] for face_index in self.face_indices.keys()})

    assert all( isinstance(val, MultiPatchBSplineGridObject) for val in self.maps.values() )
    assert all( len(self.sides[face_index]) == len(mapping.boundary_edges) for face_index, mapping in self.maps.items() )

    for face_index, mapping in self.maps.items():
      ordered_boundary_edges, = mapping.ordered_boundary_edges
      for i, side in enumerate(self.sides[face_index]):
        myknotvector = SplineEdge.merge_SplineEdges(*[self.get_edges(s) for s in side]).knotvector
        degree = myknotvector.degree
        if degree > 1:
          myknotvector = myknotvector.to_c(1)
        if not myknotvector <= mapping.get_knotvector(ordered_boundary_edges[i]):
          raise AssertionError('The maps are incompatible with the edges.')

  @cached_property
  def edgedict(self):
    return {**dict(zip(self.indices, self.edges)),
            **dict(zip([-i for i in self.indices], [-edge for edge in self.edges]))}

  @property
  def max_edge_index(self):
    return max(self.indices)

  def get_edges(self, index):
    if np.isscalar(index):
      return super().get_edges((index,))[0]
    return super().get_edges(index)

  def plot_polygons(self, ax=None, indices=True, show=True, linewidth=1):
    if ax is None:
      fig, ax = plt.subplots()
    else:
      fig = ax.figure
    ax.set_aspect('equal')
    face_colors = self._colors
    for key, pol in self._polygons.items():
      color = face_colors[key]
      XY = pol.polygon.exterior.coords.xy
      ax.plot(*XY, color=color, linewidth=linewidth)
      ax.fill(*XY, color=color, alpha=0.1)
    if indices:
      for index, edge in zip(self.indices, self.edges):
        edge = edge.as_Edge()
        point = edge.points[y // 2] if (y := len(edge.points)) > 2 else edge.points.sum(0) / 2
        ax.text(*point, str(index), fontsize=8)
      for face_index, pol in self._polygons.items():
        try:  # in case the polygon is invalid
          point = np.concatenate(ops.polylabel(pol.polygon).coords.xy)
          ax.text(*point, str(face_index), fontsize=8, color=face_colors[face_index])
        except Exception:
          pass
      for i, (index, edge) in enumerate(zip(self.indices, self.edges)):
        edge = edge.as_Edge()
        for point in edge.vertices:
          ax.scatter(*point, c='k', s=linewidth, zorder=4)
    if not show:
      return fig, ax
    plt.show()

  qplot = plot_polygons

  def plot_grids(self, ax=None, plotkwargs=None, show=True, **kwargs):
    if plotkwargs is None:
      plotkwargs = {}
    if ax is None: fig, ax = self.plot_polygons(show=False, **kwargs)
    else: fig = ax.figure
    for face_index, mg in self.maps.items():
      for g in mg.break_apart():
        plot(g, ax=ax, **plotkwargs)
    if not show:
      return fig, ax
    plt.show()

  def take(self, face_indices):
    sides = {key: self.sides[key] for key in face_indices}
    templates = {key: self.templates[key].template for key in face_indices}
    maps = {key: self.maps[key] for key in face_indices if key in self.maps}
    indices = sorted(set(itertools.chain(*map(lambda x: abs(x.ravel()), sides.values()))))
    edges = self.get_edges(indices)
    return self.__class__(indices, edges, sides, templates, maps=maps)

  def set_map(self, face_index, mapping):
    maps = dict(self.maps)
    maps[face_index] = mapping
    # XXX: make this a self.edit call
    return self.__class__(**{**self._lib(), **{'maps': maps}})

  """ Topological stuff """

  @cached_property
  def index2faces(self):
    """ {abs(index): tuple(all faces that contain it)} """
    index2faces = defaultdict(set)
    # map each positive side to all faces that contain it (in either orientation)
    for face_index, sides in self.sides.items():
      for side in abs(sides.ravel()):
        index2faces[side].update({face_index})

    return {key: set(value) for key, value in index2faces.items()}

  @cached_property
  def face_index_neighbours(self):
    """ Return a dict with dict[face_index] = {side_on_face: face_index_that_neighbours_side for all side_on_face in sides[face_index]}. """

    index2face = self.index2faces

    face_neighbours = {}

    for face_index, sides in self.sides.items():
      myneighbours = {}
      for side in abs(sides.ravel()):
        neighbours = set(index2face[side]) - {face_index}
        if len(neighbours) > 0:  # index can have at most 2 faces it's contained in
          neighbour, = neighbours  # make sure it's just one at this point. Else that's a bug
          myneighbours[side] = neighbour

      face_neighbours[face_index] = myneighbours

    return face_neighbours

  @property
  def face_neighbours(self):
    """ Return {face_index: tuple(neighbouring face indices)} """
    return {key: tuple(set(value.values())) for key, value in self.face_index_neighbours.items()}

  @cached_property
  def immediate_edge_neighbours(self):
    neighbours = defaultdict(set)
    for face_index, template in self.templates.items():
      mypairings = (tuple(a.ravel()[0] for a in pairing) for pairing in template.edge_pairings())
      for edge0, edge1 in mypairings:
        neighbours[edge0].update({edge1})
        neighbours[-edge0].update({-edge1})
        neighbours[edge1].update({edge0})
        neighbours[-edge1].update({-edge0})

    return {index: tuple(sorted(value)) for index, value in neighbours.items()}

  @cached_property
  def edge_neighbours(self):
    neighbours = {index: set(value) for index, value in self.immediate_edge_neighbours.items()}

    while True:
      newneighbours = {key: value.copy() for key, value in neighbours.items()}
      for index, myneighbours in neighbours.items():
        for myneighbour in myneighbours:
          newneighbours[index].update(neighbours[myneighbour])
      if newneighbours == neighbours:
        break
      neighbours = newneighbours

    return {index: tuple(sorted(value)) for index, value in neighbours.items()}

  def unify_edges(self):
    """ Prolong all neighbouring edges to a unified knotvector """
    edge_neighbours = self.edge_neighbours
    knotvectors = {}
    for index in self.indices:
      myneighbours = edge_neighbours[index]
      if index in knotvectors: continue
      all_kvs = [self.get_edges(i).g.knotvector[0] for i in myneighbours]
      myknotvector = reduce(lambda x, y: x + y, all_kvs)
      myknotvector_flipped = myknotvector.flip()
      for i in myneighbours:
        knotvectors[abs(i)] = myknotvector if i > 0 else myknotvector_flipped

    knotvectors = {key: ko.TensorKnotObject([val]) for key, val in knotvectors.items()}

    newedges = [ SplineEdge.from_gridobject(go.refine_GridObject(edge.g, knotvectors[index]))
                            for index, edge in zip(self.indices, self.edges) ]

    return self.edit(edges=newedges)

  """ Mixed stuff """

  def solve(self):
    fail = []
    for face_index, mg in self.maps.items():
      mg.set_cons_from_x()
      try:
        sol.forward_laplace(mg)
        sol.Blechschmidt(mg)
        sol.elliptic_partial(mg, eps=1e-6)
      except Exception:
        fail.append(face_index)

    return fail

  def ndofs(self, trunc=7):
    assert len(self.maps) == len(self.face_indices)
    dofs = set()
    for mg in self.maps.values():
      dofs.update(set(map(tuple, np.round(mg.x.reshape([-1, 2]), trunc))))

    return len(dofs)

  def refine_all_maps(self):
    if not all( len(template.patches) == 1 for template in self.templates.values() ):
      raise AssertionError('Please apply all mappings first.')

    if not set(self.maps.keys()) == set(self.face_indices.keys()):
      raise AssertionError('Please set all maps first.')

    ret = self
    for face_index in self.maps.keys():
      G = self.maps[face_index]
      myknotvectors = { (0, 2): G.get_knotvector((0, 2)).ref(1),
                        (0, 1): G.get_knotvector((0, 1)).ref(1) }
      ret = ret.set_map(face_index, ret.maps[face_index].ref_to(myknotvectors))

    return ret

  def make_maps(self, overwrite=True, baseknotvector=None, stack_c=0, **kwargs):
    maps = dict(self.maps)
    for key in self.face_indices.keys():
      if not overwrite and key in maps: continue
      maps[key] = self.templates[key] \
                      .make_MultiPatchBSplineGridObject(stack_c=stack_c, knotvectors=baseknotvector, **kwargs)
    return self.edit(maps=maps)

  def to_ascii(self, foldername):
    if not all( len(template.patches) == 1 for template in self.templates.values() ):
      raise AssertionError('Please apply all mappings first.')

    if not set(self.maps.keys()) == set(self.face_indices.keys()):
      raise AssertionError('Please set all maps first.')

    if not foldername.endswith('/'):
      foldername = foldername + '/'

    import os
    if not os.path.exists(foldername):
      os.mkdir(foldername)

    from export import to_ASCII

    for face_index, g in self.maps.items():
      to_ASCII(g.break_apart()[0], foldername + '_'.join(str(face_index)[1:-1].split(', ')))


class SplineNetwork(SplineNetworkBase):

  @classmethod
  def from_EdgeNetwork(cls, network):
    assert len(network.fits) == len(network.indices)
    assert set(network.face_indices.keys()) == set(network.templates.keys())
    # edgedict = dict(zip(network.indices, network.edges))
    edges = [SplineEdge.from_gridobject(network.fits[key]) for key in network.indices]
    templates = {key: network.templates[key].template for key in network.face_indices.keys()}
    sides = {key: network.templates[key].sides for key in network.face_indices.keys()}
    return cls(network.indices, edges, sides, templates)

  def _lib(self):
    return {'indices': self.indices,
            'edges': self.edges,
            'sides': self.sides,
            'templates': {key: value.template for key, value in self.templates.items()},
            'maps': self.maps.copy() }

  def apply_mapping(self, face_index):
    assert face_index[1] == 0, NotImplementedError
    assert face_index in self.maps
    maps = dict(self.maps)
    mapping = maps.pop(face_index)
    n = self.max_edge_index + 1
    patches = mapping.patches
    indices, edges = tuple(self.indices), tuple(self.edges)
    bindices, = mapping.ordered_boundary_edges
    newsides = dict(self.sides)
    templates = {key: value.template for key, value in self.templates.items()}
    del newsides[face_index]
    del templates[face_index]
    sides = self.sides[face_index]
    map_edge_new_index = {}
    for i, (patch, g) in enumerate(zip(patches, mapping.break_apart())):
      mysides = []
      for side, edge, sign in zip(('bottom', 'right', 'top', 'left'),
                                  ((patch[0], patch[2]), (patch[2], patch[3]), (patch[3], patch[1]), (patch[1], patch[0])),
                                  (1, 1, -1, -1)):
        if edge in bindices:
          edge_index = bindices.index(edge)
          mysides.append( sides[edge_index] )
        elif edge[::-1] in bindices:
          edge_index = bindices.index(edge[::-1])
          mysides.append(-sides[edge_index])
        elif edge in map_edge_new_index:
          mysides.append([map_edge_new_index[edge]])
        else:
          indices = indices + (n,)
          n += 1
          edges = edges + (sign * SplineEdge.from_gridobject(g(side)),)
          mysides.append([indices[-1]])
          map_edge_new_index[edge] = indices[-1]
          map_edge_new_index[edge[::-1]] = - indices[-1]
      my_face_index = face_index + (i,)
      newsides[my_face_index] = mysides
      G = singlepatch(dict(zip([(0, 2), (0, 1)], g.knotvector)))
      maps[my_face_index] = G
      G.x = g.x
      templates[my_face_index] = singlepatch_template()
    if all(len(side) == 4 for side in newsides.values()):
      return SinglePatchSplineNetwork(indices, edges, newsides, templates, maps, undo=self)
    return self.__class__(indices, edges, newsides, templates, maps, undo=self)

  def apply_all_mappings(self):
    face_indices = list(self.maps.keys())
    assert set(face_indices) == set(self.face_indices.keys()), 'Please set all maps first.'
    ret = self
    nfaces = len(face_indices)
    for i, face_index in enumerate(face_indices, 1):
      log.info('Applying map {} of {}.'.format(i, nfaces))
      ret = ret.apply_mapping(face_index)
    return ret


class SinglePatchSplineNetwork(SplineNetworkBase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert all(len(face) == 4 for face in self.face_indices.values())

  def unify_grids(self):

    if not set(self.maps.keys()) == set(self.face_indices.keys()):
      raise AssertionError('Please set all maps first.')

    knotvectors = defaultdict(set)

    # add all knotvectors of one side to that side
    for face_index, temp in self.templates.items():
      sides = self.sides[face_index].ravel()
      g, = self.maps[face_index].break_apart()
      for side, direction, sign in zip(sides, (0, 1, 0, 1), (1, 1, -1, -1)):
        knotvector = g.knotvector[direction]
        if sign == -1:
          knotvector = knotvector.flip()
        knotvectors[side].update({knotvector})
        knotvectors[-side].update({knotvector.flip()})

    edge_neighbours = self.edge_neighbours

    newknotvectors = {}

    for index in self.indices:
      myneighbours = edge_neighbours[index]
      if index in newknotvectors: continue
      all_kvs = set.union(*[knotvectors[i] for i in myneighbours])
      myknotvector = reduce(lambda x, y: x + y, all_kvs)
      myknotvector_flipped = myknotvector.flip()
      for i in myneighbours:
        newknotvectors[i] = myknotvector
        newknotvectors[-i] = myknotvector_flipped

    knotvectors = newknotvectors

    maps = {}
    for face_index, G in self.maps.items():
      g, = G.break_apart()
      sides = self.sides[face_index].ravel()
      newknotvector = knotvectors[sides[0]] * knotvectors[sides[1]]
      g = go.refine_GridObject(g, newknotvector)
      patch, = self.templates[face_index].patches
      maps[face_index] = singlepatch( {(patch[0], patch[2]): g.knotvector[0],
                                                  patch[:2]: g.knotvector[1]}, x=g.x )

    gos = [go.refine_GridObject(edge.g, ko.TensorKnotObject([knotvectors[i]]))
                                  for i, edge in zip(self.indices, self.edges)]
    newedges = [SplineEdge.from_gridobject(g) for g in gos]

    return self.edit(maps=maps, edges=newedges)

  def reverse_orientation(self, face_indices):
    sides = dict(self.sides)
    maps = dict(self.maps)

    for face_index in face_indices:
      mysides = list(sides[face_index])
      sides[face_index] = [-mysides[0], -mysides[3], -mysides[2], -mysides[1]]

      if face_index in maps:
        patch, = self.templates[face_index].patches
        g = maps[face_index]
        new_g = singlepatch( {(patch[0], patch[2]): g.knotvectors[(patch[0], patch[2])].flip(),
                              (patch[0], patch[1]): g.knotvectors[(patch[0], patch[1])]},
                              x=g.break_apart()[0].x.tensor[::-1].ravel() )
        maps[face_index] = new_g

    return self.edit(sides=sides, maps=maps)


class LinearNetwork(SinglePatchSplineNetwork):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert all( edge.g.knotvector[0].degree == 1 for edge in self.edges )

  def to_gmsh(self, **kwargs):
    return to_gmsh(self, **kwargs)


def to_gmsh_(p1: LinearNetwork, trunc=7):
  assert len(p1.maps) == len(p1.sides)
  for face_index, sides in p1.sides.items():
    sides = sides.ravel()
    g, = p1.maps[face_index].break_apart()
    for side, local_side, orientation in zip(sides, ['bottom', 'right', 'top', 'left'], [1, 1, -1, -1]):
      if not (np.round(g[local_side], trunc) == np.round(p1.get_edges(orientation * side).g.x, trunc)).all():
        raise AssertionError('Either the grids are incompatible with the edges or the truncation parameter is too high.')

  map_points_index = {}
  points = []
  n = 0
  for face_index in sorted(p1.maps):
    for x in map(tuple, np.round(p1.maps[face_index].x.reshape([-1, 2]), trunc)):
      if x in map_points_index: continue
      map_points_index[x] = n
      points.append(x)
      n += 1

  points = np.array(points)

  cell_sets = defaultdict(list)
  ncells = sum( len(mg.break_apart()[0].domain) for mg in p1.maps.values() )
  indices = np.empty((ncells, 4), dtype=int)
  n = 0
  for face_index, mg in p1.maps.items():
    cell_name = 'Face_{}'.format(face_index[0])
    g, = mg.break_apart()
    X = np.round(g.x.tensor, trunc)
    for xs in np.lib.stride_tricks.sliding_window_view(X, (2, 2), axis=(0, 1)).reshape([-1, 2, 4]):
      a, b, c, d = xs.T
      indices[n, 0] = map_points_index[tuple(a)]
      indices[n, 1] = map_points_index[tuple(c)]
      indices[n, 2] = map_points_index[tuple(d)]
      indices[n, 3] = map_points_index[tuple(b)]
      cell_sets[cell_name].append(n)
      n += 1

  import meshio
  return meshio.Mesh(points, {'quad': indices}, cell_sets=cell_sets)


def to_gmsh(p1: LinearNetwork, filename='test.msh', trunc=7):
  assert len(p1.maps) == len(p1.sides)
  for face_index, sides in p1.sides.items():
    sides = sides.ravel()
    g, = p1.maps[face_index].break_apart()
    for side, local_side, orientation in zip(sides, ['bottom', 'right', 'top', 'left'], [1, 1, -1, -1]):
      if not (np.round(g[local_side], trunc) == np.round(p1.get_edges(orientation * side).g.x, trunc)).all():
        import ipdb
        ipdb.set_trace()
        raise AssertionError('Either the grids are incompatible with the edges or the truncation parameter is too high.')

  map_points_index = {}
  points = []
  n = 1
  for face_index in sorted(p1.maps):
    for x in map(tuple, np.round(p1.maps[face_index].x.reshape([-1, 2]), trunc)):
      if x in map_points_index: continue
      map_points_index[x] = n
      points.append(x)
      n += 1

  import gmsh
  gmsh.initialize()

  try:
    # gmsh.model.add('test')
    # tags = set()
    # for edge in p1.edges:
    #   for x in map(tuple, np.round(edge.g.x.reshape([-1, 2]), trunc)):
    #     mytag = map_points_index[x]
    #     if mytag not in tags:
    #       gmsh.model.geo.addPoint(*x, 0, 1e-2, map_points_index[x])
    #       tags.update({mytag})

    # for index, edge in zip(p1.indices, p1.edges):
    #   gmsh.model.geo.addPolyline([ map_points_index[x] for x in map(tuple, np.round(edge.g.x.reshape([-1, 2]), trunc)) ], index)

    # for i, side in enumerate(p1.sides.values(), 1):
    #   gmsh.model.geo.addCurveLoop(side.ravel(), i)
    #   gmsh.model.geo.addPlaneSurface([i], i)
    #   gmsh.model.geo.addPhysicalGroup(2, [i], name='Face {}'.format(i))

    surf = gmsh.model.addDiscreteEntity(2)

    gmsh.model.mesh.addNodes(2, surf, [map_points_index[point] for point in points], np.concatenate([np.array(points), np.zeros(len(points))[:, None]], axis=1).ravel())

    points = np.array(points)

    cell_sets = defaultdict(list)
    element_groups = defaultdict(list)
    ncells = sum( len(mg.break_apart()[0].domain) for mg in p1.maps.values() )
    indices = np.empty((ncells, 4), dtype=int)
    n = 0
    for face_index, mg in p1.maps.items():
      cell_name = face_index[0] + 1
      g, = mg.break_apart()
      X = np.round(g.x.tensor, trunc)
      for xs in np.lib.stride_tricks.sliding_window_view(X, (2, 2), axis=(0, 1)).reshape([-1, 2, 4]):
        a, b, c, d = xs.T
        indices[n, 0] = map_points_index[tuple(a)]
        indices[n, 1] = map_points_index[tuple(c)]
        indices[n, 2] = map_points_index[tuple(d)]
        indices[n, 3] = map_points_index[tuple(b)]
        cell_sets[cell_name].extend([map_points_index[tuple(i)] for i in (a, c, d, b)])
        element_groups[cell_name].append(n + 1)
        n += 1

    gmsh.model.mesh.addElementsByType(surf, 3, np.arange(len(indices)) + 1, indices.ravel())

    partitions = np.concatenate([ i * np.ones(len(group), dtype=int) for i, group in element_groups.items() ])
    elementTags = np.concatenate(list(element_groups.values()))
    gmsh.model.mesh.partition(len(element_groups), elementTags=elementTags, partitions=partitions)

    gmsh.write(filename)

    gmsh.finalize()
    return surf

  except Exception as ex:
    gmsh.finalize()
    raise Exception("Failed with exception '{}'.".format(ex))


def p1_splinenetwork(snetwork: SplineNetwork):
  assert all(len(face) == 4 for face in snetwork.face_indices.values())
  assert len(snetwork.maps) == len(snetwork.faces)

  gos = [edge.g.to_p(1) for edge in snetwork.edges]

  for g, edge in zip(gos, snetwork.edges):
    g.x = edge.g.toscipy()(g.knots[0]).ravel()

  knotvectors = {index: gos[i].knotvector[0] for i, index in enumerate(snetwork.indices)}
  knotvectors.update({-index: kv.flip() for index, kv in knotvectors.items()})

  edges = [SplineEdge.from_gridobject(g) for g in gos]

  p1network = snetwork.edit(edges=edges, maps={})

  maps = {}
  for face in snetwork.maps.keys():
    sides = snetwork.sides[face].ravel()
    newknotvector = knotvectors[sides[0]] * knotvectors[sides[1]]
    g, = snetwork.maps[face].break_apart()
    X = g.toscipy()(*[kv.knots for kv in newknotvector]).ravel()
    patch, = snetwork.templates[face].patches
    maps[face] = singlepatch( {(patch[0], patch[2]): newknotvector[0],
                                          patch[:2]: newknotvector[1]}, x=X )

  return p1network.edit(maps=maps)


def collocate_edge(target: go.TensorGridObject, baseknotvector: ko.KnotObject, btol=1e-3):
  from mapping.aux import prolongation_matrix
  from nutils import function

  assert baseknotvector.degree == 1
  assert len(target.knotvector) == 1

  target_kv, = target.knotvector
  target_kv_p1 = target_kv.to_p(1)

  knotvector_p1 = baseknotvector

  while True:
    g_p1 = go.TensorGridObject(knotvector=ko.TensorKnotObject([knotvector_p1]), targetspace=2)
    g_p1.x = target.toscipy()(knotvector_p1.knots).ravel()

    knotvector_p1_prol = knotvector_p1 + target_kv_p1
    g_p1_prol = go.refine_GridObject(g_p1, ko.TensorKnotObject([knotvector_p1_prol]))

    T_p0 = prolongation_matrix(knotvector_p1.to_p(0), knotvector_p1_prol.to_p(0))
    T_p0_restrict = T_p0.T

    basis0 = g_p1_prol.domain.basis_discont(0)

    target_kv_prol = target_kv + knotvector_p1.to_p(target_kv.degree)

    target_prol = go.refine_GridObject(target, ko.TensorKnotObject([target_kv_prol]))

    residual = g_p1_prol.domain.integrate( ((g_p1_prol.mapping - target_prol.mapping)**2).sum() * basis0 * function.J(g_p1_prol.geom), degree=10 )
    residual_restrict = np.sqrt(T_p0_restrict @ residual)

    indices, = np.where(residual_restrict > btol)
    if len(indices) == 0:
      break

    log.warning('Maximum error: {:.3g}, refining {} elements'.format(residual_restrict.max(), len(indices)))

    knotvector_p1 = knotvector_p1.ref_by(indices)

  return g_p1


def collocate_SplineNetwork(snetwork, baseknotvector, btol=1e-3):
  assert all(len(face) == 4 for face in snetwork.face_indices.values())
  assert len(snetwork.maps) == len(snetwork.faces)

  nedges = len(snetwork.edges)
  knotvectors = {}
  for i, (index, edge) in enumerate(zip(snetwork.indices, snetwork.edges)):
    log.warning('Fitting edge {} of {}.'.format(i, nedges))
    fit = collocate_edge(edge.g, baseknotvector, btol=btol)
    kv = fit.knotvector[0]
    knotvectors[index] = {kv}
    knotvectors[-index] = {kv.flip()}

  # take the union across neighbouring sides
  while True:
    newknotvectors = {key: value.copy() for key, value in knotvectors.items()}
    for face_index, temp in snetwork.templates.items():
      sides = snetwork.sides[face_index].ravel()
      othersides = [-sides[2], -sides[3], -sides[0], -sides[1]]
      for side, otherside in zip(sides, othersides):
        kvs, otherkvs = newknotvectors[side], newknotvectors[otherside]
        newknotvectors[side] = newknotvectors[otherside] = set.union(kvs, otherkvs)

        kvs, otherkvs = newknotvectors[-side], newknotvectors[-otherside]
        newknotvectors[-side] = newknotvectors[-otherside] = set.union(kvs, otherkvs)

    if newknotvectors == knotvectors:
      knotvectors = newknotvectors
      break
    knotvectors = newknotvectors

  knotvectors = {edge: reduce(lambda x, y: x + y, value) for edge, value in knotvectors.items()}

  edges = []
  for i, edge in zip(snetwork.indices, snetwork.edges):
    kv_p1 = knotvectors[i]
    g_p1 = go.TensorGridObject(knotvector=ko.TensorKnotObject([kv_p1]), targetspace=2)
    g_p1.x = edge.g.toscipy()(kv_p1.knots).ravel()
    edges.append(SplineEdge.from_gridobject(g_p1))

  p1network = snetwork.edit(edges=edges, maps={})

  maps = {}
  for i, face in enumerate(snetwork.maps.keys()):
    log.warning('Collocating face {} of {}.'.format(i, nedges))
    sides = snetwork.sides[face].ravel()
    newknotvector = (knotvectors[sides[0]] + knotvectors[-sides[2]]) * (knotvectors[sides[1]] + knotvectors[-sides[3]])
    g, = snetwork.maps[face].break_apart()

    X = g.toscipy()(*[kv.knots for kv in newknotvector]).ravel()
    patch, = snetwork.templates[face].patches
    maps[face] = singlepatch( {(patch[0], patch[2]): newknotvector[0],
                                          patch[:2]: newknotvector[1]}, x=X )

  return p1network.edit(maps=maps)


def orthogonalize_SplineNetwork(snetwork, constrain_n=True, **kwargs):
  controlmaps = orthogonal_controlmaps(snetwork, **kwargs)
  fail = []
  for face_index, (f, sides) in controlmaps.items():
    g, = snetwork.maps[face_index].break_apart()
    g.set_cons_from_x()
    g.controlmap = f.mapping
    if constrain_n:
      for side in set(g.sides) - set(sides):
        direction = {'bottom': 1, 'top': 1, 'left': 0, 'right': 0}[side]
        g.cons = g.project( g.mapping.grad(g.geom)[:, direction],
                            onto=g.basis.grad(g.geom)[:, direction].vector(2),
                            domain=g.domain.boundary[side], constrain=g.cons)
    try:
      if not f.defects_discrete():
        raise AssertionError

      sol.elliptic_partial(g)
      snetwork.maps[face_index].x = g.x
    except Exception:
      fail.append(face_index)

  return fail, controlmaps