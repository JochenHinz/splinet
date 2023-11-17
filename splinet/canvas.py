from matplotlib import pyplot as plt
from shapely import geometry
from .aux import _normalized
import numbers
import numpy as np
from itertools import chain


class MatplotlibEventHandler:

  def __init__(self, ax):
    self.ax = ax
    self.funcs = tuple(func for func in dir(self) if func.startswith('on_')
                                                  and callable(getattr(self, func)))
    self.xlim0 = ax.get_xlim()
    self.ylim0 = ax.get_ylim()
    assert len(self.funcs) > 0, AssertionError

  def connect(self):
    for _func in self.funcs:
      func = getattr(self, _func)
      setattr(self, 'cid' + _func[3:], self.ax.figure.canvas.mpl_connect(func.__doc__, func))

  def disconnect(self):
    for func in self.funcs:
      self.ax.figure.canvas.mpl_disconnect('cid' + func[3:])

  def draw(self):
    self.ax.figure.canvas.draw()

  def on_key(self, event):
    'key_press_event'
    if event.key == 'w':
      ymin, ymax = self.ax.get_ylim()
      xmin, xmax = self.ax.get_xlim()
      x, y = event.xdata, event.ydata
      self.ax.set_xlim(max(self.xlim0[0], x - (xmax - xmin) / 4),
                       min(self.xlim0[1], x + (xmax - xmin) / 4))
      self.ax.set_ylim(max(self.ylim0[0], y - (ymax - ymin) / 4),
                       min(self.ylim0[1], y + (ymax - ymin) / 4))
      self.draw()
    elif event.key == 'e':
      ymin, ymax = self.ax.get_ylim()
      xmin, xmax = self.ax.get_xlim()
      x, y = event.xdata, event.ydata
      self.ax.set_xlim(max(self.xlim0[0], x - (xmax - xmin) * 2),
                       min(self.xlim0[1], x + (xmax - xmin) * 2))
      self.ax.set_ylim(max(self.ylim0[0], y - (ymax - ymin) * 2),
                       min(self.ylim0[1], y + (ymax - ymin) * 2))
      self.draw()

  def __enter__(self):
    self.connect()
    self.draw()
    plt.show()
    return self

  def close(self):
    plt.close('all')
    self.disconnect()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      plt.close('all')


class MatplotlibEventHandlerWithNetwork(MatplotlibEventHandler):

  def __init__(self, network, plot_args=None, title=None, ax=None, plot_polygons=True):

    if plot_args is None:
      plot_args = {}

    self.network = network

    # either plot the polygons or don't
    if plot_polygons:
      fig, ax = self.network.plot_polygons(show=False, ax=ax, **plot_args)
    else:
      assert ax is not None

    if title is not None:
      ax.figure.suptitle(str(title))

    super().__init__(ax)


class HighlightEdgeMixin:

  def _remove_highlighted_line(self):
    assert self.edge_highlighted
    del self.ax.lines[self.line]
    self.v0.remove()
    self.v1.remove()

  def __init__(self, *args, indices=None, efac=None, **kwargs):
    super().__init__(*args, **kwargs)  # forward to 2nd super() __init__
    if efac is None: efac = .025
    assert len(self.network.face_indices) == 1
    allindices = tuple(list(self.network.face_indices.values())[0])
    if indices is None:
      indices = allindices
    assert len(set(map(abs, indices))) == len(indices), 'Duplicate indices found.'
    assert set(map(abs, indices)).issubset(set(map(abs, allindices)))
    self.indices = tuple(indices)
    self.edges = tuple( self.network.get_edges(edge) for edge in self.indices )
    self.linestrings = tuple( geometry.LineString(edge.points) for edge in self.edges )
    self.minind = None
    self.edge_highlighted = False
    self.line = None
    self.v0, self.v1 = None, None
    if len(allindices) > 1:
      vertices = np.stack([ edge.vertices[0] for edge in self.network.get_edges(allindices) ])
      self.r = efac * np.linalg.norm([ vertices[:, i].max()
                                    - vertices[:, i].min() for i in range(2) ])
    else:
      edge, = self.network.get_edges(allindices)
      X = edge.points
      self.r = efac * np.linalg.norm([X[:, i].max() - X[:, i].min() for i in range(2)])

  def on_motion(self, event):
    'motion_notify_event'
    if event.xdata is not None:
      x = geometry.Point(np.array([event.xdata, event.ydata]))
      dist = np.array([ line.distance(x) for line in self.linestrings ])
      minind = np.argmin(dist)
      if dist[minind] < self.r:
        edge = self.edges[minind]
        if self.edge_highlighted:
          self._remove_highlighted_line()
        self.minind = minind
        self.ax.plot(*edge.points.T, linewidth=3, c='k')
        self.line = len(self.ax.lines) - 1
        self.v0 = self.ax.scatter(*edge.vertices[0], c='k', s=40)
        self.v1 = self.ax.scatter(*edge.vertices[1], c='k', s=40)
        self.edge_highlighted = True
        self.draw()
      else:
        if self.edge_highlighted:
          self._remove_highlighted_line()
          self.edge_highlighted = False
          self.draw()


class SelectEdges(HighlightEdgeMixin, MatplotlibEventHandlerWithNetwork):

  def __init__(self, *args, nclicks=None, **kwargs):
    kwargs.setdefault('title', 'Click to select edges, press space to finalize')
    super().__init__(*args, **kwargs)
    if nclicks is None:
      nclicks = np.inf
    self.nclicks = nclicks
    self.clicked_edges = []

  def on_press(self, event):
    'button_press_event'
    if self.edge_highlighted:
      minind = self.minind
      self.clicked_edges.append(self.indices[minind])
      edge = self.edges[minind]
      self.ax.plot(*edge.points.T, linewidth=3, c='r')
      self.ax.scatter(*edge.vertices[0], c='r', s=40)
      self.ax.scatter(*edge.vertices[1], c='r', s=40)
      self.indices = self.indices[:minind] + self.indices[minind + 1:]
      self.edges = self.edges[:minind] + self.edges[minind + 1:]
      self.linestrings = self.linestrings[:minind] + self.linestrings[minind+1:]
      self.draw()
      if len(self.linestrings) == 0 or len(self.clicked_edges) == self.nclicks: self.close()

  def on_key(self, event):
    'key_press_event'
    if event.key == ' ': self.close()
    super().on_key(event)


def select_edges(network, face, **kwargs):

  with SelectEdges(network.take([face]), **kwargs) as handler:
    clicked_edges = handler.clicked_edges

  return clicked_edges


class GenerateVertexFromClick(HighlightEdgeMixin, MatplotlibEventHandlerWithNetwork):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_press(self, event):
    'button_press_event'
    if self.edge_highlighted:
      data = [event.xdata, event.ydata]
      point = geometry.Point(np.array(data))
      edge_index = self.indices[self.minind]
      edge = self.edges[self.minind]
      point, index_on_edge = edge.get_nearest_point(data, return_index=True)
      self.ei = (edge_index, np.clip(index_on_edge, 1, len(edge) - 2))
      self.close()


def generate_vertex_from_click(network, face, title=None, plot=True):
  if title is None:
    title = 'Click on the desired position'

  snetwork = network.take((face,))

  with GenerateVertexFromClick(snetwork, title=title) as handler:
    ei = handler.ei

  if plot:
    snetwork.split_edge(ei).qplot()

  return network.split_edge(ei)


def generate_virtual_point_from_click(network, face, title=None, plot=True):
  if title is None:
    title = 'Click on the desired position'

  snetwork = network.take((face,))

  with GenerateVertexFromClick(snetwork, title=title, plot_args={'vpoints': True}) as handler:
    ei = handler.ei

  if plot:
    snetwork.add_virtual_point(ei)
    snetwork.qplot(vpoints=True)

  network.add_virtual_point(ei)


def half_edge(network, face, **kwargs):

  kwargs.setdefault('title', 'Select edge to split in half')
  index, = select_edges(network, face, nclicks=1, **kwargs)

  return network.split_edge((index, .5))


class HighlightFaceMixin:

  # XXX: expensive operation, make cheaper

  def _remove_highlighted_face(self):
    assert self.face_highlighted
    self.ax.patches.remove(self.fill)
    self.fill = None
    self.face_highlighted = False
    self.face = None

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)  # forward to 2nd super() __init__
    assert len(self.network.face_indices) > 1
    self.ordered_faces = tuple(sorted(self.network.face_indices.keys()))
    self.polygons = tuple(self.network.polygons[key].polygon for key in self.ordered_faces)
    self.face = None
    self.face_highlighted = False
    self.fill = None
    self.fills = self.ax.fill( *list(chain.from_iterable([[*pol.exterior.xy, self.network._colors[face]] for face, pol in zip(self.ordered_faces, self.polygons)])),
                                alpha=.8)
    self.ax.patches = self.ax.patches[:-len(self.fills)]

  def on_motion(self, event):
    'motion_notify_event'
    if event.xdata is not None:
      x = geometry.Point(np.array([event.xdata, event.ydata]))
      for i, pol in enumerate(self.polygons):
        if x.within(pol):
          face = i
          break
      else:
        face = None
      if (self.face is None and face is not None) or (face is not None and face != self.face):
        if self.face_highlighted:
          self._remove_highlighted_face()
        pol = self.polygons[face]
        self.fill = self.fills[face]
        self.ax.add_patch(self.fill)
        self.face = face
        self.face_highlighted = True
        self.draw()
      else:
        # print('here0', self.face_highlighted)
        if self.face_highlighted:
          self._remove_highlighted_face()
          self.draw()
    else:
      # print('here1', self.face_highlighted)
      if self.face_highlighted:
        self._remove_highlighted_face()
        self.draw()


class SelectFaces(MatplotlibEventHandlerWithNetwork):

  def __init__(self, *args, nclicks=None, **kwargs):
    kwargs.setdefault('title', 'Click to select faces, press space to finalize')
    super().__init__(*args, **kwargs)
    if nclicks is None:
      nclicks = len(self.network.face_indices)
    assert len(self.network.face_indices) >= nclicks
    self.ordered_faces = tuple(sorted(self.network.face_indices.keys()))
    self.polygons = tuple(self.network.polygons[key].polygon for key in self.ordered_faces)
    self.nclicks = nclicks
    self.clicked_faces = []

  def on_press(self, event):
    'button_press_event'
    if event.xdata is not None:
      x = geometry.Point(np.array([event.xdata, event.ydata]))
      for i, pol in enumerate(self.polygons):
        if x.within(pol):
          face = i
          break
    else:
      face = None
    if face is not None:
      self.clicked_faces.append(self.ordered_faces[face])
      pol = self.polygons[face]
      color = self.network._colors[self.ordered_faces[face]]
      self.ax.fill(*pol.exterior.coords.xy, color=color, alpha=0.8)
      self.ordered_faces = self.ordered_faces[:face] + self.ordered_faces[face+1:]
      self.polygons = self.polygons[:face] + self.polygons[face+1:]
      self.draw()
      if len(self.polygons) == 0 or len(self.clicked_faces) == self.nclicks: self.close()

  def on_key(self, event):
    'key_press_event'
    if event.key == ' ': self.close()
    super().on_key(event)


def select_faces(network, **kwargs):

  with SelectFaces(network, **kwargs) as handler:
    clicked_faces = handler.clicked_faces

  return clicked_faces


class GenerateClicks(MatplotlibEventHandlerWithNetwork):

  def __init__(self, *args, n=None, **kwargs):
    self.points = list()
    self.n = n and int(n)
    super().__init__(*args, **kwargs)

  def on_key(self, event):
    'key_press_event'
    if event.key == ' ' and self.n is None: self.close()
    super().on_key(event)

  def on_press(self, event):
    'button_press_event'
    x, y = event.xdata, event.ydata
    self.points.append(np.array([x, y]))

    if len(self.points) >= 2:
      self.ax.plot(*np.stack(self.points[-2:]).T, c='k')

    self.ax.scatter(x, y, c='k')
    self.draw()

    if self.n is not None:
      print('Created point {} / {}.'.format(len(self.points), self.n))
      if len(self.points) == self.n: self.close()


class GenerateClicksWithBoundaryConstrain(GenerateClicks):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert len(self.network.face_indices) == 1

  def close(self):
    for i in (0, len(self.points) - 1):
      point = self.points[i]
      self.network, self.points[i] = \
        self.network.split_at_nearest_point(point, return_point=True)
    super().close()


class HighlightVertexMixin:

  def __init__(self, *args, vfac=None, **kwargs):
    super().__init__(*args, **kwargs)  # forward to 2nd super() initializer
    if vfac is None: vfac = 0.0125
    vertices = []
    vertex_indices = []
    key, = self.network.faces
    self.nedges = len(self.network.face_indices[key])
    for i, index in enumerate(self.network.face_indices[key]):
      v0, v1 = self.network.get_edges(index).vertices
      vertices.append(v0)  # only take first vertex
      vertex_indices.append(-index)
    self.vertex_indices = tuple(vertex_indices)
    self.vertices = np.stack(vertices)
    self.rvertex = vfac * np.linalg.norm([ self.vertices[:, i].max()
                                         - self.vertices[:, i].min() for i in range(2) ])
    self.point_highlighted = False
    self.vminind = None
    self.highlightedpoint = None

  def on_motion(self, event):
    'motion_notify_event'
    if event.xdata is not None:
      x, y = event.xdata, event.ydata
      dist = np.linalg.norm(self.vertices - np.array([x, y])[None], axis=1)
      minind = np.argmin(dist)
      if dist[minind] < self.rvertex:
        point = self.vertices[minind]
        if self.point_highlighted:
          self.highlightedpoint.remove()
        self.vminind = minind
        self.highlightedpoint = self.ax.scatter(*point, c='k', s=40)
        self.point_highlighted = True
        self.draw()
      else:
        if self.point_highlighted:
          self.highlightedpoint.remove()
          self.point_highlighted = False
          self.draw()


class SelectVertices(HighlightVertexMixin, MatplotlibEventHandlerWithNetwork):

  def __init__(self, *args, nclicks=None, **kwargs):
    kwargs.setdefault('title', 'Click to select vertices, press space to finalize')
    super().__init__(*args, **kwargs)
    if nclicks is None:
      nclicks = np.inf
    self.nclicks = nclicks
    self.clicked_vertices = []

  def on_press(self, event):
    'button_press_event'
    if self.point_highlighted:
      minind = self.vminind
      self.clicked_vertices.append(self.vertex_indices[minind])
      point = self.network.get_point_on_edge(self.clicked_vertices[-1])
      self.ax.scatter(*point, c='r', s=40)
      self.vertices = np.concatenate([self.vertices[:minind], self.vertices[minind + 1:]])
      self.vertex_indices = self.vertex_indices[:minind] + self.vertex_indices[minind + 1:]
      self.draw()
      if len(self.vertex_indices) == 0 or len(self.clicked_vertices) == self.nclicks: self.close()

  def on_key(self, event):
    'key_press_event'
    if event.key == ' ': self.close()
    super().on_key(event)


def select_vertices(network, face, **kwargs):

  snetwork = network.take((face,))

  with SelectVertices(snetwork, **kwargs) as handler:
    clicked_vertices = handler.clicked_vertices

  return clicked_vertices


class GenerateClicksFromVertexToVertex(HighlightVertexMixin, GenerateClicks):

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('title', 'Connect two vertices by clicking the canvas')
    super().__init__(*args, **kwargs)  # runs GenerateClicks' __init__
    self.face_index, = self.network.faces
    self.clicked_vertices = []

  def on_press(self, event):
    'button_press_event'
    if self.point_highlighted:  # overwrite current mouse position
      if self.n is None or len(self.points) in (0, self.n - 1):
        event.xdata, event.ydata = self.vertices[self.vminind]
        self.clicked_vertices.append(
            -self.network.face_indices[self.face_index][self.vminind]
        )
    super().on_press(event)

  def on_motion(self, event):
    'motion_notify_event'
    if self.n is None or len(self.points) in (0, self.n - 1):
      super().on_motion(event)
    else: self.point_highlighted = False

  def close(self):
    super().close()
    vertices = set(map(tuple, self.vertices))
    assert {tuple(self.points[0]), tuple(self.points[-1])}.issubset(vertices)
    assert len(set(map(tuple, self.points[1:-1])) & vertices) == 0


def draw_spline_from_vertex_to_vertex(network, face_index, npoints=101, plot=True, title=None, interpolargs=None, **kwargs):
  if interpolargs is None:
    interpolargs = {}
  interpolargs.setdefault('k', 3)

  if title is None:
    title = 'Connect two vertices by a curve by clicking the canvas'

  snetwork = network.take((face_index,))
  with GenerateClicksFromVertexToVertex(snetwork, title=title, **kwargs) as handler:
    clicked_vertices = handler.clicked_vertices
    points = handler.points

  interpolargs['k'] = min(interpolargs['k'], len(points) - 1)

  from mapping import pc
  points = pc.PointCloud(np.stack(points)).toInterpolatedUnivariateSpline(**interpolargs). \
                                           toPointCloud(np.linspace(0, 1, npoints)).points
  if plot:
    snetwork.split_face(face_index, *clicked_vertices, connecting_edge=points).qplot()

  return network.split_face(face_index, *clicked_vertices, connecting_edge=points)


def draw_linears(network, face_index, npoints=101, plot=True):

  snetwork = network.take((face_index,))

  with GenerateClicksFromVertexToVertex(snetwork) as handler:
    clicked_vertices = handler.clicked_vertices
    points = handler.points

  from index import Edge

  edges = []
  for p0, p1 in zip(points, points[1:]):
    edges.append(Edge.between_points(p0, p1, npoints=npoints))

  if plot:
    snetwork.split_face(face_index, *clicked_vertices, connecting_edge=edges).qplot()

  return network.split_face(face_index, *clicked_vertices, connecting_edge=edges)


def connect_two_new_vertices_by_curve(network, face, npoints=101,
                                                     interpolargs=None,
                                                     points_to_verts=None,
                                                     plot=True, **kwargs):
  if interpolargs is None:
    interpolargs = {}
  interpolargs.setdefault('k', 3)

  if points_to_verts is None:
    points_to_verts = lambda points: None
  assert 'title' not in kwargs

  title0 = 'Draw two new vertices by clicking the edges.'

  with GenerateVertexFromClick(network.take((face,)), title=title0) as handler:
    ei = handler.ei
  network = network.split_edge(handler.ei)
  e0 = -np.sign(ei[0]) * network.max_edge_index

  with GenerateVertexFromClick(network.take((face,)), title=title0) as handler:
    ei = handler.ei
  network = network.split_edge(handler.ei)
  e1 = -np.sign(ei[0]) * network.max_edge_index

  title1 = 'Draw a curve between the two new vertices.'
  with GenerateClicksFromVertexToVertex(network.take((face,)), title=title1, **kwargs) as handler:
    points = handler.points
    clicked_vertices = handler.clicked_vertices
    snetwork = handler.network

  p0, p1 = network.get_point_on_edge(e0), network.get_point_on_edge(e1)
  assert {tuple(p0), tuple(p1)} == {tuple(points[0]), tuple(points[-1])}

  from mapping import pc
  interpolargs['k'] = min(interpolargs['k'], len(handler.points) - 1)
  verts = points_to_verts(points)
  points = pc.PointCloud(np.stack(points), verts=verts).\
                         toInterpolatedUnivariateSpline(**interpolargs). \
                         toPointCloud(np.linspace(0, 1, npoints)).points

  if plot:
    snetwork.split_face(face, *clicked_vertices, connecting_edge=points).qplot()

  network = network.split_face(face, *clicked_vertices, connecting_edge=points)

  return network


def create_diamond(network, face, npoints=101):
  assert npoints % 2 == 1  # only odd number of points
  points_to_verts = lambda points: np.linspace(0, 1, 3)
  network = connect_two_new_vertices_by_curve(network, face,
                                              npoints=npoints,
                                              interpolargs={'k': 1}, n=3,
                                              points_to_verts=points_to_verts,
                                              plot=False)
  n = network.max_edge_index
  ret = network.split_edge((n, .5))
  ret.take((face, ret.max_face_index)).qplot()
  return ret


def closest_point(func, point, x0, lower=None, upper=None, **scipyargs):

  constraints = list(scipyargs.pop('constraints', []))

  if lower is not None:
    assert isinstance(lower, numbers.Number)
    constraints.append( {'type': 'ineq', 'fun': lambda x: x - lower} )

  if upper is not None:
    assert isinstance(upper, numbers.Number)
    constraints.append( {'type': 'ineq', 'fun': lambda x: upper - x} )

  constraints = tuple(constraints)

  point = np.asarray(point)
  assert point.shape == (2,)

  myfunc = lambda x: .5 * ((func(x) - point)**2).sum()

  from scipy import optimize

  return optimize.minimize(myfunc, x0, constraints=constraints, **scipyargs)


def half_face(network, face, upper_edges=None, lower_edges=None,
                             vertices=None, orth=(), dist=(),
                             nmasterpoints=21, npoints=101,
                                       eps=1e-5, plot=True, interpolargs={}):

  snetwork = network.take((face,))

  # first select all edges of the upper (master) side
  if upper_edges is None:
    with SelectEdges(snetwork, title='Select upper boundaries') as handler:
      upper_edges = [-i for i in handler.clicked_edges]  # assuming positive orientednes, we have to reverse the order

  # select all edges of the lower (slave) side
  if lower_edges is None:
    with SelectEdges(snetwork, title='Select lower  boundaries') as handler:
      lower_edges = handler.clicked_edges

  assert len(set(map(abs, upper_edges)) & set(map(abs, lower_edges))) == 0  # make sure they are disjoint

  # select start and end vertex
  if vertices is None:
    with SelectVertices(snetwork, title='Select starting and ending vertex', nclicks=2) as handler:
      vertices = handler.clicked_vertices

  begin, end = vertices

  # select the corresponding sides
  begin_point = snetwork.get_point_on_edge(begin)
  end_point = snetwork.get_point_on_edge(end)

  from index import Edge
  upper_edges = snetwork.get_edges(upper_edges)

  # concatenate into one edge and interpolate
  upper_spline = Edge( np.concatenate([ edge.points if not i else edge.points[1:]
                                          for i, edge in enumerate(upper_edges) ]) ). \
                       toPointCloud().toInterpolatedUnivariateSpline()

  # do exactly the same for the slave side
  lower_edges = snetwork.get_edges(lower_edges)
  lower_spline = Edge( np.concatenate([ edge.points if not i else edge.points[1:]
                                      for i, edge in enumerate(lower_edges) ]) ). \
                       toPointCloud().toInterpolatedUnivariateSpline()

  funcs = (upper_spline, lower_spline)

  edges = snetwork.polygons[face]
  indices = snetwork.face_indices[face]
  begin_index = indices.index(begin, mod=True)
  end_index = indices.index(end, mod=True)

  normals = edges.average_normals()
  normal_jumps = edges.normal_jumps()

  # find x values that are closest to starting and ending point (for each spline)
  # a strong jump in the normal means the vertex protrudes into the domain.
  # in this case, we truncate the upper / lower sides to the minimum distance
  # to the point
  if np.linalg.norm(normal_jumps[begin_index], 2) > 1e-1:
    lower_bound = [closest_point(func, begin_point, 0, lower=1e-7, upper=1-1e-7).x[0] for func in funcs]
  else:
    lower_bound = [1e-7] * 2

  if np.linalg.norm(normal_jumps[end_index], 2) > 1e-1:
    upper_bound = [closest_point(func, end_point, 1-eps, lower=p+1e-7, upper=1-1e-7).x[0] for func, p in zip(funcs, lower_bound)]
  else:
    upper_bound = [1 - 1e-7] * 2

  master_points = upper_spline( np.linspace(lower_bound[0], upper_bound[0], nmasterpoints) )[1: -1]
  slave_points = []

  interval = [lower_bound[1]+eps, upper_bound[1]]
  for point in master_points:
    x = closest_point(lower_spline, point, interval[0], lower=interval[0], upper=interval[1])
    interval[0] = x.x[0] + eps
    slave_points.append(lower_spline(x.x)[0])

  if 0 in orth:
    begin_index = indices.index(begin, mod=True)
    average_normal = _normalized(-normals[begin_index])
    vert_edge0 = Edge.between_points(slave_points[0], master_points[0])
    # find distance along edge such that the vector from begin_point
    # pointing toward the point associated with this distance along the edge
    # is as close as possible to average_normal
    y0 = np.argmin(((_normalized(vert_edge0.points - begin_point[None])
                                                - average_normal[None])**2).sum(1)) / len(vert_edge0.points)
  elif 0 in dist:
    p0s = [func(val) for func, val in zip(funcs, lower_bound)]
    vert_edge0 = Edge.between_points( p0s[1].ravel(), p0s[0].ravel() )
    y0 = vert_edge0.get_nearest_point(begin_point, return_index=True)[1] / len(vert_edge0.points)
  else:
    y0 = .5

  if 1 in orth:
    end_index = indices.index(end, mod=True)
    average_normal = _normalized(normals[end_index])
    vert_edge1 = Edge.between_points(slave_points[-1], master_points[-1])
    y1 = np.argmin(((_normalized(end_point[None] - vert_edge1.points)
                                              - average_normal[None])**2).sum(1)) / len(vert_edge1.points)
  elif 1 in dist:
    p1s = [func(val) for func, val in zip(funcs, upper_bound)]
    vert_edge1 = Edge.between_points( p1s[1].ravel(), p1s[0].ravel() )
    y1 = vert_edge1.get_nearest_point(end_point, return_index=True)[1] / len(vert_edge1.points)
  else:
    y1 = .5

  # a * x**2 + b * x + c in 0 = y0, in .5 = 0.5 in 1 = y1
  a, b, c = np.linalg.solve(np.array([ [  0,  0, 1],
                                       [.25, .5, 1],
                                       [  1,  1, 1] ]), np.array([y0, .5, y1]))

  xi = np.linspace(0, 1, nmasterpoints)[1: -1]
  prefacs = (a * xi**2 + b * xi + c)[:, None]

  points = np.concatenate([ begin_point[None],
                                    prefacs * master_points
                            + (1 - prefacs) * np.stack(slave_points, axis=0),
                            end_point[None] ])

  from mapping import pc
  newedge = pc.PointCloud(points).toInterpolatedUnivariateSpline(**interpolargs)(np.linspace(0, 1, npoints))
  if plot:
    snetwork.split_face(face, begin, end, connecting_edge=newedge).qplot()

  return network.split_face(face, begin, end, connecting_edge=newedge)


def create_full_diamond(network, face, **kwargs):
  network = create_diamond(network, face)
  import ipdb
  ipdb.set_trace()
  v0 = network.max_edge_index
  network = network.half_edge(face)
  v1 = network.max_edge_index
  indices = network.face_indices[face]
  return half_face(network, face,
                   vertices=[(v0 @ indices) * v0, -(v1 @ indices) * v1],
                   orth=(0,), **kwargs)


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=2:ts=2:sts=2:foldignore=#
