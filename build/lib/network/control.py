from nutils import function, solver, util, cli
from mapping.aux import unpack_GridObject, to_csr
from mapping.sol import _A, transfinite_interpolation, forward_laplace
from scipy.sparse import linalg as splinalg
from scipy import sparse
import numpy as np
from genus import default_ids
import itertools
from copy import deepcopy


_ = np.newaxis


def apply_transfinite_to_all_patches(mg):

  gs = mg.break_apart()
  basis_disc = mg.make_basis(patchcontinuous=False)

  for g in gs:
    g.set_cons_from_x()
    transfinite_interpolation(g)

  x = np.concatenate([g.x for g in gs])
  func = basis_disc.vector(2).dot(x)

  mg.x = mg.project(func)


def greville_verts(mg):
  "On mg.geom"

  patchverts = np.asarray(mg.patchverts)

  verts = []
  for patch in mg.patches:
    kv0 = mg.knotvectors.get(patch[:2], mg.knotvectors.get(patch[:2][::-1]))
    kv1 = mg.knotvectors.get(patch[::2], mg.knotvectors.get(patch[::2][::-1]))
    Xi, Eta = np.meshgrid(kv0.greville(), kv1.greville())
    X = np.stack([Xi.ravel(), Eta.ravel()], axis=1)
    a, b, c, d = patchverts[list(patch)]
    transform = lambda x: a[_] * ((1 - x[:, 0]) * (1 - x[:, 1]))[:, _] + \
                          c[_] * (x[:, 0] * (1 - x[:, 1]))[:, _] + \
                          b[_] * ((1 - x[:, 0]) * x[:, 1])[:, _] + \
                          d[_] * (x[:, 0] * x[:, 1])[:, _]
    verts.append(transform(X))

  return np.concatenate(verts)


def greville_fit(mg, funcs, return_func=True):
  if isinstance(funcs, function.Array):
    funcs = [funcs]

  verts = greville_verts(mg)

  sample = mg.domain.locate(mg.geom, verts, eps=1e-6, tol=1e-5)
  N = sparse.csr_matrix(sample.eval(mg.basis))
  X = N.T @ N
  Ps = sample.eval(funcs)

  controlpoints = []
  for P in Ps:
    controlpoints.append(splinalg.spsolve(X, N.T @ P))

  if return_func:
    return [mg.basis.dot(C) for C in controlpoints]

  return tuple(controlpoints)


def approximate_composition(mg, map0, map1, return_func=True, mu=1e-2, **locatekwargs):
  """
    Approximate the composition of two maps in the greville abscissae.
  """

  locatekwargs.setdefault('eps', 1e-5)
  locatekwargs.setdefault('tol', 1e-5)
  locatekwargs.setdefault('scale', 2)

  # verts = greville_verts(mg)

  # sample = mg.domain.locate(mg.geom, verts, **locatekwargs)
  sample0 = mg.domain.sample('gauss', 4)
  V = sample0.eval(map0)

  sample = mg.domain.locate(mg.geom, V, **locatekwargs)
  Ps = sample.eval(map1)

  N = sparse.csr_matrix(sample0.eval(mg.basis))
  X = N.T @ N

  if mu is not None:
    dbasis = mg.basis.grad(mg.geom)
    A = sparse.csr_matrix(mg.integrate( (dbasis[:, None] * dbasis[None]).sum([2]) * function.J(mg.geom) ).export('dense'))
    X += mu * A

  controlpoints = []
  for P in Ps.T:
    controlpoints.append(splinalg.spsolve(X, N.T @ P))

  controlpoints = np.stack(controlpoints, axis=1).ravel()

  if return_func:
    return mg.basis.vector(2).dot(controlpoints)

  return controlpoints


def test_approximate_composition():
  from genus import checkers_mg

  mg = checkers_mg(2, 2)
  mg.x = mg.project(mg.geom)
  mg.set_cons_from_x()

  D = build_patch_removal_diffusivity(mg)
  multipatch_trace_penalty(mg, D=D)

  x = approximate_composition(mg, mg.geom, mg.mapping, return_func=False)
  mg.x = x

  mg.qplot()


def build_distance_function(mg, mu=1e-6):
  patchverts = np.asarray(mg.patchverts)

  # verts = []
  # for patch in mg.patches:
  #   kv0 = mg.knotvectors.get(patch[:2], mg.knotvectors.get(patch[:2][::-1]))
  #   kv1 = mg.knotvectors.get(patch[::2], mg.knotvectors.get(patch[::2][::-1]))
  #   Xi, Eta = np.meshgrid(kv0.greville(), kv1.greville())
  #   X = np.stack([Xi.ravel(), Eta.ravel()], axis=1)
  #   a, b, c, d = patchverts[list(patch)]
  #   transform = lambda x: a[_] * ((1 - x[:, 0]) * (1 - x[:, 1]))[:, _] + \
  #                         c[_] * (x[:, 0] * (1 - x[:, 1]))[:, _] + \
  #                         b[_] * ((1 - x[:, 0]) * x[:, 1])[:, _] + \
  #                         d[_] * (x[:, 0] * x[:, 1])[:, _]
  #   verts.append(transform(X))

  # verts = np.concatenate(verts)

  # sample = mg.domain.locate(mg.geom, verts, eps=1e-4, tol=1e-4, scale=2)
  sample = mg.domain.sample('vertex', 0)
  verts = sample.eval(mg.controlmap)
  N = sparse.csr_matrix(sample.eval(mg.basis))
  X = N.T @ N
  dbasis = mg.basis.grad(mg.controlmap)
  A = to_csr(mg.integrate( (dbasis[:, _] * dbasis[_]).sum([-1]) * function.J(mg.controlmap) ))

  bedges, = mg.to_template().ordered_boundary_edges
  bindices = np.array([bedge[0] for bedge in bedges])

  import shapely
  pol = shapely.Polygon(patchverts[bindices]).exterior

  D = np.array([ pol.distance(P) for P in map(shapely.Point, verts) ])

  return mg.basis.dot( splinalg.spsolve(X + mu * A, N.T @ D) )


def build_gaussian_stabilisation(mg, geom=None, mu=18, exclude_even_valence=True, include_corner_verts=False):
  """
    XXX: Docstring
  """

  if geom is None:
    geom = mg.controlmap

  vertcount = {}
  for patch in mg.patches:
    for j in patch:
      vertcount.setdefault(j, 0)
      vertcount[j] += 1

  if include_corner_verts:
    exclude_verts = set()
  else:
    exclude_verts = set([ key for key, value in vertcount.items() if value == 1 ])
  if exclude_even_valence:
    exclude_verts.update({ key for key, value in vertcount.items() if (value % 2) == 0 })

  if include_corner_verts:
    exclude_verts = set()

  verts = {}

  for i, patch in enumerate(mg.patches):
    topo = mg.domain.patches[i].topo
    for ii, j in enumerate(patch):
      if j in exclude_verts: continue
      if j in verts: continue
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = topo.boundary[side0].boundary[side1].sample('vertex', 0).eval(geom).ravel()

  verts = np.stack([ verts[index] for index in sorted(verts.keys()) ], axis=0)

  mindists = []
  for vert in verts:
    mindists.append( np.sort(((verts - vert[None])**2).sum(1)**.5)[1] )
  mindists = np.asarray(mindists)

  mat = np.stack([ np.exp(-((mu/mindists[:, None] * (verts - vert[None]))**2).sum(1)) for vert in verts ], axis=0)
  As = np.linalg.solve(mat, np.ones((mat.shape[0],)))

  return lambda x: (As * function.exp(-((mu/mindists[:, _] * (x[_] - verts))**2).sum([1]))).sum([0])


def build_distance_func(mg, mu=2):

  """
    Build scalar function that takes large values close to the boundaries
    (in the parametric domain).

    Parameters
    ----------

    mg : :class: genus.MultiPatchBSplineGridObject
    mu : :class: float
      Exponential decay rate. Bigger value => faster decay away from the boundary.
  """

  assert mu > 0

  patchverts = np.asarray(mg.patchverts)

  bedges = set(mg.boundary_edges().keys())
  bpatches = mg.boundary_patches()

  bpatch = mg.domain.basis_patch()

  dist_funcs = []

  for edge in bedges:
    mypatch, = [i for i in bpatches if len(set(edge) & set(mg.patches[i])) == 2]
    e = np.zeros((len(bpatch),))
    e[mypatch] = 1
    P0, P1 = patchverts[ list(edge) ]
    t = (P1 - P0) / np.linalg.norm(P1 - P0)
    n = np.array([-t[1], t[0]])

    def func(x, P0=P0, n=n, e=e):
      return function.exp(-mu * ((x - P0) * n).sum()**2) * bpatch.dot(e)

    dist_funcs.append(func)

  return lambda x: sum(func(x) for func in dist_funcs)


def build_patch_removal_diffusivity(mg, normalise=True):

  Jmu = mg.controlmap.grad(mg.localgeom)
  if normalise:
    Jmu = function.normalized(Jmu, axis=0)

  Ds = function.outer(Jmu[:, 0]) + function.outer(Jmu[:, 1])
  return lambda x: Ds


def build_stabilised_patch_removal_diffusivity(mg, mu=9, exclude_even_valence=True, include_corner_verts=False, **kwargs):

  geom = mg.controlmap

  D = build_patch_removal_diffusivity(mg, **kwargs)(geom)
  vertcount = {}
  for patch in mg.patches:
    for j in patch:
      vertcount.setdefault(j, 0)
      vertcount[j] += 1

  if include_corner_verts:
    exclude_verts = set()
  else:
    exclude_verts = set([ key for key, value in vertcount.items() if value == 1 ])
  if exclude_even_valence:
    exclude_verts.update({ key for key, value in vertcount.items() if (value % 2) == 0 })

  verts = {}
  mats = {}

  for i, patch in enumerate(mg.patches):
    topo = mg.domain.patches[i].topo
    for ii, j in enumerate(patch):
      if j in exclude_verts: continue
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      sample = topo.boundary[side0].boundary[side1].sample('vertex', 0)
      verts[j] = sample.eval(geom).ravel()
      mats.setdefault(j, []).append(sample.eval(D).reshape([2, 2]))

  # mats = {key: sum(D / (D**2).sum()**.5 for D in myDs) / sum(1 / (D**2).sum()**.5 for D in myDs) for key, myDs in mats.items()}
  mats = {key: sum(D for D in myDs) / len(myDs) for key, myDs in mats.items()}

  stackverts = np.stack([ verts[index] for index in sorted(verts.keys()) ], axis=0)

  mindists = {}
  for j, vert in verts.items():
    mindists[j] = np.sort(((stackverts - vert[None])**2).sum(1)**.5)[1]

  exponentials = {j: function.exp(-((mu/mindists[j] * (geom - verts[j]))**2).sum()) for j in sorted(mats.keys())}
  expsum = sum(exponentials.values())

  return lambda x: (1 - expsum) * D + sum(exponentials[j] * mats[j] for j in sorted(mats.keys()))


def build_transversal_boundary_patch_slider(mg, a=20):

  if np.isscalar(a):
    a = lambda x: a

  bpatch_ids = mg.boundary_patches()
  bedges = set(mg.boundary_edges().keys())
  bedges.update({edge[::-1] for edge in bedges})
  pbasis = mg.domain.basis_patch()

  Jmu = mg.controlmap.grad(mg.localgeom)
  Ds = []
  Ds.append(lambda x: pbasis.dot([1 if i not in bpatch_ids else 0 for i in range(len(mg.patches))]) * function.eye(2))

  for ipatch in bpatch_ids:
    e = np.zeros((len(pbasis),))
    e[ipatch] = 1
    patch = mg.patches[ipatch]
    myedges = patch[:2], patch[2:], patch[::2], patch[1::2]
    mybedges = [ edge for edge in myedges if edge in bedges ]

    if len(mybedges) >= 2:
      def func(x, e=e):
        return pbasis.dot(e) * function.eye(2)
    else:
      bedge, = mybedges
      # left or right => slide only in mu_0 direction
      # bottom or top => slide in mu_1
      direction = {0: 0, 1: 0, 2: 1, 3: 1}[ myedges.index(bedge) ]
      v1 = function.normalized(Jmu[:, direction])
      v1orth = function.stack([-v1[1], v1[0]])

      def func(x, e=e, v1=v1, v1orth=v1orth):
        return pbasis.dot(e) * (a(x) * function.outer(v1) + function.outer(v1orth)) / (a(x) + 1)

    Ds.append( func )

  return lambda x: sum(D(x) for D in Ds)


def make_unit_disc_controlmap(mg, lengths=None, method='coons'):

  assert method in ('coons', 'laplace')

  # get boundary edges
  bedges, = mg.to_template().ordered_boundary_edges

  # get boundary indices in counterclockwise direction
  bindices = np.array([bedge[0] for bedge in bedges])

  patchverts = np.asarray(mg.patchverts)
  center = patchverts[bindices].sum(0) / len(bindices)

  # center
  patchverts = patchverts - center[_]

  geom = mg.geom - center

  # compute the first angle
  start = patchverts[bindices[0]]
  x, y = start / (start**2).sum()**.5

  angle0 = np.arctan2(y, x)
  if angle0 < 0:
    angle0 = angle0 % (2 * np.pi)

  # if the lengths of the edges occupying the unit disc are not passed,
  # take them based on the length of the edge in hat{Ω}
  if lengths is None:
    lengths = [np.linalg.norm(patchverts[i] - patchverts[j]) for i, j in zip(bindices, np.roll(bindices, -1))]

  # translate the lengths to radians over a 2 * π interval
  rads = 2 * np.pi * np.array([0, *lengths]).cumsum() / sum(lengths) + angle0

  # project circle boundary correspondence onto the basis
  cons = util.NanVec([len(mg.basis)*2])
  for (r0, r1), (i, j) in zip(zip(rads, [*rads[1:], rads[0] + 2*np.pi]),
                              zip(bindices, np.roll(bindices, -1))):

    p0, p1 = patchverts[i], patchverts[j]
    mylength = np.linalg.norm(p1 - p0)

    s = ((geom - p0)**2).sum()**.5 / mylength

    cons |= mg.project_edge((i, j), function.stack([ function.cos(r0 + (r1-r0) * s),
                                                     function.sin(r0 + (r1-r0) * s) ]))

  _mg = deepcopy(mg)
  _mg.cons = cons

  # perform forward laplace to find the interior vertices in the interior
  # of \hat{Ω}^{r}
  forward_laplace(_mg)

  # get the verts in \hat{Ω}^r
  verts = {}

  for i, patch in enumerate(mg.patches):
    topo = mg.domain.patches[i].topo
    for ii, j in enumerate(patch):
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = topo.boundary[side0].boundary[side1].sample('vertex', 0).eval(_mg.mapping).ravel()

  mapped_edges = set(bedges)

  # project all remaining edges in the interior onto straight lines
  # connecting the interior vertices
  for i, patch in enumerate(mg.patches):
    for edge in (patch[:2], patch[2:], patch[::2], patch[1::2]):
      if edge in mapped_edges or edge[::-1] in mapped_edges: continue
      i, j = edge
      p0, p1 = patchverts[i], patchverts[j]
      v0, v1 = verts[i], verts[j]
      mylength = np.linalg.norm(p1 - p0)

      s = ((geom - p0)**2).sum()**.5 / mylength

      cons |= mg.project_edge((i, j), v0 + (v1 - v0) * s)

  _mg.cons = cons
  _mg.x = _mg.cons | 0

  # either use patchwise forward laplace or coon's patch
  # to parameterise each patch
  if method == 'laplace':
    forward_laplace(_mg)
  elif method == 'coons':
    apply_transfinite_to_all_patches(_mg)
  else:
    raise

  return _mg


@unpack_GridObject
def multipatch_trace_penalty(f, D=None, *, cons=None, ischeme, basis, btol=1e-5, **ignorekwargs):
  if cons is None:
    cons = f.cons

  if D is None:
    Jmu = function.normalized(f.mapping.grad(f.localgeom), axis=0)
    D = lambda r: function.matmat(Jmu, Jmu.T)

  x = f.xArgument()
  D = D(x)

  Dx = function.matmat(x.grad(f.mapping), D)

  integrand = (basis.vector(2).grad(f.mapping) * Dx[None]).sum([1, 2])
  res = f.integral( integrand * function.J(f.mapping), degree=2*ischeme )

  f.x = solver.newton('target', res, constrain=cons.where, lhs0=cons|f.x).solve(btol)


@unpack_GridObject
def multipatch_trace_penalty_(g, G=None, prefunc=None, *, ischeme, f=None, **ignorekwargs):

  if f is None:
    f = g.g_controlmap()

  if G is None:
    Jmu = function.normalized(g.geom.grad(g.localgeom), axis=0)
    G = function.matmat(Jmu, Jmu.T)

  if prefunc is None:
    prefunc = function.ones(())

  if prefunc.shape in ((), (1,)):
    prefunc = function.stack([prefunc, prefunc])

  assert prefunc.shape == (2,)

  x = f.xArgument()
  J = x.grad(g.geom)

  interior = function.trace(prefunc[:, None] * function.matmat(J, function.matmat(G, J.T)))

  costfunc = f.integral(interior * function.J(g.geom), degree=2*ischeme)

  f.x = solver.optimize('target', costfunc, constrain=f.cons)

  return f


@unpack_GridObject
def multipatch_trace_penalty_new(g, G=None, prefunc=None, *, ischeme, f=None, **ignorekwargs):

  if f is None:
    f = g.g_controlmap()

  if G is None:
    Jmu = function.normalized(g.geom.grad(g.localgeom), axis=0)
    G = function.matmat(Jmu, Jmu.T)

  if prefunc is None:
    prefunc = function.ones(())

  if prefunc.shape in ((), (1,)):
    prefunc = function.stack([prefunc, prefunc])

  assert prefunc.shape == (2,)

  x = f.xArgument()
  J = function.matmat(x.grad(g.geom), g.geom.grad(g.localgeom))
  interior = function.trace(function.matmat(J.T, J))

  costfunc = f.integral(interior * function.J(g.geom), degree=2*ischeme)

  f.x = solver.optimize('target', costfunc, constrain=f.cons)

  return f


@unpack_GridObject
def multipatch_trace_penalty_stab_(g, G=None, prefunc=None, *, ischeme, f=None, mu=1e-3, eps=1e-4, btol=1e-5, **ignorekwargs):

  if f is None:
    f = g.g_controlmap()

  if G is None:
    Jmu = function.normalized(g.geom.grad(g.localgeom), axis=0)
    G = function.matmat(Jmu, Jmu.T)

  if prefunc is None:
    prefunc = function.ones(())

  if prefunc.shape in ((), (1,)):
    prefunc = function.stack([prefunc, prefunc])

  assert prefunc.shape == (2,)

  x = f.xArgument()
  J = x.grad(g.geom)

  interior = function.trace(prefunc[:, None] * function.matmat(J, function.matmat(G, J.T)))

  res = f.integral(interior * function.J(g.geom), degree=2*ischeme).derivative('target')

  x = g.xArgument()
  A = _A(g, x)

  J = function.determinant(x.grad(g.geom))
  denom = (J + function.sqrt(4 * eps + J**2)) / 2

  res += mu * g.integral( (g.basis.vector(2).grad(g.geom) * A[None]).sum([1, 2]) * denom**(-1) * function.J(g.geom), degree=ischeme*2 )
  g.x = solver.newton('target',
                      res,
                      constrain=g.cons.where,
                      lhs0=g.x).solve(btol)

  return f


@unpack_GridObject
def multipatch_trace_penalty_stab(g, G=None, prefunc=None, *, ischeme, f=None, mu=1e-5, eps=1e-4, vl=None, vu=None, btol=1e-5, **ignorekwargs):

  if f is None:
    f = g.g_controlmap()

  if prefunc is None:
    prefunc = function.ones(())

  if prefunc.shape in ((), (1,)):
    prefunc = function.stack([prefunc, prefunc])

  if G is None:
    Jmu = function.normalized(g.controlmap.grad(g.localgeom), axis=0)
    G = function.matmat(Jmu, Jmu.T)

  assert prefunc.shape == (2,)

  x = f.xArgument()
  J = x.grad(g.controlmap)

  interior = function.trace(prefunc[:, None] * function.matmat(J, function.matmat(G, J.T)))

  res = f.integral( interior * function.J(g.controlmap), degree=2*ischeme)

  if (vl, vu) != (None, None):
    Gstab = function.matmat(J, J.T)

    detJe = lambda x: (x + function.sqrt(4 * eps**2 + x**2)) / 2
    detJ = function.determinant(x.grad(g.controlmap))

    if vl is not None:
      interior += mu * function.trace(Gstab) / detJe(detJ - vl)
    if vu is not None:
      interior += mu * function.trace(Gstab) / detJe(-detJ + vu)

  res = g.integral( interior * function.J(g.controlmap) )
  f.x = solver.optimize('target',
                        res,
                        constrain=f.cons.where,
                        lhs0=f.x, tol=btol)

  return f


@unpack_GridObject
def multipatch_trace_penalty_periodic(mg, direction=0, prefunc=None, *, ischeme, **kwargs):
  all_ids = mg.all_ids
  all_edges = mg.all_edges

  _map = {}

  for id, edge in zip(all_ids, all_edges):
    _map.setdefault(id, set()).update({edge})

  _map = {key: value for key, value in _map.items() if len(value) == 2}
  pairs = tuple(itertools.chain(*_map.values()))

  patchverts = np.array(mg.patchverts)
  centers = np.round(patchverts[ list(all_edges) ].sum(1) / 2, 7)

  map_center_orig_edge = dict(zip(map(tuple, centers), all_edges))

  A = mg.to_template()
  translate = np.array([ patchverts[:, i].max() - patchverts[:, i].min()
                         if i == direction else 0 for i in range(2) ])

  B = A.fuse(A.translate(-translate)).fuse(A.translate(translate))

  edges = set(all_edges)
  knotvectors = mg.knotvectors.copy()
  all_new_edges = tuple(itertools.chain(*default_ids(B.patches)))
  new_patchverts = np.array(B.patchverts)

  for edge in all_new_edges:
    if edge in edges:
      continue
    orig_center_min = tuple( np.round( new_patchverts[list(edge)].sum(0)/2 - translate, 7) )
    orig_center_plus = tuple( np.round( new_patchverts[list(edge)].sum(0)/2 + translate, 7) )
    orig_edge = map_center_orig_edge.get(orig_center_min, map_center_orig_edge.get(orig_center_plus))
    knotvectors[edge] = knotvectors[orig_edge]

  f = mg.__class__(B.patches, B.patchverts, knotvectors)
  f.x = f.project(f.geom)
  f.set_cons_from_x()

  Jmu = function.normalized(f.geom.grad(f.localgeom), axis=0)
  G = function.matmat(Jmu, Jmu.T)

  if prefunc is None:
    prefunc = function.ones(())

  if prefunc.shape in ((), (1,)):
    prefunc = function.stack([prefunc, prefunc])

  assert prefunc.shape == (2,)

  x = f.xArgument()
  J = x.grad(f.geom)

  cons = util.NanVec((len(f.basis),))
  for edge in pairs:
    cons |= f.project_edge(edge, f.geom[direction], onto=f.basis)

  _cons = f.cons.copy()
  _cons[direction::2] = cons
  f.cons = _cons | f.cons

  interior = function.trace(prefunc[:, None] * function.matmat(J, function.matmat(G, J.T)))

  costfunc = f.integral(interior * function.J(f.geom), degree=2*ischeme)

  f.x = solver.optimize('target', costfunc, constrain=f.cons)

  return f