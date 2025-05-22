from nutils import types, function, solver, topology, util, log, cli
from mapping.aux import unpack_GridObject
from mapping.mul import basis_spline
from mapping.ko import KnotObject
import numpy as np
from abc import abstractmethod, abstractstaticmethod
from scipy.optimize import anderson
from functools import cached_property


def coarsen_knotvector(kv: KnotObject):
  assert (kv.knotmultiplicities[1: -1] == 1).all(), NotImplementedError
  assert kv.periodic is False, NotImplementedError

  if len(kv.knotvalues) % 2 == 0:
    raise AssertionError('Cannot coarsen a knotvector with an even number of knotvalues.')

  return KnotObject(knotvalues=kv.knotvalues[::2],
                    knotmultiplicities=kv.knotmultiplicities[::2],
                    degree=kv.degree,
                    periodic=False)


def _metric_tensor(x, geom):
  jacT = function.transpose(x.grad(geom))
  return function.outer(jacT, jacT).sum(-1)


def _A(x, geom):
  stack = function.stack
  (g11, g12), (g12, g22) = _metric_tensor(x, geom)
  return stack([ stack([g22, -g12]), stack([-g12, g11]) ])


def gamma(A):
  return function.trace(A) / (A**2).sum([0, 1])


def find_controlmap(mg, basis=None, return_func=True):
  if basis is None:
    basis = mg.basis

  s = mg.xArgument()
  integrand = (basis.vector(2).grad(mg.mapping) * s.grad(mg.mapping)[None]).sum([1, 2])
  res = mg.integral(integrand * function.J(mg.mapping))
  cons = mg.project(mg.controlmap, domain=mg.domain.boundary, geometry=mg.mapping)

  weights = solver.solve_linear('target', res, constrain=cons)
  if return_func:
    return basis.vector(2).dot(weights)
  return weights


class EllipticSolver:

  def __init__(self, g, N=None):
    self.g = g
    self._cons = np.asarray(g.cons).view(util.NanVec)
    self._x0 = np.asarray(g.x)

    if N is None:
      N = len(self._x0)

    self.N = int(N)
    self.n = 2 * len(self.g.basis)

  @abstractmethod
  def prepare_namespace(self, x):
    pass

  @abstractmethod
  def _res(self, x):
    # ns = self.prepare_namespace(x)
    pass

  def res(self, x=None):
    if x is None:
      x = function.Argument('target', [self.N])
    return self._res(x)

  @property
  def x0(self):
    return self.g.x

  @property
  def cons(self):
    return self.g.cons

  def newton(self, x0=None, cons=None, btol=1e-6):
    if x0 is None:
      x0 = self.x0
    cons = self.cons

    assert x0.shape == cons.shape

    residual = self.res()
    return solver.newton('target', residual, constrain=cons.where, lhs0=x0).solve(btol)[-self.n:]

  def fixed_point_F(self, x):
    raise NotImplementedError

  def fixed_point_iter(self, x0, *args, **kwargs):
    while True:
      x0 = self.fixed_point_F(x0, *args, **kwargs)
      yield x0

  def fixed_point(self, x0=None, btol=1e-5):

    if x0 is None:
      x0 = self.x0

    fp_it = self.fixed_point_iter(x0)
    xprev = x0

    for xnow in fp_it:
      res = np.abs(xprev - xnow).max()
      log.warning('Current residual: {}'.format(res))
      if res < btol:
        break
      xprev = xnow

    return xnow[-self.n:]

  def scipy(self, solverargs=None):
    dofindices = np.isnan(self.cons)
    solverargs = dict(solverargs or {})

    def F(_x):
      x = self.cons.copy()
      x[dofindices] = _x.copy()
      ret = self.fixed_point_F(x)[dofindices] - _x
      log.warning(np.abs(ret).max())
      return ret

    x0 = self.x0[dofindices]

    from scipy.optimize import root
    x0 = root(F, x0, **solverargs)

    x = self.cons.copy()
    x[dofindices] = x0.x

    return x[-self.n:]

  def anderson_acceleration(self, solverargs=None):
    dofindices = np.isnan(self.cons)
    solverargs = dict(solverargs or {})
    solverargs.setdefault('line_search', None)

    def F(_x):
      x = self.cons.copy()
      x[dofindices] = _x.copy()
      ret = self.fixed_point_F(x)[dofindices]
      return ret - _x

    x0 = self.x0[dofindices]

    x0 = anderson(F, x0, verbose=True, **solverargs)

    x = self.cons.copy()
    x[dofindices] = x0

    return x[-self.n:]


class EllipticSolverWithControlFunction(EllipticSolver):

  def __init__(self, g, P=None):
    super().__init__(g)

    if P is None:
      P = function.zeros((2,))

    assert callable(P) or P.shape == (2,)

    self.P = P


class EllipticPartial(EllipticSolver):

  def __init__(self, g, eps=1e-4, **kwargs):
    assert eps >= 0
    super().__init__(g, **kwargs)
    self.eps = float(eps)

  def prepare_namespace(self, x):
    assert x.shape == (self.N,)
    g = self.g
    ns = function.Namespace()
    ns.trial = g.basis.vector(2)
    ns.target = x
    ns.sol_i = 'trial_ni target_n'
    ns.x = g.controlmap
    ns.J_ij = 'sol_i,j'
    ns.C = function.stack([ [ns.J[1, 1], -ns.J[0, 1]],
                            [-ns.J[1, 0], ns.J[0, 0]] ])
    ns.detJ = function.determinant(ns.J)
    ns.A_ij = 'C_ik C_jk'
    ns.eps = self.eps
    ns.denom = '(detJ + (4 eps + detJ^2)^.5) / 2' if self.eps else 'detJ'
    ns.dbasis = ns.trial.grad(ns.x)
    return ns

  def fixed_point_F(self, xprev):
    g = self.g
    nsprev = self.prepare_namespace(xprev)
    nsnow = self.prepare_namespace(function.Argument('target', [self.N]))

    ns = function.Namespace()
    ns.sol = nsnow.sol
    ns.Cold = nsprev.C
    ns.C = nsnow.C
    ns.denom = nsprev.denom
    ns.dbasis = nsprev.dbasis
    ns.solold = nsprev.sol
    ns.x = self.g.controlmap

    res = 'dbasis_nij (-sol_i,j + Cold_ik C_jk + solold_i,j) denom^(-1)' @ ns
    res = g.integral(res * function.J(ns.x))

    return solver.solve_linear('target', res, constrain=g.cons)

  def _res(self, x):
    ns = self.prepare_namespace(x)

    ischeme = self.g.ischeme
    res = 'dbasis_nij A_ij denom^(-1)' @ ns

    return self.g.integral( res * function.J(ns.x), degree=ischeme*2 )


class Blechschmidt(EllipticSolver):

  def __init__(self, g, eta=1e1, eps=0, mu=1e-5, tau='delta_c', P=None, **kwargs):
    super().__init__(g, **kwargs)
    assert eta >= 0 and eps >= 0
    self.eta = float(eta)
    self.eps = float(eps)
    assert tau in ('delta', 'delta_c', 'Id')
    self.tau = tau
    self.mu = float(mu)

    if P is None:
      P = function.zeros((2,))

    assert callable(P) or P.shape == (2,)

    self.P = P

  def prepare_namespace(self, x):
    g = self.g
    ns = function.Namespace()
    ns.target = x
    ns.trial = self.g.basis.vector(2)
    ns.sol_i = 'trial_ni target_n'
    ns.x = g.controlmap
    ns.A = _A(ns.sol, ns.x)
    ns.Astab = ns.A + (0 if self.eps == 0 else self.eps * function.exp(-function.determinant(ns.A)) * function.eye(2))
    ns.g = function.determinant(ns.A)
    ns.P = self.P(ns.sol, ns.x) if callable(self.P) else self.P

    if not hasattr(self, '_h'):
      try:
        interfaces = g.domain.interfaces
        gammabasis = interfaces.basis_discont(0)
        weights = interfaces.integrate(gammabasis * function.J(g.controlmap), degree=5)
        self._h = gammabasis.dot(weights)
      except Exception as ex:
        log.warning("Warning, domain.interfaces failed with exception '{}'.".format(ex))
        self._h = 1

    ns.h = self._h

    assert ns.P.shape == (2,)

    ns.n = ns.x.normal().normalized()
    ns.vn = function.jump(ns.trial.grad(ns.x))[..., None] * ns.n[None, None, None]
    ns.xn = function.jump(ns.sol.grad(ns.x))[..., None] * ns.n[None, None]

    tau = self.tau

    if tau == 'delta':
      test = function.laplace(g.basis.vector(2), g.geom)
    elif tau == 'delta_c':
      test = function.laplace(g.basis.vector(2), g.controlmap)
    elif tau == 'id':
      test = g.basis.vector(2)
    else:
      raise AssertionError

    ns.test = test

    return ns

  def _res(self, x):

    g = self.g
    ns = self.prepare_namespace(x)
    ns.Astab = ns.A + (0 if self.eps == 0 else self.eps * function.exp(-function.determinant(ns.A)) * function.eye(2))
    ns.gamma = gamma(ns.Astab)

    vn = ns.vn
    xn = ns.xn

    res = g.domain.integral( 'gamma test_ni (Astab_jk sol_i,jk + g P_k sol_i,k)' @ ns * function.J(g.controlmap), degree=g.ischeme*2 )

    try:
      res += self.eta * g.domain.interfaces['interpatch'].integral( (ns.h**-1) * (vn * xn[None]).sum([1, 2, 3]) * function.J(g.controlmap), degree=g.ischeme*2 )
    except Exception:
      pass

    return res

  def fixed_point_F(self, xprev):
    g = self.g
    nsprev = self.prepare_namespace(xprev)
    nsnow = self.prepare_namespace(function.Argument('target', [self.N]))
    ns = function.Namespace()
    ns.h = nsnow.h
    ns.I = self.mu * function.eye(2)
    ns.A = nsprev.A
    ns.Aoff = ns.A + ns.I
    ns.dsolold = self.mu * function.laplace(nsprev.sol, g.controlmap)
    ns.gamma = gamma(ns.Aoff)
    ns.sol = nsnow.sol
    ns.g = nsprev.g
    ns.P = nsprev.P
    ns.test = nsnow.test
    ns.solold = nsprev.sol
    ns.x = nsnow.x

    vn, xn = nsnow.vn, nsnow.xn

    interfaces = g.domain.interfaces
    if 'interpatch' in interfaces:
      interfaces = interfaces['interpatch']

    res = g.domain.integral( 'gamma test_ni (Aoff_jk sol_i,jk - dsolold_i + g P_k solold_i,k)' @ ns * function.J(g.controlmap), degree=g.ischeme*2 )
    res += self.eta * interfaces.integral( (ns.h**-1) * (vn * xn[None]).sum([1, 2, 3]) * function.J(g.controlmap), degree=g.ischeme*2 )

    return solver.solve_linear('target', res, constrain=g.cons)


class LakkisPryer(EllipticSolver):

  def __init__(self, g, eps=0, tau='Id', P=None, **kwargs):

    n = len(g.basis)
    super().__init__(g, N=10*n, **kwargs)

    self.eps = float(eps)
    assert tau in ('delta', 'delta_c', 'Id')
    self.tau = tau

    if P is None:
      P = function.zeros((2,))

    assert callable(P) or P.shape == (2,)

    self.P = P

  @property
  def x0(self):
    return np.concatenate([np.ones((8*len(self.g.basis),)), self.g.x])

  @property
  def cons(self):
    cons = util.NanVec((10*len(self.g.basis),))
    cons[-2*len(self.g.basis):] = self.g.cons
    return cons

  def prepare_namespace(self, x):

    g = self.g
    N = len(g.basis)

    ns = function.Namespace()
    ns.target = x
    ns.trial = g.basis.vector(2)
    ns.x = g.controlmap

    ns.h = ns.target[:8*N]
    ns.xvec = ns.target[8*N:]

    ns.Hbasis = g.basis.vector(8)
    ns.Hflatbasis = ns.trial
    ns.sol_i = 'trial_ni xvec_n'
    ns.H_i = 'Hbasis_ni h_n'
    ns.n = ns.x.normal()
    ns.A = _A(ns.sol, ns.x)
    ns.g = function.determinant(ns.A)
    ns.eps = self.eps
    ns.P = self.P(ns.sol, ns.x) if callable(self.P) else self.P

    if self.tau == 'delta':
      test = function.laplace(ns.trial, g.geom)
    elif self.tau == 'delta_c':
      test = function.laplace(ns.trial, ns.x)
    elif self.tau == 'Id':
      test = ns.trial
    else:
      raise AssertionError

    ns.test = test

    ns.Hdiv = function.div(ns.Hflatbasis, ns.x)
    ns.Hinn_n = 'Hflatbasis_ni n_i'

    return ns

  def _res(self, x):
    ns = self.prepare_namespace(x)

    res_hes = function.concatenate([
        (ns.Hflatbasis * ns.H[2*i + 4*j: 2*(i+1) + 4*j][None]).sum(1) +
        ns.sol[j].grad(ns.x)[i] * ns.Hdiv for j in range(2) for i in range(2)
    ])
    res_hes_bndry = - function.concatenate([ ns.sol[j].grad(ns.x)[i] * ns.Hinn for j in range(2) for i in range(2) ])

    A = ns.A

    A = ns.A + (0 if self.eps == 0 else self.eps * function.exp(-function.determinant(ns.A)) * function.eye(2))
    ns.gamma = gamma(A)

    ns.AH = function.stack([  A[0, 0] * _H[0]
                            + A[0, 1] * _H[1]
                            + A[1, 0] * _H[2]
                            + A[1, 1] * _H[3] for _H in (ns.H[:4], ns.H[4:]) ])

    res = function.concatenate([res_hes, 'gamma test_ni (AH_i + g P_k sol_i,k)' @ ns])
    res_bndry = function.concatenate([res_hes_bndry, 0 * ns.trial.sum(1)])

    g = self.g

    res = g.domain.integral( res * function.J(ns.x), degree=g.ischeme*2 )
    res += g.domain.boundary.integral( res_bndry * function.J(ns.x), degree=g.ischeme*2 )

    return res

  def fixed_point_F(self, xprev):

    nsprev = self.prepare_namespace(xprev)
    nsnow = self.prepare_namespace(function.Argument('target', [self.N]))

    ns = function.Namespace()
    ns.A = nsprev.A
    ns.gamma = gamma(ns.A)
    ns.test = nsnow.test
    ns.sol = nsnow.sol
    ns.g = nsprev.g
    ns.P = nsprev.P
    ns.Hinn = nsprev.Hinn
    ns.H = nsnow.H
    ns.Hdiv = nsnow.Hdiv
    ns.Hflatbasis = nsnow.Hflatbasis
    ns.x = nsnow.x
    ns.trial = nsnow.trial

    res_hes = function.concatenate([
        (ns.Hflatbasis * ns.H[2*i + 4*j: 2*(i+1) + 4*j][None]).sum(1) +
        ns.sol[j].grad(ns.x)[i] * ns.Hdiv for j in range(2) for i in range(2)
    ])
    res_hes_bndry = - function.concatenate([ ns.sol[j].grad(ns.x)[i] * ns.Hinn for j in range(2) for i in range(2) ])

    A = ns.A

    ns.AH = function.stack([  A[0, 0] * _H[0]
                            + A[0, 1] * _H[1]
                            + A[1, 0] * _H[2]
                            + A[1, 1] * _H[3] for _H in (ns.H[:4], ns.H[4:]) ])

    res = function.concatenate([res_hes, 'gamma test_ni (AH_i + g P_k sol_i,k)' @ ns])
    res_bndry = function.concatenate([res_hes_bndry, 0 * ns.trial.sum(1)])

    g = self.g

    res = g.domain.integral( res * function.J(ns.x), degree=g.ischeme*2 )
    res += g.domain.boundary.integral( res_bndry * function.J(ns.x), degree=g.ischeme*2 )

    return solver.solve_linear('target', res, constrain=self.cons)


def rot(vec, geom):
  # assert len(vec.shape) >= 2
  assert vec.shape[-1] == 2

  return vec.grad(geom)[..., 1, 0] - vec.grad(geom)[..., 0, 1]


def dofindices(domain, geom, basis):
  domint = domain.boundary.integrate( basis * function.J(geom),
                                      degree=10 ).export('dense')
  if len(domint.shape) == 2:
    domint = domint.sum(1)

  return ~(np.abs(domint) > 1e-10)


def epsilon(domain, A, check=10, fac=0.9):

  if not isinstance(A, (list, tuple)):
    A = [A]

  sample = domain.sample('bezier', check)
  expr = [(function.trace(a) ** 2) / (a * a).sum() for a in A]
  evalf = sample.eval(expr)
  eps = np.asarray([ e.min() - 1 for e in evalf])

  if len(eps) == 1:
    eps = eps[0]

  return fac * eps


def make_consw(domain, geom, basis, x0):

  if len(basis.shape) == 1:
    basis = basis.vector(2)

  indices = dofindices(domain, geom, basis)
  # indices = np.concatenate([indices, indices], dtype=bool)
  # indices = np.repeat(indices, 2)
  _basis = function.take(basis, ~indices, axis=0).vector(2).swapaxes(1, 2)
  proj = _basis.dot(function.Argument('target', [len(_basis)]))
  # t = function.tangent(geom, geom).normalized()
  t = function.localgradient(geom, len(geom) - 1)[:, 0].normalized()
  costfunc = (((x0.grad(geom) - proj) * t[None]).sum(1)**2).sum()
  costfunc = domain.boundary.integral(costfunc * function.J(geom), degree=10)
  _cons = solver.optimize('target', costfunc, droptol=1e-10)

  consw = util.NanVec(2 * len(basis))
  consw[~np.repeat(indices, 2)] = _cons

  return consw


def make_consw_(domain, geom, basis, x0):

  if len(basis.shape) == 1:
    basis = basis.vector(2)

  indices = dofindices(domain, geom, basis)
  # _basis = basis[~indices]
  _basis = function.take(basis, ~indices, axis=0)

  t = function.localgradient(geom, len(geom) - 1)[:, 0].normalized()
  _consw = domain.boundary.project( (x0.grad(geom) * t[None]).sum([-1]),
                                    onto=(_basis * t[None]).sum([-1]).vector(2),
                                    geometry=geom,
                                    ischeme='gauss6' )
  consw = util.NanVec( 2 * len(basis) )
  consw[:len(basis)][~indices] = _consw[:len(_basis)]
  consw[len(basis):][~indices] = _consw[len(_basis):]
  return consw


class Gallistl(EllipticSolver):

  def __init__(self, g, eps=0, mu=1e-5, basis='SG', P=None, **kwargs):
    super().__init__(g, **kwargs)
    assert basis in ('SG', 'TH')

    self.eps = float(eps)
    if basis == 'SG':
      pknotvectors = {edge: coarsen_knotvector(kv) for edge, kv in g.knotvectors.items()}
      knots = {key: value.knots for key, value in g.knotvectors.items()}
      knotmultiplicities = {key: value.knotmultiplicities for key, value in g.knotvectors.items()}
      gcoarse = g.__class__(g.patches, g.patchverts, pknotvectors)
      self.pdomain = gcoarse.domain
      self.domain = self.pdomain.refine(1)
      self._pbasis = gcoarse.basis
      self._basis = basis_spline(self.domain, knotvalues=knots,
                                              knotmultiplicities=knotmultiplicities,
                                              degree=g._degree,
                                              patchcontinuous=g._patchcontinuous)[0]
      self.controlmap = self.basis.vector(2).dot(g.project(g.controlmap))
    else:
      self._pbasis = g.basis
      pknotvectors = {key: kv.to_p(kv.degree + 1) for key, kv in self.g.knotvectors.items()}
      knotvalues = {key: kv.knots for key, kv in pknotvectors.items()}
      knotmultiplicities = {key: kv.knotmultiplicities for key, kv in pknotvectors.items()}
      degree = self.g._degree + 1
      self._basis = self.g.make_basis(knotvalues=knotvalues,
                                      knotmultiplicities=knotmultiplicities,
                                      degree=degree)
      self.domain = g.domain
      self.pdomain = g.domain
      self.controlmap = g.controlmap

    self.N = 4 * len(self.basis) + 2 * len(self.pbasis)

    if P is None:
      P = function.zeros((2,))

    assert callable(P) or P.shape == (2,)

    self.P = P

    assert mu >= 0
    self.mu = float(mu)

  @cached_property
  def basis(self):
    return self._basis

  @cached_property
  def pbasis(self):
    return self._pbasis[:-1]  # - domint

  @property
  def cons(self):
    cons = util.NanVec(self.N)
    cons[:4*len(self.g.basis)] = make_consw(self.g.domain,
                                            self.g.controlmap,
                                            self.g.basis,
                                            self.g.mapping)
    return cons

  @property
  def x0(self):
    g = self.g
    x0 = g.mapping
    n = 4 * len(self.g.basis)
    consw = self.cons[:n]
    domain = self.g.domain
    geom = self.g.controlmap
    basis = self.g.basis
    _w0 = [ domain.project( x0[i].grad(geom),
                            onto=basis.vector(2),
                            geometry=geom,
                            ischeme='gauss7',
                            constrain=_cons )
                            for i, _cons in enumerate([consw[::2], consw[1::2]]) ]
    w0 = np.zeros(len(consw))
    w0[::2] = _w0[0]
    w0[1::2] = _w0[1]
    x0 = np.concatenate([w0, np.zeros(self.N - len(w0))])
    return x0

  def prepare_namespace(self, x):

    def A_from_w(w):
      s = function.stack
      u, v = (w[:, i] for i in range(2))
      g11, g12, g22 = (u**2).sum(), (u * v).sum(), (v**2).sum()
      return s([ s([g22, -g12]), s([-g12, g11]) ])

    ns = function.Namespace()
    ns.target = x
    ns.xbasis = self.basis
    ns.basis = self.basis
    ns.pbasis = self.pbasis
    m = 2 * len(ns.pbasis)
    ns.w = ns.basis.vector(2).vector(2).swapaxes(1, 2).dot(ns.target[:-m])
    ns.p = ns.pbasis.vector(2).dot(ns.target[-m:])
    ns.x = self.controlmap
    ns.rotw = rot(ns.w, ns.x)
    ns.A = A_from_w(ns.w)
    ns.test = function.div(ns.basis.vector(2), ns.x).vector(2)
    ns.Aw_i = 'A_jk w_ij,k'
    ns.eps = self.eps
    ns.g = function.determinant(ns.A)

    if not hasattr(self, '_h'):
      dOmega = self.domain.boundary
      gammabasis = dOmega.basis_discont(0)
      weights = dOmega.integrate(gammabasis * function.J(self.controlmap), degree=5)
      self._h = gammabasis.dot(weights)

    ns.h = self._h

    # XXX: this will fail for callable self.P because ns.sol is not defined
    #      replace this by ns.w or something, not sure yet
    ns.P = self.P(ns.sol, ns.x) if callable(self.P) else self.P
    ns.t = function.localgradient(ns.x, len(ns.x) - 1)[:, 0].normalized()

    return ns

  def _res(self, x, sigma=0.1):
    raise NotImplementedError

  def fixed_point_F(self, xprev, cons):
    """ Overwrite in derived class. """
    raise NotImplementedError

  def fixed_point(self, x0=None, btol=1e-6):

    if x0 is None:
      x0 = self.x0

    fp_it = self.fixed_point_iter(x0, self.cons)
    xprev = x0

    for xnow in fp_it:
      res = np.abs(xprev - xnow).max()
      log.warning('Current residual: {}'.format(res))
      if res < btol:
        break
      xprev = xnow

    return self.recover_x(xnow)

  def recover_x(self, x):
    w = x[:4*len(self.basis)]
    basis = self.basis
    target = function.Argument('target', [2 * len(basis)])
    u = basis.vector(2).dot(target)
    _w = self.basis.vector(2).vector(2).swapaxes(1, 2).dot(w)

    res = ( basis.vector(2).grad(self.controlmap) * (u.grad(self.controlmap) - _w)[None] ).sum([-2, -1])
    res = self.domain.integral( res * function.J(self.controlmap),
                                degree=10 )
    x = solver.solve_linear('target', res, constrain=self.g.cons)
    return x

  def newton(self, x0=None, cons=None, btol=1e-5):
    raise NotImplementedError


class GallistlWeak(Gallistl):

  def __init__(self, *args, eta=1e3, **kwargs):
    super().__init__(*args, **kwargs)
    assert eta >= 0
    self.eta = float(eta)  # penalty parameter for weak imposition of Dirichlet

  @property
  def cons(self):
    return util.NanVec(self.N)

  @property
  def x0(self):
    g = self.g
    xws = [g.domain.project(g.mapping[i].grad(self.g.controlmap),
                            onto=g.basis.vector(2),
                            geometry=g.controlmap,
                            ischeme='gauss7') for i in range(2)]
    xw = np.stack(xws, axis=1).ravel()
    return np.concatenate([xw, np.zeros(self.N - len(xw))])

  def fixed_point_F(self, xprev, cons=None):
    if cons is None:
      cons = self.cons
    nsprev = self.prepare_namespace(xprev)
    nsnow = self.prepare_namespace(function.Argument('target', [self.N]))
    ns = function.Namespace()
    ns.basis = nsprev.basis
    ns.pbasis = nsprev.pbasis
    ns.rotw = nsprev.rotw
    ns.x = nsprev.x
    ns.Id = self.mu * function.eye(2)
    ns.A = nsprev.A
    ns.Aoff = ns.A + ns.Id  # offsetted A
    ns.gamma = gamma(ns.Aoff)
    ns.w = nsnow.w
    ns.p = nsnow.p
    ns.Aw_i = 'Aoff_jk w_ij,k'
    ns.wprev = nsprev.w
    ns.Idwprev_i = 'Id_jk wprev_ij,k'
    ns.test = nsnow.test
    ns.g = nsprev.g
    ns.P = nsprev.P
    ns.t = nsnow.t
    ns.h = nsnow.h

    eps = max(epsilon(self.domain, ns.Aoff), 0)
    print('using eps = {eps}'.format(eps=eps))
    lamb = 1 + np.sqrt(eps) / 2  # to fullfill |lamb - 1| < sqrt(eps)
    sigma = np.sqrt(1 - lamb/2)

    sigma = np.asarray([sigma, sigma])

    res1 = 'gamma test_ni (Aw_i + g P_k w_ik - Idwprev_i)' @ ns
    res1 += ( rot(ns.basis.vector(2), ns.x).vector(2) * ns.p[None] ).sum([-1])
    res1 += ((rot(ns.basis.vector(2), ns.x).vector(2) * sigma[None] ** 2) *
              rot(ns.w, ns.x)[None]).sum([-1])

    res2 = (rot(ns.w, ns.x)[None] * ns.pbasis.vector(2)).sum([-1])

    x0 = self.basis.vector(2).dot(self.g.cons|0)

    basist = (ns.basis.vector(2).vector(2).swapaxes(1, 2) * ns.t[None, None]).sum(-1)
    res1_bndry = self.eta * (ns.h**-1) * (((ns.w - x0.grad(ns.x)) * ns.t[None]).sum(-1)[None] * basist).sum(-1)
    res2_bndry = function.concatenate([ ns.pbasis * (v.grad(ns.x) * ns.t).sum() for v in x0 ]) * 0

    res = self.domain.integral(function.concatenate([res1, res2]) * function.J(ns.x),
                                 degree=10)
    res += self.domain.boundary.integral(function.concatenate([res1_bndry, res2_bndry]) * function.J(ns.x),
                                         degree=10)

    return solver.solve_linear('target', res, constrain=cons)

  def newton(self, x0=None, cons=None, btol=1e-5, stab=1e-5):
    raise NotImplementedError
