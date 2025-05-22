import numpy as np
from scipy import optimize
import treelog as log
from itertools import chain, count
from functools import wraps, cached_property


from .template import MultiPatchTemplate
from .aux import angle_between, frozen


_ = np.newaxis


def ex_ey(patches):
  """
    i1 ------- i3
    | th1   th3 |
    |           |
    |           |
    | th0   th2 |
    i0 ------- i2

    ex = (i0, i2), (i1, i3), (i0, i2), (i1, i3)
    ey = (i0, i1), (i0, i1), (i2, i3), (i2, i3)

    us = c[:, optimizer.ex[:, 1]] - c[:, optimizer.ex[:, 0]]
    vs = c[:, optimizer.ey[:, 1]] - c[:, optimizer.ey[:, 0]]

    us = us / np.linalg.norm(us, axis=-1)[..., None]
    vs = vs / np.linalg.norm(vs, axis=-1)[..., None]

    angles = np.arctan2(np.cross(us, vs), (us * vs).sum(-1))

    means:
    angles = [th0(patch0), th0(patch1), ..., th0(patchN), th1(patch0), ...]
  """
  ex = np.concatenate([ [patch[::2] for patch in patches],
                        [patch[1::2] for patch in patches],
                        [patch[::2] for patch in patches],
                        [patch[1::2] for patch in patches] ])
  ey = np.concatenate([ [patch[:2] for patch in patches],
                        [patch[:2] for patch in patches],
                        [patch[2:] for patch in patches],
                        [patch[2:] for patch in patches] ])
  return ex, ey


def optionally_return_template(fn):
  @wraps(fn)
  def wrapper(optimizer, *args, return_template=False, **kwargs):
    ret = fn(optimizer, *args, **kwargs)
    if return_template:
      template = optimizer.template
      if isinstance(ret, tuple):
        return (MultiPatchTemplate(template.patches, ret[0], template._knotvector_edges),) + tuple(ret[1:])
      return MultiPatchTemplate(template.patches, ret, template._knotvector_edges)
    return ret
  return wrapper


def angle_weights(template):
  """
    return (1 for boundary indices else 2) / (val(index)) where index the vertex at the corner of the angle
    under consideration. Do this for all angles according to the numbering induced by ex_ey.

    This allows testing the deviation from the optimal corner of (1 or 2) * pi / (val(index)) created by the template.
  """
  obi = set([edge[0] for edge in template.ordered_boundary_edges[0]])
  nx = template.to_networkx()
  valences = nx.degree()

  index = count()
        
  ret = np.zeros(len(template.patches) * 4)
  for i in range(4):
    for patch in template.patches:
      vertex = patch[i]
      ret[next(index)] = 1 / 2 if vertex in obi else 2 / valences[vertex]

  return ret


class TemplateOptimiser:

  def __init__(self, template, patchverts=None, eps=1e-7):
    assert isinstance(template, MultiPatchTemplate)
    self.template = template
    if patchverts is None:
      patchverts = np.asarray(self.template.patchverts)
    patchverts = np.asarray(patchverts, dtype=float)
    assert patchverts.shape[1:] == (2,)
    assert patchverts.shape[0] == len(self.template.patchverts)
    self.patchverts = frozen(patchverts)
    self.patches = frozen(np.asarray(self.template.patches, dtype=int))
    obe, = self.template.ordered_boundary_edges
    self.obi = frozen(np.asarray([edge[0] for edge in obe], dtype=int))
    self.dofindices = frozen(np.asarray([i for i in range(len(self.template.patchverts)) if i not in self.obi], dtype=int))
    self.ex, self.ey = map(frozen, ex_ey(self.template.patches))
    self.eps = float(eps)
    self.eye = self.eps * np.eye(len(self.dofindices)*2)

  @property
  def x0(self):
    return self.patchverts[self.dofindices].ravel().copy()

  def cvector(self, c):
    """ shape (nvertices, 2) """
    shape = c.shape
    if len(c.shape) == 1:
      c = c[_]
    assert c.shape[1:] == (len(self.dofindices) * 2,)
    ret = np.empty((c.shape[0], len(self.template.patchverts), 2), dtype=float)
    ret[:, self.obi] = self.patchverts[self.obi]
    ret[:, self.dofindices] = c.reshape(c.shape[0], -1, 2)
    if len(shape) == 1:
      return ret[0]
    return ret

  @cached_property
  def cross(self):
    def func(c, normalize=False):
      c = self.cvector(c)
      shape = c.shape
      if len(c.shape) == 2:
        c = c[_]

      us = c[:, self.ex[:, 1]] - c[:, self.ex[:, 0]]
      vs = c[:, self.ey[:, 1]] - c[:, self.ey[:, 0]]

      if normalize:
        us = us / np.linalg.norm(us, axis=-1)[..., None]
        vs = vs / np.linalg.norm(vs, axis=-1)[..., None]

      ret = np.cross(us, vs)
      if len(shape) == 2:
        return ret[0]
      return ret
    return func

  @cached_property
  def area(self):

    def func(c):
      cons = self.cvector(c)
      a, b, c, d = self.patches.T
      ac_s = cons[c] - cons[a]
      cd_s = cons[d] - cons[c]
      ab_s = cons[b] - cons[a]
      bd_s = cons[d] - cons[b]
      return .5 * np.abs(np.cross(ac_s, cd_s)) + .5 * np.abs(np.cross(ab_s, bd_s))

    return func

  @cached_property
  def cross_constraint(self):
    def func(c, mu=0, normalize=False):
      return self.cross(c, normalize=normalize) - mu
    return func

  @cached_property
  def untangle_costfunc(self):
    def func(c, mu=0, **kwargs):
      x = self.cross(c, **kwargs)
      shape = x.shape
      if len(shape) == 1:
        x = x[_]
      ret = np.piecewise(x, [x < mu, x >= mu], [lambda x: mu - x, lambda x: np.zeros(x.shape)]).sum(1)
      if len(shape) == 1:
        return ret[0]
      return ret
    return func

  @cached_property
  def area_homogenisation_costfunc(self):
    if not hasattr(self, '_area'):
      self._area = self.area(self.x0).sum()

    def func(c, normalize=True):
      normalization = 1 if not normalize else self._area
      return ((self.area(c) / normalization)**2).sum()

    return func

  @cached_property
  def angles(self):
    def func(c):
      c = self.cvector(c)
      us = c[self.ex[:, 1]] - c[self.ex[:, 0]]
      vs = c[self.ey[:, 1]] - c[self.ey[:, 0]]
      return angle_between(us, vs)
    return func

  @cached_property
  def angle_homogenisation_costfunc(self):
    if not hasattr(self, '_angle0'):
      self._angle0 = self.angles(self.patchverts[self.dofindices].ravel()).sum()
    
    def func(c, normalize=True):
      normalization = 1 if not normalize else self._angle0
      return (self.angles(c) ** 2).sum() / normalization

    return func

  @cached_property
  def jacobian_homogenisation_costfunc(self):
    if not hasattr(self, '_jac0'):
      self._jac0 = self.cross(self.x0, normalize=False).sum()

    def func(c, normalize=True):
      shape = c.shape
      if len(shape) == 1:
        c = c[_]
      normalization = 1 if not normalize else self._jac0
      ret = ((self.cross(c, normalize=False) / normalization)**2).sum(1)
      if len(shape) == 1:
        return ret[0]
      return ret

    return func


def template_or_optimizer(fn):
  @wraps(fn)
  def wrapper(arg0, *args, patchverts=None, **kwargs):
    if isinstance(arg0, MultiPatchTemplate):
      arg0 = TemplateOptimiser(arg0, patchverts=patchverts)
    assert isinstance(arg0, TemplateOptimiser)
    return fn(arg0, *args, **kwargs)
  return wrapper


@template_or_optimizer
@optionally_return_template
def untangle_template(optimizer, mu=0.05, normalize=True, return_funcval=False):
  func = lambda c: optimizer.untangle_costfunc(c, mu=mu, normalize=normalize)
  grad = lambda c: (func(c + optimizer.eye) - func(c - optimizer.eye)) / (2 * optimizer.eps)
  x0 = optimizer.x0

  if func(x0) == 0:
    ret = optimizer.cvector(x0)
    return (ret,) + (0,) if return_funcval else ret

  log.warning("Initial cost function value: {}.".format(func(x0)))
  x = optimize.minimize(func, x0, jac=grad)
  log.warning("Template untangling success: {}".format(x.success))
  log.warning("Untangling routine terminated with cost function value: {}.".format(func(x.x)))

  funcval = func(x.x)

  if not funcval < func(x0) or optimizer.untangle_costfunc(x.x, mu=0.01, normalize=True) > 0:
    ret = (optimizer.patchverts,)
  else:
    ret = (optimizer.cvector(x.x),)

  return ret + (funcval,) if return_funcval else ret[0]


@template_or_optimizer
@optionally_return_template
def homogenise_angles(optimizer, mu=0.3):
  func = lambda c: optimizer.angle_homogenisation_costfunc(c, normalize=True)
  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False)

  constraints = ({'type': 'ineq', 'fun': cons},)

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch angles: {}.".format(optimizer.angles(x0)))

  x = optimize.minimize(func, x0, constraints=constraints)
  log.warning("Template angle homogenisation success: {}".format(x.success))
  log.warning("Template angle homogenisation routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch angles: {}.".format(optimizer.angles(x.x)))

  if not func(x.x) < func(x0) or not (cons(x.x) >= 0).all():
    return optimizer.patchverts

  return optimizer.cvector(x.x)


@template_or_optimizer
@optionally_return_template
def homogenise_areas(optimizer, mu=0.3):
  func = lambda c: optimizer.area_homogenisation_costfunc(c, normalize=True)
  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False)

  constraints = ({'type': 'ineq', 'fun': cons},)

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch areas: {}.".format(optimizer.area(x0)))

  x = optimize.minimize(func, x0, constraints=constraints)
  log.warning("Template area homogenisation success: {}".format(x.success))
  log.warning("Template area homogenisation routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch areas: {}.".format(optimizer.area(x.x)))

  if not func(x.x) < func(x0) or not (cons(x.x) >= 0).all():
    return optimizer.patchverts

  return optimizer.cvector(x.x)


@template_or_optimizer
@optionally_return_template
def homogenise_jacobian(optimizer, mu=0.3):
  func = lambda c: optimizer.jacobian_homogenisation_costfunc(c, normalize=True)
  grad = lambda c: (func(c + optimizer.eye) - func(c - optimizer.eye)) / (2 * optimizer.eps)
  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False) - mu * relax_vec
  cons_jac = lambda x: (cons(x[_, :] + optimizer.eye) - cons(x[_, :] - optimizer.eye)).T / (2 * optimizer.eps)

  constraints = ({'type': 'ineq', 'fun': cons, 'jac': cons_jac},)

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch jacobians: {}.".format(optimizer.cross(x0)))

  x = optimize.minimize(func, x0, constraints=constraints, jac=grad)
  log.warning("Template jacobian homogenisation success: {}".format(x.success))
  log.warning("Template jacobian homogenisation routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch jacobians: {}.".format(optimizer.cross(x.x)))

  if not func(x.x) < func(x0) or not (cons(x.x) >= 0).all():
    return optimizer.patchverts

  return optimizer.cvector(x.x)


@template_or_optimizer
@optionally_return_template
def soft_maxmin(optimizer, beta=6, mu=0.5, **ignorekwargs):

  def func(c):
    c = optimizer.cvector(c)
    shape = c.shape
    if len(shape) == 2:
      c = c[_]

    us = c[:, optimizer.ex[:, 1]] - c[:, optimizer.ex[:, 0]]
    vs = c[:, optimizer.ey[:, 1]] - c[:, optimizer.ey[:, 0]]

    us = us / np.linalg.norm(us, axis=-1)[..., None]
    vs = vs / np.linalg.norm(vs, axis=-1)[..., None]

    angles = np.arctan2(np.cross(us, vs), (us * vs).sum(-1))
    angles[angles < 0] += 2 * np.pi
    # angles = np.cross(us, vs)
    expb = np.exp(beta * angles)
    expmb = np.exp(-beta * angles)
    ret = ((angles * expb).sum(1) / (expb.sum(1))) / ((angles * expmb).sum(1) / (expmb.sum(1)))
    if len(shape) == 2:
      return ret[0]
    return ret

  grad = lambda c: (func(c + optimizer.eye) - func(c - optimizer.eye)) / (2 * optimizer.eps)

  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch angles: {}.".format(optimizer.angles(x0)))

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False) - mu * relax_vec
  cons_jac = lambda x: (cons(x[_, :] + optimizer.eye) - cons(x[_, :] - optimizer.eye)).T / (2 * optimizer.eps)

  constraints = ({'type': 'ineq', 'fun': cons, 'jac': cons_jac},)

  x = optimize.minimize(func, x0, constraints=constraints, jac=grad)
  log.warning("Template angle minmax success: {}".format(x.success))
  log.warning("Template angle minmax routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch angles: {}.".format(optimizer.angles(x.x)))

  if not func(x.x) < func(x0):
    return optimizer.patchverts

  return optimizer.cvector(x.x)


@template_or_optimizer
@optionally_return_template
def weighted_soft_maxmin(optimizer, beta=6, mu=0.5, **ignorekwargs):

  weights = np.pi * angle_weights(optimizer.template)

  def func(c):
    c = optimizer.cvector(c)
    shape = c.shape
    if len(shape) == 2:
      c = c[_]

    us = c[:, optimizer.ex[:, 1]] - c[:, optimizer.ex[:, 0]]
    vs = c[:, optimizer.ey[:, 1]] - c[:, optimizer.ey[:, 0]]

    us = us / np.linalg.norm(us, axis=-1)[..., None]
    vs = vs / np.linalg.norm(vs, axis=-1)[..., None]

    angles = np.arctan2(np.cross(us, vs), (us * vs).sum(-1))
    angles[angles < 0] += 2 * np.pi

    angles /= weights

    # angles = np.cross(us, vs)
    expb = np.exp(beta * angles)
    expmb = np.exp(-beta * angles)
    ret = ((angles * expb).sum(1) / (expb.sum(1))) / ((angles * expmb).sum(1) / (expmb.sum(1)))
    if len(shape) == 2:
      return ret[0]
    return ret

  grad = lambda c: (func(c + optimizer.eye) - func(c - optimizer.eye)) / (2 * optimizer.eps)

  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch angles: {}.".format(optimizer.angles(x0)))

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False) - mu * relax_vec
  cons_jac = lambda x: (cons(x[_, :] + optimizer.eye) - cons(x[_, :] - optimizer.eye)).T / (2 * optimizer.eps)

  constraints = ({'type': 'ineq', 'fun': cons, 'jac': cons_jac},)

  x = optimize.minimize(func, x0, constraints=constraints, jac=grad)
  log.warning("Template weighted angle minmax success: {}".format(x.success))
  log.warning("Template weighted angle minmax routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch angles: {}.".format(optimizer.angles(x.x)))

  if not func(x.x) < func(x0):
    return optimizer.patchverts

  return optimizer.cvector(x.x)


@template_or_optimizer
@optionally_return_template
def optimize_custom(optimizer, func, mu=0.05):
  x0 = optimizer.x0

  cons = lambda x: optimizer.cross_constraint(x, mu=mu, normalize=False)

  constraints = ({'type': 'ineq', 'fun': cons},)

  log.warning("Initial cost function value: {}.".format(func(x0)))
  log.warning("Initial patch jacobians: {}.".format(optimizer.cross(x0)))

  x = optimize.minimize(func, x0, constraints=constraints)
  log.warning("Template jacobian homogenisation success: {}".format(x.success))
  log.warning("Template jacobian homogenisation routine terminated with cost function value: {}.".format(func(x.x)))
  log.warning("Final patch jacobians: {}.".format(optimizer.cross(x.x)))

  if not func(x.x) < func(x0) or not (cons(x.x) >= 0).all():
    return optimizer.patchverts

  return optimizer.cvector(x.x)


def default_strategy_(template):
  """
    By default we untangle (if necessary) and then we homogenise the area.
    The default factor for untangling is .3 ,i.e., if all vertex crosses are above
    this threshold, the costfunction is zero.
    Then we homogenise the areas with relaxation factor .5.
  """
  if set(chain(*template.patches)) == set(chain(*template.ordered_boundary_edges[0])):
    return template
  template = untangle_template(template, return_template=True, mu=.3)
  return homogenise_jacobian(template, return_template=True, mu=.5)


def default_strategy_old(template):
  if set(chain(*template.patches)) == set(chain(*template.ordered_boundary_edges[0])):
    return template
  # template = soft_maxmin(template, return_template=True)
  for mu in (0, .1, .2, .3):
    template = untangle_template(template, return_template=True, mu=mu, normalize=True)
  # return weighted_soft_maxmin(template, return_template=True)
  return template


def default_strategy(template):
  if set(chain(*template.patches)) == set(chain(*template.ordered_boundary_edges[0])):
    return template
  mu = 0
  temp_prev = template
  while True:
    log.warning("Untangling template with mu = {}.".format(mu))
    template, funcval = untangle_template(temp_prev, return_template=True, mu=mu, normalize=False, return_funcval=True)
    if funcval > 0:
      return temp_prev
    temp_prev = template
    mu += 0.1


if __name__ == "__main__":
  from adjacency import even_n_leaf_template

  template = even_n_leaf_template(12).normalize()
  template.qplot()

  template0 = homogenise_angles(template, mu=0.00001, return_template=True)
  template0.qplot()

  template0 = homogenise_areas(template, mu=0.00001, return_template=True)
  template0.qplot()

  template0 = homogenise_jacobian(template, mu=0.00001, return_template=True)
  template0.qplot()