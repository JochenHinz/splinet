import numpy as np
from scipy import optimize
import treelog as log
from functools import wraps, cached_property


from .template import MultiPatchTemplate
from .aux import angle_between, frozen


def ex_ey(patches):
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
      return MultiPatchTemplate(template.patches, ret, template._knotvector_edges)
    return ret
  return wrapper


class TemplateOptimiser:

  def __init__(self, template, patchverts=None):
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

  @property
  def x0(self):
    return self.patchverts[self.dofindices].ravel().copy()

  def cvector(self, c):
    """ shape (nvertices, 2) """
    assert c.shape == (len(self.dofindices) * 2,)
    ret = np.empty((len(self.template.patchverts), 2), dtype=float)
    ret[self.obi] = self.patchverts[self.obi]
    ret[self.dofindices] = c.reshape(-1, 2)
    return ret

  @cached_property
  def cross(self):
    def func(c, normalize=False):
      c = self.cvector(c)

      us = c[self.ex[:, 1]] - c[self.ex[:, 0]]
      vs = c[self.ey[:, 1]] - c[self.ey[:, 0]]

      if normalize:
        us = us / np.linalg.norm(us, axis=1)[:, None]
        vs = vs / np.linalg.norm(vs, axis=1)[:, None]

      return np.cross(us, vs)
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
      return np.piecewise(x, [x < mu, x >= mu], [lambda x: mu - x, lambda x: np.zeros(len(x))]).sum()
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
      normalization = 1 if not normalize else self._jac0
      return ((self.cross(c, normalize=False) / normalization)**2).sum()

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
def untangle_template(optimizer, mu=0.05):
  func = lambda c: optimizer.untangle_costfunc(c, mu=mu, normalize=True)
  x0 = optimizer.x0

  if func(x0) == 0:
    return optimizer.cvector(x0)

  log.warning("Initial cost function value: {}.".format(func(x0)))
  x = optimize.minimize(func, x0)
  log.warning("Template untangling success: {}".format(x.success))
  log.warning("Untangling routine terminated with cost function value: {}.".format(func(x.x)))

  if not func(x.x) < func(x0):
    return optimizer.patchverts

  return optimizer.cvector(x.x)


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
  x0 = optimizer.x0

  relax_vec = optimizer.cross(x0, normalize=False)
  assert (relax_vec > 0).all()

  cons = lambda x: optimizer.cross_constraint(x, mu=0, normalize=False) - mu * relax_vec

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


def default_strategy(template):
  """
    By default we untangle (if necessary) and then we homogenise the area.
    The default factor for untangling is .3 ,i.e., if all vertex crosses are above
    this threshold, the costfunction is zero.
    Then we homogenise the areas with relaxation factor .5.
  """
  template = untangle_template(template, return_template=True, mu=.3)
  return homogenise_areas(template, return_template=True, mu=.5)


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