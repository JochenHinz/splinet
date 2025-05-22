from mapping import sol
from nutils import function, util, solver
import numpy as np


@sol.withdefaultrequirements
@sol.unpack_GridObject
def elliptic_partial(mg, Ds=None, Dx=None, *, domain, basis, ischeme,
                                              eps=1e-4, btol=1e-5, **ignorekwargs):

  """
    Coupled version of the weak-form discretisation.

    ∂x \cdot (D^x ∂x s) = 0
    ∂r \cdot (D^s ∂r s) = 0

    Parameters
    ----------

    As in the paper.

    mg: MultiPatchBSplineGridObject to perform operations on.
    Ds: :class: `Callable`
      Diffusivity imposed on the controlmap. Must be of the form
      Ds = Ds(r, x)
    Dx: :class: `Callable`
      Diffusivity imposed on the mapping. Dx = Dx(s, x)

  """

  if Ds is None:
    Ds = lambda *args: function.eye(2)

  if Dx is None:
    Dx = lambda *args: function.eye(2)

  target = function.Argument('target', [len(basis)*4])
  N = len(basis)
  x = basis.vector(2).dot(target[:2*N])
  s = basis.vector(2).dot(target[2*N:])

  dsdr = s.grad(mg.controlmap)
  Jr = x.grad(mg.controlmap)
  C = function.matmat(function.stack([ [Jr[1, 1], -Jr[1, 0]], [-Jr[0, 1], Jr[0, 0]] ]), dsdr.T)
  Dx = Dx(s, x)
  A = function.matmat(C.T, function.matmat(Dx, C))

  detJ = function.determinant(C)
  denom = (detJ + function.sqrt(4 * eps + detJ**2)) / 2

  dsdr_inv = function.stack([ [dsdr[1, 1], -dsdr[0, 1]], [-dsdr[1, 0], dsdr[0, 0]] ])

  dbasis = (basis.vector(2).grad(mg.controlmap)[:, :, None] * dsdr_inv.T[None]).sum(-1)

  res0 = (dbasis * A[None]).sum([1, 2]) * denom**(-1) * function.J(mg.controlmap)

  Ds = Ds(mg.controlmap, x)

  if Ds.shape == (1,) or Ds.shape == ():
    Ds = Ds * function.eye(2)

  Ddx = function.stack([ (Ds * _x.grad(mg.controlmap)[None]).sum([1]) for _x in s ])

  res1 = (basis.vector(2).grad(mg.controlmap) * Ddx[None]).sum([1, 2]) * function.J(mg.controlmap)

  res = mg.integral( function.concatenate([res0, res1]), degree=10 )
  f = mg.g_controlmap()

  cons = util.NanVec(target.shape)
  cons[:2*N][mg.cons.where] = mg.cons[mg.cons.where]
  cons[2*N:][f.cons.where] = f.cons[f.cons.where]

  x0 = np.concatenate([mg.x, f.x])

  x = solver.newton('target',
                    res,
                    constrain=cons.where,
                    lhs0=x0).solve(btol)

  mg.x, f.x = x[:2*N], x[2*N:]

  return mg, f


@sol.withdefaultrequirements
@sol.unpack_GridObject
def elliptic_partial_(mg, Ds=None, *, domain, basis, ischeme,
                                      eps=1e-4, btol=1e-5, **ignorekwargs):

  """
    Coupled version of the weak-form discretisation.

    ∂x \cdot (D^x ∂x s) = 0
    ∂r \cdot (D^s ∂r s) = 0

    Parameters
    ----------

    As in the paper.

    mg: MultiPatchBSplineGridObject to perform operations on.
    Ds: :class: `Callable`
      Diffusivity imposed on the controlmap. Must be of the form
      Ds = Ds(r, x)
    Dx: :class: `Callable`
      Diffusivity imposed on the mapping. Dx = Dx(s, x)

  """

  if Ds is None:
    Ds = lambda *args: function.eye(2)

  target = function.Argument('target', [len(basis)*4])
  N = len(basis)
  x = basis.vector(2).dot(target[:2*N])
  s = basis.vector(2).dot(target[2*N:])

  res0 = (basis.vector(2).grad(s) * mg.controlmap.grad(s)[None]).sum([1, 2]) * function.J(s)

  Ds = Ds(mg.controlmap, x)

  if Ds.shape == (1,) or Ds.shape == ():
    Ds = Ds * function.eye(2)

  Ddx = function.stack([ (Ds * _x.grad(mg.controlmap)[None]).sum([1]) for _x in s ])

  res1 = (basis.vector(2).grad(mg.controlmap) * Ddx[None]).sum([1, 2]) * function.J(mg.controlmap)

  res = mg.integral( function.concatenate([res0, res1]), degree=10 )
  f = mg.g_controlmap()

  cons = util.NanVec(target.shape)
  cons[:2*N][mg.cons.where] = mg.cons[mg.cons.where]
  cons[2*N:][f.cons.where] = f.cons[f.cons.where]

  x0 = np.concatenate([mg.x, f.x])

  x = solver.newton('target',
                    res,
                    constrain=cons.where,
                    lhs0=x0).solve(btol)

  mg.x, f.x = x[:2*N], x[2*N:]

  return mg, f
