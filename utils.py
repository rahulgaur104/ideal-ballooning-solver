#!/usr/bin/env python
"""

This script contains a bunch of functions that help calculate the geometric coefficients
and the ideal ballooning growth rate


"""

import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from scipy.integrate import simps
from scipy.interpolate import CubicSpline as cubspl
from scipy.sparse.linalg import eigs

from simsopt.mhd.vmec import Vmec

# from netCDF4 import Dataset as ds

# from matplotlib import pyplot as plt
import pdb
import sys

######################################################################################################################
########################-------------------------3D EQUILIBRIUM ONLY----------------------############################
######################################################################################################################


class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """


def vmec_splines(vmec):
    """
    Initialize radial splines for a VMEC equilibrium.
    Args:
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec`.

    Returns:
        A structure with the splines as attributes.
    """
    vmec.run()
    results = Struct()

    rmnc = []
    zmns = []
    lmns = []
    psi = []
    d_rmnc_d_s = []
    d_zmns_d_s = []
    d_lmns_d_s = []
    d_psi_d_s = []

    for jmn in range(vmec.wout.mnmax):
        rmnc.append(
            InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.rmnc[jmn, :])
        )
        zmns.append(
            InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.zmns[jmn, :])
        )
        lmns.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.lmns[jmn, 1:])
        )
        d_rmnc_d_s.append(rmnc[-1].derivative())
        d_zmns_d_s.append(zmns[-1].derivative())
        d_lmns_d_s.append(lmns[-1].derivative())

    gmnc = []
    bmnc = []
    bsupumnc = []
    bsupvmnc = []
    bsubsmns = []
    bsubumnc = []
    bsubvmnc = []
    d_bmnc_d_s = []
    d_bsupumnc_d_s = []
    d_bsupvmnc_d_s = []
    for jmn in range(vmec.wout.mnmax_nyq):
        gmnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.gmnc[jmn, 1:])
        )
        bmnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bmnc[jmn, 1:])
        )
        bsupumnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupumnc[jmn, 1:])
        )
        bsupvmnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupvmnc[jmn, 1:])
        )
        # Note that bsubsmns is on the full mesh, unlike the other components:
        bsubsmns.append(
            InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.bsubsmns[jmn, :])
        )
        bsubumnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubumnc[jmn, 1:])
        )
        bsubvmnc.append(
            InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubvmnc[jmn, 1:])
        )
        d_bmnc_d_s.append(bmnc[-1].derivative())
        d_bsupumnc_d_s.append(bsupumnc[-1].derivative())
        d_bsupvmnc_d_s.append(bsupvmnc[-1].derivative())

    # Handle 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(
        vmec.s_half_grid, vmec.wout.pres[1:]
    )
    results.d_pressure_d_s = results.pressure.derivative()

    results.psi = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.chi[1:])
    results.d_psi_d_s = results.psi.derivative()

    results.iota = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:])
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = vmec.wout.phi[-1]
    variables = [
        "Aminor_p",
        "mnmax",
        "xm",
        "xn",
        "mnmax_nyq",
        "xm_nyq",
        "xn_nyq",
        "nfp",
        "raxis_cc",
    ]
    for v in variables:
        results.__setattr__(v, eval("vmec.wout." + v))

    variables = [
        "rmnc",
        "zmns",
        "lmns",
        "d_rmnc_d_s",
        "d_zmns_d_s",
        "d_lmns_d_s",
        "gmnc",
        "bmnc",
        "d_bmnc_d_s",
        "bsupumnc",
        "bsupvmnc",
        "d_bsupumnc_d_s",
        "d_bsupvmnc_d_s",
        "bsubsmns",
        "bsubumnc",
        "bsubvmnc",
    ]
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def vmec_fieldlines(
    vs, s, alpha, theta1d=None, phi1d=None, phi_center=0, plot=False, show=True
):
    r"""
    Compute field lines in a vmec configuration, and compute many
    geometric quantities of interest along the field lines. In
    particular, this routine computes the geometric quantities that
    enter the gyrokinetic equation.

    One of the tasks performed by this function is to convert between
    the poloidal angles :math:`\theta_{vmec}` and
    :math:`\theta_{pest}`. The latter is the angle in which the field
    lines are straight when used in combination with the standard
    toroidal angle :math:`\phi`. Note that all angles in this function
    have period :math:`2\pi`, not period 1.

    For the inputs and outputs of this function, a field line label
    coordinate is defined by

    .. math::

        \alpha = \theta_{pest} - \iota (\phi - \phi_{center}).

    Here, :math:`\phi_{center}` is a constant, usually 0, which can be
    set to a nonzero value if desired so the magnetic shear
    contribution to :math:`\nabla\alpha` vanishes at a toroidal angle
    different than 0.  Also, wherever the term ``psi`` appears in
    variable names in this function and the returned arrays, it means
    :math:`\psi =` the toroidal flux divided by :math:`2\pi`, so

    .. math::

        \vec{B} = \nabla\psi\times\nabla\theta_{pest} + \iota\nabla\phi\times\nabla\psi = \nabla\psi\times\nabla\alpha.

    To specify the parallel extent of the field lines, you can provide
    either a grid of :math:`\theta_{pest}` values or a grid of
    :math:`\phi` values. If you specify both or neither, ``ValueError``
    will be raised.

    Most of the arrays that are computed have shape ``(ns, nalpha,
    nl)``, where ``ns`` is the number of flux surfaces, ``nalpha`` is the
    number of field lines on each flux surface, and ``nl`` is the number
    of grid points along each field line. In other words, ``ns`` is the
    size of the input ``s`` array, ``nalpha`` is the size of the input
    ``alpha`` array, and ``nl`` is the size of the input ``theta1d`` or
    ``phi1d`` array. The output arrays are returned as attributes of the
    returned object. Many intermediate quantities are included, such
    as the Cartesian components of the covariant and contravariant
    basis vectors. Some of the most useful of these output arrays are (all with SI units):

    - ``phi``: The standard toroidal angle :math:`\phi`.
    - ``theta_vmec``: VMEC's poloidal angle :math:`\theta_{vmec}`.
    - ``theta_pest``: The straight-field-line angle :math:`\theta_{pest}` associated with :math:`\phi`.
    - ``modB``: The magnetic field magnitude :math:`|B|`.
    - ``B_sup_theta_vmec``: :math:`\vec{B}\cdot\nabla\theta_{vmec}`.
    - ``B_sup_phi``: :math:`\vec{B}\cdot\nabla\phi`.
    - ``B_cross_grad_B_dot_grad_alpha``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\alpha`.
    - ``B_cross_grad_B_dot_grad_psi``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\psi`.
    - ``B_cross_kappa_dot_grad_alpha``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\alpha`,
      where :math:`\vec{\kappa}=\vec{b}\cdot\nabla\vec{b}` is the curvature and :math:`\vec{b}=|B|^{-1}\vec{B}`.
    - ``B_cross_kappa_dot_grad_psi``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\psi`.
    - ``grad_alpha_dot_grad_alpha``: :math:`|\nabla\alpha|^2 = \nabla\alpha\cdot\nabla\alpha`.
    - ``grad_alpha_dot_grad_psi``: :math:`\nabla\alpha\cdot\nabla\psi`.
    - ``grad_psi_dot_grad_psi``: :math:`|\nabla\psi|^2 = \nabla\psi\cdot\nabla\psi`.
    - ``iota``: The rotational transform :math:`\iota`. This array has shape ``(ns,)``.
    - ``shat``: The magnetic shear :math:`\hat s= (x/q) (d q / d x)` where
      :math:`x = \mathrm{Aminor_p} \, \sqrt{s}` and :math:`q=1/\iota`. This array has shape ``(ns,)``.

    The following normalized versions of these quantities used in the
    gyrokinetic codes ``stella``, ``gs2``, and ``GX`` are also
    returned: ``bmag``, ``gbdrift``, ``gbdrift0``, ``cvdrift``,
    ``cvdrift0``, ``gds2``, ``gds21``, and ``gds22``, along with
    ``L_reference`` and ``B_reference``.  Instead of ``gradpar``, two
    variants are returned, ``gradpar_theta_pest`` and ``gradpar_phi``,
    corresponding to choosing either :math:`\theta_{pest}` or
    :math:`\phi` as the parallel coordinate.

    The value(s) of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.

    The implementation in this routine is similar to the one in the
    gyrokinetic code ``stella``.

    Example usage::

        import numpy as np
        from simsopt.mhd.vmec import Vmec
        from simsopt.mhd.vmec_diagnostics import vmec_fieldlines

        v = Vmec('wout_li383_1.4m.nc')
        theta = np.linspace(-np.pi, np.pi, 50)
        fl = vmec_fieldlines(v, 0.5, 0, theta1d=theta)
        print(fl.B_cross_grad_B_dot_grad_alpha)

    Args:
        vmec_fname: name of the input vmec file. Ex. "W7X.nc"
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        alpha: Values of the field line label :math:`\alpha` on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta1d: 1D array of :math:`\theta_{pest}` values, setting the grid points
          along the field line and the parallel extent of the field line.
        phi1d: 1D array of :math:`\phi` values, setting the grid points along the
          field line and the parallel extent of the field line.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
        plot: Whether to create a plot of the main geometric quantities. Only one field line will
          be plotted, corresponding to the leading elements of ``s`` and ``alpha``.
        show: Only matters if ``plot==True``. Whether to call matplotlib's ``show()`` function
          after creating the plot.
    """
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    shat = (-2 * s / iota) * d_iota_d_s

    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))

    ######## CAREFUL!!###########################################################
    # Everything here and in vmec_splines is designed for up-down symmetric eqlbia
    # When we start optimizing equilibria with lasym = "True"
    # we should edit this as well as vmec_splines
    lmnc = np.zeros((ns, mnmax))
    # lasym = 0

    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    bsubsmns = np.zeros((ns, mnmax_nyq))
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))
    # pdb.set_trace()
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)
        bsubsmns[:, jmn] = vs.bsubsmns[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)

    theta_pest = np.zeros((ns, nalpha, nl))
    phi = np.zeros((ns, nalpha, nl))

    if theta1d is None:
        # We are given phi. Compute theta_pest:
        for js in range(ns):
            phi[js, :, :] = phi1d[None, :]
            theta_pest[js, :, :] = alpha[:, None] + iota[js] * (
                phi1d[None, :] - phi_center
            )
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_pest[js, :, :] = theta1d[None, :]
            phi[js, :, :] = phi_center + (theta1d[None, :] - alpha[:, None]) / iota[js]

    def residual(theta_v, phi0, theta_p_target, jradius):
        """
        This function is used for computing the value of theta_vmec that
        gives a desired theta_pest.
        """
        """
        theta_p = theta_v
        for jmn in range(len(xn)):
            angle = xm[jmn] * theta_v - xn[jmn] * phi0
            theta_p += lmns[jradius, jmn] * np.sin(angle)
        return theta_p_target - theta_p
        """
        return theta_p_target - (
            theta_v + np.sum(lmns[jradius, :] * np.sin(xm * theta_v - xn * phi0))
        )

    def residual(theta_v, phi0, theta_p_target, jradius):
        """
        This function is used for computing an array of values of theta_vmec that
        give a desired theta_pest array.
        """
        return theta_p_target - (
            theta_v
            + np.sum(
                lmns[js, :, None] * np.sin(xm[:, None] * theta_v - xn[:, None] * phi0),
                axis=0,
            )
        )

    theta_vmec = np.zeros((ns, nalpha, nl))

    # This is more robust than the root finder used by GSL
    for js in range(ns):
        for jalpha in range(nalpha):
            theta_guess = theta_pest[js, jalpha, :]
            solution = newton(
                residual,
                x0=theta_guess,
                x1=theta_guess + 0.1,
                args=(phi[js, jalpha, :], theta_pest[js, jalpha, :], js),
            )
            theta_vmec[js, jalpha, :] = solution

    # print("theta_vmec_old-new", np.max(np.abs(theta_vmec_old-theta_vmec)))
    # Now that we know theta_vmec, compute all the geometric quantities
    angle = (
        xm[:, None, None, None] * theta_vmec[None, :, :, :]
        - xn[:, None, None, None] * phi[None, :, :, :]
    )
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum("ij,jikl->ikl", rmnc, cosangle)
    d_R_d_s = np.einsum("ij,jikl->ikl", d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum("ij,jikl->ikl", rmnc, msinangle)
    d_R_d_phi = np.einsum("ij,jikl->ikl", rmnc, nsinangle)

    Z = np.einsum("ij,jikl->ikl", zmns, sinangle)
    d_Z_d_s = np.einsum("ij,jikl->ikl", d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum("ij,jikl->ikl", zmns, mcosangle)
    d_Z_d_phi = -np.einsum("ij,jikl->ikl", zmns, ncosangle)

    d_lambda_d_s = np.einsum("ij,jikl->ikl", d_lmns_d_s, sinangle)
    d_lambda_d_theta_vmec = np.einsum("ij,jikl->ikl", lmns, mcosangle)
    d_lambda_d_phi = -np.einsum("ij,jikl->ikl", lmns, ncosangle)

    # Now handle the Nyquist quantities:
    angle = (
        xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :]
        - xn_nyq[:, None, None, None] * phi[None, :, :, :]
    )
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None, None, None] * cosangle
    ncosangle = xn_nyq[:, None, None, None] * cosangle
    msinangle = xm_nyq[:, None, None, None] * sinangle
    nsinangle = xn_nyq[:, None, None, None] * sinangle

    sqrt_g_vmec = np.einsum("ij,jikl->ikl", gmnc, cosangle)
    modB = np.einsum("ij,jikl->ikl", bmnc, cosangle)
    d_B_d_s = np.einsum("ij,jikl->ikl", d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum("ij,jikl->ikl", bmnc, msinangle)
    d_B_d_phi = np.einsum("ij,jikl->ikl", bmnc, nsinangle)

    B_sup_theta_vmec = np.einsum("ij,jikl->ikl", bsupumnc, cosangle)
    B_sup_phi = np.einsum("ij,jikl->ikl", bsupvmnc, cosangle)
    B_sub_s = np.einsum("ij,jikl->ikl", bsubsmns, sinangle)
    B_sub_theta_vmec = np.einsum("ij,jikl->ikl", bsubumnc, cosangle)
    B_sub_phi = np.einsum("ij,jikl->ikl", bsubvmnc, cosangle)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    # X = R * cos(phi):
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    # Y = R * sin(phi):
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (
        d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi
    ) / sqrt_g_vmec
    grad_s_Y = (
        d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi
    ) / sqrt_g_vmec
    grad_s_Z = (
        d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi
    ) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    # Form grad alpha = grad (theta_vmec + lambda - iota * phi)
    grad_alpha_X = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_X
    grad_alpha_Y = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_Y
    grad_alpha_Z = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_Z

    grad_alpha_X += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_X + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_X
    grad_alpha_Y += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_Y
    grad_alpha_Z += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_Z

    grad_B_X = (
        d_B_d_s * grad_s_X
        + d_B_d_theta_vmec * grad_theta_vmec_X
        + d_B_d_phi * grad_phi_X
    )
    grad_B_Y = (
        d_B_d_s * grad_s_Y
        + d_B_d_theta_vmec * grad_theta_vmec_Y
        + d_B_d_phi * grad_phi_Y
    )
    grad_B_Z = (
        d_B_d_s * grad_s_Z
        + d_B_d_theta_vmec * grad_theta_vmec_Z
        + d_B_d_phi * grad_phi_Z
    )

    B_X = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_X_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Y = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_Y_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Z = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_Z_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec
        )
        / sqrt_g_vmec
    )

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_s_dot_grad_alpha = (
        B_sub_phi * (1 + d_lambda_d_theta_vmec)
        - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
    ) / sqrt_g_vmec

    B_cross_grad_s_dot_grad_alpha_alternate = (
        0
        + B_X * grad_s_Y * grad_alpha_Z
        + B_Y * grad_s_Z * grad_alpha_X
        + B_Z * grad_s_X * grad_alpha_Y
        - B_Z * grad_s_Y * grad_alpha_X
        - B_X * grad_s_Z * grad_alpha_Y
        - B_Y * grad_s_X * grad_alpha_Z
    )

    B_cross_grad_B_dot_grad_alpha = (
        0
        + (
            B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
            + B_sub_theta_vmec
            * d_B_d_phi
            * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None])
            + B_sub_phi * d_B_d_s * (1 + d_lambda_d_theta_vmec)
            - B_sub_phi
            * d_B_d_theta_vmec
            * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None])
            - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None])
            - B_sub_s * d_B_d_phi * (1 + d_lambda_d_theta_vmec)
        )
        / sqrt_g_vmec
    )

    B_cross_grad_B_dot_grad_alpha_alternate = (
        0
        + B_X * grad_B_Y * grad_alpha_Z
        + B_Y * grad_B_Z * grad_alpha_X
        + B_Z * grad_B_X * grad_alpha_Y
        - B_Z * grad_B_Y * grad_alpha_X
        - B_X * grad_B_Z * grad_alpha_Y
        - B_Y * grad_B_X * grad_alpha_Z
    )

    grad_alpha_dot_grad_alpha = (
        grad_alpha_X * grad_alpha_X
        + grad_alpha_Y * grad_alpha_Y
        + grad_alpha_Z * grad_alpha_Z
    )

    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    grad_psi_dot_grad_psi = (
        grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z
    )

    B_cross_grad_B_dot_grad_psi = (
        (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec)
        / sqrt_g_vmec
        * edge_toroidal_flux_over_2pi
    )

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = (
        B_cross_grad_B_dot_grad_alpha / modB
        + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi
    )

    # stella / gs2 / gx quantities:

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    # This is half of the total beta_N. Used in GS2 as beta_ref
    beta_N = 4 * np.pi * 1e-7 * vs.pressure(s) / B_reference**2

    tprim = -1 * d_pressure_d_s * 2 * sqrt_s * 2 / 3 * 1 / vs.pressure(s)
    fprim = -1 * d_pressure_d_s * 2 * sqrt_s * 1 / 3 * 1 / vs.pressure(s)
    # tprim = -1 * d_pressure_d_s * 2 * sqrt_s * 2 / 5 * 1 / vs.pressure(s)
    # fprim = -1 * d_pressure_d_s * 2 * sqrt_s * 3 / 5 * 1 / vs.pressure(s)

    temp = (vs.pressure(s)) ** (2 / 3)
    dens = (vs.pressure(s)) ** (1 / 3)

    bmag = modB / B_reference
    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB
    gradpar_phi = L_reference * B_sup_phi / modB

    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference
    gds22 = (
        grad_psi_dot_grad_psi
        * shat[:, None, None]
        * shat[:, None, None]
        / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])
    )

    # temporary fix. Please see issue #238 and the discussion therein
    gbdrift = (
        -1.0
        * 2
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * B_cross_grad_B_dot_grad_alpha
        / (modB * modB * modB)
        * toroidal_flux_sign
    )

    gbdrift0 = (
        -1.0
        * B_cross_grad_B_dot_grad_psi
        * 2
        * shat[:, None, None]
        / (modB * modB * modB * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )

    # temporary fix. Please see issue #238 and the discussion therein
    cvdrift = 1.0 * gbdrift - 2 * B_reference * L_reference * L_reference * sqrt_s[
        :, None, None
    ] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (
        edge_toroidal_flux_over_2pi * modB * modB
    )

    cvdrift0 = gbdrift0

    # Package results into a structure to return:
    results = Struct()
    variables = [
        "ns",
        "nalpha",
        "nl",
        "s",
        "iota",
        "d_iota_d_s",
        "d_pressure_d_s",
        "shat",
        "alpha",
        "theta1d",
        "phi1d",
        "phi",
        "theta_pest",
        "d_lambda_d_s",
        "d_lambda_d_theta_vmec",
        "d_lambda_d_phi",
        "sqrt_g_vmec",
        "sqrt_g_vmec_alt",
        "theta_vmec",
        "modB",
        "d_B_d_s",
        "d_B_d_theta_vmec",
        "d_B_d_phi",
        "B_sup_theta_vmec",
        "B_sup_theta_pest",
        "B_sup_phi",
        "B_sub_s",
        "B_sub_theta_vmec",
        "B_sub_phi",
        "edge_toroidal_flux_over_2pi",
        "sinphi",
        "cosphi",
        "R",
        "d_R_d_s",
        "d_R_d_theta_vmec",
        "d_R_d_phi",
        "Z",
        "d_Z_d_s",
        "d_Z_d_theta_vmec",
        "d_Z_d_phi",
        "d_X_d_theta_vmec",
        "d_X_d_phi",
        "d_X_d_s",
        "d_Y_d_theta_vmec",
        "d_Y_d_phi",
        "d_Y_d_s",
        "grad_s_X",
        "grad_s_Y",
        "grad_s_Z",
        "grad_theta_vmec_X",
        "grad_theta_vmec_Y",
        "grad_theta_vmec_Z",
        "grad_phi_X",
        "grad_phi_Y",
        "grad_phi_Z",
        "grad_psi_X",
        "grad_psi_Y",
        "grad_psi_Z",
        "grad_alpha_X",
        "grad_alpha_Y",
        "grad_alpha_Z",
        "grad_B_X",
        "grad_B_Y",
        "grad_B_Z",
        "B_X",
        "B_Y",
        "B_Z",
        "B_cross_grad_s_dot_grad_alpha",
        "B_cross_grad_s_dot_grad_alpha_alternate",
        "B_cross_grad_B_dot_grad_alpha",
        "B_cross_grad_B_dot_grad_alpha_alternate",
        "B_cross_grad_B_dot_grad_psi",
        "B_cross_kappa_dot_grad_psi",
        "B_cross_kappa_dot_grad_alpha",
        "grad_alpha_dot_grad_alpha",
        "grad_alpha_dot_grad_psi",
        "grad_psi_dot_grad_psi",
        "L_reference",
        "B_reference",
        "toroidal_flux_sign",
        "bmag",
        "gradpar_theta_pest",
        "gradpar_phi",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
        "beta_N",
        "tprim",
        "fprim",
        "dens",
        "temp",
    ]

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(13, 7))
        nrows = 4
        ncols = 5
        variables = [
            "modB",
            "B_sup_theta_pest",
            "B_sup_phi",
            "B_cross_grad_B_dot_grad_alpha",
            "B_cross_grad_B_dot_grad_psi",
            "B_cross_kappa_dot_grad_alpha",
            "B_cross_kappa_dot_grad_psi",
            "grad_alpha_dot_grad_alpha",
            "grad_alpha_dot_grad_psi",
            "grad_psi_dot_grad_psi",
            "bmag",
            "gradpar_theta_pest",
            "gradpar_phi",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
            "gds2",
            "gds21",
            "gds22",
        ]
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(phi[0, 0, :], eval(variable + "[0, 0, :]"))
            plt.xlabel("Standard toroidal angle $\phi$")
            plt.title(variable)

        plt.figtext(0.5, 0.995, f"s={s[0]}, alpha={alpha[0]}", ha="center", va="top")
        plt.tight_layout()
        if show:
            plt.show()

    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#########################################################################################################
#######################------------------AXISYMMETRIC EQLBIA ONLY--------------------####################
#########################################################################################################


def vmec_fieldlines_axisym(
    vs, s, alpha, theta1d=None, phi1d=None, phi_center=0, plot=False, show=True
):

    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    # For axisymmetric equilibria, all field lines are identical, i.e., your choice of alpha doesn't matter
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    # mnmax_nyq = vs.mnmax_nyq
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = np.zeros(np.shape(xm_nyq))

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    shat = (-2 * s / iota) * d_iota_d_s

    d_psi_d_s = vs.d_psi_d_s(s)

    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))

    ######## CAREFUL!!###########################################################
    # Everything here and in vmec_splines is designed for up-down symmetric eqlbia
    # When we start optimizing equilibria with lasym = "True"
    # we should edit this as well as vmec_splines
    lmnc = np.zeros((ns, mnmax))
    # lasym = 0

    # pdb.set_trace()
    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    bsubsmns = np.zeros((ns, mnmax_nyq))
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))

    # pdb.set_trace()
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)

        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)
        bsubsmns[:, jmn] = vs.bsubsmns[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)

    theta_pest = np.zeros((ns, nalpha, nl))
    phi = np.zeros((ns, nalpha, nl))

    ## Solve for theta_vmec corresponding to theta_pest:
    ## Does the same calculation as the commented code above but faster
    theta_vmec = np.zeros((ns, nalpha, nl))
    for js in range(ns):
        for jalpha in range(nalpha):
            theta_vmec[js, jalpha] = np.linspace(
                np.min(theta1d), np.max(theta1d), len(theta1d)
            )

    # print("theta_vmec_old-new", np.max(np.abs(theta_vmec_old-theta_vmec)))
    # Now that we know theta_vmec, compute all the geometric quantities
    angle = (
        xm[:, None, None, None] * (theta_vmec[None, :, :, :])
        - xn[:, None, None, None] * phi[None, :, :, :]
    )
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)

    R = np.einsum("ij,jikl->ikl", rmnc, cosangle)
    Z = np.einsum("ij,jikl->ikl", zmns, sinangle)

    flipit = 0.0

    # if R is increasing AND Z is decreasing, we must be moving counter clockwise from
    # the inboard side, otherwise we need to flip the theta coordinate
    if R[0][0][0] > R[0][0][1] or Z[0][0][1] > Z[0][0][0]:
        # if R[0][0][0] > R[0][0][1]:
        flipit = 1

    if flipit == 1:
        angle = (
            xm[:, None, None, None] * (theta_vmec[None, :, :, :] + np.pi)
            - xn[:, None, None, None] * phi[None, :, :, :]
        )
    else:
        angle = (
            xm[:, None, None, None] * (theta_vmec[None, :, :, :])
            - xn[:, None, None, None] * phi[None, :, :, :]
        )
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum("ij,jikl->ikl", rmnc, cosangle)
    d_R_d_s = np.einsum("ij,jikl->ikl", d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum("ij,jikl->ikl", rmnc, msinangle)
    d_R_d_phi = np.einsum("ij,jikl->ikl", rmnc, nsinangle)

    lambdas = np.einsum("ij,jikl->ikl", lmns, sinangle)

    d_lambda_d_s = np.einsum("ij,jikl->ikl", d_lmns_d_s, sinangle)
    d_lambda_d_theta_vmec = np.einsum("ij,jikl->ikl", lmns, mcosangle)
    d_lambda_d_phi = -np.einsum("ij,jikl->ikl", lmns, ncosangle)

    Z = np.einsum("ij,jikl->ikl", zmns, sinangle)
    d_Z_d_s = np.einsum("ij,jikl->ikl", d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum("ij,jikl->ikl", zmns, mcosangle)
    d_Z_d_phi = -np.einsum("ij,jikl->ikl", zmns, ncosangle)

    R_mag_ax = vs.raxis_cc

    # geometric theta; denotest the actual poloidal angle
    theta_geo = np.array([np.arctan2(Z[i], R[i] - R_mag_ax) for i in range(ns)])

    # Instead of finding theta_vmec for a given theta_PEST, we take a uniformly spaced theta_vmec,
    # find the corresponding theta_vmec, and interpolate the coeffiecients to the input theta_PEST
    # grid
    theta_pest = theta_vmec + lambdas
    for js in range(ns):
        # pdb.set_trace()
        phi[js, :, :] = phi_center + (theta_pest[js] - alpha[:, None]) / iota[js]

    # Now handle the Nyquist quantities:
    if flipit == 1:
        angle = xm_nyq[:, None, None, None] * (theta_vmec[None, :, :, :] + np.pi)
    else:
        angle = xm_nyq[:, None, None, None] * (theta_vmec[None, :, :, :])

    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None, None, None] * cosangle
    ncosangle = xn_nyq[:, None, None, None] * cosangle
    msinangle = xm_nyq[:, None, None, None] * sinangle
    nsinangle = xn_nyq[:, None, None, None] * sinangle

    sqrt_g_vmec = np.einsum("ij,jikl->ikl", gmnc, cosangle)
    modB = np.einsum("ij,jikl->ikl", bmnc, cosangle)
    d_B_d_s = np.einsum("ij,jikl->ikl", d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum("ij,jikl->ikl", bmnc, msinangle)
    d_B_d_phi = np.einsum("ij,jikl->ikl", bmnc, nsinangle)

    B_sup_theta_vmec = np.einsum("ij,jikl->ikl", bsupumnc, cosangle)
    B_sup_phi = np.einsum("ij,jikl->ikl", bsupvmnc, cosangle)
    B_sub_s = np.einsum("ij,jikl->ikl", bsubsmns, sinangle)
    B_sub_theta_vmec = np.einsum("ij,jikl->ikl", bsubumnc, cosangle)
    B_sub_phi = np.einsum("ij,jikl->ikl", bsubvmnc, cosangle)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    # RG: instead of doing this, I adjust the sign of the radial poloidal flux gradient
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    # X = R * cos(phi):
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    # Y = R * sin(phi):
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (
        d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi
    ) / sqrt_g_vmec
    grad_s_Y = (
        d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi
    ) / sqrt_g_vmec
    grad_s_Z = (
        d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi
    ) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    # Form grad alpha = grad (theta_vmec + lambda - iota * phi)
    grad_alpha_X = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_X
    grad_alpha_Y = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_Y
    grad_alpha_Z = (
        d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]
    ) * grad_s_Z

    grad_alpha_X += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_X + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_X
    grad_alpha_Y += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_Y
    grad_alpha_Z += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z + (
        -iota[:, None, None] + d_lambda_d_phi
    ) * grad_phi_Z

    grad_B_X = (
        d_B_d_s * grad_s_X
        + d_B_d_theta_vmec * grad_theta_vmec_X
        + d_B_d_phi * grad_phi_X
    )
    grad_B_Y = (
        d_B_d_s * grad_s_Y
        + d_B_d_theta_vmec * grad_theta_vmec_Y
        + d_B_d_phi * grad_phi_Y
    )
    grad_B_Z = (
        d_B_d_s * grad_s_Z
        + d_B_d_theta_vmec * grad_theta_vmec_Z
        + d_B_d_phi * grad_phi_Z
    )

    B_X = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_X_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Y = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_Y_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Z = (
        edge_toroidal_flux_over_2pi
        * (
            (1 + d_lambda_d_theta_vmec) * d_Z_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec
        )
        / sqrt_g_vmec
    )

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_s_dot_grad_alpha = (
        B_sub_phi * (1 + d_lambda_d_theta_vmec)
        - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
    ) / sqrt_g_vmec

    B_cross_grad_s_dot_grad_alpha_alternate = (
        0
        + B_X * grad_s_Y * grad_alpha_Z
        + B_Y * grad_s_Z * grad_alpha_X
        + B_Z * grad_s_X * grad_alpha_Y
        - B_Z * grad_s_Y * grad_alpha_X
        - B_X * grad_s_Z * grad_alpha_Y
        - B_Y * grad_s_X * grad_alpha_Z
    )

    B_cross_grad_B_dot_grad_alpha = (
        0
        + (
            B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
            + B_sub_theta_vmec
            * d_B_d_phi
            * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None])
            + B_sub_phi * d_B_d_s * (1 + d_lambda_d_theta_vmec)
            - B_sub_phi
            * d_B_d_theta_vmec
            * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None])
            - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None])
            - B_sub_s * d_B_d_phi * (1 + d_lambda_d_theta_vmec)
        )
        / sqrt_g_vmec
    )

    B_cross_grad_B_dot_grad_alpha_alternate = (
        0
        + B_X * grad_B_Y * grad_alpha_Z
        + B_Y * grad_B_Z * grad_alpha_X
        + B_Z * grad_B_X * grad_alpha_Y
        - B_Z * grad_B_Y * grad_alpha_X
        - B_X * grad_B_Z * grad_alpha_Y
        - B_Y * grad_B_X * grad_alpha_Z
    )

    grad_alpha_dot_grad_alpha = (
        grad_alpha_X * grad_alpha_X
        + grad_alpha_Y * grad_alpha_Y
        + grad_alpha_Z * grad_alpha_Z
    )

    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    grad_psi_dot_grad_psi = (
        grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z
    )

    B_cross_grad_B_dot_grad_psi = (
        (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec)
        / sqrt_g_vmec
        * edge_toroidal_flux_over_2pi
    )

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = (
        B_cross_grad_B_dot_grad_alpha / modB
        + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi
    )

    # stella / gs2 / gx quantities:

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    # This is half of the total beta_N. Used in GS2 as beta_ref
    beta_N = 4 * np.pi * 1e-7 * vs.pressure(s) / B_reference**2

    tprim = -1 * d_pressure_d_s * 2 * sqrt_s * 2 / 3 * 1 / vs.pressure(s)
    fprim = -1 * d_pressure_d_s * 2 * sqrt_s * 1 / 3 * 1 / vs.pressure(s)
    # tprim = -1 * d_pressure_d_s * 2 * sqrt_s * 2 / 5 * 1 / vs.pressure(s)
    # fprim = -1 * d_pressure_d_s * 2 * sqrt_s * 3 / 5 * 1 / vs.pressure(s)

    temp = (vs.pressure(s)) ** (2 / 3)
    dens = (vs.pressure(s)) ** (1 / 3)

    B_p = np.sqrt(
        B_sub_theta_vmec
        * abs(edge_toroidal_flux_over_2pi)
        * np.reshape(iota, (-1, 1, 1))
    )

    bmag = modB / B_reference
    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB
    gradpar_phi = L_reference * B_sup_phi / modB

    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference
    gds22 = (
        grad_psi_dot_grad_psi
        * shat[:, None, None]
        * shat[:, None, None]
        / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])
    )

    # temporary fix. Please see issue #238 and the discussion therein
    gbdrift = (
        -1.0
        * 2
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * B_cross_grad_B_dot_grad_alpha
        / (modB * modB * modB)
        * toroidal_flux_sign
    )

    gbdrift0 = (
        -1
        * B_cross_grad_B_dot_grad_psi
        * 2
        * shat[:, None, None]
        / (modB * modB * modB * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )

    # temporary fix. Please see issue #238 and the discussion therein
    # The dpsi/drho = 2 * sqrt_s. Another factor of 2 comes from B_ref * L_ref * L_ref
    cvdrift = gbdrift - 2.0 * B_reference * L_reference * L_reference * sqrt_s[
        :, None, None
    ] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (
        edge_toroidal_flux_over_2pi * modB * modB
    )

    cvdrift0 = gbdrift0

    cvdrift0_1 = np.interp(theta1d, theta_pest[0][0], cvdrift0[0][0])
    gbdrift0_1 = np.interp(theta1d, theta_pest[0][0], cvdrift0[0][0])

    gradpar_theta_pest_1 = np.interp(
        theta1d, theta_pest[0][0], gradpar_theta_pest[0][0]
    )
    bmag_1 = np.interp(theta1d, theta_pest[0][0], bmag[0][0])
    B_p_1 = np.interp(theta1d, theta_pest[0][0], B_p[0][0])

    cvdrift_1 = np.interp(theta1d, theta_pest[0][0], cvdrift[0][0])
    gbdrift_1 = np.interp(theta1d, theta_pest[0][0], gbdrift[0][0])

    gds21_1 = np.interp(theta1d, theta_pest[0][0], gds21[0][0])
    gds22_1 = np.interp(theta1d, theta_pest[0][0], gds22[0][0])
    gds2_1 = np.interp(theta1d, theta_pest[0][0], gds2[0][0])

    R_1 = np.interp(theta1d, theta_pest[0][0], R[0][0])
    Z_1 = np.interp(theta1d, theta_pest[0][0], Z[0][0])

    Rprime_1 = (
        np.interp(theta1d, theta_pest[0][0], d_R_d_s[0][0])
        * 1
        / edge_toroidal_flux_over_2pi
        * 1
        / iota
        * R_1
        * B_p_1
    )
    Zprime_1 = (
        np.interp(theta1d, theta_pest[0][0], d_Z_d_s[0][0])
        * 1
        / edge_toroidal_flux_over_2pi
        * 1
        / iota
        * R_1
        * B_p_1
    )

    # loc_shr    = dermv(gds21_1/gds22_1, theta1d, ch="l", par="e")[0]
    loc_shr = gds21_1 * 0
    # loc_shr   = theta1d
    # Package results into a structure to return:
    results = Struct()
    variables = [
        "ns",
        "nalpha",
        "nl",
        "s",
        "iota",
        "d_iota_d_s",
        "d_pressure_d_s",
        "shat",
        "alpha",
        "theta1d",
        "phi1d",
        "phi",
        "theta_pest",
        "d_lambda_d_s",
        "d_lambda_d_theta_vmec",
        "d_lambda_d_phi",
        "sqrt_g_vmec",
        "sqrt_g_vmec_alt",
        "theta_vmec",
        "modB",
        "d_B_d_s",
        "d_B_d_theta_vmec",
        "d_B_d_phi",
        "B_sup_theta_vmec",
        "B_sup_theta_pest",
        "B_sup_phi",
        "B_sub_s",
        "B_sub_theta_vmec",
        "B_sub_phi",
        "edge_toroidal_flux_over_2pi",
        "sinphi",
        "cosphi",
        "R",
        "R_1",
        "Rprime_1",
        "d_R_d_s",
        "d_R_d_theta_vmec",
        "d_R_d_phi",
        "Z",
        "Z_1",
        "Zprime_1",
        "d_Z_d_s",
        "d_Z_d_theta_vmec",
        "d_Z_d_phi",
        "d_X_d_theta_vmec",
        "d_X_d_phi",
        "d_X_d_s",
        "d_Y_d_theta_vmec",
        "d_Y_d_phi",
        "d_Y_d_s",
        "grad_s_X",
        "grad_s_Y",
        "grad_s_Z",
        "grad_theta_vmec_X",
        "grad_theta_vmec_Y",
        "grad_theta_vmec_Z",
        "grad_phi_X",
        "grad_phi_Y",
        "grad_phi_Z",
        "grad_psi_X",
        "grad_psi_Y",
        "grad_psi_Z",
        "grad_alpha_X",
        "grad_alpha_Y",
        "grad_alpha_Z",
        "grad_B_X",
        "grad_B_Y",
        "grad_B_Z",
        "B_X",
        "B_Y",
        "B_Z",
        "B_cross_grad_s_dot_grad_alpha",
        "B_cross_grad_s_dot_grad_alpha_alternate",
        "B_cross_grad_B_dot_grad_alpha",
        "B_cross_grad_B_dot_grad_alpha_alternate",
        "B_cross_grad_B_dot_grad_psi",
        "B_cross_kappa_dot_grad_psi",
        "B_cross_kappa_dot_grad_alpha",
        "grad_alpha_dot_grad_alpha",
        "grad_alpha_dot_grad_psi",
        "grad_psi_dot_grad_psi",
        "L_reference",
        "B_reference",
        "toroidal_flux_sign",
        "beta_N",
        "tprim",
        "fprim",
        "dens",
        "temp",
        "bmag",
        "gradpar_theta_pest",
        "gradpar_phi",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
        "bmag_1",
        "B_p_1",
        "gradpar_theta_pest_1",
        "gds2_1",
        "gds21_1",
        "gds22_1",
        "gbdrift_1",
        "gbdrift0_1",
        "cvdrift_1",
        "cvdrift0_1",
        "loc_shr",
    ]

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(13, 7))
        nrows = 4
        ncols = 5
        variables = [
            "modB",
            "B_sup_theta_pest",
            "B_sup_phi",
            "B_cross_grad_B_dot_grad_alpha",
            "B_cross_grad_B_dot_grad_psi",
            "B_cross_kappa_dot_grad_alpha",
            "B_cross_kappa_dot_grad_psi",
            "grad_alpha_dot_grad_alpha",
            "grad_alpha_dot_grad_psi",
            "grad_psi_dot_grad_psi",
            "R_1",
            "Z_1",
            "bmag",
            "gradpar_theta_pest",
            "gradpar_phi",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
            "gds2",
            "gds21",
            "gds22",
            "bmag_1",
            "gradpar_theta_pest_1",
            "gds2_1",
            "gds21_1",
            "gds22_1",
            "gbdrift_1",
            "gbdrift0_1",
            "cvdrift_1",
            "cvdrift0_1",
        ]
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(phi[0, 0, :], eval(variable + "[0, 0, :]"))
            plt.xlabel("Standard toroidal angle $\phi$")
            plt.title(variable)

        plt.figtext(0.5, 0.995, f"s={s[0]}, alpha={alpha[0]}", ha="center", va="top")
        plt.tight_layout()
        if show:
            plt.show()

    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#####################################################################################################################
########################--------------------BALLOONING SOLVER FUNCTION--------------------------#####################
#####################################################################################################################


def gamma_ball_full(
    dPdrho, theta_PEST, B, gradpar, cvdrift, gds2, vguess=None, sigma0=0.42
):
    # Inputs  : geometric coefficients(normalized with a_N, and B_N)
    #           on an equispaced theta_PEST grid
    # Outputs : maximum ballooning growth rate gamma
    theta_ball = theta_PEST
    ntheta = len(theta_ball)

    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    g = np.abs(gradpar) * gds2 / (B)
    c = -1 * dPdrho * cvdrift * 1 / (np.abs(gradpar) * B)
    f = gds2 / B**2 * 1 / (np.abs(gradpar) * B)

    len1 = len(g)

    ##Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)

    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)

    # uniform theta_ball on half points with half the size, i.e., only from [0, (2*nperiod-1)*np.pi]
    theta_ball_u_half = (theta_ball_u[:-1] + theta_ball_u[1:]) / 2
    h = np.diff(theta_ball_u_half)[2]
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g)
    g_u1 = g_u[:]
    c_u1 = c_u[:]
    f_u1 = f_u[:]

    len2 = int(len1) - 2
    A = np.zeros((len2, len2))

    A = (
        np.diag(g_u_half[1:-1] / f_u1[2:-1] * 1 / h**2, -1)
        + np.diag(
            -(g_u_half[1:] + g_u_half[:-1]) / f_u1[1:-1] * 1 / h**2
            + c_u1[1:-1] / f_u1[1:-1],
            0,
        )
        + np.diag(g_u_half[1:-1] / f_u1[1:-2] * 1 / h**2, 1)
    )

    # Method without M is approx 3 X faster with Arnoldi iteration
    # Perhaps, we should try dstemr as suggested by Max Ruth. However, I doubt if
    # that will give us a significant speedup
    w, v = eigs(A, 1, sigma=sigma0, v0=vguess, tol=5.0e-7, OPpart="r")
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')

    ### Richardson extrapolation
    X = np.zeros((len2 + 2,))
    dX = np.zeros((len2 + 2,))
    # X[1:-1]     = np.reshape(v[:, idx_max].real, (-1,))/np.max(np.abs(v[:, idx_max].real))
    X[1:-1] = np.reshape(v[:, 0].real, (-1,)) / np.max(np.abs(v[:, 0].real))

    X[0] = 0.0
    X[-1] = 0.0

    dX[0] = (-1.5 * X[0] + 2 * X[1] - 0.5 * X[2]) / h
    dX[1] = (X[2] - X[0]) / (2 * h)

    dX[-2] = (X[-1] - X[-3]) / (2 * h)
    dX[-1] = (0.5 * X[-3] - 2 * X[-2] + 1.5 * 0.0) / (h)

    dX[2:-2] = 2 / (3 * h) * (X[3:-1] - X[1:-3]) - (X[4:] - X[0:-4]) / (12 * h)

    Y0 = -g_u1 * dX**2 + c_u1 * X**2
    Y1 = f_u1 * X**2
    # plt.plot(range(len3+2), X, range(len3+2), dX); plt.show()
    gam = simps(Y0) / simps(Y1)

    # return np.sign(gam)*np.sqrt(abs(gam)), X, dX, g_u1, c_u1, f_u1
    return gam, X, dX, g_u1, c_u1, f_u1


#####################################################################################################################
############################------------------OPTIMIZATION-RELATED FUNCTIONS--------------###########################
#####################################################################################################################


def obj_w_grad(x0, vs, rho_val, theta, vguess00, sigma00=0.42):
    # Read as "objective with grad".
    # Outputs the objective function and its gradient (jacobian)
    # so that we can find the maximum growth rate on each surface.
    # The gradient is calculated using an adjoint method

    alpha_val, theta0_val = x0
    del_alpha = 0.004

    f1 = vmec_fieldlines(
        vs,
        rho_val,
        np.array([alpha_val - 0.5 * del_alpha, alpha_val, alpha_val + 0.5 * del_alpha]),
        theta1d=theta,
    )

    # All the relevant signs have been flipped in vmec_fieldlines
    bmag = f1.bmag[0][1]
    gbdrift = f1.gbdrift[0][1]
    cvdrift = f1.cvdrift[0][1]
    cvdrift0 = f1.cvdrift0[0][1]
    gds2 = f1.gds2[0][1]
    gds21 = f1.gds21[0][1]
    gds22 = f1.gds22[0][1]
    gradpar = f1.gradpar_theta_pest[0][1]
    dPdrho = -1.0 * 0.5 * np.mean((cvdrift - gbdrift) * bmag**2)

    cvdrift_fth = cvdrift + theta0_val * cvdrift0
    gds2_fth = gds2 + 2 * theta0_val * gds21 + theta0_val**2 * gds22

    temp_RE, X_arr, dX_arr, g_arr, c_arr, f_arr = gamma_ball_full(
        dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth, vguess00, sigma00
    )

    Y1 = simps(f_arr * X_arr**2)

    # derivatives of various coefficietns w.r.t theta0
    g_theta0 = np.abs(gradpar) * (2 * gds21 + 2 * theta0_val * gds22) / (bmag)
    c_theta0 = -1 * dPdrho * cvdrift0 * 1 / (np.abs(gradpar) * bmag)
    f_theta0 = (
        (2 * gds21 + 2 * theta0_val * gds22) / bmag**2 * 1 / (np.abs(gradpar) * bmag)
    )

    # derivative of the growth rate w.r.t theta0
    jac_theta0 = (
        simps(c_theta0 * X_arr**2) / Y1
        - simps(g_theta0 * dX_arr**2) / Y1
        - temp_RE * simps(f_theta0 * X_arr**2) / Y1
    )
    # jac_theta0     = simps(c_theta0*X_arr**2)/Y1 - simps(g_theta0*dX_arr**2)/Y1 - np.sign(temp_RE)*temp_RE**2*simps(f_theta0*X_arr**2)/Y1

    bmag_r = f1.bmag[0][2]
    gbdrift_r = f1.gbdrift[0][2]
    cvdrift_r = f1.cvdrift[0][2]
    cvdrift0_r = f1.cvdrift0[0][2]
    gds2_r = f1.gds2[0][2]
    gds21_r = f1.gds21[0][2]
    gds22_r = f1.gds22[0][2]
    gradpar_r = f1.gradpar_theta_pest[0][2]
    dPdrho_r = -1.0 * 0.5 * np.mean((cvdrift_r - gbdrift_r) * bmag_r**2)
    cvdrift_fth_r = cvdrift_r + theta0_val * cvdrift0_r
    gds2_fth_r = gds2_r + 2 * theta0_val * gds21_r + theta0_val**2 * gds22_r

    bmag_l = f1.bmag[0][0]
    gbdrift_l = f1.gbdrift[0][0]
    cvdrift_l = f1.cvdrift[0][0]
    cvdrift0_l = f1.cvdrift0[0][0]
    gds2_l = f1.gds2[0][0]
    gds21_l = f1.gds21[0][0]
    gds22_l = f1.gds22[0][0]
    gradpar_l = f1.gradpar_theta_pest[0][0]
    dPdrho_l = -1.0 * 0.5 * np.mean((cvdrift_l - gbdrift_l) * bmag_l**2)
    cvdrift_fth_l = cvdrift_l + theta0_val * cvdrift0_l
    gds2_fth_l = gds2_l + 2 * theta0_val * gds21_l + theta0_val**2 * gds22_l

    g_r = np.abs(gradpar_r) * gds2_fth_r / (bmag_r)
    c_r = -1 * dPdrho_r * cvdrift_fth_r * 1 / (np.abs(gradpar_r) * bmag_r)
    f_r = gds2_fth_r / bmag_r**2 * 1 / (np.abs(gradpar_r) * bmag_r)

    g_l = np.abs(gradpar_l) * gds2_fth_l / (bmag_l)
    c_l = -1 * dPdrho_l * cvdrift_fth_l * 1 / (np.abs(gradpar_l) * bmag_l)
    f_l = gds2_fth_l / bmag_l**2 * 1 / (np.abs(gradpar_l) * bmag_l)

    # derivatives of various coefficietns w.r.t alpha
    g_alpha = (g_r - g_l) / (del_alpha)
    c_alpha = (c_r - c_l) / (del_alpha)
    f_alpha = (f_r - f_l) / (del_alpha)

    # derivative of the growth rate w.r.t alpha
    jac_alpha = (
        simps(c_alpha * X_arr**2) / Y1
        - simps(g_alpha * dX_arr**2) / Y1
        - temp_RE * simps(f_alpha * X_arr**2) / Y1
    )
    # jac_alpha      = simps(c_alpha*X_arr**2)/Y1 - simps(g_alpha*dX_arr**2)/Y1 - np.sign(temp_RE)*temp_RE**2*simps(f_alpha*X_arr**2)/Y1

    return -1 * temp_RE, np.array([-1 * jac_alpha, -1 * jac_theta0])
    # return -1*temp_RE, np.array([-1*jac_alpha, -1*jac_theta0])/(2*np.abs(temp_RE))


####################################################################################################################
#######################-----------------FINITE-DIFFERENCE GRADIENT ROUTINE-------------------#######################
####################################################################################################################


def derm(arr, ch, par="e"):
    # Finite difference subroutine
    # ch = 'l' means difference along the flux surface
    # ch = 'r' mean difference across the flux surfaces
    # par = 'e' means even parity of the arr. PARITY OF THE INPUT ARRAY
    # par = 'o' means odd parity
    temp = np.shape(arr)
    if (
        len(temp) == 1 and ch == "l"
    ):  # finite diff along the flux surface for a single array
        # pdb.set_trace()
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            diff_arr[0, 1:-1] = np.diff(arr[0, :-1], axis=0) + np.diff(
                arr[0, 1:], axis=0
            )
        else:
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 2 * (arr[0, 1] - arr[0, 0])
            diff_arr[0, -1] = 2 * (arr[0, -1] - arr[0, -2])
            diff_arr[0, 1:-1] = np.diff(arr[0, :-1], axis=0) + np.diff(
                arr[0, 1:], axis=0
            )

    elif len(temp) == 1 and ch == "r":  # across surfaces for a single array
        # pdb.set_trace()
        d1, d2 = np.shape(arr)[0], 1
        diff_arr = np.zeros((d1, d2))
        arr = np.reshape(arr, (d1, d2))
        diff_arr[0, 0] = 2 * (
            arr[1, 0] - arr[0, 0]
        )  # single dimension arrays like psi, F and q don't have parity
        diff_arr[-1, 0] = 2 * (arr[-1, 0] - arr[-2, 0])
        diff_arr[1:-1, 0] = np.diff(arr[:-1, 0], axis=0) + np.diff(arr[1:, 0], axis=0)

    else:
        d1, d2 = np.shape(arr)[0], np.shape(arr)[1]

        diff_arr = np.zeros((d1, d2))
        if ch == "r":  # across surfaces for multi-dim array
            # pdb.set_trace()
            diff_arr[0, :] = 2 * (arr[1, :] - arr[0, :])
            diff_arr[-1, :] = 2 * (arr[-1, :] - arr[-2, :])
            diff_arr[1:-1, :] = np.diff(arr[:-1, :], axis=0) + np.diff(
                arr[1:, :], axis=0
            )

        else:  # along a surface for a multi-dim array
            # pdb.set_trace()
            if par == "e":
                # pdb.set_trace()
                diff_arr[:, 0] = np.zeros((d1,))
                diff_arr[:, -1] = np.zeros((d1,))
                diff_arr[:, 1:-1] = np.diff(arr[:, :-1], axis=1) + np.diff(
                    arr[:, 1:], axis=1
                )
            else:
                diff_arr[:, 0] = 2 * (arr[:, 1] - arr[:, 0])
                diff_arr[:, -1] = 2 * (arr[:, -1] - arr[:, -2])
                diff_arr[:, 1:-1] = np.diff(arr[:, :-1], axis=1) + np.diff(
                    arr[:, 1:], axis=1
                )

    arr = np.reshape(diff_arr, temp)
    return diff_arr


def dermv(arr, brr, ch, par="e"):
    # Finite difference subroutine
    # brr is the independent variable arr. Needed for weighted finite-difference
    # ch = 'l' means difference along the flux surface
    # ch = 'r' mean difference across the flux surfaces
    # par = 'e' means even parity of the arr. PARITY OF THE INPUT ARRAY
    # par = 'o' means odd parity
    # pdb.set_trace()
    temp = np.shape(arr)
    if (
        len(temp) == 1 and ch == "l"
    ):  # finite diff along the flux surface for a single array
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            # diff_arr[0, 1:-1] = np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            # pdb.set_trace()
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))

            h1 = np.abs(brr[0, 1]) - np.abs(brr[0, 0])
            h0 = np.abs(brr[0, -1]) - np.abs(brr[0, -2])
            diff_arr[0, 0] = (4 * arr[0, 1] - 3 * arr[0, 0] - arr[0, 2]) / (
                2 * (brr[0, 1] - brr[0, 0])
            )

            # diff_arr[0, -1] = (-4*arr[0,-1]+3*arr[0, -2]+arr[0, -3])/(2*(brr[0, -1]-brr[0, -2]))
            diff_arr[0, -1] = (-4 * arr[0, -2] + 3 * arr[0, -1] + arr[0, -3]) / (
                2 * (brr[0, -1] - brr[0, -2])
            )
            # diff_arr[0, -1] = 2*(arr[0, -1] - arr[0, -2])/(2*(brr[0, -1] - brr[0, -2]))
            # diff_arr[0, 1:-1] = np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)

    elif len(temp) == 1 and ch == "r":  # across surfaces for a single array
        pdb.set_trace()
        d1, d2 = np.shape(arr)[0], 1
        diff_arr = np.zeros((d1, d2))
        arr = np.reshape(arr, (d1, d2))
        diff_arr[0, 0] = (
            2 * (arr[1, 0] - arr[0, 0]) / (2 * (brr[1, 0] - brr[0, 0]))
        )  # single dimension arrays like psi, F and q don't have parity
        diff_arr[-1, 0] = (
            2 * (arr[-1, 0] - arr[-2, 0]) / (2 * (brr[-1, 0] - brr[-2, 0]))
        )
        # diff_arr[1:-1, 0] = np.diff(arr[:-1,0], axis=0) + np.diff(arr[1:,0], axis=0)
        for i in range(1, d1 - 1):
            h1 = brr[i + 1, 0] - brr[i, 0]
            h0 = brr[i, 0] - brr[i - 1, 0]
            diff_arr[i, 0] = (
                arr[i + 1, 0] / h1**2
                - arr[i, 0] * (1 / h0**2 - 1 / h1**2)
                - arr[i - 1, 0] / h0**2
            ) / (1 / h1 + 1 / h0)

    else:
        d1, d2 = np.shape(arr)[0], np.shape(arr)[1]

        diff_arr = np.zeros((d1, d2))
        if ch == "r":  # across surfaces for multi-dim array
            # pdb.set_trace()
            diff_arr[0, :] = 2 * (arr[1, :] - arr[0, :]) / (2 * (brr[1, :] - brr[0, :]))
            diff_arr[-1, :] = (
                2 * (arr[-1, :] - arr[-2, :]) / (2 * (brr[-1, :] - brr[-2, :]))
            )
            # diff_arr[1:-1, :] = (np.diff(arr[:-1,:], axis=0) + np.diff(arr[1:,:], axis=0))
            for i in range(1, d1 - 1):
                h1 = brr[i + 1, :] - brr[i, :]
                h0 = brr[i, :] - brr[i - 1, :]
                diff_arr[i, :] = (
                    arr[i + 1, :] / h1**2
                    + arr[i, :] * (1 / h0**2 - 1 / h1**2)
                    - arr[i - 1, :] / h0**2
                ) / (1 / h1 + 1 / h0)

        else:  # along a surface for a multi-dim array
            # pdb.set_trace()
            if par == "e":
                # pdb.set_trace()
                diff_arr[:, 0] = np.zeros((d1,))
                diff_arr[:, -1] = np.zeros((d1,))
                # diff_arr[:, 1:-1] = (np.diff(arr[:,:-1], axis=1) + np.diff(arr[:,1:], axis=1))
                for i in range(1, d2 - 1):
                    h1 = brr[:, i + 1] - brr[:, i]
                    h0 = brr[:, i] - brr[:, i - 1]
                    diff_arr[:, i] = (
                        arr[:, i + 1] / h1**2
                        + arr[:, i] * (1 / h0**2 - 1 / h1**2)
                        - arr[:, i - 1] / h0**2
                    ) / (1 / h1 + 1 / h0)
                    # pdb.set_trace()
            else:
                # pdb.set_trace()
                diff_arr[:, 0] = (
                    2 * (arr[:, 1] - arr[:, 0]) / (2 * (brr[:, 1] - brr[:, 0]))
                )
                diff_arr[:, -1] = (
                    2 * (arr[:, -1] - arr[:, -2]) / (2 * (brr[:, -1] - brr[:, -2]))
                )
                # diff_arr[:, 1:-1] = (np.diff(arr[:,:-1], axis=1) + np.diff(arr[:,1:], axis=1))
                for i in range(1, d2 - 1):
                    h1 = brr[:, i + 1] - brr[:, i]
                    h0 = brr[:, i] - brr[:, i - 1]
                    diff_arr[:, i] = (
                        arr[:, i + 1] / h1**2
                        + arr[:, i] * (1 / h0**2 - 1 / h1**2)
                        - arr[:, i - 1] / h0**2
                    ) / (1 / h1 + 1 / h0)

    arr = np.reshape(diff_arr, temp)

    return diff_arr
