#!/usr/bin/env python3
"""
Numerics for the lecture notes
  "A Qubit in a Bosonic Bath: Keldysh, Nakajima-Zwanzig and Mori"

Model (hbar = 1):
  H_S = (w0/2) sigma_z,   S = cos(theta) sigma_z + sin(theta) sigma_x,
  H_I = S (x) B,          bath Gaussian with correlation function
      C(t) = <B(t)B(0)> = g^2 exp(-i Omega t - gamma |t| / 2)      (T = 0, Lorentzian J)
  so   S(t) = g^2 e^{-gamma|t|/2} cos(Omega t),   chi(t) = 2 g^2 theta(t) e^{-gamma t/2} sin(Omega t).

Exact reference: pseudomode mapping (Garraway 1997; Tamascelli et al. 2018).  The reduced
qubit dynamics is reproduced exactly by qubit + one damped oscillator,
  d rho/dt = -i[H_S + Omega a^dag a + g S (a + a^dag), rho] + gamma D[a] rho,
started in the oscillator vacuum.

Conventions follow the notes: column-stacking vectorisation,  A^+ X = A X -> (1 (x) A),
A^- X = X A -> (A^T (x) 1);  A^cl = (A^+ + A^-)/2,  A^qu = A^+ - A^-.
Liouvillian  L X = -i[H, X].

Produces (in the directory of this file):
  fig_bath_kernels.pdf      S(t), chi(t), J(w), S(w) for the Lorentzian bath
  fig_dynamics.pdf          <sigma_z>(t), |rho_01|(t): exact vs NZ2, TCL2, Markov (Redfield)
  fig_kernels.pdf           exact memory kernel / TCL generator vs second order
  fig_filter.pdf            filter functions and dephasing Gamma(t) for Ramsey / echo / CPMG
and prints a table of second-order rates used in the text.
"""
import numpy as np
from scipy.linalg import expm, solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
w0 = 1.0          # qubit splitting
Omega = 1.0       # bath (pseudomode) frequency, resonant with the qubit
gamma = 0.5       # bath linewidth
theta = np.pi / 4 # mixing angle of the coupling operator
g_weak, g_strong = 0.10, 0.30
T_MAX, DT = 40.0, 0.02
N_OSC = 14        # pseudomode truncation (convergence checked against N_OSC=20)

# ----------------------------------------------------------------------------
# Basic matrices
# ----------------------------------------------------------------------------
I2 = np.eye(2)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
H_S = 0.5 * w0 * sz
S_op = np.cos(theta) * sz + np.sin(theta) * sx


def left(A):   # A^+  : X -> A X
    n = A.shape[0]
    return np.kron(np.eye(n), A)


def right(A):  # A^-  : X -> X A
    n = A.shape[0]
    return np.kron(A.T, np.eye(n))


def cl(A):
    return 0.5 * (left(A) + right(A))


def qu(A):
    return left(A) - right(A)


def liouvillian(H):
    return -1j * (left(H) - right(H))


def dissipator(c, rate):
    n = c.shape[0]
    cdc = c.conj().T @ c
    return rate * (np.kron(c.conj(), c) - 0.5 * left(cdc) - 0.5 * right(cdc))


def vec(X):
    return X.reshape(-1, order="F")


def unvec(v, n):
    return v.reshape((n, n), order="F")


L_S = liouvillian(H_S)
Sq, Sc = qu(S_op), cl(S_op)

# ----------------------------------------------------------------------------
# Bath kernels (analytic, T = 0 Lorentzian)
# ----------------------------------------------------------------------------
def S_noise(t, g):
    return g**2 * np.exp(-gamma * np.abs(t) / 2) * np.cos(Omega * t)


def chi_resp(t, g):
    return 2 * g**2 * (t > 0) * np.exp(-gamma * t / 2) * np.sin(Omega * t)


def J_spec(w, g):  # Lorentzian spectral density, chi''(w) = J(w)   (defined for all w)
    return g**2 * (gamma / 2) / ((w - Omega) ** 2 + gamma**2 / 4)


def S_omega(w, g):  # symmetrised noise spectrum at T = 0: S(w) = J(w) + J(-w)
    return J_spec(w, g) + J_spec(-w, g)


# ----------------------------------------------------------------------------
# Second-order (Born) kernel and derived generators,  K2(tau) = -S^q e^{L_S tau}[S(tau) S^q - i chi(tau) S^cl]
# ----------------------------------------------------------------------------
def K2(tau, g):
    return -Sq @ expm(L_S * tau) @ (S_noise(tau, g) * Sq - 1j * chi_resp(tau, g) * Sc)


def tcl2_generator(ts, g):
    """G2(t) = int_0^t K2(tau) e^{-L_S tau} dtau on the grid ts (cumulative trapezoid)."""
    integrand = np.array([K2(tau, g) @ expm(-L_S * tau) for tau in ts])
    G = np.zeros_like(integrand)
    for n in range(1, len(ts)):
        G[n] = G[n - 1] + 0.5 * (integrand[n] + integrand[n - 1]) * (ts[n] - ts[n - 1])
    return G


def propagate_tcl(ts, Gs, rho0):
    rho = vec(rho0).astype(complex)
    out = [rho.copy()]
    for n in range(1, len(ts)):
        dt = ts[n] - ts[n - 1]
        Gmid = L_S + 0.5 * (Gs[n] + Gs[n - 1])
        rho = expm(Gmid * dt) @ rho
        out.append(rho.copy())
    return np.array(out)


def propagate_nz2(ts, g, rho0):
    """Volterra equation  rho'(t) = L_S rho + int_0^t K2(t-s) rho(s) ds, trapezoid + implicit Euler-type step."""
    dt = ts[1] - ts[0]
    Ks = np.array([K2(tau, g) for tau in ts])
    rho = [vec(rho0).astype(complex)]
    I4 = np.eye(4)
    I_prev = np.zeros(4, complex)  # memory integral at t_{n-1}

    def partial_integral(n):
        """trapezoid approximation of int_0^{t_n} K(t_n - s) rho(s) ds WITHOUT the (dt/2) K(0) rho_n term."""
        R = np.array(rho)
        val = 0.5 * Ks[n] @ R[0]
        if n > 1:
            val += np.einsum("mij,mj->i", Ks[n - 1:0:-1], R[1:n])
        return dt * val

    for n in range(1, len(ts)):
        Ipart = partial_integral(n)
        # trapezoid rule in time for rho' = L_S rho + I(t), with rho_n treated implicitly
        A = I4 - 0.5 * dt * L_S - 0.25 * dt * dt * Ks[0]
        b = (I4 + 0.5 * dt * L_S) @ rho[n - 1] + 0.5 * dt * (I_prev + Ipart)
        rho_n = solve(A, b)
        rho.append(rho_n)
        I_prev = Ipart + 0.5 * dt * Ks[0] @ rho_n
    return np.array(rho)


# ----------------------------------------------------------------------------
# Exact reference: qubit + damped pseudomode
# ----------------------------------------------------------------------------
def pseudomode_setup(g, N=N_OSC):
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    IN = np.eye(N)
    H = np.kron(H_S, IN) + Omega * np.kron(I2, a.conj().T @ a) + g * np.kron(S_op, a + a.conj().T)
    Ltot = liouvillian(H) + dissipator(np.kron(I2, a), gamma)
    # projector P X = Tr_osc(X) (x) |0><0|  as a matrix on the (2N)^2 space
    D = 2 * N
    P = np.zeros((D * D, D * D), complex)
    vac = np.zeros((N, N)); vac[0, 0] = 1
    for i in range(2):
        for j in range(2):
            Eij = np.zeros((2, 2)); Eij[i, j] = 1
            # column of P acting on basis element |i,n><j,m|: Tr_osc gives delta_nm |i><j| (x) |0><0|
            for n in range(N):
                Enn = np.zeros((N, N)); Enn[n, n] = 1
                src = vec(np.kron(Eij, Enn))
                dst = vec(np.kron(Eij, vac))
                P[:, np.argmax(np.abs(src))] += dst  # src is a unit vector
    return a, H, Ltot, P


def embed(X_S, N=N_OSC):
    vac = np.zeros((N, N)); vac[0, 0] = 1
    return vec(np.kron(X_S, vac))


def partial_trace_osc(v, N=N_OSC):
    X = unvec(v, 2 * N).reshape(2, N, 2, N)
    return np.einsum("injn->ij", X)


def exact_propagator(ts, g, N=N_OSC):
    """Return Pi(t) (4x4), dPi/dt (4x4), and the exact NZ kernel K(t) (4x4) on the grid."""
    a, H, Ltot, P = pseudomode_setup(g, N)
    dt = ts[1] - ts[0]
    U = expm(Ltot * dt)
    basis = []
    for j in range(2):          # column-stacking order: vec index = i + 2 j
        for i in range(2):
            E = np.zeros((2, 2), complex); E[i, j] = 1
            basis.append(embed(E, N))
    cols = np.array(basis).T  # (2N)^2 x 4, columns ordered as vec(|i><j|)
    Pi, dPi = [], []
    cur = cols.copy()
    for n in range(len(ts)):
        Pi.append(np.array([vec(partial_trace_osc(cur[:, k], N)) for k in range(4)]).T)
        dcur = Ltot @ cur
        dPi.append(np.array([vec(partial_trace_osc(dcur[:, k], N)) for k in range(4)]).T)
        cur = U @ cur
    # exact kernel  K(tau) = P L_I e^{Q L tau} Q L_I P  (restricted to the system block)
    Q = np.eye(P.shape[0]) - P
    L_I = liouvillian(g * np.kron(S_op, a + a.conj().T))
    UQ = expm(Q @ Ltot @ Q * dt)
    Kex = []
    cur = Q @ L_I @ cols
    for n in range(len(ts)):
        tmp = L_I @ cur
        Kex.append(np.array([vec(partial_trace_osc(tmp[:, k], N)) for k in range(4)]).T)
        cur = UQ @ cur
    return np.array(Pi), np.array(dPi), np.array(Kex)


# ----------------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------------
def expect_sz(rhos):
    return np.real([np.trace(sz @ unvec(r, 2)) for r in rhos])


def coh01(rhos):
    return np.array([unvec(r, 2)[0, 1] for r in rhos])


def bloch_component(M, i, j):
    """Matrix element <<sigma_i| M |sigma_j>> / 2 in the orthonormal Pauli basis (I,x,y,z)/sqrt2."""
    paulis = [I2, sx, sy, sz]
    vi, vj = vec(paulis[i]) / np.sqrt(2), vec(paulis[j]) / np.sqrt(2)
    return vi.conj() @ M @ vj


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_bath_kernels():
    g = 1.0
    t = np.linspace(-15, 15, 1201)
    w = np.linspace(-3, 3, 1201)
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    ax[0].plot(t, S_noise(t, g), label=r"$S(t)/g^2$")
    ax[0].plot(t, chi_resp(t, g), label=r"$\chi(t)/g^2$")
    ax[0].set_xlabel(r"$t\,\omega_0$"); ax[0].legend(frameon=False); ax[0].axhline(0, lw=0.5, color="k")
    ax[0].set_title("time domain", fontsize=10)
    ax[1].plot(w, J_spec(w, g), label=r"$\chi''(\omega)/g^2$")
    ax[1].plot(w, S_omega(w, g), label=r"$S(\omega)/g^2$ ($T=0$)")
    ax[1].set_xlabel(r"$\omega/\omega_0$"); ax[1].legend(frameon=False); ax[1].axhline(0, lw=0.5, color="k")
    ax[1].set_title(r"frequency domain, $\Omega=\omega_0$, $\gamma=0.5\,\omega_0$", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_bath_kernels.pdf")); plt.close(fig)


def run_dynamics(g, ts, rho0):
    Pi, dPi, Kex = exact_propagator(ts, g)
    ex = np.array([Pi[n] @ vec(rho0) for n in range(len(ts))])
    G2 = tcl2_generator(ts, g)
    tcl = propagate_tcl(ts, G2, rho0)
    Gmark = np.array([G2[-1]] * len(ts))
    mark = propagate_tcl(ts, Gmark, rho0)
    nz = propagate_nz2(ts, g, rho0)
    return dict(Pi=Pi, dPi=dPi, Kex=Kex, exact=ex, tcl2=tcl, markov=mark, nz2=nz, G2=G2)


def fig_dynamics(results, ts):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)
    for col, (g, R) in enumerate(results.items()):
        for key, lab, ls in [("exact", "exact (pseudomode)", "-"), ("tcl2", "TCL2", "--"),
                             ("nz2", "NZ2", "-."), ("markov", "Markov (Redfield)", ":")]:
            ax[0, col].plot(ts, expect_sz(R[key]), ls, label=lab)
            ax[1, col].plot(ts, np.abs(coh01(R[key])), ls, label=lab)
        ax[0, col].set_title(rf"$g={g}\,\omega_0$, $\theta=\pi/4$", fontsize=10)
        ax[1, col].set_xlabel(r"$t\,\omega_0$")
    ax[0, 0].set_ylabel(r"$\langle\sigma_z\rangle$"); ax[1, 0].set_ylabel(r"$|\rho_{01}|$")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_dynamics.pdf")); plt.close(fig)


def fig_kernels(results, ts):
    """Compare exact memory kernel and exact TCL generator with second order, for g_strong."""
    g = g_strong
    R = results[g]
    K2s = np.array([K2(tau, g) for tau in ts])
    Gex = np.array([(R["dPi"][n] - L_S @ R["Pi"][n]) @ np.linalg.inv(R["Pi"][n]) for n in range(len(ts))])
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    # kernel: population-population element <<z|K|z>> and the response-driven element <<z|K|1>>
    lab = [(3, 3, r"$K_{zz}$"), (3, 0, r"$K_{z\mathbb{1}}$")]
    for (i, j, name), c in zip(lab, ["C0", "C1"]):
        ax[0].plot(ts, np.real([bloch_component(R["Kex"][n], i, j) for n in range(len(ts))]), c, label=name + " exact")
        ax[0].plot(ts, np.real([bloch_component(K2s[n], i, j) for n in range(len(ts))]), c + "--", label=name + r" $K^{(2)}$")
    ax[0].set_xlim(0, 15); ax[0].set_xlabel(r"$\tau\,\omega_0$"); ax[0].set_title("memory kernel", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    for (i, j, name), c in zip(lab, ["C0", "C1"]):
        gname = name.replace("K", r"\mathcal{G}")
        ax[1].plot(ts, np.real([bloch_component(Gex[n], i, j) for n in range(len(ts))]), c, label=gname + " exact")
        ax[1].plot(ts, np.real([bloch_component(R["G2"][n], i, j) for n in range(len(ts))]), c + "--", label=gname + r" TCL2")
    ax[1].set_xlim(0, 12); ax[1].set_xlabel(r"$t\,\omega_0$"); ax[1].set_title("TCL generator", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_kernels.pdf")); plt.close(fig)


def filter_function(w, t, seq):
    """|int_0^t y(s) e^{i w s} ds|^2 * w^2/2 for sign sequences; returns F(w,t)."""
    if seq == "ramsey":
        flips = []
    elif seq == "hahn":
        flips = [t / 2]
    elif seq == "cpmg4":
        flips = [(k + 0.5) * t / 4 for k in range(4)]
    edges = [0.0] + flips + [t]
    y = 1.0
    acc = np.zeros_like(w, complex)
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        acc += y * (np.exp(1j * w * b) - np.exp(1j * w * a)) / (1j * w)
        y = -y
    return 0.5 * w**2 * np.abs(acc) ** 2


def fig_filter():
    g = 0.2
    w = np.linspace(1e-3, 4, 4000)
    t = 20.0
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for seq, lab in [("ramsey", "Ramsey"), ("hahn", "Hahn echo"), ("cpmg4", "CPMG-4")]:
        ax[0].plot(w, filter_function(w, t, seq) / w**2, label=lab)
    ax[0].plot(w, S_omega(w, g) / S_omega(w, g).max() * 10, "k:", label=r"$S(\omega)$ (scaled)")
    ax[0].set_xlabel(r"$\omega/\omega_0$"); ax[0].set_ylabel(r"$F(\omega,t)/\omega^2$,  $t=20/\omega_0$")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-2, 400)
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    tt = np.linspace(0.05, 40, 400)
    ww = np.linspace(-8, 8, 8000) + 1e-3
    for seq, lab in [("ramsey", "Ramsey"), ("hahn", "Hahn echo"), ("cpmg4", "CPMG-4")]:
        Gam = []
        for t_ in tt:
            F = filter_function(ww, t_, seq)
            Gam.append(np.trapezoid(S_omega(ww, g) / ww**2 * F, ww) / (2 * np.pi))
        ax[1].plot(tt, Gam, label=lab)
    ax[1].set_xlabel(r"$t\,\omega_0$"); ax[1].set_ylabel(r"$\Gamma(t)$,  $g=0.2\,\omega_0$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_filter.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------
# Check of Theorem 5.4 at fourth order:  K(tau) - K2(tau) = K4(tau) + O(g^6),
# with K4 = P L_I e^{L0} Q L_I e^{L0} Q L_I e^{L0} Q L_I P evaluated in the pseudomode space
# (the single surviving P-insertion in the middle gap is what Q supplies; see Example 5.1).
# ----------------------------------------------------------------------------
def check_fourth_order(N=8, n=50):
    def setup(g):
        a, H, Ltot, P = pseudomode_setup(g, N)
        LI = liouvillian(g * np.kron(S_op, a + a.conj().T)); L0 = Ltot - LI; Q = np.eye(P.shape[0]) - P
        cols = []
        for j in range(2):
            for i in range(2):
                E = np.zeros((2, 2), complex); E[i, j] = 1; cols.append(embed(E, N))
        return L0, LI, Q, np.array(cols).T

    def toS(out):
        return np.array([vec(partial_trace_osc(out[:, k], N)) for k in range(4)]).T

    def K_exact(tau, g):
        L0, LI, Q, cols = setup(g)
        return toS(LI @ expm(Q @ (L0 + LI) @ Q * tau) @ (Q @ LI @ cols))

    def K4(tau, g):
        L0, LI, Q, cols = setup(g)
        s = np.linspace(0, tau, n + 1); ds = s[1] - s[0]
        E = [expm(L0 * x) for x in s]
        acc = np.zeros((4, 4), complex)
        for i in range(n + 1):
            for j in range(i):
                w = (0.5 if i in (0, n) else 1) * (0.5 if j in (0, n) else 1)
                v = E[n - i] @ (Q @ LI @ (E[i - j] @ (Q @ LI @ (E[j] @ (Q @ LI @ cols)))))
                acc += w * toS(LI @ v)
        return acc * ds * ds

    print("Theorem 5.4 at fourth order: exact kernel vs K2 + K4 (trapezoid, n=%d)" % n)
    for g in (0.05, 0.1, 0.2):
        for tau in (1.0, 2.5):
            Kx = K_exact(tau, g); K2v = K2(tau, g); K4v = K4(tau, g)
            print("  g=%.2f tau=%.1f  |K-K2|=%.2e  |K4|=%.2e  |K-K2-K4|=%.2e" % (
                g, tau, np.abs(Kx - K2v).max(), np.abs(K4v).max(), np.abs(Kx - K2v - K4v).max()))


def check_rwa_kernel(N=4):
    """Lecture 5, Sec. 'Can the kernel be computed exactly?': the damped Jaynes-Cummings
    model (theta = pi/2 in the rotating-wave approximation, T = 0, Lorentzian bath, resonance).
    Compares the exact kernel P L_I e^{Q L tau} Q L_I P of the RWA pseudomode model with the
    closed forms  K_eg(tau) = -g^2 e^{-gamma tau/2}  (coherence) and
    K_ee(tau) = -2 g^2 e^{-3 gamma tau/4} [cos(nu tau) + gamma/(4 nu) sin(nu tau)],
    nu = sqrt(32 g^2 - gamma^2)/4  (population), both in the frame rotating at omega_0,
    and the fourth-order term of the geometric series with K - K2 at small g."""
    sp_ = np.array([[0, 1], [0, 0]], complex); sm_ = sp_.T.conj()
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex); IN = np.eye(N)
    D = 2 * N
    H0 = np.kron(H_S, IN) + Omega * np.kron(I2, a.conj().T @ a)
    vac = np.zeros((N, N)); vac[0, 0] = 1

    def P_matrix():
        P = np.zeros((D * D, D * D), complex)
        for k in range(D * D):
            E = np.zeros(D * D, complex); E[k] = 1
            RS = partial_trace_osc(E, N)
            P[:, k] = vec(np.kron(RS, vac))
        return P

    P = P_matrix(); Q = np.eye(D * D) - P
    Ee = np.diag([1, 0]).astype(complex)         # |e><e|, e = upper level of sigma_z
    Eeg = np.array([[0, 1], [0, 0]], complex)    # |e><g|

    def kernels(g, taus):
        HI = g * (np.kron(sp_, a) + np.kron(sm_, a.conj().T))
        LI = liouvillian(HI); L0 = liouvillian(H0) + dissipator(np.kron(I2, a), gamma); L = L0 + LI
        out = []
        for tau in taus:
            Kt = P @ LI @ expm(Q @ L @ Q * tau) @ (Q @ LI @ P)
            # rotate back to the frame rotating at omega_0 (populations are unaffected)
            R = expm(-L_S * tau)
            kee = partial_trace_osc(Kt @ vec(np.kron(Ee, vac)), N)[0, 0]
            keg = (R @ vec(partial_trace_osc(Kt @ vec(np.kron(Eeg, vac)), N)))[2]  # vec index of |e><g| is 2
            out.append((kee, keg))
        return np.array(out)

    taus = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    print("Damped Jaynes-Cummings model (RWA, T=0, Lorentzian, resonance): exact NZ kernel vs closed forms")
    for g in (0.1, 0.3):
        nu = np.sqrt(32 * g ** 2 - gamma ** 2 + 0j) / 4
        kee_f = -2 * g ** 2 * np.exp(-3 * gamma * taus / 4) * (np.cos(nu * taus) + gamma / (4 * nu) * np.sin(nu * taus))
        keg_f = -g ** 2 * np.exp(-gamma * taus / 2)
        num = kernels(g, taus)
        print("  g=%.2f  max|K_ee - formula| = %.1e   max|K_eg - formula| = %.1e" % (
            g, np.abs(num[:, 0] - kee_f).max(), np.abs(num[:, 1] - keg_f).max()))
    # fourth-order term of the geometric series vs  K_ee - K_ee^(2)
    print("  fourth-order check (population):  K - K2  vs  K4 = (8g^4/gamma) tau e^{-gamma tau/2} - (16g^4/gamma^2)(e^{-gamma tau/2} - e^{-gamma tau})")
    for g in (0.05, 0.1, 0.2):
        num = kernels(g, taus)
        k2 = -2 * g ** 2 * np.exp(-gamma * taus / 2)
        k4 = 8 * g ** 4 / gamma * taus * np.exp(-gamma * taus / 2) - 16 * g ** 4 / gamma ** 2 * (np.exp(-gamma * taus / 2) - np.exp(-gamma * taus))
        print("    g=%.2f  max|K-K2| = %.2e  max|K-K2-K4| = %.2e" % (g, np.abs(num[:, 0] - k2).max(), np.abs(num[:, 0] - k2 - k4).max()))



def check_influence_kernel():
    """Lecture 7, Sec. 'From the influence functional to the memory kernel' (Ivander, Lindoy and Lee 2024).
    Discretized Feynman-Vernon path sum in the S eigenbasis (symmetric Trotter split, step dt):
      U_N = G [prod I_0] [prod F] [prod_{k>k'} I_{k-k'}] G,   I_n(x',x) = exp[-(s'^+ - s'^-)(eta_n s^+ - eta_n^* s^-)],
    with eta_n the double cell integral of C(t) = g^2 e^{-(i Omega + gamma/2) t}.  Checks:
      (i)   U_N^{Trotter} vs the exact pseudomode map (O(dt^3) per step);
      (ii)  the discrete NZ kernels K_n (transfer tensors) from exact propagators vs the continuous K(n dt);
      (iii) K_n from the transfer-tensor recursion == K_n from the chain (Dyck-path) rule, to machine precision;
      (iv)  the inverse map K -> I recovers the influence functions;
      (v)   Hamiltonian learning: (Omega, gamma, g^2) from eta_{1+m}/eta_1, with Trotter-consistent and with exact data."""
    import itertools
    R = expm(-1j * theta / 2 * sy)                     # S = R sz R^dag
    HSp = R.conj().T @ H_S @ R                         # H_S in the S eigenbasis
    LSp = liouvillian(HSp)
    Rs = np.kron(R.T, R.conj().T); Rsi = np.linalg.inv(Rs)   # vec(R^dag X R) = Rs vec X
    sval = np.array([1, -1])
    spl = lambda x: sval[x % 2]      # s^+ (ket index)
    smi = lambda x: sval[x // 2]     # s^- (bra index)
    a_b = 1j * Omega + gamma / 2

    def eta(g, dt, n):
        if n == 0:
            return g ** 2 * (dt - (1 - np.exp(-a_b * dt)) / a_b) / a_b
        return 4 * g ** 2 * np.sinh(a_b * dt / 2) ** 2 / a_b ** 2 * np.exp(-a_b * n * dt)

    def I_mat(g, dt, n):
        e = eta(g, dt, n)
        if n == 0:
            return np.array([np.exp(-(spl(x) - smi(x)) * (e * spl(x) - np.conj(e) * smi(x))) for x in range(4)])
        return np.array([[np.exp(-(spl(xl) - smi(xl)) * (e * spl(xe) - np.conj(e) * smi(xe))) for xe in range(4)] for xl in range(4)])

    def U_discrete(g, dt, N):
        G = expm(LSp * dt / 2); F = G @ G
        I0 = I_mat(g, dt, 0); Ik = {n: I_mat(g, dt, n) for n in range(1, N)}
        U = np.zeros((4, 4), complex)
        for path in itertools.product(range(4), repeat=N):
            w = 1.0 + 0j
            for n in range(N): w *= I0[path[n]]
            for n in range(N - 1): w *= F[path[n + 1], path[n]]
            for k in range(N):
                for kp in range(k + 1, N): w *= Ik[kp - k][path[kp], path[k]]
            U += w * np.outer(G[:, path[-1]], G[path[0], :])
        return U

    def chains(n):
        out = []
        def rec(ch):
            a, b = ch[-1]
            if b == n: out.append(list(ch)); return
            for a2 in range(a + 1, b + 1):
                for b2 in range(b + 1, n + 1): rec(ch + [(a2, b2)])
        for b in range(1, n + 1): rec([(0, b)])
        return out

    def ttm(U, L, dt):
        K = []
        for N in range(1, len(U)):
            rest = U[N] - L @ U[N - 1]
            for mm in range(2, N + 1): rest -= dt ** 2 * K[N - mm] @ U[mm - 1]
            K.append(rest / dt ** 2)
        return K

    def chain_core(n, I0, Ik, F, skip_crest_arc=False):
        """Sum over chains of the dressed influence-function products; returns (crest dressing, rest) if skip_crest_arc."""
        arcs = [(a, b) for a in range(n + 1) for b in range(a + 1, n + 1)]
        crest_d = np.zeros((4, 4), complex); rest = np.zeros((4, 4), complex)
        for ch in chains(n):
            crest = (ch == [(0, n)])
            for path in itertools.product(range(4), repeat=n + 1):
                w = 1.0 + 0j
                for p in range(n + 1): w *= I0[path[p]]
                for p in range(n): w *= F[path[p + 1], path[p]]
                for (a, b) in arcs:
                    if skip_crest_arc and crest and (a, b) == (0, n): continue
                    if (a, b) in ch: w *= Ik[b - a][path[b], path[a]] - 1
                    elif any(c[0] <= a and b <= c[1] for c in ch): w *= Ik[b - a][path[b], path[a]]
                (crest_d if crest else rest)[path[-1], path[0]] += w
        return crest_d, rest

    def K_chain(g, dt, n):
        G = expm(LSp * dt / 2); F = G @ G
        I0 = I_mat(g, dt, 0); Ik = {k: I_mat(g, dt, k) for k in range(1, n + 1)}
        if n == 0: return G @ np.diag(I0 - 1) @ G / dt ** 2
        c, r = chain_core(n, I0, Ik, F)
        return G @ (c + r) @ G / dt ** 2

    def invert(dt, K):
        G = expm(LSp * dt / 2); F = G @ G; Gi = np.linalg.inv(G)
        I0 = np.diag(Gi @ (dt ** 2 * K[0] + F) @ Gi)
        Irec = {0: I0}
        for n in range(1, len(K)):
            core = Gi @ K[n] @ Gi * dt ** 2
            dress, rest = chain_core(n, I0, Irec, F, skip_crest_arc=True)
            Irec[n] = 1 + (core - rest) / dress
        return Irec

    def eta_from_I(I):   # x = i + 2j: (+,-) -> 2, (+,+) -> 0
        return (-np.log(I[2, 2]) / 4).real + 1j * (np.log(I[2, 0]) / (-4j)).real

    g = g_strong
    print("Influence functional -> memory kernel (Ivander, Lindoy, Lee 2024), S eigenbasis, g=%.2f, theta=pi/4" % g)
    print("  (i) Trotterized path sum vs exact pseudomode map:")
    for dt in (0.2, 0.1, 0.05):
        ts = np.arange(0, 4 * dt + 1e-9, dt); Pi, _, _ = exact_propagator(ts, g, N=12)
        errs = [np.abs(U_discrete(g, dt, N) - Rs @ Pi[N] @ Rsi).max() for N in (1, 2, 4)]
        print("     dt=%.2f  N=1,2,4: %.1e %.1e %.1e" % (dt, *errs))
    print("  (ii) discrete kernel K_n from EXACT propagators vs continuous K(n dt), tau=1,2 (error O(dt)):")
    for dt in (0.2, 0.1, 0.05):
        ts = np.arange(0, 3.0001, dt); Pi, _, Kex = exact_propagator(ts, g, N=12)
        Kd = ttm([Pi[n] for n in range(len(ts))], expm(L_S * dt), dt)
        print("     dt=%.2f  " % dt + "  ".join("tau=%.0f: %.2e" % (tau, np.abs(Kd[int(round(tau / dt))] - Kex[int(round(tau / dt))]).max()) for tau in (1.0, 2.0)))
    dt = 0.1
    print("  (iii) transfer-tensor recursion vs chain rule, dt=%.1f:" % dt)
    U = [np.eye(4)] + [U_discrete(g, dt, N) for N in range(1, 5)]
    Kt = ttm(U, expm(LSp * dt), dt)
    for n in range(4):
        print("     n=%d  max|K_ttm - K_chain| = %.1e   (|K_n| = %.3f)" % (n, np.abs(Kt[n] - K_chain(g, dt, n)).max(), np.abs(Kt[n]).max()))
    print("  (iv) inverse map K -> I (Trotter-consistent data):")
    Irec = invert(dt, Kt)
    print("     " + "  ".join("n=%d: %.1e" % (n, np.abs(Irec[n] - I_mat(g, dt, n)).max()) for n in range(4)))
    e = {n: eta_from_I(Irec[n]) for n in (1, 2)}
    a = -np.log(e[2] / e[1]) / dt; g2 = e[1] * a ** 2 / (4 * np.sinh(a * dt / 2) ** 2 * np.exp(-a * dt))
    print("     learned: Omega=%.6f gamma=%.6f g^2=%.6f   (exact %.1f, %.1f, %.2f)" % (a.imag, 2 * a.real, g2.real, Omega, gamma, g ** 2))
    print("  (v) Hamiltonian learning from the EXACT map (Trotter mismatch), dt=%.1f: Omega, gamma from eta_{1+m}/eta_1" % dt)
    nmax = 4
    ts = np.arange(0, (nmax + 1) * dt + 1e-9, dt); Pi, _, _ = exact_propagator(ts, g, N=12)
    Kx = ttm([Rs @ Pi[n] @ Rsi for n in range(nmax + 2)], expm(LSp * dt), dt)
    Irec = invert(dt, Kx); e = {n: eta_from_I(Irec[n]) for n in range(1, nmax + 1)}
    for n in range(1, nmax + 1):
        print("     eta_%d: relative error %.2e" % (n, abs(e[n] / eta(g, dt, n) - 1)))
    for mm in range(1, nmax):
        a = -np.log(e[1 + mm] / e[1]) / (mm * dt)
        print("     m=%d: Omega=%.4f gamma=%.4f" % (mm, a.imag, 2 * a.real))


# ----------------------------------------------------------------------------
def main():
    ts = np.arange(0, T_MAX + DT / 2, DT)
    rho0 = np.array([[1, 1], [1, 1]], complex) / 2  # |+x>
    fig_bath_kernels()
    results = {}
    for g in (g_weak, g_strong):
        results[g] = run_dynamics(g, ts, rho0)
    fig_dynamics(results, ts)
    fig_kernels(results, ts)
    fig_filter()

    # ---- table of second-order Markov rates (Bloch form) and comparison with analytic formulas
    print("Second-order Markov generator in the Pauli basis (I,x,y,z), g = %.2f, theta = pi/4" % g_strong)
    Gm = results[g_strong]["G2"][-1]
    M = np.array([[bloch_component(L_S + Gm, i, j) for j in range(4)] for i in range(4)])
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(np.real(M)); print(np.imag(M))
    # analytic: Gamma_down = 2 sin^2 theta J(w0)(1+n), Gamma_up = 2 sin^2 theta J(-w0) ... (T=0: J(-w0) extension)
    g = g_strong
    Gdown = 2 * np.sin(theta) ** 2 * J_spec(w0, g)
    Gup = 2 * np.sin(theta) ** 2 * J_spec(-w0, g)
    Gphi = 2 * np.cos(theta) ** 2 * S_omega(0.0, g)  # pure-dephasing rate  = 2 cos^2 theta S(omega=0)
    print("analytic  Gamma_down = %.5f  Gamma_up = %.5f  Gamma_phi(pure) = %.5f" % (Gdown, Gup, Gphi))
    print("numeric   -M_zz = %.5f (= 1/T1),  -(M_xx+M_yy)/2 = %.5f (= 1/T2, secular average)" % (-np.real(M[3, 3]), -0.5 * np.real(M[1, 1] + M[2, 2])))
    print("expected  1/T1 = %.5f,  1/T2 = 1/(2T1) + Gamma_phi = %.5f" % (Gdown + Gup, 0.5 * (Gdown + Gup) + Gphi))
    print("numeric   precession frequency sqrt(-M_xy M_yx) = %.5f (Lamb-shifted omega_0)" % np.sqrt(-np.real(M[1, 2] * M[2, 1])))
    print("numeric   steady-state <sigma_z> = %.5f, expected -(Gdown-Gup)/(Gdown+Gup) = %.5f" % (
        -np.real(M[3, 0]) / np.real(M[3, 3]) * np.sqrt(2) / np.sqrt(2), -(Gdown - Gup) / (Gdown + Gup)))
    # convergence check of the pseudomode truncation
    Pi20, _, _ = exact_propagator(ts[:200], g_strong, N=20)
    err = np.max(np.abs(Pi20 - results[g_strong]["Pi"][:200]))
    print("pseudomode truncation check (N=14 vs 20), max |dPi| over t<4: %.2e" % err)
    check_fourth_order()
    check_rwa_kernel()
    check_influence_kernel()


if __name__ == "__main__":
    main()
