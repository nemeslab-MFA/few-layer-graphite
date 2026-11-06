#!/usr/bin/env python3
"""
Tip-induced band bending across the (+4-) | (+5) stacking wall with the
``bandbending`` engine (https://github.com/nemeslab-MFA/bandbending).

Local approximation: three laterally uniform six-layer stacks under the same
tip, each solved self-consistently (Hartree + gamma0/gamma1 tight binding, the
engine's ``GrapheneMultilayer`` with ``coupling="scf"``) on 90 nm SiO2 over
the grounded Si:

  1. (+5)  ABCABC, far from the wall;
  2. the wall, where the flat band bends: the top-layer dimer coupling shared
     between B5-A6 (weight 1 - s, the (+5) bond) and A5-B6 (weight s, the
     (+4-) bond), the interpolated Hamiltonian of ``stacking_wall.py`` in the
     engine's gamma0/gamma1 model;
  3. (+4-) ABCABA, far from the wall.

The engine reduces every zone integral to one dimension because the gamma0/
gamma1 bond graph of a pure stacking is a tree.  The wall Hamiltonian closes a
loop A5-B5-A6-B6, whose flux is twice the phase of the graphene structure
factor f(k), so ``WallTB`` keeps the complex Hamiltonian H(eps, phi) with
phi = arg f and averages over phi.  Near K, where all states within the
spectral window live, phi is uniformly distributed on the contours of
|f|, so a midpoint rule on [0, pi) (H depends on exp(2 i phi)) is used at
every eps node of the engine's quadrature; the end members s = 0 and s = 1
are trees and reproduce ``MultilayerTB`` exactly.

Procedure (``main``):
  * the doping is set from the measured flat-band energy of the (+5) side
    (``GrapheneMultilayer(fermi_level=-E_FB)``: the isolated stack is neutral
    with its flat band E_FB above the Fermi level); the fixed charge per layer
    this implies is then used unchanged for the wall and the (+4-) stack;
  * the tip contact potential (W_tip - W_sample, the same for the three
    stacks) is scanned, the simulated top-layer dI/dV of the (+5) stack
    (``tunneling_spectrum``: band-bending term, WKB barrier, lock-in
    broadening) is compared with the measured spectrum far from the wall and
    the best value kept;
  * the wall and the (+4-) stack are solved with these parameters and their
    spectra compared with the measurement at the corresponding positions.

    python stacking_wall_tibb.py --sm4 LINE.sm4 --tag 514 --x-sf 22 --x-rg 55 \
        [--cpot -0.2 -0.1 0 0.1 0.2] [--s 0.75] [--stage fit|all]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library
from bandbending.materials.multilayer import (
    CHUNK, DEGENERATE_DE, G_SPIN, MultilayerTB, TBParameters, _gauss_smooth, _ramp_accumulate,
)
from bandbending.results.spectroscopy import tunneling_spectrum

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")

SEQ_RG, SEQ_SF = "ABCABC", "ABCABA"
WALL_NAME = "WALL"


# =============================================================================
# the wall stack in the engine's tight-binding model
# =============================================================================

class WallTB(MultilayerTB):
    """Six-layer stack whose top dimer is shared between the (+5) bond B5-A6, weight
    1 - s, and the (+4-) bond A5-B6, weight s (gamma1 times the weight).

    H(eps, phi) is complex Hermitian: eps exp(i phi) on every intralayer bond (phi the
    phase of f(k)), the dimers real; the loop A5-B5-A6-B6 carries the flux 2 phi.  Zone
    averages use the engine's eps quadrature times ``n_phi`` midpoint nodes of phi on
    [0, pi).  Charges, their Jacobian (Lindhard sum) and the projected DOS are evaluated
    with numpy's batched complex eigensolver, vectorised over radial rows.
    """

    def __init__(self, params: TBParameters | None = None, *, s: float = 0.5, n_phi: int = 6,
                 stacking: str = SEQ_RG, **kw):
        params = library.TB_MCELLISTRIM2023 if params is None else params
        super().__init__(stacking, params, **kw)
        if self.N != 6:
            raise ValueError("WallTB is written for six layers")
        self.s = float(s)
        self.n_phi = int(n_phi)
        self._chain = None                                   # never the tridiagonal path
        self._spec.update(s=self.s, n_phi=self.n_phi)
        phi = (np.arange(self.n_phi) + 0.5) * math.pi / self.n_phi
        g = self.grid
        self.nodes2 = np.repeat(g.nodes, self.n_phi)         # (P n_phi,)
        self.phi2 = np.tile(phi, g.nodes.size)
        self.weights2 = np.repeat(g.weights, self.n_phi) / self.n_phi
        self.cells2 = np.repeat(g.cells, self.n_phi, axis=0)
        # dimers of ABCABC without the top one, plus the two interpolated top bonds
        self._dimers = tuple(d for d in self.stacking.dimers if 10 not in d and 11 not in d)
        self._top = ((9, 10, 1.0 - self.s), (8, 11, self.s))   # (B5, A6), (A5, B6)

    @classmethod
    def from_spec(cls, spec: dict) -> "WallTB":
        d = dict(spec)
        params = TBParameters(**d.pop("params"))
        stacking = d.pop("stacking")
        return cls(params, s=d.pop("s"), n_phi=d.pop("n_phi"), stacking=stacking, **d)

    def __repr__(self) -> str:
        return (f"WallTB(s={self.s:.3f}, n_phi={self.n_phi}, gamma0={self.params.gamma0:.4g}, "
                f"gamma1={self.params.gamma1:.4g}, T={self.temperature:g} K, "
                f"broadening={self.broadening:.3g} eV, nodes={self.nodes2.size})")

    # -- Hamiltonian ---------------------------------------------------------------
    def hamiltonian(self, eps, U=None, phi=0.0) -> np.ndarray:
        """(P, 12, 12) complex Hermitian H(eps, phi) with layer potentials U."""
        return self._hamiltonian_rows(np.atleast_2d(self._potential(U)), eps, phi)[0]

    def _hamiltonian_rows(self, U_rows, eps, phi) -> np.ndarray:
        """(R, P, 12, 12) for R rows of layer potentials and P (eps, phi) nodes."""
        U_rows = np.asarray(U_rows, dtype=np.float64)
        eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
        phase = np.exp(1j * np.broadcast_to(np.asarray(phi, dtype=np.float64), eps.shape))
        R, P, n = U_rows.shape[0], eps.size, 2 * self.N
        H = np.zeros((R, P, n, n), dtype=np.complex128)
        idx = np.arange(n)
        H[:, :, idx, idx] = np.repeat(U_rows, 2, axis=1)[:, None, :]
        t = eps * phase
        for l in range(self.N):
            a, b = 2 * l, 2 * l + 1
            H[:, :, a, b] = t[None, :]
            H[:, :, b, a] = np.conj(t)[None, :]
        g1 = self.params.gamma1
        for i, j in self._dimers:
            H[:, :, i, j] = H[:, :, j, i] = g1
        for i, j, w in self._top:
            H[:, :, i, j] = H[:, :, j, i] = g1 * w
        return H

    def eigen(self, U=None, eps=None):
        """Eigenpairs on the (eps, phi) nodes (default) or on eps at phi = 0."""
        if eps is None:
            return np.linalg.eigh(self.hamiltonian(self.nodes2, U, self.phi2))
        return np.linalg.eigh(self.hamiltonian(eps, U))

    # -- charges --------------------------------------------------------------------
    def _core(self, U_rows, mu0: float, jacobian: bool, batch: int = 8192, threads: int | None = None):
        """(excess occupation (R, N), Lindhard sum (R, N, N) or None) over the 2D nodes,
        the node batches shared between threads (the eigensolver releases the GIL)."""
        from concurrent.futures import ThreadPoolExecutor

        U_rows = np.asarray(U_rows, dtype=np.float64)
        R, N, B = U_rows.shape[0], self.N, 2 * self.N
        P_nodes = self.nodes2.size
        per = max(1, batch // max(R, 1))                       # nodes per batch
        starts = list(range(0, P_nodes, per))
        threads = (os.cpu_count() or 1) if threads is None else int(threads)
        threads = max(1, min(threads, len(starts)))

        def work(my_starts):
            occ = np.zeros((R, N))
            dn = np.zeros((R, N, N)) if jacobian else None
            for s in my_starts:
                eps, phi, W = self.nodes2[s:s + per], self.phi2[s:s + per], self.weights2[s:s + per]
                E, P = np.linalg.eigh(self._hamiltonian_rows(U_rows, eps, phi))   # (R, p, B), (R, p, B, B)
                f, df = self.smearing.evaluate(E - mu0)
                P4 = P.reshape(R, eps.size, N, 2, B)
                w = (np.abs(P4) ** 2).sum(axis=3)                                   # (R, p, N, B)
                occ += np.einsum("p,rpb,rplb->rl", W, f, w) - W.sum()
                if jacobian:
                    dE = E[..., :, None] - E[..., None, :]
                    deg = np.abs(dE) < DEGENERATE_DE
                    with np.errstate(divide="ignore", invalid="ignore"):
                        F = np.where(deg, 0.5 * (df[..., :, None] + df[..., None, :]),
                                     (f[..., :, None] - f[..., None, :]) / np.where(deg, 1.0, dE))
                    Fp = np.sqrt(np.maximum(-F, 0.0))                              # F <= 0: f decreases
                    A = np.einsum("rplsa,rplsb->rplab", np.conj(P4), P4)            # (R, p, N, B, B)
                    G = (A * Fp[:, :, None, :, :]).reshape(R, eps.size, N, B * B)
                    dn -= np.real(np.einsum("p,rplk,rpmk->rlm", W, G, np.conj(G)))
            return occ, dn

        if threads == 1:
            return work(starts)
        with ThreadPoolExecutor(threads) as pool:
            parts = list(pool.map(work, [starts[i::threads] for i in range(threads)]))
        occ = np.sum([o for o, _ in parts], axis=0)
        dn = np.sum([d for _, d in parts], axis=0) if jacobian else None
        return occ, dn

    def _accumulate_dense(self, U, mu0: float, jacobian: bool):
        occ, dn = self._core(self._potential(U)[None, :], mu0, jacobian)
        return occ[0], (None if dn is None else dn[0])

    def charges_many(self, U, mu0: float = 0.0, doping=0.0, *, jacobian: bool = False):
        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2 or U.shape[1] != self.N:
            raise ValueError(f"U must have shape (n_r, {self.N}), got {U.shape}")
        if not np.all(np.isfinite(U)):
            raise ValueError("U must be finite")
        k = G_SPIN * self.params.per_cell_to_cm2
        dop = np.broadcast_to(np.asarray(doping, dtype=np.float64), (self.N,))
        occ, dn = self._core(U, mu0, jacobian)
        return dop[None, :] - k * occ, (None if dn is None else -k * dn)

    # -- spectra --------------------------------------------------------------------
    def band_edges(self, U=None) -> tuple[float, float]:
        g = self.grid
        fine = g.nodes <= g.eps_fine
        eps = np.concatenate([[0.0], np.repeat(g.nodes[fine], self.n_phi)])
        phi = np.concatenate([[0.0], np.tile((np.arange(self.n_phi) + 0.5) * math.pi / self.n_phi,
                                             int(fine.sum()))])
        E = np.linalg.eigvalsh(self.hamiltonian(eps, U, phi))
        N = self.N
        return float(E[:, N - 1].max()), float(E[:, N].min())

    def bands(self, U=None, eps=None):
        if eps is None:
            eps = np.linspace(0.0, 3.0 * self.params.gamma0, 600)
        eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
        E, P = np.linalg.eigh(self.hamiltonian(eps, U))
        w = (np.abs(P) ** 2).reshape(P.shape[0], self.N, 2, 2 * self.N).sum(axis=2)
        return eps, E, w

    def ldos(self, U=None, energies=None, *, broadening: float = 0.0, resolve: str = "layer"):
        """Projected DOS as ``MultilayerTB.ldos``, over the (eps, phi) nodes."""
        if energies is None:
            energies = np.arange(-1.0, 1.0 + 1e-12, 5e-4)
        E_out = np.asarray(energies, dtype=np.float64)
        dE = (E_out[-1] - E_out[0]) / (E_out.size - 1)
        if not dE > 0 or not np.allclose(np.diff(E_out), dE, rtol=1e-6, atol=1e-12):
            raise ValueError("energies must be uniformly spaced and increasing")
        U = self._potential(U)
        B = 2 * self.N
        edges0 = E_out[0] - 0.5 * dE
        nb = E_out.size
        cum = np.zeros((nb + 1, B))
        for s in range(0, self.nodes2.size, CHUNK):
            eps, phi = self.nodes2[s:s + CHUNK], self.phi2[s:s + CHUNK]
            W, cell = self.weights2[s:s + CHUNK], self.cells2[s:s + CHUNK]
            E, P = np.linalg.eigh(self.hamiltonian(eps, U, phi))
            # dE_n/d eps = <n| dH/d eps |n> = 2 Re sum_l conj(psi_A) psi_B exp(i phi)
            slope = 2.0 * np.real(np.einsum("cls,cls,c->cs", np.conj(P[:, 0::2, :]), P[:, 1::2, :],
                                            np.exp(1j * phi)))
            e1 = E + slope * (cell[:, 0] - eps)[:, None]
            e2 = E + slope * (cell[:, 1] - eps)[:, None]
            mass = W[:, None, None] * np.abs(P) ** 2
            lo = (np.minimum(e1, e2) - edges0) / dE
            hi = (np.maximum(e1, e2) - edges0) / dE
            _ramp_accumulate(cum, lo.ravel(), hi.ravel(), mass.transpose(0, 2, 1).reshape(-1, B))
        D = np.diff(cum, axis=0) / dE * (G_SPIN * self.params.per_cell_to_cm2)
        if broadening > 0:
            D = _gauss_smooth(D, broadening / dE)
        if resolve == "site":
            return E_out, D
        layer = D.reshape(nb, self.N, 2).sum(2)
        if resolve == "layer":
            return E_out, layer
        if resolve == "total":
            return E_out, layer.sum(1)
        raise ValueError("resolve must be 'layer', 'site' or 'total'")


class WallSimulation(Simulation):
    """``Simulation`` whose multilayer named ``WALL_NAME`` is evaluated with ``WallTB``."""

    def __init__(self, sample, tip, temperature, *, wall_s: float, n_phi: int = 6, **kw):
        self._wall_s = float(wall_s)
        self._n_phi = int(n_phi)
        super().__init__(sample, tip, temperature, **kw)

    def _multilayer_tb(self, spec, settings=None, e_window=None):
        if spec.name != WALL_NAME:
            return super()._multilayer_tb(spec, settings, e_window)
        settings = self._solver if settings is None else settings
        gamma = settings.effective_broadening(spec.broadening)
        kB = settings.constants_set.kT(1.0)
        window = float(spec.e_window if e_window is None else max(e_window, spec.e_window))
        key = (WALL_NAME, self._wall_s, self._n_phi, spec.params, self._T, gamma, kB, window)
        store = self._store
        tb = store.get(store.multilayer, key)
        if tb is None:
            tb = WallTB(spec.params, s=self._wall_s, n_phi=self._n_phi, temperature=self._T,
                        broadening=gamma, e_window=window, kB=kB)
            store.put(store.multilayer, key, tb, 16)
            store.stats["multilayer_builds"] += 1
        return tb


# =============================================================================
# samples, sweeps and spectra
# =============================================================================

def make_sample(stacking: str, name: str, *, doping=None, fermi_level=None, broadening: float,
                oxide: float = 90.0, e_window: float = 0.6) -> Sample:
    """Six-layer stack on ``oxide`` nm of SiO2 over the grounded Si (a gate at 0 V)."""
    kw = {"doping": tuple(float(d) for d in doping)} if doping is not None else {"fermi_level": float(fermi_level)}
    ml = GrapheneMultilayer(stacking, name=name, broadening=broadening, e_window=e_window, **kw)
    return Sample(surface=ml, layers=(Layer(oxide, library.SIO2, name="SiO2"),), gate=Gate())


def run_sweep(sample, tip, biases, *, temperature: float, wall_s=None, grid="normal", n_phi=6):
    """Self-consistent bias sweep at V_g = 0; returns (MultilayerSweep, seconds)."""
    surface = sample.surface
    name = (surface[0] if isinstance(surface, (tuple, list)) else surface).name
    t0 = time.perf_counter()
    if wall_s is None:
        sim = Simulation(sample, tip, temperature=temperature, grid=grid)
    else:
        sim = WallSimulation(sample, tip, temperature, wall_s=wall_s, n_phi=n_phi, grid=grid)
    sw = sim.sweep(np.asarray(biases, dtype=np.float64), 0.0, fields="none")
    m = sw.multilayers[name]
    fails = sw.diagnostics.get("failures", []) if hasattr(sw, "diagnostics") else []
    return m, time.perf_counter() - t0, fails


def spectrum_of(m, bias, *, modulation: float, barrier: float, separation: float, broadening: float,
                temperature: float):
    """Simulated top-layer dI/dV on ``bias`` from a sweep (potentials interpolated)."""
    return m.spectrum(0, bias=np.asarray(bias, dtype=np.float64), modulation=modulation, barrier=barrier,
                      separation=separation, broadening=broadening, temperature=temperature)


def sweep_arrays(m) -> dict:
    """The sweep's apex and far-field layer potentials, serialisable."""
    return {"bias": m.bias.tolist(), "U0": m.U0[0].tolist(), "U_far": m.U_far[0].tolist(),
            "mu0": float(m.mu0), "doping": np.asarray(m.doping).tolist(), "stacking": m.stacking}


# =============================================================================
# comparison with the measured spectra
# =============================================================================

def measured_spectra(exp: dict, x_sf: float, x_rg: float, x_wall: float, half: float = 2.0) -> dict:
    """Mean normalised dI/dV far on the (+4-) side (dist < x_sf), at the wall (|dist - x_wall|
    <= half) and far on the (+5) side (dist > x_rg)."""
    bias, dist, didv = exp["bias"], exp["dist"], exp["didv"]
    return {
        "bias": bias,
        "sf": np.nanmean(didv[:, dist < x_sf], axis=1),
        "wall": np.nanmean(didv[:, np.abs(dist - x_wall) <= half], axis=1),
        "rg": np.nanmean(didv[:, dist > x_rg], axis=1),
    }


def shape_distance(bias_exp, y_exp, bias_sim, y_sim, window=(-0.45, 0.45)) -> tuple[float, float]:
    """Least-squares scale a and the relative residual of a y_sim against y_exp on the window."""
    sel = (bias_exp >= window[0]) & (bias_exp <= window[1]) & np.isfinite(y_exp)
    ys = np.interp(bias_exp[sel], bias_sim, y_sim)
    ye = y_exp[sel]
    a = float(np.dot(ys, ye) / np.dot(ys, ys))
    return a, float(np.sqrt(np.sum((ye - a * ys) ** 2) / np.sum(ye ** 2)))


def peak_energy(bias, y, window=(-0.12, 0.12)) -> float:
    sel = (bias >= window[0]) & (bias <= window[1])
    return float(bias[sel][np.nanargmax(y[sel])])


# =============================================================================
# driver
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm4", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--x-sf", type=float, required=True)
    p.add_argument("--x-rg", type=float, required=True)
    p.add_argument("--x-wall", type=float, default=None, help="position of the bend (nm); default from the fit json")
    p.add_argument("--s", type=float, default=0.75, help="(+4-) weight of the wall Hamiltonian")
    p.add_argument("--cpot", type=float, nargs="+", default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    p.add_argument("--radius", type=float, default=20.0)
    p.add_argument("--separation", type=float, default=0.7)
    p.add_argument("--barrier", type=float, default=4.6)
    p.add_argument("--temperature", type=float, default=9.0)
    p.add_argument("--broadening", type=float, default=0.010, help="Gaussian width of the TB occupation and LDOS (eV)")
    p.add_argument("--modulation", type=float, default=3.75e-3, help="lock-in modulation (V rms)")
    p.add_argument("--bias-max", type=float, default=0.45)
    p.add_argument("--bias-step", type=float, default=0.025)
    p.add_argument("--grid", default="normal")
    p.add_argument("--n-phi", type=int, default=6)
    p.add_argument("--stage", choices=["fit", "wall", "all"], default="all")
    p.add_argument("--out", default=None, help="output json (default examples/stacking_wall_tibb_TAG.json)")
    return p.parse_args()


def main():
    import stacking_wall as sw

    args = parse_args()
    out_path = args.out or os.path.join(EXAMPLES, f"stacking_wall_tibb_{args.tag}.json")
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}

    exp = sw.load_line_spectrum(args.sm4)
    fit_path = os.path.join(EXAMPLES, f"stacking_wall_{args.tag}.json")
    fit = json.load(open(fit_path))["fit"] if os.path.exists(fit_path) else None
    if args.x_wall is None:
        if fit is None:
            raise SystemExit("--x-wall needed (no fit json)")
        # the bend is strongest about half a wall length on the (+4-) side of the centre
        args.x_wall = fit["x0"] - 0.55 * fit["ell"]
    meas = measured_spectra(exp, args.x_sf, args.x_rg, args.x_wall)
    E_fb = peak_energy(meas["bias"], meas["rg"])
    print(f"[{args.tag}] measured flat band (+5) side {1e3 * E_fb:+.1f} mV; wall spectrum at {args.x_wall:.1f} nm")

    biases = np.round(np.arange(-args.bias_max, args.bias_max + 1e-9, args.bias_step), 4) + 0.0
    fine = np.round(np.arange(-args.bias_max, args.bias_max + 1e-9, 2.5e-3), 4)
    spec_kw = dict(modulation=args.modulation, barrier=args.barrier, separation=args.separation,
                   broadening=args.broadening, temperature=args.temperature)
    common = {"radius": args.radius, "separation": args.separation, "barrier": args.barrier,
              "temperature": args.temperature, "broadening": args.broadening, "modulation": args.modulation,
              "bias_step": args.bias_step, "grid": args.grid, "fermi_level": -E_fb, "E_fb_measured": E_fb,
              "x_wall": args.x_wall, "s": args.s, "n_phi": args.n_phi}
    state["settings"] = common
    state["measured"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meas.items()}

    # ---- 1. (+5): doping from the flat band, contact-potential scan ----------------
    if args.stage in ("fit", "all"):
        rg = make_sample(SEQ_RG, "RG", fermi_level=-E_fb, broadening=args.broadening)
        scan = state.setdefault("scan", {})
        for cpot in args.cpot:
            key = f"{cpot:+.3f}"
            if key in scan:
                continue
            tip = Tip(radius=args.radius, separation=args.separation, contact_potential=cpot)
            m, dt, fails = run_sweep(rg, tip, biases, temperature=args.temperature, grid=args.grid)
            sp = spectrum_of(m, fine, **spec_kw)
            a, res = shape_distance(meas["bias"], meas["rg"], fine, sp.didv)
            scan[key] = {"cpot": cpot, "seconds": dt, "failures": len(fails), "scale": a, "residual": res,
                         "peak_sim": peak_energy(fine, sp.didv), "didv": sp.didv.tolist(),
                         "didv_frozen": sp.didv_frozen.tolist(), "sweep": sweep_arrays(m)}
            state["doping"] = np.asarray(m.doping).tolist()
            print(f"[{args.tag}] CPot {cpot:+.2f} V: residual {res:.4f}, peak {1e3 * peak_energy(fine, sp.didv):+.1f} mV, "
                  f"{dt:.0f} s, {len(fails)} failures")
            json.dump(state, open(out_path, "w"))
        best = min(scan.values(), key=lambda d: d["residual"])
        state["best_cpot"] = best["cpot"]
        print(f"[{args.tag}] best contact potential {best['cpot']:+.2f} V (residual {best['residual']:.4f})")
        json.dump(state, open(out_path, "w"))

    # ---- 2, 3. the wall and the (+4-) stack with the same doping and tip -------------
    if args.stage in ("wall", "all"):
        cpot = state["best_cpot"]
        doping = state["doping"]
        tip = Tip(radius=args.radius, separation=args.separation, contact_potential=cpot)
        runs = state.setdefault("runs", {})
        for name, stacking, wall_s, mkey in (("SF", SEQ_SF, None, "sf"), (WALL_NAME, SEQ_RG, args.s, "wall")):
            rkey = f"{mkey}_s{args.s:.2f}" if wall_s is not None else mkey
            if rkey in runs:
                continue
            sample = make_sample(stacking, name, doping=doping, broadening=args.broadening)
            m, dt, fails = run_sweep(sample, tip, biases, temperature=args.temperature, wall_s=wall_s,
                                     grid=args.grid, n_phi=args.n_phi)
            sp = spectrum_of(m, fine, **spec_kw)
            a, res = shape_distance(meas["bias"], meas[mkey], fine, sp.didv)
            runs[rkey] = {"stacking": stacking, "s": wall_s, "seconds": dt, "failures": len(fails), "scale": a,
                          "residual": res, "peak_sim": peak_energy(fine, sp.didv), "didv": sp.didv.tolist(),
                          "didv_frozen": sp.didv_frozen.tolist(), "sweep": sweep_arrays(m)}
            print(f"[{args.tag}] {rkey}: residual {res:.4f}, {dt:.0f} s, {len(fails)} failures, "
                  f"mu0 = {1e3 * m.mu0:+.1f} meV")
            json.dump(state, open(out_path, "w"))
    state["fine_bias"] = fine.tolist()
    json.dump(state, open(out_path, "w"))
    print(f"[{args.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
