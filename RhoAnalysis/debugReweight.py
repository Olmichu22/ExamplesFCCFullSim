#!/usr/bin/env python3
"""
debugReweight.py
----------------
Debug script to compare tau polarization reweighting (SM x weight_M1)
against directly generated P1 sample.

NOTE on weight naming convention:
  weight_M1 stored in the SM tree is the weight that REPRODUCES the P1
  (helicity +1) distribution. The naming is inverted w.r.t. what you
  might expect. This script uses weight_M1 to reweight SM -> P1.

Usage:
    python3 debugReweight.py
"""

import os
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SM_FILE = (
    "/nfs/cms/arqolmo/ExamplesFCCFullSim/Results/RhoAnalysis/"
    "PolAnalysis_GEN_SM_ALL_tau_trained0.4_tph0.35_tpi0_n3_g0.0/"
    "tau_traineddecayAll_0.4_tph0.35_tpi0_n3_g0.0.root"
)
P1_FILE = (
    "/nfs/cms/arqolmo/ExamplesFCCFullSim/Results/RhoAnalysis/"
    "PolAnalysis_GEN_P1_ALL_tau_trained0.4_tph0.35_tpi0_n3_g0.0/"
    "tau_traineddecayAll_0.4_tph0.35_tpi0_n3_g0.0.root"
)
TREE_NAME = "outtree_original"
OUT_DIR = (
    "/nfs/cms/arqolmo/ExamplesFCCFullSim/Results/RhoAnalysis/"
)

BRANCHES = [
    "tau1_decayID", "tau2_decayID",
    "tau1_tauPDG",  "tau2_tauPDG",
    "tau1_cos_theta", "tau2_cos_theta",
    "tau1_cos_psi",   "tau2_cos_psi",
    "tau1_cos_beta",  "tau2_cos_beta",
    "tau1_omega",     "tau2_omega",
    "tau1_weight_P1", "tau2_weight_P1",
    "tau1_weight_M1", "tau2_weight_M1",
    "tau1_cos_theta_tau", "tau2_cos_theta_tau",
    "tau1_visM", "tau2_visM",
]


# ---------------------------------------------------------------------------
# Corrected weight computation (gv_ga sign fixed, no original code modified)
# ---------------------------------------------------------------------------
_MTAU    = 1.7769
_SIN2    = 0.2312
_ALPHA   = 0.46

def _compute_corrected_weight_M1(cos_theta_tau, z, cos_psi, cos_beta, mRho,
                                  sin2=_SIN2, New_Atau=-1.0):
    """
    Recompute weight_M1 with gv_ga = 1 - 4*sin2 (corrected sign).
    All inputs are numpy arrays. Returns array of weights.
    """
    gv_ga   = 1.0 - 4.0 * sin2          # CORRECTED (code uses -1+4*sin2)
    Ae_sm   = 2.0 * gv_ga / (1.0 + gv_ga**2)
    Atau_sm = Ae_sm

    ct2 = cos_theta_tau**2
    Ptau_sm = -(Atau_sm*(1+ct2) + 2*Ae_sm*cos_theta_tau) / \
               (1+ct2 + 2*Ae_sm*Atau_sm*cos_theta_tau)
    Pnew    = -(New_Atau*(1+ct2) + 2*Ae_sm*cos_theta_tau) / \
               (1+ct2 + 2*Ae_sm*New_Atau*cos_theta_tau)

    mtau = _MTAU
    ratioM2 = mtau**2 / np.where(mRho > 0, mRho**2, 1.0)

    anglePsi   = np.arccos(np.clip(cos_psi, -1, 1))
    sin2psi    = np.sin(2.0 * anglePsi)
    theta_rho  = np.arccos(np.clip(z, -1, 1))
    sin_theta  = np.sin(theta_rho)
    P2beta     = (3.0*cos_beta**2 - 1.0) / 2.0
    P2psi      = (3.0*cos_psi**2  - 1.0) / 2.0

    def _terms(P):
        ta = 2.0/3.0*((1-P*z) - ratioM2*(1+P*z)) + ratioM2*(1+P*z)
        tb = -2.0/3.0*((1-P*z - ratioM2*(1+P*z))*P2psi
                        - 1.5*np.sqrt(ratioM2)*P*sin2psi*sin_theta) * P2beta
        return ta + tb

    den = _terms(Ptau_sm)
    num = _terms(Pnew)

    weight = np.where(np.abs(den) > 1e-10, num / den, 1.0)
    return weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample(filepath: str):
    """Load relevant branches from a ROOT file into numpy arrays."""
    print(f"  Loading: {filepath}")
    with uproot.open(filepath) as f:
        t = f[TREE_NAME]
        data = t.arrays(BRANCHES, library="np")
    return data


def select_rho_muon(data: dict):
    """
    Select events where exactly one tau decays to rho (decayID==1)
    and the other to muon (decayID==-13).
    Returns arrays for the rho-tau variables (possibly from tau1 or tau2).
    """
    t1 = data["tau1_decayID"]
    t2 = data["tau2_decayID"]

    # rho is tau1, muon is tau2
    m12 = (t1 == 1) & (t2 == -13)
    # rho is tau2, muon is tau1
    m21 = (t2 == 1) & (t1 == -13)

    def pick(var_base):
        """Return array for the rho tau, choosing tau1 or tau2 branch."""
        arr1 = data[f"tau1_{var_base}"]
        arr2 = data[f"tau2_{var_base}"]
        return np.where(m12, arr1, np.where(m21, arr2, np.nan))

    mask = m12 | m21
    sel = {
        "cos_theta":     pick("cos_theta")[mask],
        "cos_psi":       pick("cos_psi")[mask],
        "cos_beta":      pick("cos_beta")[mask],
        "omega":         pick("omega")[mask],
        "weight_P1":     pick("weight_P1")[mask],
        "weight_M1":     pick("weight_M1")[mask],
        "cos_theta_tau": pick("cos_theta_tau")[mask],
        "tauPDG":        pick("tauPDG")[mask],
        "visM":          pick("visM")[mask],
    }
    # Corrected weight computed on-the-fly (gv_ga sign fixed)
    sel["weight_M1_corr"] = _compute_corrected_weight_M1(
        sel["cos_theta_tau"], sel["cos_theta"],
        sel["cos_psi"], sel["cos_beta"], sel["visM"],
    )
    n_total = len(t1)
    n_sel   = mask.sum()
    print(f"    Events total: {n_total:,}  |  rho+muon selected: {n_sel:,}  "
          f"({100.*n_sel/n_total:.1f}%)")
    return sel


def chi2_ndf(h_obs, h_exp):
    """
    Compute chi2/ndf between two (normalised) histograms.
    Bins with zero expected content are skipped.
    """
    valid = h_exp > 0
    chi2  = np.sum((h_obs[valid] - h_exp[valid])**2 / h_exp[valid])
    ndf   = valid.sum() - 1
    return chi2, ndf


def normed_hist(values, bins, weights=None):
    """Return bin centres and density-normalised histogram (area=1)."""
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(edges)
    total  = counts.sum()
    if total > 0:
        density = counts / (total * widths)
    else:
        density = counts.copy().astype(float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density, edges


def plot_comparison(ax_main, ax_ratio,
                    sm_vals, sm_w, p1_vals,
                    bins, xlabel, label_sm="SM nominal",
                    label_rew="SM × weight_M1 (corr)",
                    label_p1="P1 direct"):
    """
    Draw shape comparison on ax_main and ratio (SM×w / P1) on ax_ratio.
    Returns chi2/ndf string.
    """
    centres, h_sm,  edges = normed_hist(sm_vals,  bins)
    _,       h_rew, _     = normed_hist(sm_vals,  bins, weights=sm_w)
    _,       h_p1,  _     = normed_hist(p1_vals,  bins)

    widths = np.diff(edges)

    ax_main.bar(edges[:-1], h_sm,  width=widths, align="edge",
                alpha=0.35, color="steelblue",  label=label_sm)
    ax_main.bar(edges[:-1], h_rew, width=widths, align="edge",
                alpha=0.35, color="darkorange", label=label_rew)
    ax_main.step(edges[:-1], h_p1, where="post",
                 color="black", linewidth=1.5,  label=label_p1)

    chi2, ndf = chi2_ndf(h_rew, h_p1)
    chi2_str  = rf"$\chi^2$/ndf = {chi2:.1f}/{ndf}"

    ax_main.set_ylabel("Normalised entries / bin width")
    ax_main.legend(fontsize=8)

    # Ratio
    ratio = np.where(h_p1 > 0, h_rew / h_p1, np.nan)
    ax_ratio.step(edges[:-1], ratio, where="post", color="darkorange")
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("(SM×w) / P1")
    ax_ratio.set_xlabel(xlabel)

    return chi2_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading SM sample ...")
    sm_raw = load_sample(SM_FILE)
    print("Selecting rho+muon events (SM) ...")
    sm = select_rho_muon(sm_raw)

    print("Loading P1 sample ...")
    p1_raw = load_sample(P1_FILE)
    print("Selecting rho+muon events (P1) ...")
    p1 = select_rho_muon(p1_raw)
    print("=" * 60)

    sm_w = sm["weight_M1_corr"]   # corrected weight (gv_ga sign fixed)

    # -----------------------------------------------------------------------
    # Stage 1 — cos_theta comparison
    # -----------------------------------------------------------------------
    print("\n[Stage 1] cos_theta comparison ...")
    bins_ct = np.linspace(-1, 1, 21)

    fig1 = plt.figure(figsize=(8, 7))
    gs   = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1  = fig1.add_subplot(gs[0])
    ax2  = fig1.add_subplot(gs[1], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    chi2_str = plot_comparison(ax1, ax2,
                               sm["cos_theta"], sm_w, p1["cos_theta"],
                               bins_ct, r"$\cos\theta_\rho$")
    ax1.set_title(f"Stage 1 — $\\cos\\theta_\\rho$ shape comparison\n{chi2_str}")

    out1 = os.path.join(OUT_DIR, "debug_reweight_stage1.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # -----------------------------------------------------------------------
    # Stage 2 — weight_M1 distribution + 2D scatter
    # -----------------------------------------------------------------------
    print("\n[Stage 2] weight_M1 distribution and 2D plot ...")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2a: 1D histogram of weight_M1
    ax = axes[0]
    bins_w = np.linspace(0, 2, 51)
    counts, edges = np.histogram(sm_w, bins=bins_w)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="steelblue", alpha=0.7)
    ax.axvline(sm_w.mean(), color="red", linestyle="--",
               label=f"mean = {sm_w.mean():.3f}")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8, label="w = 1")
    ax.set_xlabel("weight_M1")
    ax.set_ylabel("Events")
    ax.set_title("Stage 2a — weight_M1 distribution (SM)")
    ax.legend(fontsize=9)

    # 2b: 2D scatter weight_M1 vs cos_theta_tau
    ax = axes[1]
    h2d, xedges, yedges = np.histogram2d(
        sm["cos_theta_tau"], sm_w,
        bins=[40, 40],
        range=[[-1, 1], [0, 2]]
    )
    im = ax.pcolormesh(xedges, yedges, h2d.T, cmap="viridis")
    fig2.colorbar(im, ax=ax, label="Events")
    ax.set_xlabel(r"$\cos\theta_\tau$ (production angle)")
    ax.set_ylabel("weight_M1")
    ax.set_title("Stage 2b — weight_M1 vs $\\cos\\theta_\\tau$ (SM)")

    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "debug_reweight_stage2.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # -----------------------------------------------------------------------
    # Stage 3 — cos_theta sliced in 3 bins of cos_theta_tau
    # -----------------------------------------------------------------------
    print("\n[Stage 3] cos_theta sliced in cos_theta_tau bins ...")

    ct_tau_sm = sm["cos_theta_tau"]
    ct_tau_p1 = p1["cos_theta_tau"]

    slices = [
        ("bin A:  $\\cos\\theta_\\tau < -0.33$",
         ct_tau_sm < -0.33,   ct_tau_p1 < -0.33),
        ("bin B:  $-0.33 \\leq \\cos\\theta_\\tau < 0.33$",
         (ct_tau_sm >= -0.33) & (ct_tau_sm < 0.33),
         (ct_tau_p1 >= -0.33) & (ct_tau_p1 < 0.33)),
        ("bin C:  $\\cos\\theta_\\tau \\geq 0.33$",
         ct_tau_sm >= 0.33,   ct_tau_p1 >= 0.33),
    ]

    fig3, all_axes = plt.subplots(
        2, 3, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.3}
    )
    # all_axes shape: (2, 3)

    for col, (title, msm, mp1) in enumerate(slices):
        ax_m = all_axes[0, col]
        ax_r = all_axes[1, col]
        plt.setp(ax_m.get_xticklabels(), visible=False)

        sm_vals_sl = sm["cos_theta"][msm]
        sm_w_sl    = sm_w[msm]
        p1_vals_sl = p1["cos_theta"][mp1]

        if len(sm_vals_sl) < 5 or len(p1_vals_sl) < 5:
            ax_m.text(0.5, 0.5, "Not enough events",
                      ha="center", va="center", transform=ax_m.transAxes)
            ax_r.set_visible(False)
            ax_m.set_title(title)
            continue

        chi2_str = plot_comparison(ax_m, ax_r,
                                   sm_vals_sl, sm_w_sl, p1_vals_sl,
                                   bins_ct, r"$\cos\theta_\rho$")

        n_sm = msm.sum()
        n_p1 = mp1.sum()
        ax_m.set_title(f"Stage 3 — {title}\n"
                       f"N(SM)={n_sm:,}  N(P1)={n_p1:,}\n{chi2_str}",
                       fontsize=8)

    out3 = os.path.join(OUT_DIR, "debug_reweight_stage3.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out3}")

    # -----------------------------------------------------------------------
    # Stage 4 — omega comparison
    # -----------------------------------------------------------------------
    print("\n[Stage 4] omega comparison ...")

    # Clip omega to physical range (extreme values arise from numerical issues when w_c≈0)
    bins_om = np.linspace(-3, 3, 51)

    fig4 = plt.figure(figsize=(8, 7))
    gs4  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1  = fig4.add_subplot(gs4[0])
    ax2  = fig4.add_subplot(gs4[1], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    chi2_str = plot_comparison(ax1, ax2,
                               sm["omega"], sm_w, p1["omega"],
                               bins_om, r"$\omega$ (optimal variable)")
    ax1.set_title(f"Stage 4 — $\\omega$ shape comparison\n{chi2_str}")

    out4 = os.path.join(OUT_DIR, "debug_reweight_stage4.png")
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {out4}")

    # -----------------------------------------------------------------------
    # Stage 5 — cos_theta split by rho-tau charge (tau PDG = ±15)
    # -----------------------------------------------------------------------
    print("\n[Stage 5] cos_theta split by rho-tau charge ...")

    charges = [
        (r"$\tau^-\to\rho$ (PDG=+15)", sm["tauPDG"] == 15,  p1["tauPDG"] == 15),
        (r"$\tau^+\to\rho$ (PDG=-15)", sm["tauPDG"] == -15, p1["tauPDG"] == -15),
    ]

    fig5, all_axes5 = plt.subplots(
        2, 2, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.3}
    )

    for col, (title, msm, mp1) in enumerate(charges):
        ax_m = all_axes5[0, col]
        ax_r = all_axes5[1, col]
        plt.setp(ax_m.get_xticklabels(), visible=False)

        sm_vals_sl = sm["cos_theta"][msm]
        sm_w_sl    = sm["weight_M1_corr"][msm]
        p1_vals_sl = p1["cos_theta"][mp1]

        if len(sm_vals_sl) < 5 or len(p1_vals_sl) < 5:
            ax_m.text(0.5, 0.5, "Not enough events",
                      ha="center", va="center", transform=ax_m.transAxes)
            ax_r.set_visible(False)
            ax_m.set_title(title)
            continue

        chi2_str = plot_comparison(ax_m, ax_r,
                                   sm_vals_sl, sm_w_sl, p1_vals_sl,
                                   bins_ct, r"$\cos\theta_\rho$")
        n_sm = msm.sum()
        n_p1 = mp1.sum()
        ax_m.set_title(f"Stage 5 — {title}\n"
                       f"N(SM)={n_sm:,}  N(P1)={n_p1:,}\n{chi2_str}",
                       fontsize=9)

    out5 = os.path.join(OUT_DIR, "debug_reweight_stage5.png")
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved: {out5}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All stages complete. Output files:")
    for f in [out1, out2, out3, out4, out5]:
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
