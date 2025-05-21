import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from scipy.ndimage import gaussian_filter1d

from os import listdir 
from os.path import join
from pathlib import Path

import formula

def _latex_float(x):
    float_str = "{0:.2g}".format(x)
    # print(float_str)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def get_eos(a, H):
    """
    get equation of state parameter from scalar factor and Hubble parameter
    """
    return np.diff(np.log((H/H[0])**2)) / np.diff(np.log(a)) / (-3) - 1

def plot_back_single(dn):
    fn = join(dn, "eom.npz")
    data = np.load(fn)

    t = data["tau"]
    a = data["a"]
    H = data["H"]
    app_a = data["app_a"]
    phi = data["phi"]
    Omega_r = data["Omega_r"]
    Omega_ϕ = data["Omega_phi"]
    N = np.log(a)
    
    a_e = data["a_e"]
    a_rh = data["a_rh"]
    H_e = data["H_e"]
    N_e = np.log(a_e)

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # ax1.plot(t, a)
    ax1.plot(N, phi, c="k")
    w = get_eos(a, H)
    w_smooth = gaussian_filter1d(w, 100)
    # print(w_smooth[::100])
    ax1.plot(N[:-1], w, c="tab:blue", alpha=0.3)
    ax1.plot(N[:-1], w_smooth, c="tab:blue", alpha=1.0)
    # ax1.set_ylim(-1, 1)
    # ax1.legend()
    ax1.set_xlabel("$ln(a)$")
    ax1.set_ylabel(r"$\phi/m_{pl}$")

    ax2.plot(N, Omega_r, label=r"$\Omega_r$")
    ax2.plot(N, Omega_ϕ, label=r"$\Omega_{\phi}$")
    ax2.set_xlabel("$ln(a)$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-10, 2)
    ax2.legend()

    plt.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "EOM.pdf")
    # print(fig_fn)
    plt.savefig(fig_fn, bbox_inches="tight")

    plt.close()
    
    ###############################
    # rho^2 / H
    ###############################
    fig, ax = plt.subplots()
    rho2_H = (Omega_ϕ * 3 * H**2)**2 / H
    rho2_H_ana = (3 * H_e**2 * (a_e / a)**3)**2 / (H_e*(a/a_e)**(-3/2))
    ax.plot(N[N > N_e], rho2_H[N > N_e], color="k")
    ax.plot(N[N > N_e], rho2_H_ana[N > N_e], color="k", ls="--")
    # ax.plot(N[N > N_e], np.interp(N[N > N_e], ))
    
    ax.set_xlabel(r"$ln(a)$")
    ax.set_ylabel(r"$\rho_\phi^2 / H$")
    ax.set_yscale("log")
    # ax.set_ylim(1e-24, 1e-11)

    plt.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "rho2_H.pdf")
    # print(fig_fn)
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()
    
    ###############################
    # m_eff
    ###############################
    try:
        m_eff = data["m_eff"]
        fig, ax = plt.subplots()
        ax.plot(N, m_eff, c="k", label=r"$m_{\text{eff}}$")
        ax.plot(N[0:-1], np.diff(m_eff)/np.diff(N), c="k", ls="--", label=r"$dm_{\text{eff}}/dN$")

        ax.set_xlabel(r"$ln(a)$")
        # ax.set_ylabel(r"$m_{\text{eff}}$")
        ax.legend()

        fig_fn = join(out_dn, "m_eff.pdf")
        plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()
    except KeyError:
        print("m_eff not found. Skipping...")


    ###############################
    # a''/a
    ###############################
    fig, ax = plt.subplots()
    ax.plot(N, app_a, c="k")
    ax.set_xlabel(r"$ln(a)$")
    ax.set_ylabel(r"$a''/a$")
    fig_fn = join(out_dn, "app_a.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()



def plot_spec_single(dn):
    fn = join(dn, "spec.npz")
    data = np.load(fn)
    # print(data)

    k = data["k"]
    n = data["n"]
    ρ = data["rho"]
    error = data["error"]

    fig, ax = plt.subplots()

    ax.plot(k, n, label=r"$|\beta_k|^2$")
    ax.plot(k, n[0] * (k/k[0])**(-9/2), label=r"$k^{-9/2}$", color="gray", ls="--")
    ax.plot(k, ρ, label=r"$\rho_k$")
    ax.plot(k, error, label="error")

    ax.set_xlabel("$k/a_e H_e$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((1e-15, 1e1))
    ax.legend()

    fig.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "specs.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig=fig)

def draw_spec(dn, AX, AX2, label_pref, m_phi, Γ, c, ls):
    # print(dn, label)
    fn = join(dn, "spec.npz")
    data = np.load(fn)
    # print(data)

    k = data["k"]
    ρ = data["rho"]
    n = data["n"]
    # n_boltz = data["n_boltz"]
    # print(k.shape, n_boltz.shape)

    fn_ode = join(dn, "eom.npz")
    # print(fn_ode)
    data_eom = np.load(fn_ode)
    a = data_eom["a"]
    H_e = data_eom["H_e"]
    a_e = data_eom["a_e"]
    a_rh = data_eom["a_rh"]
    H = data_eom["H"]
    ρ_ϕ = data_eom["Omega_phi"] * 3 * H**2
    
    try:
        # try to read out m_eff
        m_eff = data_eom["m_eff"]
        f_exact_boltz = formula.get_f_exact_boltz(k, a, ρ_ϕ, H, m_eff, a_e, H_e)
    except KeyError:
        # if not found, then no m_eff, just use m_phi
        f_exact_boltz = formula.get_f_exact_boltz(k, a, ρ_ϕ, H, m_phi, a_e, H_e)


    AX.plot(k[f_exact_boltz > 0], f_exact_boltz[f_exact_boltz > 0], color=c, ls="dotted", label=label_pref + "exact Boltz.")

    # AX.plot(k[n_boltz != 0], n_boltz[n_boltz !=0], color="grey", ls="dotted")
    AX.plot(k, n, ls=ls, color=c, label=label_pref+"Bogo.", alpha=0.5)
    
    if AX2 is not None:
        f = formula.get_f(k, a_e/a_rh, H_e, Γ)
        Ω = formula.get_Ω_gw0(ρ, a_e/a_rh, H_e, Γ)
        AX2.plot(f, Ω, label=label_pref, color=c, ls=ls)

    return None

def _get_m_Γ_list(fns):
    """
    get list of numerical values of m and Γ
    """
    # read the directory name first
    m = []
    Γm = []
    for fn in fns:
        # m_i, Γ_i = fn.replace("m=", "").split("-Γ=")
        m_i, Γ_i = fn.replace("r=", "").split("-Γ=")
        print(m_i, Γ_i)
        m.insert(0, float(m_i))
        Γm.insert(0, round(float(Γ_i)/float(m_i), 4))
    m = list(set(m))
    m.sort()
    Γm = list(set(Γm))
    Γm.sort()
    # print(m_array, Γm_array)

    return m, Γm

def plot_all(dn):
    # remove test
    fns = [x for x in listdir(dn) if x not in ["test"]]
    # print(fns)
    
    # for ρ_k plot
    fig = plt.figure(1)
    ax = fig.add_subplot()

    # for Ω_gw plot 
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    
    # linestyles for different Γ/m values
    ls_array = ["solid", "dashed", "dashdot", "dotted"]
   
    m_array, Γm_array = _get_m_Γ_list(fns)

    # get color from m_array
    cmap = colormaps['viridis']
    color_array = cmap(np.log10(m_array)/np.log10(np.amin(m_array)))
    # print(color_array)
    
    for fn in fns:
        # print(fn)
        # m, Γ = fn.replace("m=", "").split("-Γ=")
        m, Γ = fn.replace("r=", "").split("-Γ=")
        m = float(m)
        Γ = float(Γ)

        full_dn = join(dn, fn)
        plot_back_single(full_dn)
        
        if Γm_array.index(round(Γ/m, 4)) == 0:
        # show the legend when the ls is solid
            label = rf"$m_\phi={_latex_float(m)}" + "m_{pl}$"
            # + rf", \Gamma={_latex_float(Γ/m)} m_\phi$"
        else:
            label = ""
        # print(label)
        plot_spec_single(full_dn)
        color = color_array[m_array.index(m)]
        # ls = ls_array[Γm_array.index(round(Γ/m, 4))]
        ls = None
        draw_spec(full_dn, ax, ax2, label, m, Γ, color, ls)
  
    ax.set_xlabel(r"$k/a_e H_e$")
    ax.set_ylabel(r"$f=|\beta_k|^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(1, 1e2)
    # ax.set_ylim(1e-10, 1e1)
    
    # only showing first three handles; avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    # print(handles, labels)
    ax.legend(handles=handles[0:3], loc="upper right")

    fig.tight_layout()
    fig.savefig(dn.replace("data", "figs") + "spec_all.pdf", bbox_inches="tight")
    plt.close(1)

    # ax.set_xlabel("$k/a_e H_e$")
    # ax.set_ylabel(r"$a_0^4 \rho_{h, k} / (a_eH_e)^4$")
    ax2.set_xlabel("$f/Hz$")
    ax2.set_ylabel(r"$\Omega_{gw, 0}h^2 $")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-33, 1e-24)
    ax2.legend(loc="upper right")

    fig2.tight_layout()
    fig2.savefig(dn.replace("data", "figs") + "GW_spec_all.pdf", bbox_inches="tight")
    plt.close(2)

def check_H(dn):
    fn = join(dn, "eom.npz")
    data = np.load(fn)
    # print(data)

    t = data["tau"]
    a = data["a"]
    H = data["H"]
    a_e = data["a_e"]
    a_rh = data["a_rh"]
    H_e = data["H_e"]

    fig, ax = plt.subplots()

    ax.plot(np.log(a), H_e * (a/a_e)**(-3/2), color="grey", ls="--")
    ax.plot(np.log(a), H, color="k")
    ax.plot(np.log([a_e, a_e]), [0, 1], color="tab:blue", alpha=0.5)
    ax.plot(np.log([a_rh, a_rh]), [0, 1], color="tab:orange", alpha=0.5)

    ax.set_xlabel("$N$")
    ax.set_ylabel("$H$")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-3)
    
    fig.tight_layout()

    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "Hubble.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close()

def draw_n2_H(dn, ax, m_phi, c, offset):
    fn = join(dn, "eom.npz")
    data = np.load(fn)

    N = np.log(data["a"])
    N_e = np.log(data["a_e"])
    Omega_ϕ = data["Omega_phi"]
    H = data["H"]

    try:
        m_eff = data["m_eff"]
        n2_H_m = (Omega_ϕ * 3 * H**2)**2 / H / m_eff**3
    except KeyError:
        # no m_eff defined
        n2_H_m = (Omega_ϕ * 3 * H**2)**2 / H / m_phi**3

    ax.plot(N - N_e + offset, n2_H_m, c)


def plot_comp_chaotic_tmodel():
    """
    compare the spectra from chaotic and T model 
    """
    dn1 = "../data/Chaotic2/"
    fns1 = [x for x in listdir(dn1) if x not in ["test"]]
    dn2 = "../data/TModel-n=2/"
    fns2 = [x for x in listdir(dn2) if x not in ["test"]]
    # print(fns1, fns2)
    
    fig, ax = plt.subplots()
    m = 1e-5 
    
    fig2, ax2 = plt.subplots()

    for fn in fns1:
        m, Γ = fn.replace("m=", "").split("-Γ=")
        m = float(m)
        Γ = float(Γ)
        if m == 1e-5:
            full_dn = join(dn1, fn)
            draw_spec(full_dn, ax, None, "chaotic, ", m, Γ, "k", None)
            draw_n2_H(full_dn, ax2, m, "k", 0)

    for fn in fns2:
        # print(m)
        full_dn = join(dn2, fn)
        draw_spec(full_dn, ax, None, "T model $n=2$, ", m, 0.0, "tab:blue", None)
        draw_n2_H(full_dn, ax2, m, "tab:blue", 0.15)

    ax.set_xlabel(r"$k/a_e H_e$")
    ax.set_ylabel(r"$f=|\beta_k|^2$")
    # ax.set_ylabel(r"$f=|\beta_k|^2 (k/a_e H_e)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e2)
    ax.set_ylim(1e-10, 1e0)

    handles, labels = ax.get_legend_handles_labels()
    # print(handles, labels)
    # labels_nodup = list(set(labels))
    ax.legend(handles=[handles[0], handles[1], handles[6], handles[7]], loc="upper right")
    
    fig.tight_layout()
    fig.savefig("../figs/spec_comp_chaotic_tmodel.pdf", bbox_inches="tight")
    plt.close(fig=1)
    
    ax2.set_xlim(0, 5)
    ax2.set_xlabel(r"N - N_e")
    ax2.set_ylabel(r"$\sim \rho^2/H/m^3$")
    ax2.set_yscale("log")
    fig2.tight_layout()
    fig2.savefig("../figs/n2_H_chaotic_tmodel.pdf", bbox_inches="tight")
    plt.close(fig=2)
        

if __name__ == "__main__":
    # plot_back_single("../data/Chaotic2/test")
    # plot_spec_single("../data/Chaotic2/test")
    # plot_all("../data/Chaotic2/")

    # check_H("../data/Chaotic2/m=1.0e-04-Γ=1.0e-07/")
    # check_H("../data/Chaotic2/m=1.0e-05-Γ=1.0e-06/")

    # plot_all("../data/TModel-n=2/")

    # plot_comp_chaotic_tmodel()
    # check_H("../data/TModel-n=2/r=4.5e-03-Γ=1.0e-06/")

    # plot_back_single("../data/Chaotic4/test")
    # plot_spec_single("../data/Chaotic4/test")
    plot_all("../data/Chaotic4/")
