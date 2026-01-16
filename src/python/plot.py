from matplotlib.patches import bbox_artist
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
    
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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

'''
def get_eos(a, H):
    """
    get equation of state parameter from scalar factor and Hubble parameter
    """
    return np.diff(np.log((H/H[0])**2)) / np.diff(np.log(a)) / (-3) - 1
'''

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
    print("a_rh/a_e = ", a_rh / a_e)
    error = data["error"]

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # ax1.plot(t, a)
    ax1.plot(N, phi, c="k")
    # w = get_eos(a, H)
    w = data["w"]
    w_smooth = gaussian_filter1d(w, w.size/1000)
    # print(w_smooth[::100])
    ax1.plot(N, w, c="tab:blue", alpha=0.3)
    ax1.plot(N, w_smooth, c="tab:blue", alpha=1.0)
    ax1.set_ylim(-1, 2)
    # ax1.legend()
    ax1.set_xlabel(r"$\ln(a)$")
    ax1.set_ylabel(r"$\phi/m_\textrm{pl}$")

    ax2.plot(N, Omega_r, label=r"$\Omega_r$")
    ax2.plot(N, Omega_ϕ, label=r"$\Omega_{\phi}$")
    ax2.set_xlabel(r"$\ln(a)$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 2)
    ax2.legend(loc="lower left")

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
    
    ax.set_xlabel(r"$\ln(a)$")
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
        ax.plot(N, m_eff, c="k", label=r"$m_{\textrm{eff}}$")
        # ax.plot(N, m_eff, c="k")
        ax.plot(N[0:-1], np.diff(m_eff)/np.diff(N), c="k", ls="--", label=r"$dm_{\textrm{eff}}/dN$")
        # ax.plot(N[0:-1], np.diff(m_eff)/np.diff(N), c="k", ls="--")

        ax.set_xlabel(r"$\ln(a)$")
        # ax.set_ylabel(r"$m_{\text{eff}}$")
        ax.legend()

        fig_fn = join(out_dn, "m_eff.pdf")
        plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()
    except KeyError:
        print("m_eff not found. Skipping...")


    ###############################
    # a''/a and H
    ###############################
    fig, (ax, ax2) = plt.subplots(ncols=2)
    ax.plot(N, app_a, c="k")
    ax.plot([N[0], N[-1]], [a_e*H_e, a_e*H_e], c="k")
    ax.set_xlabel(r"$\ln(a)$")
    ax.set_ylabel(r"$a''/a$")
    ax.set_yscale("log")

    ax2.plot(N, H, c="k")
    ax2.set_xlabel(r"$\ln(a)$")
    ax2.set_ylabel(r"$H$")
    ax2.set_yscale("log")
    
    plt.tight_layout()
    fig_fn = join(out_dn, "app_a.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()

    #############################
    # error 
    #############################
    fig, ax = plt.subplots()
    ax.plot(N[::100], error[::100])
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("error for $a''/a$")
    plt.tight_layout()
    fig_fn = join(out_dn, "error.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()


def plot_spec_single(dn):
    fn = join(dn, "spec_bogo.npz")
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

    try:
        fn = join(dn, "spec_boltz.npz")
        # print(fn)
        data = np.load(fn)
    except:
        print("No boltzmann resutls found. SKIPPING...")
    else:
        k_boltz = data["k"]
        f_boltz = data["f"]
        mask = f_boltz > 1e-15
        ax.plot(k_boltz[mask], f_boltz[mask], color="k", ls="dotted", label="exact Boltz.")

    """
    get power index for the IR end (if exists)
    """
    if k[0] <= 5e-2:
        k_to_fit = k[k <= 0.05]
        ρ_to_fit = ρ[k <= 0.05]
        # print(k_to_fit, ρ_to_fit)

        popt, pcov = curve_fit(lambda x, a, b: a*x + b, np.log(k_to_fit), np.log(ρ_to_fit))
        # print(popt, pcov)
        perr = np.sqrt(np.diag(pcov))
        print(f"The IR end of the energy spectrum has the power: {popt[0]} +- {perr[0]}")
        plt.plot(k, k**(popt[0])*np.exp(popt[1]), label=rf"$\propto k^{{ {popt[0]:.2f} }}$", ls="-.", color="gray")

        k_to_fit = k[(k >= 0.2) & (k <= 0.7)]
        ρ_to_fit = ρ[(k >= 0.2) & (k <= 0.7)]
        # print(k_to_fit, ρ_to_fit)

        popt, pcov = curve_fit(lambda x, a, b: a*x + b, np.log(k_to_fit), np.log(ρ_to_fit))
        # print(popt, pcov)
        perr = np.sqrt(np.diag(pcov))
        print(f"The near-IR end of the energy spectrum has the power: {popt[0]} +- {perr[0]}")
    
    fn = join(dn, "spec_bogo_ana.npz")
    data = np.load(fn)
    ax.plot(data["k"], data["f"], label="s.p.a.")
    # ax.plot(data["k"], data["f_pure"], label="Bogo. pure ana")

    ax.set_xlabel("$k/a_e H_e$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((1e-15, 1e1))
    ax.legend(loc="lower left")

    fig.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "specs.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig=fig)

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(k, n, label="Bogo.", color="k")
    ax.plot(k_boltz[mask], f_boltz[mask], color="tab:orange", ls="--", label="exact Boltz.")
    ax.plot(data["k"], data["f"], label="s.p.a.")
    # ax.plot(data["k"], data["f_pure"], color="tab:red", ls="dotted", label="ana. Bogo.")
    ax.plot(data["k"], data["f_fast"], color="tab:red", ls="dotted", label="fast Bogo.")
    # print(data["f_fast"])
    ax.plot(data["k"], formula.get_f_ana_slow(data["k"], 1/2), color="tab:red", ls="dashed", label="slow Bogo.")
    draw_Boltzmann_single_j(dn, ax)

    ax.set_xlabel("$k/a_e H_e$")
    ax.set_ylabel(r"$|\beta_k|^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim((1e-1, 1e2))
    ax.set_ylim((1e-15, 1e5))
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "f.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig=fig)


def draw_spec(dn, AX, AX2, label_pref, m_phi, Γ, c, ls):
    # print(dn, label)
    fn = join(dn, "spec_bogo.npz")
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
    mask = f_exact_boltz > 0.0

    AX.plot(k, n, ls=ls, color=c, label=label_pref+"Bogo.")
    AX.plot(k[mask], f_exact_boltz[mask], color="tab:cyan", ls="dotted", label="approx. Boltz.")
    # AX.plot(k[mask], f_exact_boltz[mask], color=c, label="approx. Boltz.")
 
    try:
        fn = join(dn, "spec_boltz.npz")
        # print(fn)
        data = np.load(fn)
    except:
        print("No boltzmann resutls found. SKIPPING...")
    else:
        k_ex_boltz = data["k"]
        f_ex_boltz = data["f"]
        mask = f_ex_boltz > 0.0
        AX.plot(k_ex_boltz[mask], f_ex_boltz[mask], color="tab:orange", ls="dotted", label=label_pref + "exact Boltz.")
    

    if AX2 is not None:
        f = formula.get_f(k, a_e/a_rh, H_e, Γ)
        Ω = formula.get_Ω_gw0(ρ, a_e/a_rh, H_e, Γ)
        AX2.plot(f, Ω, label=label_pref, color=c, ls=ls)

    return None

def _get_r_T_list(fn):
    """
    get list of numerical values of m and Γ
    """
    # read the directory name first
    x = fn.split("/")[3]
    r, T = x.replace("r=", "").split("-T=")
    return r, T

"""
compare beta^2 for quadratic with varied reheating temp
"""
def plot_all_quadratic(dns):
    # print(dns)
    # print(_get_r_T_list(dns))
    
    # for ρ_k plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot()

    # get color from m_array
    # cmap = colormaps['viridis']
    # color_array = cmap(np.log10(m_array)/np.log10(np.amin(m_array)))
    # print(color_array)
    
    for dn in dns:
        r, T = _get_r_T_list(dn)
        m = 4.6e-6
        Γ = 0.0
        draw_spec(dn, ax, None, "", m, Γ, "k", "-")
  
    ax.set_xlabel(r"$k/a_e H_e$")
    ax.set_ylabel(r"$f=|\beta_k|^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 4e2)
    ax.set_ylim(1e-12, 1e1)
    
    # only showing first three handles; avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    # print(handles, labels)
    # avoid duplicate legends
    ax.legend(handles=handles[0:3], loc="upper right")

    fig.tight_layout()
    fig.savefig("../figs/TModel-n=2/spec_all.pdf", bbox_inches="tight")
    plt.close(1)
    
    """
    ax2.set_xlabel("$f/Hz$")
    ax2.set_ylabel(r"$\Omega_{gw, 0}h^2 $")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-33, 1e-24)
    ax2.legend(loc="upper right")

    fig2.tight_layout()
    fig2.savefig(dn.replace("data", "figs") + "GW_spec_all.pdf", bbox_inches="tight")
    plt.close(2)
    """

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
    print(fns1, fns2)
    
    fig, ax = plt.subplots()
    m = 1e-5 
    
    fig2, ax2 = plt.subplots()

    for fn in fns1:
        m, Γ = fn.replace("m=", "").split("-Γ=")
        m = float(m)
        Γ = float(Γ)
        if m == 4.6e-6:
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
    ax.legend(handles=[handles[0], handles[1], handles[3], handles[4]], loc="upper right")
    # ax.legend(loc="upper right")
    
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

"""
plot β^2 evolution
"""
def plot_k_every(dn):
    fns = [x for x in listdir(dn) if x not in ["eom.npz", "eom.dat", "spec.npz"]]
    # print(fns)
    
    fig, ax = plt.subplots()
    for fn in fns:
        k = float(fn.replace("k=", "").replace(".npz", ""))
        full_path = join(dn, fn)
        # print(full_path, k)
        data = np.load(full_path)

        ax.plot(data["N"], data["n"], c="k")
        ax.plot(data["N"], data["error"], c="gray")
    
    ax.set_xlabel("$N$")
    ax.set_yscale("log")
    fig.tight_layout()

    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "k_every.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()

def plot_all_n():
    """
    plot the energy parameter of models with different n's together
    Fix the reheating temp also
    """
    r = 0.01 
    ns = [2, 4, 6]
    Ts = [1e-5, 1e-5, 1e-5]

    fig, ax = plt.subplots()
    for n, T in zip(ns, Ts):
        dn = f"../data/TModel-n={n}/r={r:.1e}-T={T:.1e}/"

        # print(dn)
        # plot_back_single(dn)
        # plot_spec_single(dn)

        fn = join(dn, "eom.npz")
        data = np.load(fn)
        H_e = data["H_e"]
        print("He = ", H_e)
        a_e = data["a_e"]
        a_rh = data["a_rh"]
        print("a_e/a_rh = ", a_e/a_rh)

        fn = join(dn, "spec_bogo.npz")
        data = np.load(fn)
        k = data["k"]
        ρ = data["rho"]

        f = formula.get_f(k, a_e/a_rh, H_e, T)
        Ω = formula.get_Ω_gw0(ρ, a_e/a_rh, H_e, T)
    
        n = dn.split("/")[2][-1]
        ax.plot(f, Ω, label=f"$n={n}$")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((1e-20, 1e-10))
    ax.set_xlabel(r"$f/\textrm{Hz}$")
    ax.set_ylabel(r"$\Omega_{\textrm{gw}, 0} h^2$")
    plt.legend()
    plt.savefig("../figs/Omega_all_n.pdf", bbox_inches="tight")
    plt.close()

def compare_k_rh():
    Ts = [1e-04, 1e-05]
    labels = [r"$10^{-4} m_\textrm{pl}$", r"$10^{-5} m_\textrm{pl}$"]
    # print(dn)

    fig, ax = plt.subplots(figsize=(4,3))
    for (T, label) in zip(Ts, labels):
        dn = f"../data/TModel-n=2/r=1.0e-02-T={T:.1e}/"
        fn = join(dn, "spec_bogo.npz")
        data = np.load(fn)
        # print(data)

        k = data["k"]
        ρ = data["rho"]
        # n = data["n"]

        ax.plot(k, ρ, label=label)

        data = np.load(join(dn, "eom.npz"))
        a_rh = data["a_rh"]
        a_e = data["a_e"]
        H_e = data["H_e"]

        a = data["a"]
        H = data["H"]
        H_rh = np.interp(a_rh, a, H)
        k_rh = a_rh * H_rh / a_e / H_e
        ax.plot([k_rh, k_rh], [1e-10, 1], color="gray", ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/a_eH_e$")
    ax.set_ylabel(r"$|\beta_k|^2 k^4/\pi^2 $")
    
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig_fn = join("../figs/TModel-n=2/", "spec_k_rh.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig=fig)


def draw_Boltzmann_single_j(dn, ax):
    """
    plot Boltzmann spectrum but to each Fourier modes
    """    
    '''
    data = np.load(join(dn, "m_tilde.npz"))
    a_m = data["a"]
    # take the first m_tilde as m_tilde(a_e)
    m_tilde = data["m"]
    a_m_e = a_m[0] * m_tilde[0] 
    print(np.log(a_m[0]))

    H_e = np.load(join(dn, "eom.npz"))["H_e"]
    print(H_e, m_tilde[0])
    '''
    
    fn_prefix = "spec_boltz_j="
    fns = [x for x in listdir(dn) if fn_prefix in x]
    js = [int(x.replace(fn_prefix, "").replace(".npz", "")) for x in fns]
    # print(fns)
    cmap = colormaps["magma"]

    for (fn, j) in zip(fns, js):
        c = cmap(j/np.amax(js)) 

        data = np.load(join(dn, fn))
        k = data["X"]
        f = data["Y"]
        ax.plot(k, f, alpha=0.5, color="gray")
        # popt, pcov = curve_fit(lambda x, a, b: a*x + b, np.log(k[0:4]), np.log(f[0:4]))
        # popt2, pcov2 = curve_fit(lambda x, a, b: a*x + b, np.log(k[-100:]), np.log(f[-100:]))
        # print(popt, popt2)
    

def plot_mtilde(dn):
    fn = join(dn, "eom.npz")
    data = np.load(fn)
    N = np.log(data["a"])
    H = data["H"]
    Ωphi = data["Omega_phi"]
    ρphi = 3 * H**2 * Ωphi
    a_e = data["a_e"]
    H_e = data["H_e"]

    data = np.load(join(dn, "m_tilde.npz"))
    m = data["m"]
    a_n = data["a"]
    
    # WARNING: need to change n
    T = 1e-05 
    n = 2
    Γ = (7-n) / (np.sqrt(3)*n) * T**(4/n) * (30/106/np.pi**2)**(-1/n) * ρphi**(1/2 - 1/n)

    fig, ax = plt.subplots()
    ax.plot([np.log(a_e), np.log(a_e)], [1e-12, 1e-4], color="gray", ls="--")
    ax.plot(N, H, label=r"$H$")
    ax.plot(N, Γ, label=r"$\Gamma$")
    ax.plot(np.log(a_n)[:-1], m, label=r"$\tilde{m}$", marker="+")

    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_xlim((np.log(a_e), N[-1]))
    ax.set_ylim((1e-6, 1e-5))
    
    out_dn = dn.replace("data", "figs")
    fig_fn = join(out_dn, "mtilde.pdf")
    ax.legend(loc="lower left")
    plt.savefig(fig_fn, bbox_inches="tight") 
    plt.close()

def plot_fourier(dn):
    data = np.load(join(dn, "four_coef.npz"))

    plt.plot(np.log(data["c_n"]))
    plt.savefig(dn.replace("data", "figs")+"four_coef.pdf")

if __name__ == "__main__":
    # dn = "../data/TModel-n=2/r=1.0e-02-T=1.0e-05/"
    dn = "../data/TModel-n=4/r=1.0e-02-T=1.0e-05/"
    # dn = "../data/TModel-n=6/r=1.0e-02-T=1.0e-05/"
    # dn = "../data/Chaotic2/r=1.0e-02-T=5.0e-05/"
    # dn = "../data/Chaotic4/r=1.0e-02-T=1.0e-05/"
    # dn = "../data/Chaotic6/r=1.0e-02-T=1.0e-06/"
    # plot_back_single(dn)
    plot_spec_single(dn)
    # plot_fourier(dn)
    # plot_Boltzmann_single_j(dn)
    # plot_mtilde(dn)

    # plot_all_quadratic([f"../data/TModel-n=2/r=1.0e-02-T={x}" for x in ["1.0e-04", "5.0e-05", "1.0e-05"]])

    ################################## n = 2
    # plot_all("../data/Chaotic2/")

    # check_H("../data/Chaotic2/m=1.0e-04-Γ=1.0e-07/")
    # check_H("../data/Chaotic2/m=1.0e-05-Γ=1.0e-06/")

    # plot_all("../data/TModel-n=2/")
    
    # plot_comp_chaotic_tmodel()
    
    ################################### n = 4
    # plot_all("../data/Chaotic4/") 
    # plot_all("../data/TModel-n=4/")
    ################################### n = 6
    # plot_back_single("../data/Chaotic6/r=4.5e-03-Γ=1.0e-12")
    # plot_back_single("../data/Chaotic6/r=4.5e-03-Γ=1.0e-10")
    # plot_all("../data/Chaotic6/")

    # plot_all_n()
    # compare_k_rh()
