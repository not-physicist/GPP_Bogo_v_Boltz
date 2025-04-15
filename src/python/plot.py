import matplotlib.pyplot as plt
import numpy as np

from os import listdir 
from os.path import join
from pathlib import Path

def _latex_float(x):
    float_str = "{0:.2g}".format(x)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def plot_back_single(dn):
    # dn = "../data/Chaotic2/"
    fn = join(dn, "eom.npz")
    data = np.load(fn)
    # print(data)

    t = data["tau"]
    a = data["a"]
    app_a = data["app_a"]
    phi = data["phi"]
    Omega_r = data["Omega_r"]
    Omega_ϕ = data["Omega_phi"]
    N = np.log(a)
    
    a_e = data["a_e"]
    a_rh = data["a_rh"]
    print(f"a_end = {a_e/a_rh}")

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # ax1.plot(t, a)
    ax1.plot(N, phi, c="k")
    # ax1.plot(N, R, c="k")
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # ax1.set_xlim((5e4, 6e4))
    # ax1.set_ylim((-1e-10,1e-10))
    ax1.set_xlabel("$ln(a)$")
    ax1.set_ylabel(r"$\phi/m_{pl}$")

    ax2.plot(N, Omega_r, label=r"$\Omega_r$")
    ax2.plot(N, Omega_ϕ, label=r"$\Omega_{\phi}$")
    ax2.set_xlabel("$ln(a)$")
    ax2.set_yscale("log")
    # ax2.set_ylim(1e-10, 2)
    ax2.legend()

    plt.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "EOM.pdf")
    # print(fig_fn)
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
    ax.plot(k, n[0] * k**(-9/2), label=r"$k^{-9/2}$", color="gray", ls="--")
    ax.plot(k, ρ, label=r"$\omega_k |\beta_k|^2 / a_e H_e$")
    ax.plot(k, error, label="error")

    ax.set_xlabel("$k/a_e H_e$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    fig_fn = join(out_dn, "specs.pdf")
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig=fig)


def draw_spec(dn, AX, label, m_phi):
    # print(dn, label)
    fn = join(dn, "spec.npz")
    data = np.load(fn)
    # print(data)

    k = data["k"]
    ρ = data["rho"]

    fn_ode = join(dn, "eom.npz")
    H_e = np.load(fn_ode)["H_e"]
    a_e = np.load(fn_ode)["a_e"]
    a_rh = np.load(fn_ode)["a_rh"]

    AX.plot(k, ρ, label=label)
    
    # analytical results
    # print(m_phi, H_e)
    C = 1.5
    C_k = 1.5
    k_high = C_k * m_phi * a_rh / (a_e * H_e)
    # print(k_high)
    ρ_ana = C*3/(64*np.pi) * (m_phi / H_e)**(3/2) * k**(-1/2) * np.where(k < 0.1 * k_high, 1, np.exp(- 2 * (k/k_high)**2))
    # ρ_ana2 = C*3/(64*np.pi) * (m_phi / H_e)**(3/2) * k**(-1/2) * np.exp(-2* (k/k_high)**2)
    AX.plot(k, ρ_ana, color="gray", ls="--")
    # AX.plot(k, ρ_ana2, color="gray", ls="--")

    return None

def plot_all(dn):
    fns = [x for x in listdir(dn)]
    fig, ax = plt.subplots()

    for fn in fns:
        # print(fn)
        m, Γ = fn.replace("m=", "").split("-Γ=")
        m = float(m)
        Γ = float(Γ)
        # print(m, Γ)

        full_dn = join(dn, fn)
        # print(full_dn)
        plot_back_single(full_dn)
       
        label = rf"$m_\phi={_latex_float(m)}" + "m_{pl}" + rf", \Gamma={_latex_float(Γ/m)} m_\phi$"
        # print(label)
        # plot_spec_single(full_dn, ax, label)
        draw_spec(full_dn, ax, label, m)

    ax.set_xlabel("$k/a_e H_e$")
    ax.set_ylabel(r"$a_0^4 \rho_{h, k} / (a_eH_e)^4$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(dn.replace("data", "figs") + "spec_all.pdf", bbox_inches="tight")
    plt.close()


# plot_single("../data/Chaotic2/m=1.0e-06-Γ=1.0e-07")
plot_all("../data/Chaotic2/")
