import matplotlib.pyplot as plt
import numpy as np

from os.path import join
from pathlib import Path

dn = "../data/Chaotic2/"
fn = join(dn, "ode.npz")
data = np.load(fn)
# print(data)

t = data["tau"]
a = data["a"]
R = data["R"]
phi = data["phi"]
Omega_r = data["Omega_r"]
Omega_ϕ = data["Omega_phi"]
N = np.log(a)

fig, (ax1, ax2) = plt.subplots(ncols=2)

# ax1.plot(t, a)
# ax1.plot(N, phi, c="k")
ax1.plot(N, R, c="k")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlim((5e4, 6e4))
# ax1.set_ylim((-1e-10,1e-10))
ax1.set_xlabel("$ln(a)$")
ax1.set_ylabel("$\phi/m_{pl}$")

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
print(fig_fn)
plt.savefig(fig_fn, bbox_inches="tight")

plt.close
