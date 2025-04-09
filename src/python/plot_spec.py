import matplotlib.pyplot as plt
import numpy as np

from os.path import join
from pathlib import Path

dn = "../data/Chaotic2/"
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

plt.tight_layout()
out_dn = dn.replace("data", "figs")
Path(out_dn).mkdir(parents=True, exist_ok=True)
fig_fn = join(out_dn, "specs.pdf")
plt.savefig(fig_fn)
plt.close()
