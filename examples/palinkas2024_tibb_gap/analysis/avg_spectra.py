import os, warnings, json, numpy as np, rhkpy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
D = os.path.join(os.environ.get("ZENODO_10931524", "zenodo_10931524"), "Figure 3 and 4", "STM data") + os.sep
F = {"3L_647": "MK_ABC_FLG_26_9K_2023_05_22_12_02_03_647.sm4", "3L_280": "MK_ABC_FLG_26_9K_2023_05_19_10_41_20_280.sm4",
     "4L_209": "MK_ABC_FLG_33_9K_2023_06_20_10_21_28_209.sm4", "4L_660": "MK_ABC_FLG_33_9K_2023_06_21_12_39_49_660.sm4"}
out = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, (k, f) in enumerate(F.items()):
    m = rhkpy.rhkdata(D + f)
    lia = m.msw().mrep().spectra.lia
    a = lia.mean(dim=["specpos_x", "specpos_y"])
    V = a.bias.values; y = a.values
    o = np.argsort(V); V, y = V[o], y[o]
    sd = lia.std(dim=["specpos_x", "specpos_y"]).values[o]
    out[k] = dict(bias=V.tolist(), didv=y.tolist(), std=sd.tolist(), li_amp_mV=float(m.spectra.attrs["LI amplitude"]),
                  setpoint_V=float(m.spectra.attrs["bias"]), setpoint_pA=float(m.spectra.attrs["setpoint"]))
    for j, lim in enumerate([(-0.8, 0.8), (-0.12, 0.12)]):
        s = (V > lim[0]) & (V < lim[1])
        ax[j].plot(V[s], y[s] / np.median(y) + 1.0 * i, label=k)
for j in range(2): ax[j].legend(); ax[j].grid(alpha=.3); ax[j].set_xlabel("bias (V)")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "avg_spectra.png"), dpi=110)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stm_3L4L_avg.json"), "w"))
