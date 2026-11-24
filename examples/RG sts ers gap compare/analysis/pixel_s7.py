"""Per-pixel S7 gaps of the four maps with the multi-start fitter (rolling means as in the Zenodo notebook)."""
import os, warnings, sys, json, numpy as np, rhkpy
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s7
D = os.path.join(os.environ.get("ZENODO_10931524", "zenodo_10931524"), "Figure 3 and 4", "STM data") + os.sep
CFG = {"3L_647": ("MK_ABC_FLG_26_9K_2023_05_22_12_02_03_647.sm4", 5, (-0.6, -0.15), (0.2, 0.7)),
       "3L_280": ("MK_ABC_FLG_26_9K_2023_05_19_10_41_20_280.sm4", 5, (-0.6, -0.15), (0.2, 0.7)),
       "4L_209": ("MK_ABC_FLG_33_9K_2023_06_20_10_21_28_209.sm4", 3, (-0.7, -0.2), (0.2, 0.7)),
       "4L_660": ("MK_ABC_FLG_33_9K_2023_06_21_12_39_49_660.sm4", 5, (-0.6, -0.2), (0.2, 0.7))}
out = {}
for k, (f, roll, wh, we) in CFG.items():
    lia = rhkpy.rhkdata(D + f).msw().mrep().spectra.lia
    avg = lia.rolling(specpos_x=roll, specpos_y=roll, center=True).mean().transpose("specpos_y", "specpos_x", "bias")
    V = avg.bias.values; o = np.argsort(V); V = V[o]; A = avg.values[:, :, o]
    sp = []
    for iy in range(A.shape[0]):
        for ix in range(A.shape[1]):
            y = A[iy, ix]
            if not np.all(np.isfinite(y)):
                continue
            h = s7.fit_edge(V, y, wh, n_seeds=9); e = s7.fit_edge(V, y, we, n_seeds=9)
            sp.append((h, e))
    sp = np.array(sp); g = 1e3 * (sp[:, 1] - sp[:, 0])
    Va = V; ya = A.reshape(-1, V.size); ya = ya[np.all(np.isfinite(ya), axis=1)].mean(axis=0)
    ha, ea, ga = s7.gap(Va, ya, wh, we)
    out[k] = dict(mean=float(np.nanmean(g)), std=float(np.nanstd(g)), median=float(np.nanmedian(g)), n=int(np.isfinite(g).sum()),
                  h=float(1e3 * np.nanmean(sp[:, 0])), e=float(1e3 * np.nanmean(sp[:, 1])), avg_gap=1e3 * ga, gaps=g.tolist())
    print(f"{k}: per-pixel gap mean {out[k]['mean']:.1f} median {out[k]['median']:.1f} std {out[k]['std']:.1f} n {out[k]['n']} | "
          f"x0_h {out[k]['h']:+.1f} x0_e {out[k]['e']:+.1f} | gap of the averaged (rolled) spectrum {1e3*ga:.1f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixel_s7.json"), "w"))
