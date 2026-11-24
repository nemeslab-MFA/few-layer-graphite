"""Reproduce the S7 gap statistics of the paper with the notebook's settings (rhkpy.peakfit)."""
import os, warnings, numpy as np, rhkpy
warnings.filterwarnings("ignore")
D = os.path.join(os.environ.get("ZENODO_10931524", "zenodo_10931524"), "Figure 3 and 4", "STM data") + os.sep

def lsgl(x, x0=0, slope_below=-1, slope_above=-0.25, ampl_gau=0.15, width=0.1, ampl_step=0.2, spread=0.02, offset=0):
    y = np.zeros_like(x)
    b, a = x < x0, x >= x0
    y[b] = offset + slope_below * (x[b] - x0)
    y[a] = offset + slope_above * (x[a] - x0)
    return y + ampl_step / (np.exp((x - x0) / spread) + 1) + ampl_gau * np.exp(-2 * np.log(2) * (x - x0) ** 2 / width ** 2)

CFG = {
 "647": dict(f="MK_ABC_FLG_26_9K_2023_05_22_12_02_03_647.sm4", roll=5, h=(-0.15, -0.6), e=(0.7, 0.2),
             se=dict(x0=0.39, slope_below=0.5, slope_above=6, ampl_gau=0.3, offset=1.5),
             sh=dict(x0=-0.38, slope_below=-5.3, slope_above=-0.23, ampl_gau=0.3, offset=1.5)),
 "280": dict(f="MK_ABC_FLG_26_9K_2023_05_19_10_41_20_280.sm4", roll=5, h=(-0.15, -0.6), e=(0.7, 0.2),
             se=dict(x0=0.39, slope_below=1.7, slope_above=6, ampl_gau=0.3, offset=1.5),
             sh=dict(x0=-0.38, slope_below=-5.3, slope_above=-0.23, ampl_gau=0.3, offset=1.5)),
 "209": dict(f="MK_ABC_FLG_33_9K_2023_06_20_10_21_28_209.sm4", roll=3, h=(-0.2, -0.7), e=(0.7, 0.2),
             se=dict(x0=0.37, ampl_gau=0.2, slope_below=1.5, slope_above=4, offset=0.5),
             sh=dict(x0=-0.32, ampl_gau=0.2, slope_below=-5, slope_above=1.2, offset=0.7)),
 "660": dict(f="MK_ABC_FLG_33_9K_2023_06_21_12_39_49_660.sm4", roll=5, h=(-0.2, -0.6), e=(0.7, 0.2),
             se=dict(x0=0.37, ampl_gau=0.2, slope_below=0, slope_above=3.5, offset=1),
             sh=dict(x0=-0.34, ampl_gau=0.2, slope_below=-3.5, slope_above=1, offset=0.2)),
}
out = {}
for k, c in CFG.items():
    m = rhkpy.rhkdata(D + c["f"])
    lia = m.msw().mrep().spectra.lia
    avg = lia.rolling(specpos_x=c["roll"], specpos_y=c["roll"], center=True).mean()
    fe = rhkpy.peakfit(avg.sel(bias=slice(*c["e"])), func=lsgl, stval=c["se"], bounds={"ampl_gau": [0, 0.5]})
    fh = rhkpy.peakfit(avg.sel(bias=slice(*c["h"])), func=lsgl, stval=c["sh"], bounds={"ampl_gau": [0, 0.5]})
    xe = fe.curvefit_coefficients[:, :, 0].values
    xh = fh.curvefit_coefficients[:, :, 0].values
    sp = 1000 * (xe - xh)
    print(f"map {k}: splitting mean {np.nanmean(sp):.2f} mV, std {np.nanstd(sp):.2f} mV, n {np.isfinite(sp).sum()}, "
          f"x0_h mean {1e3*np.nanmean(xh):.1f}, x0_e mean {1e3*np.nanmean(xe):.1f}", flush=True)
