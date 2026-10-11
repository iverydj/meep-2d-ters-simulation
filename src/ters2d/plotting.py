"""Figures (matplotlib). Result plots read the compact CSV/NPZ tables; schematics are computed.

Colors: categorical slots in fixed order (blue, orange, aqua; at most three series per
panel, always with a legend and distinct markers), one-hue blue ramp for magnitudes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .results import FieldMap, read_csv

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METAL, METAL_EDGE, DIEL, LAYER = "#e8c766", "#8a6d1d", "#d7e6f5", "#eb6834"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.titlesize": 9.5,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blue_ramp", ["#ffffff"] + BLUE_RAMP)


def save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _tex(name: str) -> str:
    return name.replace("WS2", "WS$_2$").replace("SiO2", "SiO$_2$")


def _fem_rows(fem_csv: Path, group: str, sample: str) -> dict:
    """Rows of the FEM reference table for one group and sample."""
    f = read_csv(fem_csv)
    m = (np.array(f["group"]) == group) & (np.array(f["sample"]) == sample)
    if not m.any():
        raise ValueError(f"no FEM rows for group {group!r}, sample {sample!r}")
    return {k: np.asarray(v)[m] for k, v in f.items()}


# ----------------------------------------------------------------------------- results


def plot_spectra(
    cases: list[tuple[str, Path, str]],
    out_png: Path,
    fem_csv: Path | None = None,
    fem_group: str = "spectrum",
    title_suffix: str = "",
) -> Path:
    """EF (relative to the tip-free field) and |E|^2 relative to the incident beam vs wavelength.

    cases: [(label, FDTD csv, FEM sample name or ''), ...]; FDTD markers, FEM lines (rows of
    ``fem_group`` in the FEM table).
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for i, (label, csv_path, fem_sample) in enumerate(cases):
        d = read_csv(csv_path)
        for ax, key in zip(axes, ("EF_probe", "I_probe_tip_over_inc")):
            ax.plot(d["wavelength_nm"], d[key], color=SERIES[i], ls="none", marker="os^"[i], ms=4, label=f"{label}, FDTD")
            if fem_csv is not None and fem_sample:
                r = _fem_rows(fem_csv, fem_group, fem_sample)
                o = np.argsort(r["wavelength_nm"])
                ax.plot(r["wavelength_nm"][o], r[key][o], color=SERIES[i], lw=1.0, label=f"{label}, FEM")
    axes[0].set_ylabel(r"EF $=|E|^2/|E_\mathrm{ref}|^2$ at gap centre")
    axes[1].set_ylabel(r"$|E|^2/|E_\mathrm{inc}(\mathrm{focus})|^2$ at gap centre")
    axes[0].set_title("(a) relative to the tip-free field" + title_suffix)
    axes[1].set_title("(b) relative to the incident beam" + title_suffix)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("wavelength (nm)")
    axes[0].legend(fontsize=7)
    return save(fig, out_png)


def plot_sweep(
    csv_path: Path,
    xkey: str,
    xlabel: str,
    out_png: Path,
    fem_csv: Path | None = None,
    fem_group: str = "",
    flag_below: float | None = None,
    logx: bool = False,
    logy: bool = True,
) -> Path:
    """EF at the gap centre vs one parameter (one wavelength): FDTD markers, FEM line."""
    plt = _plt()
    d = read_csv(csv_path)
    wl = d["wavelength_nm"][0]
    if np.any(d["wavelength_nm"] != wl):
        raise ValueError("plot_sweep expects a single wavelength")
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    flag = d[xkey] < flag_below if flag_below is not None else np.zeros(d[xkey].shape, bool)
    ax.plot(d[xkey][~flag], d["EF_probe"][~flag], color=SERIES[0], ls="none", marker="o", ms=4.5, label=f"FDTD, {wl:.0f} nm")
    if flag.any():
        ax.plot(
            d[xkey][flag],
            d["EF_probe"][flag],
            color=SERIES[0],
            ls="none",
            marker="o",
            mfc="white",
            ms=4.5,
            label="FDTD, gap = 1 cell",
        )
    if fem_csv is not None:
        r = _fem_rows(fem_csv, fem_group, "Au")
        o = np.argsort(r[xkey])
        ax.plot(r[xkey][o], r["EF_probe"][o], color=SERIES[0], lw=1.0, label="FEM reference")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"EF $=|E|^2/|E_\mathrm{ref}|^2$ at gap centre")
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if logx:
        ax.set_xscale("log")
        ticks = sorted(set(d[xkey]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.legend(fontsize=7.5)
    return save(fig, out_png)


def plot_ws2(csv_path: Path, out_png: Path, fem_csv: Path | None = None) -> Path:
    """EF(WS2/Au) / EF(Au/Au) vs wavelength (FDTD small domain; FEM production and small
    domain), and the in-plane absorption of the WS2 model."""
    from .materials import WS2_MONOLAYER

    plt = _plt()
    d = read_csv(csv_path)
    fig, axes = plt.subplots(
        2, 1, figsize=(4.6, 4.4), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    if fem_csv is not None:
        for group, ls, label in (("ws2", "-", "FEM, production domain"), ("ws2_small", "--", "FEM, small domain")):
            ws, au = _fem_rows(fem_csv, group, "WS2/Au"), _fem_rows(fem_csv, group, "Au")
            if not np.array_equal(ws["wavelength_nm"], au["wavelength_nm"]):
                raise ValueError("FEM WS2 and Au rows do not match")
            axes[0].plot(ws["wavelength_nm"], ws["EF_probe"] / au["EF_probe"], color=SERIES[0], ls=ls, lw=1.0, label=label)
    o = np.argsort(d["wavelength_nm"])
    axes[0].plot(
        d["wavelength_nm"][o], d["ratio_EF_probe"][o], color=SERIES[1], ls="none", marker="s", ms=4.5, label="FDTD, small domain"
    )
    axes[0].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    axes[0].set_ylabel("EF(WS$_2$/Au) / EF(Au/Au)")
    axes[0].legend(fontsize=7)
    wl = np.linspace(480, 900, 300)
    axes[1].plot(wl, np.imag(WS2_MONOLAYER.in_plane.epsilon(wl)), color=MUTED, lw=1.2)
    axes[1].set_ylabel(r"Im $\varepsilon_\parallel$ (WS$_2$)")
    axes[1].set_xlabel("wavelength (nm)")
    return save(fig, out_png)


def plot_resolution_series(csv_path: Path, out_png: Path) -> Path:
    """EF relative to the finest grid vs grid spacing (small domain)."""
    plt = _plt()
    d = read_csv(csv_path)
    case = np.array(d["case"])
    keys = list(dict.fromkeys(zip(case, d["wavelength_nm"])))
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    colors = SERIES + ["#e87ba4"]
    for i, (c, wl) in enumerate(keys):
        m = (case == c) & (d["wavelength_nm"] == wl)
        o = np.argsort(d["dx_nm"][m])
        x, y = d["dx_nm"][m][o], d["EF_probe"][m][o]
        g = d["gap_nm"][m][0]
        label = f"{_tex(c.split('_')[1].replace('-', '/'))}, gap {g:.0f} nm, {wl:.0f} nm"
        ax.plot(x, y / y[0], color=colors[i], marker="os^Dv"[i], ms=4, label=label)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.text(1.0, ax.get_ylim()[1], " production", fontsize=7, color=INK2, va="top")
    ax.set_xlabel("grid spacing (nm)")
    ax.set_ylabel("EF / EF(finest grid)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.8)
    return save(fig, out_png)


def plot_field_map(fmap: FieldMap | Path, out_png: Path, vmin: float = 1e-2) -> Path:
    """|E|^2 / |E_ref(probe)|^2 around the apex (log scale)."""
    from matplotlib.colors import LogNorm

    if not isinstance(fmap, FieldMap):
        fmap = FieldMap.load_npz(fmap)
    plt = _plt()
    x, y, e2 = fmap.x_nm, fmap.y_nm, fmap.e2_rel
    fig, ax = plt.subplots(figsize=(5.4, 2.8), constrained_layout=True)
    finite = e2[np.isfinite(e2)]
    im = ax.pcolormesh(
        x, y, e2.T, shading="nearest", cmap=blue_cmap(), norm=LogNorm(vmin=max(finite.min(), vmin), vmax=finite.max())
    )
    o = fmap.outline_nm
    ax.fill(o[:, 0], o[:, 1], color="#d9d7cf", ec=INK2, lw=0.6)
    layers = np.asarray(fmap.layers_nm).reshape(-1, 2)
    sub_top = layers[0][0] if layers.size else 0.0
    ax.axhspan(y.min(), sub_top, color="#d9d7cf")
    for y0, y1 in layers:
        ax.axhline(y1, color=LAYER, lw=0.6)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    m = fmap.meta
    pol = m.get("polarization", "p")
    if m.get("normalization", "reference") == "incident":
        cb.set_label(r"$|E|^2 / |E_\mathrm{inc}(\mathrm{focus})|^2$")
        what = f"$|E|^2/|E_\\mathrm{{inc}}|^2$ at gap centre = {m['I_probe_tip_over_inc']:.3g}"
    else:
        cb.set_label(r"$|E|^2 / |E_\mathrm{ref}(\mathrm{gap\ centre})|^2$")
        what = f"EF at gap centre = {m['EF_probe']:.0f}"
    ax.set_title(f"{_tex(m['label'])}, {pol}-pol., {m['wavelength_nm']:.0f} nm, {what}")
    return save(fig, out_png)


def plot_fresnel(csv_path: Path, out_png: Path, incidence_deg: float = 60.0) -> Path:
    """Tip-free reflectance at the beam's central angle: Meep (markers) vs analytic (lines)."""
    plt = _plt()
    d = read_csv(csv_path)
    sel = d["incidence_deg"] == incidence_deg
    samples = list(dict.fromkeys(np.array(d["sample"])[sel]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for i, s in enumerate(samples):
        m = sel & (np.array(d["sample"]) == s)
        o = np.argsort(d["wavelength_nm"][m])
        wl = d["wavelength_nm"][m][o]
        axes[0].plot(wl, d["R_fresnel"][m][o], color=SERIES[i], lw=2.4, alpha=0.35)
        axes[0].plot(wl, d["R_sim"][m][o], color=SERIES[i], ls="none", marker="os^"[i], ms=3.5, label=_tex(s))
        axes[1].plot(wl, d["abs_r_err"][m][o], color=SERIES[i], marker="os^"[i], ms=3.5, label=_tex(s))
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"reflectance at {incidence_deg:.0f}° (line: analytic)")
    axes[1].set_ylabel(r"$|r_\mathrm{Meep} - r_\mathrm{analytic}|$")
    for ax in axes:
        ax.set_xlabel("wavelength (nm)")
    axes[0].legend(fontsize=7.5)
    return save(fig, out_png)


def plot_nanogap(fdtd_csv: Path, fem_csv: Path, out_png: Path) -> Path:
    """Cylinder above Au: FDTD EF relative to the converged curved-element FEM value."""
    plt = _plt()
    d = read_csv(fdtd_csv)
    fem = read_csv(fem_csv)

    def ref(g, r, wl):
        m = (fem["gap_nm"] == g) & (fem["radius_nm"] == r) & (fem["wavelength_nm"] == wl) & (fem["half_width_um"] == 0.5)
        if not m.any():
            raise ValueError(f"no FEM reference for gap {g}, R {r}, {wl} nm")
        return fem["EF_probe"][m][np.argmax(fem["ndof"][m])]

    ratio = np.array([e / ref(g, r, w) for e, g, r, w in zip(d["EF_probe"], d["gap_nm"], d["radius_nm"], d["wavelength_nm"])])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    series = list(dict.fromkeys(zip(d["gap_nm"], d["radius_nm"], d["wavelength_nm"])))
    i = 0
    for g, r, w in series:
        m = (d["gap_nm"] == g) & (d["radius_nm"] == r) & (d["wavelength_nm"] == w)
        if m.sum() < 2:
            continue
        o = np.argsort(d["dx_nm"][m])
        axes[0].plot(
            d["dx_nm"][m][o],
            ratio[m][o],
            color=SERIES[i],
            marker="os^"[i],
            ms=4,
            label=f"gap {g:.0f} nm, R {r:.0f} nm, {w:.0f} nm",
        )
        i += 1
    m = d["resolution"] == 1000
    for j, r in enumerate(sorted(set(d["radius_nm"][m]))):
        mm = m & (d["radius_nm"] == r) & (d["wavelength_nm"] == 633.0)
        o = np.argsort(d["gap_nm"][mm])
        axes[1].plot(d["gap_nm"][mm][o], ratio[mm][o], color=SERIES[j], ls="none", marker="os^"[j], ms=5, label=f"R {r:.0f} nm")
    for ax in axes:
        ax.axhline(1.0, color=INK2, lw=1.0, ls="--")
        ax.set_ylabel(r"EF$_\mathrm{FDTD}$ / EF$_\mathrm{FEM}$")
    axes[0].set_xlabel("FDTD grid spacing (nm)")
    axes[0].set_xlim(left=0)
    axes[0].set_title("(a) grid convergence")
    axes[1].set_xlabel("gap (nm)")
    axes[1].set_title("(b) 1 nm grid (production), 633 nm")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    return save(fig, out_png)


def plot_lorentz_fit(csv_path: Path, out_png: Path) -> Path:
    """n, k data vs passive Lorentz fit (columns component, wavelength_um, n_data, k_data, n_fit, k_fit)."""
    plt = _plt()
    d = read_csv(csv_path)
    comp = np.array(d["component"])
    tags = list(dict.fromkeys(comp))
    fig, axes = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 2.8), constrained_layout=True, squeeze=False)
    for ax, tag in zip(axes[0], tags):
        m = comp == tag
        wl = d["wavelength_um"][m] * 1e3
        ax.plot(wl, d["n_data"][m], color=SERIES[0], lw=2.6, alpha=0.35, label="n data")
        ax.plot(wl, d["n_fit"][m], color=SERIES[0], lw=1.2, label="n fit")
        ax.plot(wl, d["k_data"][m], color=SERIES[1], lw=2.6, alpha=0.35, label="k data")
        ax.plot(wl, d["k_fit"][m], color=SERIES[1], lw=1.2, ls="--", label="k fit")
        ax.set_xlabel("wavelength (nm)")
        ax.set_title({"inplane": "in-plane (monolayer data)", "outofplane": "out-of-plane (bulk data)"}.get(tag, tag))
    axes[0][0].set_ylabel("n, k")
    axes[0][0].legend(ncol=2, fontsize=7.5)
    return save(fig, out_png)


# ----------------------------------------------------------------------------- schematics


def _draw_domain(ax, layout, annotate=True):
    """Free region, PML, conductor-in-PML, absorbers, drive ramp, tip and sample of a Layout (um)."""
    from matplotlib.patches import Polygon, Rectangle

    m = layout.model
    num = m.numerics
    (cx0, cy0), (cx1, cy1) = layout.cell_lo, layout.cell_hi
    (px0, py0), (px1, py1) = layout.phys_lo, layout.phys_hi
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fc="#efeee9", ec=AXIS, lw=0.8))
    ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fc="white", ec=INK2, lw=0.8))
    sub_col = METAL if layout.substrate_metallic else DIEL
    top = -layout.thickness
    ax.add_patch(
        Rectangle(
            (px0, cy0 if not layout.bottom_pml else py0),
            px1 - px0,
            top - (cy0 if not layout.bottom_pml else py0),
            fc=sub_col,
            ec="none",
        )
    )
    hatch = dict(fc="none", ec=SERIES[2], lw=0, hatch="////", alpha=0.75)
    ax.add_patch(Rectangle((px0, layout.shaft_absorber_start), px1 - px0, py1 - layout.shaft_absorber_start, **hatch))
    if layout.side_absorbers:
        w = px1 - num.side_absorber_start_um
        for x0 in (px0, num.side_absorber_start_um):
            ax.add_patch(Rectangle((x0, 0.0), w, layout.side_absorber_top, **hatch))
    tip = layout.tip
    o = tip.outline(py1)
    ax.add_patch(Polygon(o, closed=True, fc=METAL, ec=METAL_EDGE, lw=0.6))
    hw = float(tip.half_width(py1))
    hw2 = float(tip.half_width(cy1))
    cond = "#b9a36a"
    ax.add_patch(Polygon([[-hw, py1], [hw, py1], [hw2, cy1], [-hw2, cy1]], closed=True, fc=cond, ec=METAL_EDGE, lw=0.6))
    if layout.substrate_metallic:
        for x0, x1 in ((cx0, px0), (px1, cx1)):
            ax.add_patch(Rectangle((x0, cy0), x1 - x0, top - cy0, fc=cond, ec="none"))
    r0, r1 = layout.drive_ramp
    ax.plot([0, 0], [0.12, r0], color=SERIES[1], lw=2.4, solid_capstyle="butt")
    ax.plot([0, 0], [r0, r1], color=SERIES[1], lw=2.4, ls=(0, (1.5, 1)), solid_capstyle="butt")
    if annotate:
        kw = dict(fontsize=6.6, color=INK2)
        box = dict(fc="white", ec="none", pad=0.6, alpha=0.9)
        ax.text(0.5 * (cx0 + cx1), cy1 - 0.5 * (cy1 - py1), "PML (shaft: conductor)", ha="center", va="center", **kw)
        ax.text(cx1 - 0.5 * (cx1 - px1), 0.5 * (py0 + py1), "PML", rotation=90, ha="center", va="center", **kw)
        ax.text(cx0 + 0.5 * (px0 - cx0), 0.5 * (py0 + py1), "PML", rotation=90, ha="center", va="center", **kw)
        ax.text(
            px0 + 0.05,
            0.5 * (layout.shaft_absorber_start + py1),
            "shaft\nplasmon\nabsorber",
            ha="left",
            va="center",
            bbox=box,
            **kw,
        )
        if layout.side_absorbers:
            ax.text(
                px1 - 0.05, layout.side_absorber_top + 0.08, "side plasmon\nabsorber", ha="right", va="bottom", bbox=box, **kw
            )
        ax.text(0.06, 0.5 * (r0 + r1), "drive\nramp", ha="left", va="center", color=SERIES[1], fontsize=6.6, bbox=box)
        ax.text(
            0.06,
            0.5 * (0.12 + r0),
            "tip driven by\nJ = −iω(ε−1)E$_\\mathrm{bg}$",
            ha="left",
            va="center",
            color=SERIES[1],
            fontsize=6.6,
            bbox=box,
        )
        lab = f"{_tex(m.sample.substrate.name)} substrate" + ("" if layout.bottom_pml else " (no bottom PML)")
        ax.text(
            0.5 * (px0 + px1), 0.5 * (top + (cy0 if not layout.bottom_pml else py0)), lab, ha="center", va="center", fontsize=7.5
        )
        if layout.bottom_pml:
            ax.text(0.5 * (cx0 + cx1), 0.5 * (cy0 + py0), "PML", ha="center", va="center", **kw)
    ax.set_xlim(cx0, cx1)
    ax.set_ylim(cy0, cy1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")


def _draw_beam_arrows(ax, ill, focus=(0.0, 0.0), length=1.25):
    th0 = np.deg2rad(ill.incidence_deg)
    ap = np.deg2rad(ill.divergence_deg)
    fx, fy = focus
    for dth, lw in ((-ap, 0.8), (0.0, 1.6), (ap, 0.8)):
        th = th0 + dth
        ax.annotate(
            "",
            xy=(fx - 0.08 * np.sin(th), fy + 0.08 * np.cos(th)),
            xytext=(fx - length * np.sin(th), fy + length * np.cos(th)),
            arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=lw),
        )


def _draw_apex(ax, layout, half_width_nm=40.0, height_nm=62.0):
    """Apex, gap, layers and probe in nm."""
    from matplotlib.patches import Polygon

    m = layout.model
    tip = layout.tip
    o = tip.outline(height_nm * 1e-3) * 1e3
    ax.add_patch(Polygon(o, closed=True, fc=METAL, ec=METAL_EDGE, lw=0.6))
    top = -layout.thickness * 1e3
    ax.axhspan(-15, top, color=METAL if layout.substrate_metallic else DIEL)
    for lay, (y0, y1) in zip(m.sample.layers, layout.layers):
        ax.axhspan(y0 * 1e3, y1 * 1e3, color=LAYER, alpha=0.55, lw=0)
    ax.plot(layout.probe[0] * 1e3, layout.probe[1] * 1e3, "o", color=SERIES[0], ms=4)
    ax.annotate(
        "probe (gap centre)",
        xy=(layout.probe[0] * 1e3, layout.probe[1] * 1e3),
        xytext=(8, -9),
        fontsize=7.5,
        color=SERIES[0],
        arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.6),
    )
    R = tip.radius * 1e3
    a = np.degrees(tip.half_angle)
    ax.text(0, 0.62 * height_nm, f"R = {R:g} nm\nhalf-angle {a:g}°", fontsize=7.5, color=INK2, ha="center")
    txt = f"air gap {m.gap_nm:.3g} nm"
    if m.sample.layers:
        txt += "\nlayer " + " + ".join(f"{(y1 - y0) * 1e3:.3g}" for y0, y1 in layout.layers) + " nm"
    ax.text(-half_width_nm + 1.5, 0.93 * height_nm, txt, fontsize=7.5, color=INK, va="top")
    ax.set_xlim(-half_width_nm, half_width_nm)
    ax.set_ylim(-10, height_nm)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")


def plot_geometry(layout, wavelength_nm: float, out_png: Path) -> Path:
    """(a) domain with boundaries and absorbers, (b) the analytic tip-free field |E_bg|^2, (c) apex and gap."""
    from .background import beam_fields_grid
    from .sources import background_beam

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    _draw_domain(axes[0], layout)
    _draw_beam_arrows(axes[0], layout.model.illumination)
    axes[0].set_title("(a) isolated tip, boundaries and absorbers")
    ax = axes[1]
    (px0, py0), (px1, py1) = layout.phys_lo, layout.phys_hi
    xs = np.linspace(px0, px1, 281)
    ys = np.linspace(0.0005, py1, 341)
    _, ex, ey = beam_fields_grid(background_beam(layout, wavelength_nm), layout.stack(wavelength_nm), xs, ys)
    e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    im = ax.pcolormesh(xs, ys, (e2 / e2.max()).T, shading="nearest", cmap=blue_cmap(), vmin=0, vmax=1)
    o = layout.tip.outline(py1)
    ax.plot(o[:, 0], o[:, 1], color=INK, lw=0.7, ls="--")
    ax.axhspan(py0, 0, color=METAL if layout.substrate_metallic else DIEL)
    ax.set_xlim(px0, px1)
    ax.set_ylim(py0, py1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (µm)")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label(r"$|E_\mathrm{bg}|^2$ (normalized)")
    ax.set_title(f"(b) tip-free beam at {wavelength_nm:.0f} nm (= reference)")
    _draw_apex(axes[2], layout)
    axes[2].set_title("(c) apex and gap")
    return save(fig, out_png)


def plot_scattered_field_idea(layout, wavelength_nm: float, fmap: FieldMap | Path, out_png: Path) -> Path:
    """E = E_bg + E_s: (a) analytic tip-free field, (b) the tip's polarization current that drives Meep,
    (c) the total field of a finished run."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    from .background import beam_fields_grid
    from .sources import background_beam, drive_weight

    if not isinstance(fmap, FieldMap):
        fmap = FieldMap.load_npz(fmap)
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), constrained_layout=True)
    beam, stack = background_beam(layout, wavelength_nm), layout.stack(wavelength_nm)
    xs = np.linspace(-0.15, 0.15, 301)
    ys = np.linspace(0.0, 0.12, 121)
    _, ex, ey = beam_fields_grid(beam, stack, xs, ys)
    e2b = np.abs(ex) ** 2 + np.abs(ey) ** 2
    probe_ref = float(np.interp(layout.probe[1], ys, e2b[np.argmin(np.abs(xs))]))
    o = layout.tip.outline(0.12) * 1e3
    norm = LogNorm(vmin=1e-2, vmax=float(np.nanmax(fmap.e2_rel)))
    ax = axes[0]
    ax.pcolormesh(xs * 1e3, ys * 1e3, (e2b / probe_ref).T, shading="nearest", cmap=blue_cmap(), norm=norm)
    ax.plot(o[:, 0], o[:, 1], color=INK, lw=0.7, ls="--")
    ax.set_title("(a) $E_\\mathrm{bg}$: beam + exact Fresnel response\n(analytic, no tip)")
    ax = axes[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    inside = layout.tip.contains(X, Y)
    jmag = np.where(inside, np.sqrt(e2b) * drive_weight(Y, layout.drive_ramp), np.nan)
    ax.add_patch(Polygon(o, closed=True, fc="none", ec=INK2, lw=0.6))
    ax.pcolormesh(xs * 1e3, ys * 1e3, jmag.T, shading="nearest", cmap="Oranges")
    ax.set_title(
        "(b) Meep: $E_s$ driven by $J=-i\\omega(\\varepsilon_\\mathrm{tip}-1)E_\\mathrm{bg}$\non the tip's own grid points"
    )
    ax = axes[2]
    im = ax.pcolormesh(fmap.x_nm, fmap.y_nm, fmap.e2_rel.T, shading="nearest", cmap=blue_cmap(), norm=norm)
    ax.add_patch(Polygon(fmap.outline_nm, closed=True, fc="#d9d7cf", ec=INK2, lw=0.6))
    ax.set_title(f"(c) total $E=E_\\mathrm{{bg}}+E_s$ (EF = {fmap.meta['EF_probe']:.0f})")
    fig.colorbar(im, ax=axes[2], shrink=0.85).set_label(r"$|E|^2/|E_\mathrm{ref}(\mathrm{gap\ centre})|^2$ (a, c)")
    for ax in axes:
        ax.axhspan(-10, 0, color=METAL if layout.substrate_metallic else DIEL)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-10, 120)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    return save(fig, out_png)


def plot_boundaries(layout, out_png: Path) -> Path:
    """Boundary and absorber layout with the absorber conductivity profile and the drive ramp."""
    from .sources import drive_weight

    plt = _plt()
    fig = plt.figure(figsize=(8.6, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    _draw_domain(ax, layout)
    ax.set_title("(a) boundary treatment of the production domain")
    num = layout.model.numerics
    ax1 = fig.add_subplot(gs[0, 1])
    n = num.absorber_segments
    s = np.linspace(0, 1, 400)
    ax1.plot(s, num.absorber_conductivity * s**num.absorber_power, color=MUTED, lw=1.0, label="continuous (FEM)")
    k = np.minimum((s * n).astype(int), n - 1)
    ax1.step(
        s,
        num.absorber_conductivity * ((k + 0.5) / n) ** num.absorber_power,
        where="mid",
        color=SERIES[2],
        lw=1.4,
        label=f"{n} steps (FDTD)",
    )
    ax1.set_xlabel("depth into the absorber / absorber length")
    ax1.set_ylabel("σ$_D$ = σ$_B$ (Meep units)")
    ax1.set_title(f"(b) matched lossy-air absorber, σ ∝ (s/L)$^{num.absorber_power}$")
    ax1.legend(fontsize=7)
    ax2 = fig.add_subplot(gs[1, 1])
    y = np.linspace(0, layout.y_top, 600)
    ax2.plot(y, drive_weight(y, layout.drive_ramp), color=SERIES[1], lw=1.6)
    ax2.axvspan(layout.shaft_absorber_start, layout.y_top, color=SERIES[2], alpha=0.12, lw=0)
    ax2.text(layout.shaft_absorber_start + 0.05, 0.5, "shaft\nabsorber", fontsize=7, color=INK2)
    ax2.set_xlabel("height on the shaft y (µm)")
    ax2.set_ylabel("drive weight")
    ax2.set_title("(c) scattered-field drive of the shaft (cos² ramp)")
    return save(fig, out_png)


def plot_beam_synthesis(ill, wavelength_nm: float, stack, out_png: Path) -> Path:
    """Gaussian angular spectrum and the resulting incident and total tip-free beam."""
    from .background import GaussianBeam, Stack, beam_fields_grid

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 1, 1]})
    beam = GaussianBeam(wavelength_nm * 1e-3, ill.incidence_deg, ill.divergence_deg, (0.0, 0.0), ill.n_waves)
    th, w = beam.angles_weights()
    ax = axes[0]
    ax.plot(np.degrees(th), w / w.max(), color=SERIES[0], lw=1.6)
    for v in (-1, 1):
        ax.axvline(ill.incidence_deg + v * ill.divergence_deg, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("plane-wave angle θ$_j$ (deg)")
    ax.set_ylabel("weight exp(−((θ−θ$_0$)/θ$_d$)²)")
    ax.set_title(f"(a) {ill.n_waves} plane waves, cut at θ$_0$ ± 2θ$_d$")
    xs = np.linspace(-1.8, 1.8, 241)
    ys = np.linspace(-0.2, 3.0, 241)
    _, ix, iy = beam_fields_grid(beam, Stack.isotropic(1.0), np.array([0.0]), np.array([0.0]))
    i0 = float(abs(ix[0, 0]) ** 2 + abs(iy[0, 0]) ** 2)
    panels = (
        (axes[1], Stack.isotropic(1.0), "(b) incident beam alone (no sample)"),
        (axes[2], stack, "(c) with the sample: exact reflection"),
    )
    for ax, st, title in panels:
        _, ex, ey = beam_fields_grid(beam, st, xs, ys)
        e2 = (np.abs(ex) ** 2 + np.abs(ey) ** 2) / i0
        e2 = np.where(ys[None, :] < 0, np.nan, e2) if st is stack else e2
        im = ax.pcolormesh(xs, ys, e2.T, shading="nearest", cmap=blue_cmap(), vmin=0)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x (µm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(r"$|E|^2/|E_\mathrm{inc}(\mathrm{focus})|^2$")
    axes[1].set_ylabel("y (µm)")
    return save(fig, out_png)


def plot_p_vs_s(cases: list, out_png: Path, fem_csv: Path) -> Path:
    """|E|^2 at the gap centre relative to the incident beam, p and s, vs wavelength.

    cases: [(label, color slot, p FDTD csv, s FDTD csv, FEM sample), ...]; color = sample,
    filled markers / solid FEM line = p, open markers / dashed FEM line = s.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    for label, slot, p_csv, s_csv, fem_sample in cases:
        color = SERIES[slot]
        for pol, csv_path, group, ls, mfc in (("p", p_csv, "spectrum", "-", color), ("s", s_csv, "spectrum_s", "--", "white")):
            r = _fem_rows(fem_csv, group, fem_sample)
            o = np.argsort(r["wavelength_nm"])
            ax.plot(r["wavelength_nm"][o], r["I_probe_tip_over_inc"][o], color=color, ls=ls, lw=1.0, label=f"{label}, {pol}, FEM")
            d = read_csv(csv_path)
            ax.plot(
                d["wavelength_nm"],
                d["I_probe_tip_over_inc"],
                color=color,
                ls="none",
                marker="o",
                mfc=mfc,
                ms=4,
                label=f"{label}, {pol}, FDTD",
            )
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(r"$|E|^2/|E_\mathrm{inc}(\mathrm{focus})|^2$ at gap centre")
    ax.legend(fontsize=6.5, ncol=2)
    return save(fig, out_png)


def plot_field_maps_shared(maps: list, titles: list, out_png: Path, vmin: float = 1e-4) -> Path:
    """Several field maps (same normalization) side by side on one log color scale."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    plt = _plt()
    fig, axes = plt.subplots(len(maps), 1, figsize=(5.6, 2.5 * len(maps)), constrained_layout=True, squeeze=False)
    vmax = max(float(np.nanmax(m.e2_rel)) for m in maps)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for ax, fmap, title in zip(axes[:, 0], maps, titles):
        im = ax.pcolormesh(fmap.x_nm, fmap.y_nm, fmap.e2_rel.T, shading="nearest", cmap=blue_cmap(), norm=norm)
        ax.add_patch(Polygon(fmap.outline_nm, closed=True, fc="#d9d7cf", ec=INK2, lw=0.6))
        ax.axhspan(float(fmap.y_nm.min()), 0.0, color="#d9d7cf")
        ax.set_xlim(fmap.x_nm.min(), fmap.x_nm.max())
        ax.set_ylim(fmap.y_nm.min(), fmap.y_nm.max())
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_ylabel("y (nm)")
        ax.set_title(title)
    axes[-1, 0].set_xlabel("x (nm)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8).set_label(r"$|E|^2 / |E_\mathrm{inc}(\mathrm{focus})|^2$")
    return save(fig, out_png)
