"""Shared figure style.

Colour choices are Okabe-Ito, which is the standard palette that stays distinguishable under
deuteranopia, protanopia and tritanopia as well as in greyscale print. Two rules follow from that
and are enforced by the builders rather than left to good intentions:

1. **Colour never carries information alone.** Every series that a reader must tell apart also
   differs in hatch (bars), marker (lines) or position, so the figure survives a monochrome
   printout and any form of colour blindness.
2. **Related quantities differ in lightness, not only hue.** `rho_null` and `rho_seed` are the two
   controls a reader must compare, so they are given a dark/light pair that separates even when hue
   is lost entirely.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe & Ito (2008), "Color Universal Design". Safe under all common colour-vision deficiencies.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

PALETTE = {
    "delta": OKABE_ITO["vermillion"],   # the measured effect
    "rho_null": OKABE_ITO["blue"],      # the control the paper argues for
    "rho_seed": OKABE_ITO["sky"],       # the control prior work uses; light pair with rho_null
    # A neutral grey fails here: under deuteranopia the sky blue collapses onto grey
    # (separation 0.12). Okabe-Ito yellow separates from all three others by >=0.37
    # under every simulated deficiency and in greyscale.
    "nu": OKABE_ITO["yellow"],
    "omega": OKABE_ITO["green"],        # warranted change
    "accent": OKABE_ITO["orange"],
}

# Redundant encoding, so the bars remain readable in greyscale and under any CVD.
HATCH = {"nu": "", "rho_null": "//", "rho_seed": "..", "delta": "xx"}
BAR_EDGE = "0.25"   # dark edges: hatch takes the edge colour, and it must read in print
MARKER = {"delta": "o", "rho_null": "s", "rho_seed": "^", "omega": "D", "nu": "v"}

QUANTITY_LABEL = {
    "nu": r"explainer noise $\nu$",
    "rho_null": r"matched null $\rho_{\mathrm{null}}$",
    "rho_seed": r"seed floor $\rho_{\mathrm{seed}}$",
    "delta": r"shift-induced $\Delta$",
}

EXPLAINER_LABEL = {
    "integrated_gradients": "IG",
    "expected_gradients": "EG",
    "gradient_x_input": "Grad$\\times$Input",
    "saliency": "Saliency",
    "smoothgrad": "SmoothGrad",
    "kernel_shap": "KernelSHAP",
    "lime": "LIME",
    "grad_cam": "Grad-CAM",
}
FAMILY_LABEL = {
    "none": "no shift (placebo)",
    "covariate": "covariate",
    "concept": "concept",
    "shortcut": "shortcut removal",
}


def use_style():
    """Type is sized so the figure can sit at ICLR's ~5.5in text width without shrinking.

    Figures are authored close to their printed size rather than large-and-downscaled: a
    7in-wide figure placed at 5.5in shrinks every label by 0.79x, which is how 8.5pt type ends
    up at 6.7pt on the page.
    """
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 400,          # camera-ready; 200 visibly softens thin rules in print
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 10.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlepad": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,      # grid behind the data, never over it
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9.0,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "lines.linewidth": 1.3,
        "hatch.linewidth": 0.6,
        "patch.linewidth": 0.4,
    })


def label(name: str) -> str:
    return EXPLAINER_LABEL.get(name, name)


def explainer_ticks(ax, names, rotation=30):
    """Explainer names are long and collide at 18 degrees; 30 with right-anchoring does not."""
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label(n) for n in names], rotation=rotation, ha="right",
                       rotation_mode="anchor")


def save(fig, path, data=None):
    """Every figure writes the numbers behind it next to the image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    plt.close(fig)
    return path
