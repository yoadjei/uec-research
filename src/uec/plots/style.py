import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "delta": "#B23A48",
    "rho_null": "#2E5E8C",
    "rho_seed": "#7A9CC6",
    "nu": "#9AA0A6",
    "omega": "#3F7D53",
    "accent": "#C9821B",
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
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "lines.linewidth": 1.3,
    })


def label(name: str) -> str:
    return EXPLAINER_LABEL.get(name, name)


def save(fig, path, data=None):
    """Every figure writes the numbers behind it next to the image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    plt.close(fig)
    return path
