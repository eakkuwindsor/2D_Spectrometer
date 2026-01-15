# This is perfect simulated data, variations/noises will probably be in a different file for the sake
# of it; ask if I should make the data look more cleaner?

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Speed of light in vacuum (m/s)
c = 2.9979e8

# Sellmeier dispersion model
# n(λ) = refractive index as a function of wavelength

def n_sellmeier(lam_um, B1, B2, B3, C1, C2, C3):
    lam2 = lam_um * lam_um
    
    # Suppress warnings near Sellmeier poles
    with np.errstate(divide="ignore", invalid="ignore"):
        expr = (
            1
            + B1 * lam2 / (lam2 - C1)
            + B2 * lam2 / (lam2 - C2)
            + B3 * lam2 / (lam2 - C3)
        )
    # Enforce physical (real, non-negative) refractive index
    return np.sqrt(np.maximum(expr, 0.0))

# BK7 optical glass dispersion
def n_BK7(lam_um):
    return n_sellmeier(
        lam_um,
        1.03961212, 0.231792344, 1.01046945,
        0.00600069867, 0.0200179144, 103.560653
    )

# Fused silica (SiO₂) dispersion
def n_FS(lam_um):
    return n_sellmeier(
        lam_um,
        0.6961663, 0.4079426, 0.8974794,
        0.0684043**2, 0.1162414**2, 9.896161**2
    )

# Calcium fluoride (CaF₂) dispersion
def n_CaF2(lam_um):
    return n_sellmeier(
        lam_um,
        0.5675888, 0.4710914, 3.8484723,
        0.050263605**2, 0.1003909**2, 34.649040**2
    )

# Dictionary mapping material name → dispersion function
MATERIALS = {
    "BK7": n_BK7,
    "FS": n_FS,
    "CaF2": n_CaF2,
}


# Interferogram simulation
# Generates a 2D interferogram: delay × frequency

def simulate(
    material,              # Material name (BK7, FS, CaF2)
    L=1e-3,                # Sample thickness (meters)
    lambda0_nm=600.0,      # Central wavelength (nm)
    Dw_frac=0.30,          # Spectral bandwidth as fraction of wc
    Nt=256,                # Number of delay (time) samples
    Nw=256,                # Number of frequency samples
    t_min_fs=-5.0,         # Minimum delay (fs)
    t_max_fs=5.0,          # Maximum delay (fs)
    left_sigmas=1.9,       # Frequency window extent
    right_sigmas=1.7,      # Frequency window extent
):
    # Time-delay axis (seconds)
    t = np.linspace(t_min_fs, t_max_fs, Nt) * 1e-15

    # Central angular frequency wc
    wc = 2 * np.pi * c / (lambda0_nm * 1e-9)

    # Spectral width (Gaussian envelope)
    Dw = Dw_frac * wc

    # Frequency window around ωc (slightly asymmetric for framing)
    w_min = wc - left_sigmas * Dw
    w_max = wc + right_sigmas * Dw
    w = np.linspace(w_min, w_max, Nw)
    dw = w[1] - w[0]

    # Gaussian spectral intensity envelope I(ω)
    Iw = np.exp(-((w - wc) ** 2) / (Dw ** 2))
    Iw /= Iw.max()

    # Convert frequency → wavelength (μm) for dispersion model
    lam_um = (2 * np.pi * c / w) * 1e6

    # Refractive index n(ω)
    n = MATERIALS[material](lam_um)
    n = np.clip(np.nan_to_num(n, nan=1.0, posinf=2.0, neginf=1.0), 1.0, 2.0)

    # Group index ng = n + ω dn/dω
    ng = n + w * np.gradient(n, dw)

    # Reference group index at ωc
    ngc = ng[np.argmin(np.abs(w - wc))]

    # Allocate interferogram array
    spec = np.empty((Nt, Nw), float)

    # Interference phase and intensity
    for i, tau in enumerate(t):
        # Phase term: material dispersion + delay
        phi = 0.5 * (w / c) * L * (n - ngc) + 0.5 * w * tau

        # Interferogram intensity
        spec[i] = Iw * (np.cos(phi) ** 2)

    # Clip for image storage
    return np.clip(spec, 0.0, 1.0)

# -------------------------------------------------
# Training dataset generator
# -------------------------------------------------
def generate_train_dataset(
    out_dir=os.path.join("datasets", "train_dataset"),
    n_per=200,                  # Images per material
    seed=42,                    # RNG seed for reproducibility
    thickness_mm_mean=1.0,      # Mean thickness (mm)
    thickness_jitter_frac=0.03, # Thickness variation (by fractions)
    lambda0_nm=600.0,           # Central wavelength (nm)
    bandwidth_frac=0.30,        # Spectral bandwidth fraction
    left_sigmas=1.9,            # Frequency framing (left)
    right_sigmas=1.7,           # Frequency framing (right)
    Nt=256,                     # Delay samples
    Nw=256,                     # Frequency samples
    t_min_fs=-5.0,              # Min delay (fs)
    t_max_fs=5.0                # Max delay (fs)
):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Assign numeric class IDs
    material_ids = {name: i for i, name in enumerate(MATERIALS)}
    labels_path = os.path.join(out_dir, "labels.csv")

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename","material","material_id",
            "thickness_mm","lambda0_nm","bandwidth_frac",
            "left_sigmas","right_sigmas","Nt","Nw","t_min_fs","t_max_fs"
        ])

        for mat in MATERIALS:
            for i in range(n_per):
                # Randomize thickness
                frac = rng.uniform(-thickness_jitter_frac, thickness_jitter_frac)
                thickness_mm = thickness_mm_mean * (1.0 + frac)
                L_m = thickness_mm * 1e-3

                # Simulate interferogram
                img = simulate(
                    mat,
                    L=L_m,
                    lambda0_nm=lambda0_nm,
                    Dw_frac=bandwidth_frac,
                    Nt=Nt,
                    Nw=Nw,
                    t_min_fs=t_min_fs,
                    t_max_fs=t_max_fs,
                    left_sigmas=left_sigmas,
                    right_sigmas=right_sigmas
                )

                # Save image
                fname = f"{mat}_{i:05d}.png"
                plt.imsave(os.path.join(out_dir, fname),
                           img, cmap="gray", vmin=0.0, vmax=1.0)

                # Save metadata
                writer.writerow([
                    fname, mat, material_ids[mat],
                    f"{thickness_mm:.6f}", f"{lambda0_nm:.6f}", f"{bandwidth_frac:.6f}",
                    f"{left_sigmas:.6f}", f"{right_sigmas:.6f}",
                    Nt, Nw, f"{t_min_fs:.6f}", f"{t_max_fs:.6f}"
                ])

            print(f"{mat}: {n_per} images")

    print("Done ->", out_dir)
    print("Labels ->", labels_path)

if __name__ == "__main__":
    generate_train_dataset(n_per=200)
