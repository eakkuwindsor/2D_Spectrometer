import os
import csv
import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8

def n_sellmeier(lam_um, B1, B2, B3, C1, C2, C3):
    lam2 = lam_um * lam_um
    with np.errstate(divide="ignore", invalid="ignore"):
        expr = 1 + B1*lam2/(lam2-C1) + B2*lam2/(lam2-C2) + B3*lam2/(lam2-C3)
    return np.sqrt(np.maximum(expr, 0.0))

def n_BK7(lam_um):
    return n_sellmeier(lam_um, 1.03961212, 0.231792344, 1.01046945,
                       0.00600069867, 0.0200179144, 103.560653)

def n_FS(lam_um):
    return n_sellmeier(lam_um, 0.6961663, 0.4079426, 0.8974794,
                       0.0684043**2, 0.1162414**2, 9.896161**2)

def n_CaF2(lam_um):
    return n_sellmeier(lam_um, 0.5675888, 0.4710914, 3.8484723,
                       0.050263605**2, 0.1003909**2, 34.649040**2)

MATERIALS = {"BK7": n_BK7, "FS": n_FS, "CaF2": n_CaF2}

def simulate(
    material,
    L=1e-3,
    lambda0_nm=600.0,
    Dw_frac=0.30,
    Nt=256,
    Nw=256,
    t_min_fs=-5.0,
    t_max_fs=5.0,
    left_sigmas=1.9,
    right_sigmas=1.7,
):
    t = np.linspace(t_min_fs, t_max_fs, Nt) * 1e-15

    wc = 2*np.pi*c/(lambda0_nm*1e-9)
    Dw = Dw_frac * wc

    w_min = wc - left_sigmas * Dw
    w_max = wc + right_sigmas * Dw
    w = np.linspace(w_min, w_max, Nw)
    dw = w[1] - w[0]

    Iw = np.exp(-((w - wc)**2)/(Dw**2))
    Iw /= Iw.max()

    lam_um = (2*np.pi*c / w) * 1e6
    n = MATERIALS[material](lam_um)
    n = np.clip(np.nan_to_num(n, nan=1.0, posinf=2.0, neginf=1.0), 1.0, 2.0)

    ng = n + w*np.gradient(n, dw)
    ngc = ng[np.argmin(np.abs(w - wc))]

    spec = np.empty((Nt, Nw), float)
    for i, tau in enumerate(t):
        phi = 0.5*(w/c)*L*(n - ngc) + 0.5*w*tau
        spec[i] = Iw*(np.cos(phi)**2)

    return np.clip(spec, 0.0, 1.0)

def generate_prediction_dataset(
    out_dir=os.path.join("datasets", "prediction_dataset"),
    n_per=10,
    seed=123,
    thickness_mm_mean=1.0,
    thickness_jitter_frac=0.03,
    lambda0_nm=600.0,
    bandwidth_frac=0.30,
    left_sigmas=1.9,
    right_sigmas=1.7,
    Nt=256,
    Nw=256,
    t_min_fs=-5.0,
    t_max_fs=5.0
):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    material_ids = {name: i for i, name in enumerate(MATERIALS)}
    labels_path = os.path.join(out_dir, "labels.csv")

    # EXACT SAME CSV HEADER AS TRAIN
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename","material","material_id",
            "thickness_mm","lambda0_nm","bandwidth_frac",
            "left_sigmas","right_sigmas","Nt","Nw","t_min_fs","t_max_fs"
        ])

        img_id = 0
        for mat in MATERIALS:
            for _ in range(n_per):
                frac = rng.uniform(-thickness_jitter_frac, thickness_jitter_frac)
                thickness_mm = thickness_mm_mean * (1.0 + frac)
                L_m = thickness_mm * 1e-3

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

                # TEST FILE NAMES
                fname = f"test_{img_id:05d}.png"
                img_id += 1

                plt.imsave(os.path.join(out_dir, fname), img,
                           cmap="gray", vmin=0.0, vmax=1.0)

                writer.writerow([
                    fname, mat, material_ids[mat],
                    f"{thickness_mm:.6f}", f"{lambda0_nm:.6f}", f"{bandwidth_frac:.6f}",
                    f"{left_sigmas:.6f}", f"{right_sigmas:.6f}",
                    Nt, Nw, f"{t_min_fs:.6f}", f"{t_max_fs:.6f}"
                ])

            print(f"{mat}: {n_per} test images")

    print("Done ->", out_dir)
    print("Labels ->", labels_path)

if __name__ == "__main__":
    generate_prediction_dataset(n_per=10)
