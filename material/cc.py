from dataclasses import dataclass

import numpy as np
import bpy

CC1_C = {
    "0": 0,
    "1": 1,
    "COMBINED": 2,
    "TEXEL0": 3,
    "TEXEL1": 4,
    "PRIMITIVE": 5,
    "SHADE": 6,
    "ENVIRONMENT": 7,
    "CENTER": 8,
    "SCALE": 9,
    "COMBINED_ALPHA": 10,
    "TEXEL0_ALPHA": 11,
    "TEXEL1_ALPHA": 12,
    "PRIMITIVE_ALPHA": 13,
    "SHADE_ALPHA": 14,
    "ENV_ALPHA": 15,
    "LOD_FRACTION": 16,
    "PRIM_LOD_FRAC": 17,
    "NOISE": 18,
    "K4": 19,
    "K5": 20,
}

CC2_C = {  # TEX0 and TEX1 are swapped
    "0": 0,
    "1": 1,
    "COMBINED": 2,
    "TEXEL1": 3,
    "TEXEL0": 4,
    "PRIMITIVE": 5,
    "SHADE": 6,
    "ENVIRONMENT": 7,
    "CENTER": 8,
    "SCALE": 9,
    "COMBINED_ALPHA": 10,
    "TEXEL1_ALPHA": 11,
    "TEXEL0_ALPHA": 12,
    "PRIMITIVE_ALPHA": 13,
    "SHADE_ALPHA": 14,
    "ENV_ALPHA": 15,
    "LOD_FRACTION": 16,
    "PRIM_LOD_FRAC": 17,
    "NOISE": 18,
    "K4": 19,
    "K5": 20,
}

CC1_A = {
    "0": 0,
    "1": 1,
    "COMBINED": 2,
    "TEXEL0": 3,
    "TEXEL1": 4,
    "PRIMITIVE": 5,
    "SHADE": 6,
    "ENVIRONMENT": 7,
    "LOD_FRACTION": 8,
    "PRIM_LOD_FRAC": 9,
}

CC2_A = {  # TEX0 and TEX1 are swapped
    "0": 0,
    "1": 1,
    "COMBINED": 2,
    "TEXEL1": 3,
    "TEXEL0": 4,
    "PRIMITIVE": 5,
    "SHADE": 6,
    "ENVIRONMENT": 7,
    "LOD_FRACTION": 8,
    "PRIM_LOD_FRAC": 9,
}

SOLID_CC = (0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5)


# Fetches CC settings from a given fast64-material
def get_cc_settings(f3d_mat) -> np.ndarray:
    c0 = f3d_mat.combiner1
    c1 = f3d_mat.combiner2

    if f3d_mat.rdp_settings.g_mdsft_cycletype == "G_CYC_1CYCLE":
        c1 = c0

    return np.array(
        [
            CC1_C[c0.A],
            CC1_C[c0.B],
            CC1_C[c0.C],
            CC1_C[c0.D],
            CC1_A[c0.A_alpha],
            CC1_A[c0.B_alpha],
            CC1_A[c0.C_alpha],
            CC1_A[c0.D_alpha],
            CC2_C[c1.A],
            CC2_C[c1.B],
            CC2_C[c1.C],
            CC2_C[c1.D],
            CC2_A[c1.A_alpha],
            CC2_A[c1.B_alpha],
            CC2_A[c1.C_alpha],
            CC2_A[c1.D_alpha],
        ],
        dtype=np.int32,
    )
