import functools

BL_INP = {
    "G_BL_0": 0,
    "G_BL_1": 1,
    "G_BL_CLR_IN": 2,
    "G_BL_CLR_MEM": 3,
    "G_BL_CLR_BL": 4,
    "G_BL_CLR_FOG": 5,
    "G_BL_A_IN": 6,
    "G_BL_A_FOG": 7,
    "G_BL_A_SHADE": 8,
    "G_BL_1MA": 9,
    "G_BL_A_MEM": 10,
}


@functools.cache
def get_blender_settings(blend_cycle1: tuple[str, str, str, str], blend_cycle2: tuple[str, str, str, str]) -> tuple:
    return tuple(BL_INP[x] for x in blend_cycle1 + blend_cycle2)
