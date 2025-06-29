import bpy
import gpu

from dataclasses import dataclass
import numpy as np

TEX_FLAG_MONO = 1 << 0
TEX_FLAG_4BIT = 1 << 1
TEX_FLAG_3BIT = 1 << 2


@dataclass
class F64Texture:
    values: tuple[float, float, float, float, float, float, float, float, int]
    buff: gpu.types.GPUTexture


def get_tile_conf(tex: "TextureProperty") -> F64Texture:
    flags = 0
    if tex.tex is not None:
        # Note: doing 'gpu.texture.from_image' seems to cost nothing, caching is not needed
        buff = gpu.texture.from_image(tex.tex)
        if tex.tex_format in {"I4", "I8"}:
            flags |= TEX_FLAG_MONO
        if tex.tex_format in {"I4", "IA8"}:
            flags |= TEX_FLAG_4BIT
        if tex.tex_format == "IA4":
            flags |= TEX_FLAG_3BIT
    else:
        buff = gpu.texture.from_image(bpy.data.images["f64render_missing_texture"])
        flags |= TEX_FLAG_MONO

    conf = np.array(
        [
            tex.S.mask,
            tex.T.mask,
            tex.S.shift,
            tex.T.shift,
            tex.S.low,
            -tex.T.low,
            tex.S.high,
            tex.T.high,
        ],
        dtype=np.float32,
    )

    conf[0:4] = 2 ** conf[0:4]  # mask/shift are exponents, calc. 2^x
    conf[2:4] = 1 / conf[2:4]  # shift is inverted

    # quantize the low/high values into 0.25 pixel increments
    conf[4:] = np.round(conf[4:] * 4) / 4

    # if clamp is on, negate the mask value
    if tex.S.clamp:
        conf[0] = -conf[0]
    if tex.T.clamp:
        conf[1] = -conf[1]

    # if mirror is on, negate the high value
    if tex.S.mirror:
        conf[6] = -conf[6]
    if tex.T.mirror:
        conf[7] = -conf[7]

    return F64Texture((*conf, flags), buff)
