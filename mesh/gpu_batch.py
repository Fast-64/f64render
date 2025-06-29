import numpy as np
import gpu


def create_vert_buf(
    vbo_format, buff_vert: np.ndarray, buff_norm: np.ndarray, buff_color: np.ndarray, buff_uv: np.ndarray
) -> list[gpu.types.GPUBatch]:
    vbo = gpu.types.GPUVertBuf(vbo_format, len(buff_vert))

    vbo.attr_fill("pos", buff_vert)
    vbo.attr_fill("inNormal", buff_norm)
    vbo.attr_fill("inColor", buff_color)
    vbo.attr_fill("inUV", buff_uv)

    return vbo


# Stripped down version of blender own batch function, specific to our layout
def batch_for_shader(vbo: gpu.types.GPUVertBuf, indices: np.ndarray) -> list[gpu.types.GPUBatch]:
    typ = "TRIS"
    ibo = gpu.types.GPUIndexBuf(type=typ, seq=indices)
    return gpu.types.GPUBatch(type=typ, buf=vbo, elem=ibo)
