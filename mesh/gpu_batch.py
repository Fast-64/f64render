import numpy as np
import gpu


def create_vert_buf(
    shader: gpu.types.GPUShader,
    vbo_format,
    buff_vert: np.ndarray,
    buff_norm: np.ndarray,
    buff_color: np.ndarray,
    buff_uv: np.ndarray,
) -> list[gpu.types.GPUBatch]:
    def fill_attr(name: str, data: np.ndarray):
        aid = shader.attr_from_name(name)
        vbo.attr_fill(aid, data)

    vbo = gpu.types.GPUVertBuf(vbo_format, len(buff_vert))
    fill_attr("pos", buff_vert)
    fill_attr("inNormal", buff_norm)
    fill_attr("inColor", buff_color)
    fill_attr("inUV", buff_uv)

    return vbo


# Stripped down version of blender own batch function, specific to our layout
def batch_for_shader(vbo: gpu.types.GPUVertBuf, indices: np.ndarray) -> list[gpu.types.GPUBatch]:
    typ = "TRIS"
    ibo = gpu.types.GPUIndexBuf(type=typ, seq=indices)
    return gpu.types.GPUBatch(type=typ, buf=vbo, elem=ibo)
