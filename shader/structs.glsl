// NOTE: this file is included by blender via 'shader_info.typedef_source(...)'

struct Light
{
  vec4 color;
  vec3 dir;
};

struct TileConf
{
  vec2 mask;  // clamped if < 0, mask = abs(mask)
  vec2 shift;
  vec2 low;
  vec2 high;  // if negative, mirrored, high = abs(high)
  uint flags; // TODO: correct impl of this would be to process textures ahead of time, but this has to be done in fast64 first
};

struct UBO_Material
{
  TileConf texConfs[8]; // TODO: should these two be part of separate uniform buffers?
  Light lights[8];
  ivec4 blender[2];

  // color-combiner
  ivec4 cc0Color;
  ivec4 cc0Alpha;
  ivec4 cc1Color;
  ivec4 cc1Alpha;

  ivec4 modes; // geo, other-low, other-high, flags

  vec4 primColor;
  vec2 primLod; // x is frac, y is min
  vec2 primDepth;
  vec4 env;
  vec4 ambientColor;
  vec3 ckCenter;
  float alphaClip;
  vec3 ckScale;
  uint numLights;
  vec3 ckWidth;
  int mipCount;
  vec4 k0123;
  vec2 k45;
  uvec2 texSize;
};

#define GEO_MODE     material.modes.x
#define OTHER_MODE_L material.modes.y
#define OTHER_MODE_H material.modes.z
#define DRAW_FLAGS   material.modes.w
