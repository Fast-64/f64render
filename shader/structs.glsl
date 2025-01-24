// NOTE: this file is included by blender via 'shader_info.typedef_source(...)'

struct Light
{
  vec4 color;
  vec3 dir;
};

struct UBO_Material
{
  ivec4 blender[2];

  //Tile settings: xy = TEX0, zw = TEX1
  vec4 mask; // clamped if < 0, mask = abs(mask)
  vec4 shift;
  vec4 low;
  vec4 high; // if negative, mirrored, high = abs(high)

  // color-combiner
  ivec4 cc0Color;
  ivec4 cc0Alpha;
  ivec4 cc1Color;
  ivec4 cc1Alpha;

  ivec4 modes; // geo, other-low, other-high, flags
  Light lights[8];
  vec4 prim_color;
  vec4 primLodDepth;
  vec4 env;
  vec4 ambientColor;
  vec4 ck_center;
  vec4 ck_scale;
  vec4 k_0123;
  vec3 k45AlphaClip;
  int numLights;
};

#define GEO_MODE     material.modes.x
#define OTHER_MODE_L material.modes.y
#define OTHER_MODE_H material.modes.z
#define DRAW_FLAGS   material.modes.w
#define ALPHA_CLIP   material.k45AlphaClip.z
#define K45          material.k45AlphaClip.xy
#define NUM_LIGHTS   material.numLights
