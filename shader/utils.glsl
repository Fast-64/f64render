#define GAMMA_FACTOR 2.2

vec3 gammaToLinear(in vec3 color) {
  return mix(
    color * (1.0 / 12.92),
    pow((color + 0.055) * (1.0 / 1.055), vec3(2.4)),
    step(0.04045, color)
  );
}

vec3 linearToGamma(in vec3 color) {
  return mix(
    color * 12.92,
    1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055,
    step(0.0031308, color)
  );
}

#define mixSelect(amount, a, b) (mix(a, b, float(amount)))
#define flagSelect(flags, flag_mask, a, b) (mixSelect((flags & flag_mask) != 0, a, b))
#define drawFlagSelect(flag_mask, a, b) flagSelect(DRAW_FLAGS, flag_mask, a, b)
#define geoModeSelect(flag_mask, a, b) flagSelect(GEO_MODE, flag_mask, a, b)
#define othermodeHSelect(flag_mask, a, b) flagSelect(material.othermodeH, flag_mask, a, b)

#define zSource()   (OTHER_MODE_L & (1 << G_MDSFT_ZSRCSEL))
#define cycleType() (OTHER_MODE_H & (3 << G_MDSFT_CYCLETYPE))
#define texFilter() (OTHER_MODE_H & (3 << G_MDSFT_TEXTFILT))
#define textPersp() (OTHER_MODE_H & (1 << G_MDSFT_TEXTPERSP))
#define textLOD()   (OTHER_MODE_H & (1 << G_MDSFT_TEXTLOD))
#define textDetail()(OTHER_MODE_H & (3 << G_MDSFT_TEXTDETAIL))

#define boolSelect(cond, a, b) (bool(mix(a, b, cond)))

float noise(in vec2 uv)
{
  return fract(sin(dot(uv, vec2(12.9898, 78.233)))* 43758.5453);
}

vec2 mirrorUV(vec2 uvEnd, vec2 uvIn)
{
  vec2 uvMod2 = mod(uvIn, uvEnd * 2.0 + 1.0);
  return mix(uvMod2, (uvEnd * 2.0) - uvMod2, step(uvEnd, uvMod2));
}

vec4 wrappedMirrorSample(const sampler2D tex, ivec2 uv, const vec2 mask, const vec2 highMinusLow,const  vec2 isClamp, const vec2 isMirror, const vec2 isForceClamp)
{
  const ivec2 texSize = textureSize(tex, 0);

  // first apply clamping if enabled (clamp S/T, low S/T -> high S/T)
  const vec2 uvClamp = clamp(uv, vec2(0.0), highMinusLow);
  uv = ivec2(mix(uv, uvClamp, isClamp));

  // then mirror the result if needed (mirror S/T)
  const vec2 uvMirror = mirrorUV(mask - vec2(0.5), vec2(uv));
  uv = ivec2(mix(vec2(uv), uvMirror, isMirror));
  
  // clamp again (mask S/T), this is also done to avoid OOB texture access
  uv = ivec2(mod(uv, min(texSize+1, mask)));

  uv.y = texSize.y - uv.y - 1; // invert Y back

  return texelFetch(tex, uv, 0);
}
