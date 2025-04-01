// allows for a per-pixel atomic access to the depth texture (needed for decals & blending)
#ifdef USE_SHADER_INTERLOCK
  #extension GL_ARB_fragment_shader_interlock : enable
  layout(pixel_interlock_unordered) in;
#endif

#define DECAL_DEPTH_DELTA 100

vec4 quantize3Bit(in vec4 color) {
  return vec4(round(color.rgb * 8.0) / 8.0, step(0.5, color.a));
}

vec4 quantize4Bit(in vec4 color) {
  return round(color * 16.0) / 16.0; // (16 seems more accurate than 15)
}

vec4 quantizeTexture(int flags, vec4 color) {
  vec4 colorQuant = flagSelect(flags, TEX_FLAG_4BIT, color, quantize4Bit(color));
  colorQuant = flagSelect(flags, TEX_FLAG_3BIT, colorQuant, quantize3Bit(colorQuant));
  return flagSelect(flags, TEX_FLAG_MONO, colorQuant.rgba, colorQuant.rrrr);
}

vec4 sampleSampler(in const sampler2D tex, in const TileConf tileConf, in vec2 uvCoord, in const uint texFilter) {
  // https://github.com/rt64/rt64/blob/61aa08f517cd16c1dbee4e097768b08e2a060307/src/shaders/TextureSampler.hlsli#L156-L276
  // TODO: Next pixel cycle 1 bug, probably best to implement in some cpu pass for textures integrated into fast64 (same for quantization)
  const ivec2 texSize = textureSize(tex, 0);

  // Compensates the lack of perspective division for the UVs based on observed hardware behavior.
#ifdef TEXTURE_RECT
  const bool applyCorrection = true;
#else
  const bool applyCorrection = textPersp() == G_TP_PERSP;
#endif

  uvCoord.y = texSize.y - uvCoord.y; // invert Y
  uvCoord *= (applyCorrection ? 1.0 : 2.0) * tileConf.shift;

#ifdef SIMULATE_LOW_PRECISION
  // Simulates the lower precision of the hardware's coordinate interpolation.
  uvCoord = round(uvCoord * LOW_PRECISION) / LOW_PRECISION;
#endif

  uvCoord -= tileConf.low;
  const ivec2 texelBaseInt = ivec2(floor(uvCoord));

  const vec2 isClamp      = step(tileConf.mask, vec2(1.0));
  const vec2 isMirror     = step(tileConf.high, vec2(0.0));
  const vec2 isForceClamp = step(tileConf.mask, vec2(1.0)); // mask == 0 forces clamping
  const vec2 mask = mix(abs(tileConf.mask), vec2(256), isForceClamp); // if mask == 0, we also have to ignore it
  const vec2 highMinusLow = abs(tileConf.high) - abs(tileConf.low);

  if (texFilter != G_TF_POINT) {
    const vec4 sample00 = wrappedMirrorSample(tex, texelBaseInt,               mask, highMinusLow, isClamp, isMirror, isForceClamp);
    const vec4 sample01 = wrappedMirrorSample(tex, texelBaseInt + ivec2(0, 1), mask, highMinusLow, isClamp, isMirror, isForceClamp);
    const vec4 sample10 = wrappedMirrorSample(tex, texelBaseInt + ivec2(1, 0), mask, highMinusLow, isClamp, isMirror, isForceClamp);
    const vec4 sample11 = wrappedMirrorSample(tex, texelBaseInt + ivec2(1, 1), mask, highMinusLow, isClamp, isMirror, isForceClamp);
    const vec2 fracPart = uvCoord - texelBaseInt;
#ifdef USE_LINEAR_FILTER
    return quantizeTexture(tileConf.flags, mix(mix(sample00, sample10, fracPart.x), mix(sample01, sample11, fracPart.x), fracPart.y));
#else
    if (texFilter == G_TF_AVERAGE && all(lessThanEqual(vec2(1 / LOW_PRECISION), abs(fracPart - 0.5)))) {
        return quantizeTexture(tileConf.flags, (sample00 + sample01 + sample10 + sample11) / 4.0f);
    }
    else {
      // Originally written by ArthurCarvalho
      // Sourced from https://www.emutalk.net/threads/emulating-nintendo-64-3-sample-bilinear-filtering-using-shaders.54215/
      vec4 tri0 = mix(sample00, sample10, fracPart.x) + (sample01 - sample00) * fracPart.y;
      vec4 tri1 = mix(sample11, sample01, 1.0 - fracPart.x) + (sample10 - sample11) * (1.0 - fracPart.y);
      return quantizeTexture(tileConf.flags, mix(tri0, tri1, step(1.0, fracPart.x + fracPart.y)));
    }
#endif
  }
  else {
    return quantizeTexture(tileConf.flags, wrappedMirrorSample(tex, texelBaseInt, mask, highMinusLow, isClamp, isMirror, isForceClamp));
  }
}

const sampler2D getTextureSampler(in const uint textureIndex) {
  switch (textureIndex) {
    default: return tex0;
    case 1: return tex1;
    case 2: return tex2;
    case 3: return tex3;
    case 4: return tex4;
    case 5: return tex5;
    case 6: return tex6;
    case 7: return tex7;
  }
}

vec3 cc_fetchColor(in int val, in vec4 shade, in vec4 comb, in vec4 texData0, in vec4 texData1)
{
       if(val == CC_C_COMB       ) return comb.rgb;
  else if(val == CC_C_TEX0       ) return texData0.rgb;
  else if(val == CC_C_TEX1       ) return texData1.rgb;
  else if(val == CC_C_PRIM       ) return material.prim_color.rgb;
  else if(val == CC_C_SHADE      ) return shade.rgb;
  else if(val == CC_C_ENV        ) return material.env.rgb;
  else if(val == CC_C_CENTER     ) return material.ck_center.rgb;
  else if(val == CC_C_SCALE      ) return material.ck_scale;
  else if(val == CC_C_COMB_ALPHA ) return comb.aaa;
  else if(val == CC_C_TEX0_ALPHA ) return texData0.aaa;
  else if(val == CC_C_TEX1_ALPHA ) return texData1.aaa;
  else if(val == CC_C_PRIM_ALPHA ) return material.prim_color.aaa;
  else if(val == CC_C_SHADE_ALPHA) return linearToGamma(shade.aaa);
  else if(val == CC_C_ENV_ALPHA  ) return material.env.aaa;
  // else if(val == CC_C_LOD_FRAC   ) return vec3(0.0); // @TODO
  else if(val == CC_C_PRIM_LOD_FRAC) return vec3(material.primLodDepth[1]);
  else if(val == CC_C_NOISE      ) return vec3(noise(posScreen*0.25));
  else if(val == CC_C_K4         ) return vec3(material.k45[0]);
  else if(val == CC_C_K5         ) return vec3(material.k45[1]);
  else if(val == CC_C_1          ) return vec3(1.0);
  return vec3(0.0); // default: CC_C_0
}

float cc_fetchAlpha(in int val, vec4 shade, in vec4 comb, in vec4 texData0, in vec4 texData1)
{
       if(val == CC_A_COMB ) return comb.a;
  else if(val == CC_A_TEX0 ) return texData0.a;
  else if(val == CC_A_TEX1 ) return texData1.a;
  else if(val == CC_A_PRIM ) return material.prim_color.a;
  else if(val == CC_A_SHADE) return shade.a;
  else if(val == CC_A_ENV  ) return material.env.a;
  // else if(val == CC_A_LOD_FRAC) return 0.0; // @TODO
  else if(val == CC_A_PRIM_LOD_FRAC) return material.primLodDepth[1];
  else if(val == CC_A_1    ) return 1.0;
  return 0.0; // default: CC_A_0
}

/**
 * handles both CC clamping and overflow:
 *        x ≤ -0.5: wrap around
 * -0.5 ≥ x ≤  1.5: clamp to 0-1
 *  1.5 < x       : wrap around
 */
vec4 cc_overflowValue(in vec4 value)
{
  return mod(value + 0.5, 2.0) - 0.5;
}
vec4 cc_clampValue(in vec4 value)
{
  return clamp(value, 0.0, 1.0);
}

#ifdef BLEND_EMULATION
vec4 blender_fetch(
  in int val, in vec4 colorBlend, in vec4 colorFog, in vec4 colorFB, in vec4 colorCC,
  in vec4 blenderA
)
{
       if (val == BLENDER_1      ) return vec4(1.0);
  else if (val == BLENDER_CLR_IN ) return colorCC;
  else if (val == BLENDER_CLR_MEM) return colorFB;
  else if (val == BLENDER_CLR_BL ) return colorBlend;
  else if (val == BLENDER_CLR_FOG) return colorCC;//colorFog; //@TODO: implemnent fog
  else if (val == BLENDER_A_IN   ) return colorCC.aaaa;
  else if (val == BLENDER_A_FOG  ) return colorFog.aaaa;
  else if (val == BLENDER_A_SHADE) return cc_shade.aaaa;
  else if (val == BLENDER_1MA    ) return 1.0 - blenderA.aaaa;
  else if (val == BLENDER_A_MEM  ) return colorFB.aaaa;
  return vec4(0.0); // default: BLENDER_0
}

vec4 blendColor(in vec4 oldColor, vec4 newColor)
{
  vec4 colorBlend = vec4(0.0); // @TODO
  vec4 colorFog = vec4(1.0, 0.0, 0.0, 1.0); // @TODO

  vec4 P = blender_fetch(material.blender[0][0], colorBlend, colorFog, oldColor, newColor, vec4(0.0));
  vec4 A = blender_fetch(material.blender[0][1], colorBlend, colorFog, oldColor, newColor, vec4(0.0));
  vec4 M = blender_fetch(material.blender[0][2], colorBlend, colorFog, oldColor, newColor, A);
  vec4 B = blender_fetch(material.blender[0][3], colorBlend, colorFog, oldColor, newColor, A);

  vec4 res = ((P * A) + (M * B)) / (A + B);
  res.a = gammaToLinear(newColor.aaa).r; // preserve for 'A_IN'

  P = blender_fetch(material.blender[1][0], colorBlend, colorFog, oldColor, res, vec4(0.0));
  A = blender_fetch(material.blender[1][1], colorBlend, colorFog, oldColor, res, vec4(0.0));
  M = blender_fetch(material.blender[1][2], colorBlend, colorFog, oldColor, res, A);
  B = blender_fetch(material.blender[1][3], colorBlend, colorFog, oldColor, res, A);

  return ((P * A) + (M * B)) / (A + B);
}

// This implements the part of the fragment shader that touches depth/color.
// All of this must happen in a way that guarantees coherency.
// This is either done via the shader interlock extension, or with a re-try loop as a fallback.
bool color_depth_blending(
  in bool alphaTestFailed, in int writeDepth, in int currDepth, in vec4 ccValue,
  out uint oldColorInt, out uint writeColor
)
{
  ivec2 screenPosPixel = ivec2(trunc(gl_FragCoord.xy));
  int oldDepth = imageAtomicMax(depth_texture, screenPosPixel, writeDepth);
  int depthDiff = int(mixSelect(zSource() == G_ZS_PRIM, abs(oldDepth - currDepth), material.primLodDepth.w));

  bool depthTest = currDepth >= oldDepth;
  if((DRAW_FLAGS & DRAW_FLAG_DECAL) != 0) {
    depthTest = depthDiff <= DECAL_DEPTH_DELTA;
  }

  oldColorInt = imageLoad(color_texture, screenPosPixel).r;
  vec4 oldColor = unpackUnorm4x8(oldColorInt);
  oldColor.a = 0.0;
  
  vec4 ccValueBlended = blendColor(oldColor, vec4(ccValue.rgb, pow(ccValue.a, 1.0 / GAMMA_FACTOR)));
  
  bool shouldDiscard = alphaTestFailed || !depthTest;

  vec4 ccValueWrite = mixSelect(shouldDiscard, ccValueBlended, oldColor);
  ccValueWrite.a = 1.0;
  writeColor = packUnorm4x8(ccValueWrite);

  if(shouldDiscard)oldColorInt = writeColor;
  return shouldDiscard;
}
#endif

void main()
{
  const uint cycleType = cycleType();
  const uint texFilter = texFilter();
  vec4 cc0[4]; // inputs for 1. cycle
  vec4 cc1[4]; // inputs for 2. cycle
  vec4 ccValue = vec4(0.0); // result after 1/2 cycle

  vec4 ccShade = geoModeSelect(G_SHADE_SMOOTH, cc_shade_flat, cc_shade);

  const vec2 uvCoord = inputUV * textureSize(getTextureSampler(material.uvBasis), 0);
  vec4 texData0 = sampleSampler(getTextureSampler(0), material.texConfs[0], uvCoord, texFilter);
  vec4 texData1 = sampleSampler(getTextureSampler(1), material.texConfs[1], uvCoord, texFilter);

  texData0.rgb = linearToGamma(texData0.rgb);
  texData1.rgb = linearToGamma(texData1.rgb);

  // @TODO: emulate other formats, e.g. quantization?

  cc0[0].rgb = cc_fetchColor(material.cc0Color.x, ccShade, ccValue, texData0, texData1);
  cc0[1].rgb = cc_fetchColor(material.cc0Color.y, ccShade, ccValue, texData0, texData1);
  cc0[2].rgb = cc_fetchColor(material.cc0Color.z, ccShade, ccValue, texData0, texData1);
  cc0[3].rgb = cc_fetchColor(material.cc0Color.w, ccShade, ccValue, texData0, texData1);

  cc0[0].a = cc_fetchAlpha(material.cc0Alpha.x, ccShade, ccValue, texData0, texData1);
  cc0[1].a = cc_fetchAlpha(material.cc0Alpha.y, ccShade, ccValue, texData0, texData1);
  cc0[2].a = cc_fetchAlpha(material.cc0Alpha.z, ccShade, ccValue, texData0, texData1);
  cc0[3].a = cc_fetchAlpha(material.cc0Alpha.w, ccShade, ccValue, texData0, texData1);

  ccValue = cc_overflowValue((cc0[0] - cc0[1]) * cc0[2] + cc0[3]);

  if(cycleType == G_CYC_2CYCLE) {
    cc1[0].rgb = cc_fetchColor(material.cc1Color.x, ccShade, ccValue, texData0, texData1);
    cc1[1].rgb = cc_fetchColor(material.cc1Color.y, ccShade, ccValue, texData0, texData1);
    cc1[2].rgb = cc_fetchColor(material.cc1Color.z, ccShade, ccValue, texData0, texData1);
    cc1[3].rgb = cc_fetchColor(material.cc1Color.w, ccShade, ccValue, texData0, texData1);
    
    cc1[0].a = cc_fetchAlpha(material.cc1Alpha.x, ccShade, ccValue, texData0, texData1);
    cc1[1].a = cc_fetchAlpha(material.cc1Alpha.y, ccShade, ccValue, texData0, texData1);
    cc1[2].a = cc_fetchAlpha(material.cc1Alpha.z, ccShade, ccValue, texData0, texData1);
    cc1[3].a = cc_fetchAlpha(material.cc1Alpha.w, ccShade, ccValue, texData0, texData1);

    ccValue = (cc1[0] - cc1[1]) * cc1[2] + cc1[3];
  }

  ccValue = cc_clampValue(cc_overflowValue(ccValue));
  ccValue.rgb = gammaToLinear(ccValue.rgb);

  bool alphaTestFailed = ccValue.a < material.alphaClip;
#ifdef BLEND_EMULATION
  // Depth / Decal handling:
  // We manually write & check depth values in an image in addition to the actual depth buffer.
  // This allows us to do manual compares (e.g. decals) and discard fragments based on that.
  // In order to guarantee proper ordering, we both use atomic image access as well as an invocation interlock per pixel.
  // If those where not used, a race-condition will occur where after a depth read happens, the depth value might have changed,
  // leading to culled faces writing their color values even though a new triangles has a closer depth value already written.
  // If no interlock is available, we use a re-try loop to ensure that the correct color value is written.
  // Note that this fallback can create small artifacts since depth and color are not able to be synchronized together.
  ivec2 screenPosPixel = ivec2(trunc(gl_FragCoord.xy));

  int currDepth = int(mixSelect(zSource() == G_ZS_PRIM, gl_FragCoord.w * 0xFFFFF, material.primLodDepth.z));
  int writeDepth = int(drawFlagSelect(DRAW_FLAG_DECAL, currDepth, -0xFFFFFF));

  if((DRAW_FLAGS & DRAW_FLAG_ALPHA_BLEND) != 0) {
    writeDepth = -0xFFFFFF;
  }

  if(alphaTestFailed)writeDepth = -0xFFFFFF;

  #ifdef USE_SHADER_INTERLOCK
    beginInvocationInterlockARB();
  #else
    if(alphaTestFailed)discard; // discarding in interlock seems to cause issues, only do it here
  #endif

  {
    uint oldColorInt = 0;
    uint writeColor = 0;
    color_depth_blending(alphaTestFailed, writeDepth, currDepth, ccValue, oldColorInt, writeColor);

    #ifdef USE_SHADER_INTERLOCK
      imageAtomicCompSwap(color_texture, screenPosPixel, oldColorInt, writeColor.r);
    #else
      int count = 4;
      while(imageAtomicCompSwap(color_texture, screenPosPixel, oldColorInt, writeColor.r) != oldColorInt && count > 0)  {
        --count;
        if(color_depth_blending(alphaTestFailed, writeDepth, currDepth, ccValue, oldColorInt, writeColor)) {
          break;
        }
      }
      // Debug: write solod color in case we failed with our loop (seems to never happen)
      //if(count <= 0)imageAtomicExchange(color_texture, screenPosPixel, 0xFFFF00FF);
    #endif
  }
  
  #ifdef USE_SHADER_INTERLOCK
    endInvocationInterlockARB();
  #endif

  // Since we only need our own depth/color textures, there is no need to actually write out fragments.
  // It may seem like we could use the calc. color from out texture and set it here,
  // but it will result in incoherent results (e.g. blocky artifacts due to depth related race-conditions)
  // This is most prominent on decals.
  discard;
#else
  if (alphaTestFailed) discard;
  if((DRAW_FLAGS & DRAW_FLAG_ALPHA_BLEND) == 0) {
    ccValue.a = 1.0;
  }
  FragColor = ccValue;
#endif
}