#ifdef GL_ARB_derivative_control
  #extension GL_ARB_derivative_control : enable
#endif

vec4 quantize3Bit(in vec4 color) {
  return vec4(round(color.rgb * 8.0) / 8.0, step(0.5, color.a));
}

vec4 quantize4Bit(in vec4 color) {
  return round(color * 16.0) / 16.0; // (16 seems more accurate than 15)
}

vec4 quantizeTexture(uint flags, vec4 color) {
  vec4 colorQuant = flagSelect(flags, TEX_FLAG_4BIT, color, quantize4Bit(color));
  colorQuant = flagSelect(flags, TEX_FLAG_3BIT, colorQuant, quantize3Bit(colorQuant));
  colorQuant.rgb = linearToGamma(colorQuant.rgb);
  return flagSelect(flags, TEX_FLAG_MONO, colorQuant.rgba, colorQuant.rrrr);
}

vec4 sampleSampler(in const sampler2D tex, in const TileConf tileConf, in vec2 uvCoord, in const uint texFilter) {
  // https://github.com/rt64/rt64/blob/61aa08f517cd16c1dbee4e097768b08e2a060307/src/shaders/TextureSampler.hlsli#L156-L276
  const ivec2 texSize = textureSize(tex, 0);

  uvCoord.y = texSize.y - uvCoord.y; // invert Y
  uvCoord *= tileConf.shift;

#ifdef SIMULATE_LOW_PRECISION
  // Simulates the lower precision of the hardware's coordinate interpolation.
  uvCoord = round(uvCoord * LOW_PRECISION) / LOW_PRECISION;
#endif

  uvCoord -= tileConf.low;

  const vec2 isClamp      = step(tileConf.mask, vec2(1.0));
  const vec2 isMirror     = step(tileConf.high, vec2(0.0));
  const vec2 isForceClamp = step(tileConf.mask, vec2(1.0)); // mask == 0 forces clamping
  const vec2 mask = mix(abs(tileConf.mask), vec2(256), isForceClamp); // if mask == 0, we also have to ignore it
  const vec2 highMinusLow = abs(tileConf.high) - abs(tileConf.low);

  if (texFilter != G_TF_POINT) {
    uvCoord -= 0.5 * tileConf.shift;
    const ivec2 texelBaseInt = ivec2(floor(uvCoord));
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
    return quantizeTexture(tileConf.flags, wrappedMirrorSample(tex, ivec2(floor(uvCoord)), mask, highMinusLow, isClamp, isMirror, isForceClamp));
  }
}

vec4 sampleIndex(in const uint textureIndex, in const vec2 uvCoord, in const uint texFilter) {
  TileConf tileConf = material.texConfs[textureIndex];
  switch (textureIndex) {
    default: return sampleSampler(tex0, tileConf, uvCoord, texFilter);
    case 1: return sampleSampler(tex1, tileConf, uvCoord, texFilter);
    case 2: return sampleSampler(tex2, tileConf, uvCoord, texFilter);
    case 3: return sampleSampler(tex3, tileConf, uvCoord, texFilter);
    case 4: return sampleSampler(tex4, tileConf, uvCoord, texFilter);
    case 5: return sampleSampler(tex5, tileConf, uvCoord, texFilter);
    case 6: return sampleSampler(tex6, tileConf, uvCoord, texFilter);
    case 7: return sampleSampler(tex7, tileConf, uvCoord, texFilter);
  }
}

float computeLOD(inout uint tileIndex0, inout uint tileIndex1) {
  // https://github.com/rt64/rt64/blob/0ca92eeb6c2f58ce3581c65f87f7261b8ac0fea0/src/shaders/TextureSampler.hlsli#L18
  if (textLOD() == G_TL_TILE)
    return 1.0f;
  const uint texDetail = textDetail();
  const bool lodSharpen = texDetail == G_TD_SHARPEN;
  const bool lodDetail = texDetail == G_TD_DETAIL;
  const bool lodSharpDetail = lodSharpen || lodDetail;

#ifdef GL_ARB_derivative_control
  const vec2 dfd = abs(vec2(dFdxCoarse(inputUV.x), dFdyCoarse(inputUV.y)));
#else
  const vec2 dfd = abs(vec2(dFdx(inputUV.x), dFdy(inputUV.y)));
#endif
  float maxDst = max(dfd.x, dfd.y);

  if (lodSharpDetail) 
    maxDst = max(maxDst, material.primLod.y);

  int tileBase = int(floor(log2(maxDst)));
  float lodFraction = maxDst / pow(2, max(tileBase, 0)) - 1.0;

  if (lodSharpen && maxDst < 1.0)
    lodFraction = maxDst - 1.0;

  if (lodDetail) {
    if (lodFraction < 0.0)
      lodFraction = maxDst;
    tileBase += 1;
  } else if (tileBase >= material.mipCount)
    lodFraction = 1.0;

  if (lodSharpDetail) 
    tileBase = max(tileBase, 0);
  else 
    lodFraction = max(lodFraction, 0.0);

  tileIndex0 = clamp(tileBase, 0, material.mipCount);
  tileIndex1 = clamp(tileBase + 1, 0, material.mipCount);
  return lodFraction;
}