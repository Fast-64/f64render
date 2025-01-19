
void main() 
{
  // Directional light
  vec3 norm = inNormal;
  vec3 normScreen = normalize(matNorm * norm);

  cc_shade = inColor;

  vec4 lightTotal = vec4(material.ambientColor.rgb, 0.0);
  for(int i=0; i<2; ++i) {
    float lightStren = max(dot(norm, material.lightDir[i].xyz), 0.0);
    lightTotal += material.lightColor[i] * lightStren;
  }

  lightTotal.rgb = clamp(lightTotal.rgb, 0.0, 1.0);
  lightTotal.a = 1.0;

  vec3 shadeWithLight = geoModeSelect(G_PACKED_NORMALS, lightTotal.rgb, cc_shade.rgb * lightTotal.rgb);
  cc_shade.rgb = geoModeSelect(G_LIGHTING, cc_shade.rgb, shadeWithLight);
  cc_shade = clamp(cc_shade, 0.0, 1.0);
  cc_shade_flat = cc_shade;

  vec2 uvGen = geoModeSelect(G_TEX_GEN, inUV, normScreen.xy * 0.5 + 0.5);

  // turn UVs ionto pixel-space, apply first tile settings
  ivec4 texSize = ivec4(textureSize(tex0, 0), textureSize(tex1, 0));
  // we simulate UVs in pixel-space, since there is only one UV set, we scale by the first texture size
  uv = uvGen.xyxy * texSize.xyxy;
  // apply material.shift from top left of texture:
  uv.yw = texSize.yw - uv.yw - 1;
  uv *= material.shift;
  uv.yw = texSize.yw - uv.yw - 1;

  uv = uv - (material.shift * 0.5) - material.low;

  tileSize = abs(material.high) - abs(material.low);

  // @TODO: uvgen (f3d + t3d)
  // forward CC (@TODO: do part of this here? e.g. prim/env/shade etc.)

  vec3 posQuant = pos;
  //posQuant = round(posQuant * 10) * 0.1;
  
  gl_Position = matMVP * vec4(posQuant, 1.0);
  posScreen = gl_Position.xy / gl_Position.w;
  
  // quantize depth to what the RDP has (16bit)
  gl_Position.z = ceil(gl_Position.z * 0x7FFF) / 0x7FFF;
  // move the depth ever so slightly to avoid z-fighting with blenders own overlays
  // e.g. transparent faces in face-edit mode, lines & points
  // don't offset decals to make manual depth checks work later
  float z = flagSelect(DRAW_FLAG_DECAL, gl_Position.z * 1.00000018, gl_Position.z);
#ifdef BLEND_EMULATION
  gl_Position.z = z;
#else // reduce offset if z-test is disabled without blend emulation
  gl_Position.z = flagSelect(DRAW_FLAG_ZTEST, z, gl_Position.z * 0.99999979);
#endif
}