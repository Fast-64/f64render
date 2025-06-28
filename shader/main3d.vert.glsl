
void main() 
{
  // Directional light
  vec3 norm = inNormal;
  vec3 normScreen = normalize(matNorm * norm);

#if VIEWSPACE_LIGHTING
  vec3 light_norm = normScreen;
#else 
  vec3 light_norm = norm;
#endif

  cc_shade = inColor;

  vec4 lightTotal = vec4(material.ambientColor.rgb, 0.0);
  for(int i=0; i<material.numLights; i++) {
    Light light = material.lights[i];
    float lightStren = max(dot(light_norm, light.dir), 0.0);
    lightTotal += light.color * lightStren;
  }

  lightTotal.rgb = clamp(lightTotal.rgb, 0.0, 1.0);
  lightTotal.a = 1.0;

  vec3 shadeWithLight = geoModeSelect(G_PACKED_NORMALS, lightTotal.rgb, cc_shade.rgb * lightTotal.rgb);
  cc_shade.rgb = geoModeSelect(G_LIGHTING, cc_shade.rgb, shadeWithLight);
  cc_shade = clamp(cc_shade, 0.0, 1.0);
  cc_shade_flat = cc_shade;

  inputUV = geoModeSelect(G_TEX_GEN, inUV, normScreen.xy * 0.5 + 0.5);

  inputUV *= material.texSize;

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

#ifdef BLEND_EMULATION
// don't offset decals to make manual depth checks work later
  gl_Position.z += drawFlagSelect(DRAW_FLAG_DECAL, 0.00006, 0.0);
#else
  gl_Position.z = gl_Position.z * 1.00000018;
#endif
}