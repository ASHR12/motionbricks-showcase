import * as THREE from 'three';
import { PAVILION_SPEC, STUDIO_THEMES } from './studioSpec';

/**
 * Polished architectural micro-cement mocap studio floor:
 * - 2-octave fbm poured-concrete organic mottling (±3.5% luminance) + subtle per-slab variation
 * - Crisp screen-space fwidth 2.0 m architectural expansion-joint seams + subtle 0.5 m mocap ticks
 * - Warm radial overhead skylight reflection pool under the pavilion center + wall-base contact AO
 * - Color-managed ACES/sRGB output with seamless theme switching (dark concrete & warm travertine light)
 */
export function createStudioFloor(): {
  group: THREE.Group;
  hitPlane: THREE.Mesh;
  updateCenter: (x: number, z: number) => void;
  setTheme: (light: boolean) => void;
} {
  const group = new THREE.Group();
  const darkTheme = STUDIO_THEMES.dark;

  const floorMat = new THREE.ShaderMaterial({
    transparent: false,
    depthWrite: true,
    uniforms: {
      uCenter: { value: new THREE.Vector2(0, 0) },
      uFloorBase: { value: new THREE.Color(darkTheme.floorBase) },
      uMinorColor: { value: new THREE.Color(darkTheme.floorMinor) },
      uMajorColor: { value: new THREE.Color(darkTheme.floorMajor) },
      uAxisColor: { value: new THREE.Color(darkTheme.floorAxis) },
      uFogColor: { value: new THREE.Color(darkTheme.fog) },
      uFogDensity: { value: darkTheme.fogDensity },
      uSkylightColor: { value: new THREE.Color(darkTheme.skylightEmissive) },
      uIsLight: { value: 0.0 },
      uBackWallZ: { value: PAVILION_SPEC.backWallZ },
      uFrontWallZ: { value: PAVILION_SPEC.frontWallZ },
      uLeftWallX: { value: PAVILION_SPEC.leftWallX },
      uRightColX: { value: PAVILION_SPEC.rightColonnadeX },
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vWorldPos;
      uniform vec2 uCenter;
      uniform vec3 uFloorBase;
      uniform vec3 uMinorColor;
      uniform vec3 uMajorColor;
      uniform vec3 uAxisColor;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform vec3 uSkylightColor;
      uniform float uIsLight;
      uniform float uBackWallZ;
      uniform float uFrontWallZ;
      uniform float uLeftWallX;
      uniform float uRightColX;

      float hash21(vec2 p) {
        p = fract(p * vec2(234.34, 435.345));
        p += dot(p, p + 34.23);
        return fract(p.x * p.y);
      }

      float valueNoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        float a = hash21(i);
        float b = hash21(i + vec2(1.0, 0.0));
        float c = hash21(i + vec2(0.0, 1.0));
        float d = hash21(i + vec2(1.0, 1.0));
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
      }

      // 2-octave architectural micro-cement fbm
      float fbm2(vec2 p) {
        float v = 0.0;
        v += 0.65 * valueNoise(p);
        v += 0.35 * valueNoise(p * 2.13 + vec2(17.4, 9.2));
        return v;
      }

      float gridLine(vec2 coord, float scale, float lineWidth) {
        vec2 p = coord / scale;
        vec2 fw = max(fwidth(p), vec2(0.0005));
        vec2 g = abs(fract(p - 0.5) - 0.5) / fw;
        float d = min(g.x, g.y);
        return 1.0 - min(d / lineWidth, 1.0);
      }

      // Precision 0.5 m mocap calibration cross-ticks (+) at grid intersections
      float calibCrossTicks(vec2 coord, float scale, float halfArm, float lineWidth) {
        vec2 cell = (fract(coord / scale + 0.5) - 0.5) * scale;
        vec2 fw = max(fwidth(coord), vec2(0.0005));
        float armX = 1.0 - smoothstep(halfArm - fw.x, halfArm + fw.x, abs(cell.x));
        float armZ = 1.0 - smoothstep(halfArm - fw.y, halfArm + fw.y, abs(cell.y));
        float lineH = (1.0 - min(abs(cell.y) / (fw.y * lineWidth), 1.0)) * armX;
        float lineV = (1.0 - min(abs(cell.x) / (fw.x * lineWidth), 1.0)) * armZ;
        return max(lineH, lineV);
      }

      void main() {
        vec2 xz = vWorldPos.xz;
        float distChar = length(xz - uCenter);
        float distPavilion = length(xz - vec2(0.0, -3.1));
        float detailFade = 1.0 - smoothstep(14.0, 34.0, distChar);

        // 1. Poured architectural micro-cement organic mottling (±3.5% luminance) + per-2m-slab cast shift
        float mottleMacro = fbm2(xz * 0.42) - 0.5;
        float mottleMicro = (valueNoise(xz * 3.1) - 0.5) * 0.35;
        float slabTint = (hash21(floor(xz / 2.0)) - 0.5) * 0.036;
        float concreteMod = 1.0 + (mottleMacro * 0.07 + mottleMicro * 0.025 + slabTint) * mix(1.0, 0.55, uIsLight);

        vec3 col = uFloorBase * concreteMod;

        // 2. Warm radial overhead skylight pool beneath pavilion skylight bays (centered at x=0, z=-3.1)
        vec2 skyDelta = (xz - vec2(0.0, -3.1)) * vec2(0.24, 0.165);
        float skylightPool = exp(-dot(skyDelta, skyDelta) * 1.1);

        // Honed-cement grazing sheen + elongated soft vertical reflection streak along camera view ray toward (0.0, -5.0)
        vec3 viewVec = cameraPosition - vWorldPos;
        float viewDist = length(viewVec);
        vec3 viewDir = viewVec / max(viewDist, 0.0001);
        float grazing = pow(1.0 - clamp(viewDir.y, 0.0, 1.0), 2.1);

        vec2 rayFwd = normalize(vec2(0.0, -5.0) - cameraPosition.xz + vec2(0.0, -0.0001));
        vec2 rayRight = vec2(-rayFwd.y, rayFwd.x);
        vec2 toAnchor = xz - vec2(0.0, -5.0);
        float alongRay = dot(toAnchor, rayFwd);
        float acrossRay = dot(toAnchor, rayRight);
        float streakAcross = exp(-acrossRay * acrossRay * 0.14);
        float streakAlong = exp(-pow((alongRay + 1.4) * 0.19, 2.0));
        float viewStreak = streakAcross * streakAlong * (0.38 + 0.62 * grazing);

        // Subtle troweled micro-cement sheen breakup so reflection feels organic and honed
        float sheenTexture = 0.92 + 0.16 * (mottleMacro + 0.5);
        float sheenFactor = ((0.58 + 0.42 * grazing) * skylightPool + viewStreak * 0.48) * sheenTexture;

        // Warm overhead skylight bounce lift (calibrated in linear space: cool slate floor + warm sheen)
        float poolStrength = mix(0.115, 0.085, uIsLight);
        col += uSkylightColor * sheenFactor * poolStrength;

        // Subtle cool-warm studio grounding falloff away from center toward perimeter walls
        float perimeterDim = mix(1.0, mix(0.72, 0.90, uIsLight), smoothstep(5.2, 14.5, distPavilion));
        col *= perimeterDim;

        // Subtle architectural floor-wall contact AO along all 4 perimeter walls & Right Colonnade
        float backAO = mix(0.76, 1.0, smoothstep(0.0, 1.45, xz.y - uBackWallZ));
        float frontAO = mix(0.76, 1.0, smoothstep(0.0, 1.45, uFrontWallZ - xz.y));
        float leftAO = mix(0.78, 1.0, smoothstep(0.0, 1.45, xz.x - uLeftWallX));
        float rightColBand = 1.0 - smoothstep(0.0, 1.35, abs(xz.x - uRightColX));
        float rightAO = 1.0 - rightColBand * 0.16 * step(uBackWallZ, xz.y) * step(xz.y, uFrontWallZ);
        col *= backAO * frontAO * leftAO * rightAO;

        // 3. Refined gallery-grade 2.0 m architectural concrete expansion-joint slab seams
        float majorJoint = gridLine(xz, 2.0, 1.2);
        float majorBevel = gridLine(xz + vec2(0.012, 0.012), 2.0, 0.9) * (1.0 - majorJoint);

        if (uIsLight > 0.5) {
          // Warm travertine light mode: subtle honed stone expansion joints
          vec3 darkJointCol = uMajorColor * 0.72;
          col = mix(col, darkJointCol, majorJoint * 0.58 * detailFade);
          col = mix(col, uFloorBase * 1.05, majorBevel * 0.22 * detailFade);
        } else {
          // Dark honed concrete mode: understated architectural saw-cut joints
          col = mix(col, uFloorBase * 0.38, majorJoint * 0.45 * detailFade);
          col = mix(col, uMajorColor, majorBevel * 0.22 * detailFade);
        }

        // 4. Whisper-fine 0.5 m mocap calibration cross-ticks + barely-there sub-grid (zero visual clutter)
        float arenaFocus = 1.0 - smoothstep(6.0, 14.5, distPavilion);
        float minorLine = gridLine(xz, 0.5, 0.85) * 0.04 * arenaFocus;
        float minorTicks = calibCrossTicks(xz, 0.5, 0.042, 1.15) * 0.14 * arenaFocus;
        vec3 tickCol = mix(uMinorColor * 1.18, uAxisColor, uIsLight);
        col = mix(col, uMinorColor, minorLine * detailFade);
        col = mix(col, tickCol, minorTicks * detailFade);

        // Whisper-subtle studio datum axes within the primary capture zone
        vec2 fwXZ = max(fwidth(xz), vec2(0.0005));
        float axX = (1.0 - min(abs(xz.y) / (fwXZ.y * 1.25), 1.0)) * (1.0 - smoothstep(5.0, 7.5, abs(xz.x)));
        float axZ = (1.0 - min(abs(xz.x) / (fwXZ.x * 1.25), 1.0)) * (1.0 - smoothstep(5.0, 7.5, abs(xz.y)));
        float axisMask = max(axX, axZ) * 0.16 * detailFade;
        col = mix(col, uAxisColor, axisMask);

        // 5. Seamless exponential scene fog matching scene.fog (FogExp2)
        float fogFactor = 1.0 - exp(-uFogDensity * uFogDensity * viewDist * viewDist);
        col = mix(col, uFogColor, clamp(fogFactor, 0.0, 1.0));

        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });

  const gridMesh = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), floorMat);
  gridMesh.rotation.x = -Math.PI / 2;
  gridMesh.position.y = -0.002;
  group.add(gridMesh);

  // Grounded soft shadow receiver plane right on y = 0
  const shadowMat = new THREE.ShadowMaterial({ opacity: darkTheme.shadowOpacity });
  const shadowMesh = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), shadowMat);
  shadowMesh.rotation.x = -Math.PI / 2;
  shadowMesh.position.y = 0.0;
  shadowMesh.receiveShadow = true;
  group.add(shadowMesh);

  // Invisible raycast plane for Click-to-Move
  const hitPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshBasicMaterial({ visible: false })
  );
  hitPlane.rotation.x = -Math.PI / 2;
  group.add(hitPlane);

  return {
    group,
    hitPlane,
    updateCenter(x: number, z: number) {
      floorMat.uniforms.uCenter.value.set(x, z);
      gridMesh.position.x = Math.floor(x / 2) * 2;
      gridMesh.position.z = Math.floor(z / 2) * 2;
      shadowMesh.position.x = x;
      shadowMesh.position.z = z;
    },
    setTheme(light: boolean) {
      const theme = STUDIO_THEMES[light ? 'light' : 'dark'];
      floorMat.uniforms.uFloorBase.value.set(theme.floorBase);
      floorMat.uniforms.uMinorColor.value.set(theme.floorMinor);
      floorMat.uniforms.uMajorColor.value.set(theme.floorMajor);
      floorMat.uniforms.uAxisColor.value.set(theme.floorAxis);
      floorMat.uniforms.uFogColor.value.set(theme.fog);
      floorMat.uniforms.uFogDensity.value = theme.fogDensity;
      floorMat.uniforms.uSkylightColor.value.set(theme.skylightEmissive);
      floorMat.uniforms.uIsLight.value = light ? 1.0 : 0.0;
      shadowMat.opacity = theme.shadowOpacity;
    },
  };
}
