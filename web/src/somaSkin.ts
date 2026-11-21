import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

/** Canonical NVIDIA SOMA standing Hips height in meters (Y-up). */
const SOMA_REST_HIPS_Y = 0.99804;
/** Canonical NVIDIA SOMA standing full character height in meters (Y-up). */
const SOMA_CANONICAL_HEIGHT = 1.758;
/** Default translucent hologram opacity when X-Ray skeleton inspection is enabled. */
const DEFAULT_XRAY_OPACITY = 0.22;

export type RestPoseCorrectionMode = 'none' | 'postmultiply' | 'premultiply';

export interface SomaSkinLoadOptions {
  /** Target opacity in X-Ray hologram mode (default 0.22). */
  xrayOpacity?: number;
  /** Optional hologram accent hex color when X-Ray mode is active (default '#38bdf8'). */
  xrayAccentHex?: string;
  /**
   * Optional rest-pose quaternion delta correction for non-SOMA GLBs whose bind bone
   * local axes differ from SOMA BVH identity. Defaults to `'none'` (direct SOMA parity).
   */
  restPoseCorrection?: RestPoseCorrectionMode;
  /** Optional explicit per-bone rest delta quaternions keyed by normalized bone name. */
  restPoseDeltas?: ReadonlyMap<string, THREE.Quaternion>;
}

export interface SomaCalibrationInfo {
  /** Uniform scale factor applied to the loaded GLB root (1.0 for meters-native SOMA GLB). */
  modelScale: number;
  /** Vertical grounding offset (meters) added to Hips Y so soles rest cleanly on y = 0. */
  hipsYOffset: number;
  /** Measured rest-pose Hips world height after scaling (meters). */
  restHipsHeight: number;
  /** Measured rest-pose lowest mesh Y after scaling (meters). */
  restSoleY: number;
  /** Total indexed THREE.Bone count. */
  boneCount: number;
  /** Total Mesh / SkinnedMesh count. */
  meshCount: number;
}

interface ManagedMaterialState {
  material: THREE.MeshStandardMaterial;
  baseOpacity: number;
  baseTransparent: boolean;
  baseDepthWrite: boolean;
  baseEmissive: THREE.Color;
  baseEmissiveIntensity: number;
  baseSide: THREE.Side;
}

const _vPos = new THREE.Vector3();
const _qSoma = new THREE.Quaternion();
const _qWork = new THREE.Quaternion();
const _mGroupInv = new THREE.Matrix4();
const _mHipsParentInGroup = new THREE.Matrix4();
const _pDecomp = new THREE.Vector3();
const _qDecomp = new THREE.Quaternion();
const _sDecomp = new THREE.Vector3();

/**
 * Normalize a GLB/USD/Mixamo bone name by stripping path segments, colon namespaces
 * (`mixamorig:`, `soma:`, `actor:rig:`), common prefixes (`mixamorig_`, `soma_`), and
 * Blender duplicate numeric suffixes (`.001`).
 */
export function normalizeBoneName(raw: string): string {
  let name = raw.trim();
  const slash = name.lastIndexOf('/');
  if (slash >= 0) name = name.slice(slash + 1);
  const colon = name.lastIndexOf(':');
  if (colon >= 0) name = name.slice(colon + 1);
  name = name.replace(/^(mixamorig[_.]?|soma[_.]?)/i, '');
  name = name.replace(/\.\d+$/, '');
  return name;
}

function isFlatOrDefaultMaterial(mat: THREE.Material | undefined): boolean {
  if (!mat) return true;
  if (
    mat instanceof THREE.MeshBasicMaterial ||
    mat instanceof THREE.MeshLambertMaterial
  ) {
    return true;
  }
  if (mat instanceof THREE.MeshStandardMaterial) {
    // Detect unstyled glTF default fallback material (unnamed pure white with no map)
    const isUnnamed = !mat.name || mat.name === 'default';
    const isPureWhite =
      mat.color.r > 0.98 && mat.color.g > 0.98 && mat.color.b > 0.98;
    const isUntextured = !mat.map && !mat.normalMap && !mat.roughnessMap;
    if (isUnnamed && isPureWhite && isUntextured && mat.roughness >= 0.95) {
      return true;
    }
    return false;
  }
  return true;
}

function createStudioPbrMaterial(
  hasVertexColors: boolean
): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    name: 'SOMA_StudioPBR',
    color: hasVertexColors ? 0xffffff : 0xd6dee8,
    roughness: 0.44,
    metalness: 0.08,
    vertexColors: hasVertexColors,
  });
}

function buildResourceBasePath(url: string): string {
  const clean = url.split('?')[0].split('#')[0];
  const idx = clean.lastIndexOf('/');
  return idx >= 0 ? clean.slice(0, idx + 1) : '';
}

/**
 * SOMA SkinnedMesh Character Retargeter & Renderer.
 *
 * Loads a rigged SOMA GLB (`/models/soma_human.glb`), calibrates scale and floor
 * grounding automatically, drives bone local quaternions directly from ONNX SOMA
 * poses with zero per-frame allocations, and supports smooth translucent X-Ray
 * hologram mode to reveal the MotionBricks skeleton underneath.
 */
export class SomaSkinnedCharacter {
  public readonly group = new THREE.Group();
  public readonly modelRoot: THREE.Object3D;
  public readonly bones: ReadonlyMap<string, THREE.Bone>;
  public readonly boneNames: readonly string[];
  public readonly skinnedMeshes: readonly THREE.SkinnedMesh[];
  public readonly calibration: SomaCalibrationInfo;

  private readonly boneMap = new Map<string, THREE.Bone>();
  private readonly restLocalQuats = new Map<string, THREE.Quaternion>();
  private readonly restLocalPositions = new Map<string, THREE.Vector3>();
  private readonly hipsBone: THREE.Bone | null;
  private readonly hipsParentInvMatrix = new THREE.Matrix4();
  private readonly hipsParentInvQuat = new THREE.Quaternion();

  private readonly allMeshes: THREE.Mesh[] = [];
  private readonly managedMaterials: ManagedMaterialState[] = [];

  private restPoseMode: RestPoseCorrectionMode;
  private customRestDeltas: ReadonlyMap<string, THREE.Quaternion> | null;
  private hipsYOffset: number;

  // Cached bone lookup aligned to caller's `boneNames` array for zero-GC hot loop
  private cachedInputNames: readonly string[] | null = null;
  private cachedInputLen = -1;
  private cachedBones: Array<THREE.Bone | null> = [];
  private cachedRestDeltas: Array<THREE.Quaternion | null> = [];
  private cachedHipsIndex = -1;

  // Smooth X-Ray hologram state
  private xrayTarget = 0;
  private xrayMix = 0;
  private readonly xrayOpacity: number;
  private readonly xrayAccent = new THREE.Color('#38bdf8');
  private lastXRayTickMs = 0;
  private updatedExternally = false;

  // Articulated 48-bone SOMA hand fist curling (relaxed 0.18 in locomotion, 1.0 tight boxer fist in combat/punch)
  private readonly fingerEntries: Array<{
    bone: THREE.Bone;
    restQuat: THREE.Quaternion;
    fistQuat: THREE.Quaternion;
  }> = [];
  private fistTarget = 0.18;
  private fistMix = 0.18;

  private constructor(
    sceneRoot: THREE.Object3D,
    options: SomaSkinLoadOptions = {}
  ) {
    this.modelRoot = sceneRoot;
    this.group.name = 'SomaSkinnedCharacter';
    this.group.add(this.modelRoot);

    this.xrayOpacity = options.xrayOpacity ?? DEFAULT_XRAY_OPACITY;
    if (options.xrayAccentHex) {
      this.xrayAccent.set(options.xrayAccentHex);
    }
    this.restPoseMode = options.restPoseCorrection ?? 'none';
    this.customRestDeltas = options.restPoseDeltas ?? null;

    const skinnedList: THREE.SkinnedMesh[] = [];
    const seenMaterials = new Set<THREE.Material>();

    // 1. Traverse scene: enable shadows, disable SkinnedMesh frustum culling,
    //    ensure studio PBR finish, and index every THREE.Bone.
    this.modelRoot.traverse((obj) => {
      if (obj instanceof THREE.Bone) {
        this.registerBone(obj);
      }

      if (obj instanceof THREE.Mesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
        if (!obj.geometry.attributes.normal) {
          obj.geometry.computeVertexNormals();
        }
        if (obj instanceof THREE.SkinnedMesh) {
          // SkinnedMesh bind bounds stay near origin while Hips travels across the studio
          obj.frustumCulled = false;
          skinnedList.push(obj);
          if (obj.skeleton) {
            for (const bone of obj.skeleton.bones) {
              this.registerBone(bone);
            }
          }
        }
        this.allMeshes.push(obj);
        this.prepareMeshMaterials(obj, seenMaterials);
      }
    });

    this.skinnedMeshes = skinnedList;
    this.bones = this.boneMap;
    this.boneNames = Array.from(this.boneMap.keys());
    this.hipsBone =
      this.boneMap.get('Hips') ??
      this.findBoneCaseInsensitive('hips') ??
      this.findRootBone();

    // 2. Auto-measure height/scale and floor grounding calibration
    this.group.updateMatrixWorld(true);
    let box = new THREE.Box3().setFromObject(this.modelRoot);
    let meshHeight = box.isEmpty() ? SOMA_CANONICAL_HEIGHT : box.max.y - box.min.y;
    let modelScale = 1.0;

    if (meshHeight > 10.0) {
      // Raw cm (100..250) or mm export without root unit scale
      modelScale =
        meshHeight >= 100 && meshHeight <= 250
          ? 0.01
          : SOMA_CANONICAL_HEIGHT / meshHeight;
    } else if (meshHeight > 0.01 && (meshHeight < 1.35 || meshHeight > 2.25)) {
      // Non-human-scale custom rig: normalize to SOMA canonical height
      modelScale = SOMA_CANONICAL_HEIGHT / meshHeight;
    }

    if (Math.abs(modelScale - 1.0) > 1e-4) {
      this.modelRoot.scale.multiplyScalar(modelScale);
      this.group.updateMatrixWorld(true);
      box = new THREE.Box3().setFromObject(this.modelRoot);
      meshHeight = box.isEmpty() ? SOMA_CANONICAL_HEIGHT : box.max.y - box.min.y;
    }

    const restSoleY = box.isEmpty() ? 0 : box.min.y;
    let restHipsHeight = SOMA_REST_HIPS_Y;
    if (this.hipsBone) {
      this.hipsBone.getWorldPosition(_vPos);
      restHipsHeight = _vPos.y;
    }

    // Distance from Hips down to the lowest sole vertex in rest pose.
    // Calibrates vertical offset so feet rest cleanly on y = 0 regardless of shoe thickness
    // or pelvis-centered vs floor-centered GLB origins.
    const hipsToSole = restHipsHeight - restSoleY;
    let hipsYOffset = 0;
    if (hipsToSole > 0.3 && hipsToSole < 1.8) {
      hipsYOffset = hipsToSole - SOMA_REST_HIPS_Y;
      // Clamp extreme outliers so malformed bounds can never catapult the character
      hipsYOffset = Math.max(-0.25, Math.min(0.25, hipsYOffset));
    } else {
      hipsYOffset = -restSoleY;
    }
    this.hipsYOffset = hipsYOffset;

    // 3. Precompute inverse transform of Hips.parent relative to `this.group`
    //    so `hipsWorldPos` & root quat map cleanly even if an Armature wrapper
    //    node carries scale (e.g. 0.01) or Z-up -> Y-up rotation.
    this.computeHipsParentBasis();
    this.buildFingerCurlRig();
    this.applyHandFist();

    this.calibration = {
      modelScale,
      hipsYOffset: this.hipsYOffset,
      restHipsHeight,
      restSoleY,
      boneCount: this.boneMap.size,
      meshCount: this.allMeshes.length,
    };
  }

  /**
   * Asynchronously load a SOMA-rigged GLB/GLTF character.
   * Returns `null` gracefully if the asset is missing (404), HTML-fallback, or invalid.
   */
  public static async load(
    url = '/models/soma_human.glb',
    options: SomaSkinLoadOptions = {}
  ): Promise<SomaSkinnedCharacter | null> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return null;
      }

      const contentType = response.headers.get('content-type') ?? '';
      if (contentType.includes('text/html')) {
        // Vite dev SPA fallback returns 200 HTML for missing public files
        return null;
      }

      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 12) {
        return null;
      }

      const lowerUrl = url.split('?')[0].toLowerCase();
      if (lowerUrl.endsWith('.glb')) {
        // Verify glTF binary magic header ('glTF' == 0x46546C67 LE)
        const magic = new DataView(buffer, 0, 4).getUint32(0, true);
        if (magic !== 0x46546c67) {
          return null;
        }
      }

      const loader = new GLTFLoader();
      const gltf = await loader.parseAsync(buffer, buildResourceBasePath(url));
      if (!gltf?.scene) {
        return null;
      }

      return new SomaSkinnedCharacter(gltf.scene, options);
    } catch {
      return null;
    }
  }

  private registerBone(bone: THREE.Bone): void {
    const normalized = normalizeBoneName(bone.name);
    if (!normalized) return;
    if (!this.boneMap.has(normalized)) {
      this.boneMap.set(normalized, bone);
      this.restLocalQuats.set(normalized, bone.quaternion.clone());
      this.restLocalPositions.set(normalized, bone.position.clone());
    }
  }

  private findBoneCaseInsensitive(lowerName: string): THREE.Bone | null {
    for (const [k, bone] of this.boneMap) {
      if (k.toLowerCase() === lowerName) return bone;
    }
    return null;
  }

  private findRootBone(): THREE.Bone | null {
    for (const bone of this.boneMap.values()) {
      if (!(bone.parent instanceof THREE.Bone)) {
        return bone;
      }
    }
    return null;
  }

  private prepareMeshMaterials(
    mesh: THREE.Mesh,
    seen: Set<THREE.Material>
  ): void {
    const hasVertexColors = Boolean(mesh.geometry.attributes.color);
    const applyOne = (mat: THREE.Material | undefined): THREE.MeshStandardMaterial => {
      if (isFlatOrDefaultMaterial(mat)) {
        const pbr = createStudioPbrMaterial(hasVertexColors);
        if (mat instanceof THREE.MeshBasicMaterial || mat instanceof THREE.MeshLambertMaterial) {
          if (mat.map) pbr.map = mat.map;
          if (!hasVertexColors && (mat.color.r < 0.98 || mat.color.g < 0.98 || mat.color.b < 0.98)) {
            pbr.color.copy(mat.color);
          }
        }
        return pbr;
      }
      const std = mat as THREE.MeshStandardMaterial;
      // Calibrate SOMA athlete materials for rich sculptural studio lighting (prevents chalk-white skin blowout)
      const matName = (std.name || '').toLowerCase();
      const meshName = (mesh.name || '').toLowerCase();
      if (matName.includes('eye')) {
        std.color.set('#0b0d12');
        std.roughness = 0.08;
        std.metalness = 0.05;
        std.envMapIntensity = 1.25;
      } else if (matName.includes('skin') || meshName === 'soma_human' || meshName.includes('body')) {
        std.color.set('#b48264');
        std.roughness = 0.54;
        std.metalness = 0.02;
        std.envMapIntensity = 0.55;
      } else if (matName.includes('tee') || meshName.includes('techtee') || meshName.includes('shirt')) {
        std.roughness = 0.82;
        std.metalness = 0.04;
        std.envMapIntensity = 0.55;
      } else if (matName.includes('short') || meshName.includes('short')) {
        std.roughness = 0.76;
        std.metalness = 0.05;
        std.envMapIntensity = 0.5;
      }
      return std;
    };

    if (Array.isArray(mesh.material)) {
      const upgraded = mesh.material.map((m) => applyOne(m));
      mesh.material = upgraded;
      for (const m of upgraded) this.trackMaterial(m, seen);
    } else {
      const upgraded = applyOne(mesh.material);
      mesh.material = upgraded;
      this.trackMaterial(upgraded, seen);
    }
  }

  private trackMaterial(
    material: THREE.MeshStandardMaterial,
    seen: Set<THREE.Material>
  ): void {
    if (seen.has(material)) return;
    seen.add(material);
    this.managedMaterials.push({
      material,
      baseOpacity: material.opacity,
      baseTransparent: material.transparent,
      baseDepthWrite: material.depthWrite,
      baseEmissive: material.emissive.clone(),
      baseEmissiveIntensity: material.emissiveIntensity,
      baseSide: material.side,
    });
  }

  private computeHipsParentBasis(): void {
    this.hipsParentInvMatrix.identity();
    this.hipsParentInvQuat.identity();

    if (!this.hipsBone || !this.hipsBone.parent) return;

    this.group.updateMatrixWorld(true);
    _mGroupInv.copy(this.group.matrixWorld).invert();
    _mHipsParentInGroup.multiplyMatrices(
      _mGroupInv,
      this.hipsBone.parent.matrixWorld
    );
    this.hipsParentInvMatrix.copy(_mHipsParentInGroup).invert();
    _mHipsParentInGroup.decompose(_pDecomp, _qDecomp, _sDecomp);
    this.hipsParentInvQuat.copy(_qDecomp).invert();
  }

  private resolveBoneCache(boneNames: readonly string[]): void {
    if (
      this.cachedInputNames === boneNames &&
      this.cachedInputLen === boneNames.length
    ) {
      return;
    }

    const n = boneNames.length;
    this.cachedInputNames = boneNames;
    this.cachedInputLen = n;
    this.cachedBones = new Array<THREE.Bone | null>(n);
    this.cachedRestDeltas = new Array<THREE.Quaternion | null>(n);
    this.cachedHipsIndex = -1;

    for (let i = 0; i < n; i++) {
      const normalized = normalizeBoneName(boneNames[i]);
      const bone =
        this.boneMap.get(normalized) ??
        this.findBoneCaseInsensitive(normalized.toLowerCase());
      this.cachedBones[i] = bone;

      if (bone && bone === this.hipsBone) {
        this.cachedHipsIndex = i;
      }

      const custom = this.customRestDeltas?.get(normalized) ?? null;
      if (custom) {
        this.cachedRestDeltas[i] = custom;
      } else if (this.restPoseMode !== 'none') {
        this.cachedRestDeltas[i] = this.restLocalQuats.get(normalized) ?? null;
      } else {
        this.cachedRestDeltas[i] = null;
      }
    }
  }

  /**
   * Configure optional rest-pose quaternion delta correction for GLBs whose rest
   * bone frames differ from SOMA BVH identity. Pass `'none'` for SOMA-native GLBs.
   */
  public setRestPoseCorrection(
    mode: RestPoseCorrectionMode,
    customDeltas?: ReadonlyMap<string, THREE.Quaternion>
  ): void {
    this.restPoseMode = mode;
    this.customRestDeltas = customDeltas ?? null;
    this.cachedInputNames = null;
  }

  /** Override the auto-calibrated vertical grounding offset (meters). */
  public setGroundOffset(offsetMeters: number): void {
    this.hipsYOffset = offsetMeters;
  }

  /** Sync subtle X-Ray hologram tint with the active MotionBricks locomotion style color. */
  public setAccentColor(hex: string): void {
    this.xrayAccent.set(hex);
    if (this.xrayMix > 0.005) {
      this.applyXRayVisuals();
    }
  }

  public setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  /**
   * Smoothly toggle translucent hologram X-Ray mode (`opacity = 0.22`) so the user
   * can inspect the live NVIDIA MotionBricks skeleton inside the skinned character.
   */
  public setXRayMode(xray: boolean, immediate = false): void {
    this.xrayTarget = xray ? 1 : 0;
    if (immediate) {
      this.xrayMix = this.xrayTarget;
      this.applyXRayVisuals();
    }
  }

  public isXRayMode(): boolean {
    return this.xrayTarget > 0.5;
  }

  /**
   * Advance smooth visual transitions (such as X-Ray hologram opacity fade).
   * Safe to call every frame in the render loop; also auto-ticks inside `applySomaPose`
   * if not called explicitly.
   */
  private buildFingerCurlRig(): void {
    const deg = THREE.MathUtils.degToRad;
    const fingerSpecs: Array<{ suffix: string; rx: number; ry: number; rz: number }> = [
      // MCP knuckles (2), PIP mid-joints (3), DIP tips (4) — verified SOMA -Y flexion into palm on both hands
      { suffix: 'HandIndex2', rx: 0, ry: -86, rz: 4 },
      { suffix: 'HandIndex3', rx: 0, ry: -96, rz: 0 },
      { suffix: 'HandIndex4', rx: 0, ry: -62, rz: 0 },
      { suffix: 'HandMiddle2', rx: 0, ry: -90, rz: 0 },
      { suffix: 'HandMiddle3', rx: 0, ry: -100, rz: 0 },
      { suffix: 'HandMiddle4', rx: 0, ry: -66, rz: 0 },
      { suffix: 'HandRing2', rx: 0, ry: -92, rz: -3 },
      { suffix: 'HandRing3', rx: 0, ry: -100, rz: 0 },
      { suffix: 'HandRing4', rx: 0, ry: -66, rz: 0 },
      { suffix: 'HandPinky2', rx: 0, ry: -94, rz: -6 },
      { suffix: 'HandPinky3', rx: 0, ry: -98, rz: 0 },
      { suffix: 'HandPinky4', rx: 0, ry: -64, rz: 0 },
      // Thumb wraps firmly across the curled index/middle knuckles
      { suffix: 'HandThumb1', rx: 24, ry: -46, rz: -26 },
      { suffix: 'HandThumb2', rx: 0, ry: -52, rz: 0 },
      { suffix: 'HandThumb3', rx: 0, ry: -60, rz: 0 },
    ];

    const euler = new THREE.Euler();
    const delta = new THREE.Quaternion();
    for (const side of ['Left', 'Right'] as const) {
      for (const spec of fingerSpecs) {
        const name = `${side}${spec.suffix}`;
        const bone = this.boneMap.get(name);
        if (!bone) continue;
        const restQuat = (this.restLocalQuats.get(name) ?? bone.quaternion).clone();
        euler.set(deg(spec.rx), deg(spec.ry), deg(spec.rz), 'XYZ');
        delta.setFromEuler(euler);
        const fistQuat = restQuat.clone().multiply(delta).normalize();
        this.fingerEntries.push({ bone, restQuat, fistQuat });
      }
    }
  }

  /**
   * Smoothly curl all 48 SOMA finger bones between open palm (0), natural relaxed curl (0.18),
   * and tight boxer's clenched fist (1.0).
   */
  public setHandFistBlend(target01: number, immediate = false): void {
    this.fistTarget = THREE.MathUtils.clamp(target01, 0, 1);
    if (immediate) {
      this.fistMix = this.fistTarget;
      this.applyHandFist();
    }
  }

  private stepHandFist(dt: number): void {
    const diff = this.fistTarget - this.fistMix;
    if (Math.abs(diff) < 1e-4) return;
    const clampedDt = dt > 0 ? Math.min(dt, 0.05) : 1 / 60;
    this.fistMix += diff * (1 - Math.exp(-clampedDt * 18));
    if (Math.abs(this.fistTarget - this.fistMix) < 0.004) {
      this.fistMix = this.fistTarget;
    }
    this.applyHandFist();
  }

  private applyHandFist(): void {
    const mix = this.fistMix;
    for (let i = 0; i < this.fingerEntries.length; i++) {
      const f = this.fingerEntries[i];
      f.bone.quaternion.copy(f.restQuat).slerp(f.fistQuat, mix);
    }
  }

  public update(dt: number): void {
    this.updatedExternally = true;
    this.stepXRayTransition(dt);
    this.stepHandFist(dt);
  }

  private stepXRayTransition(dt: number): void {
    const diff = this.xrayTarget - this.xrayMix;
    if (Math.abs(diff) < 1e-4) {
      if (this.xrayMix !== this.xrayTarget) {
        this.xrayMix = this.xrayTarget;
        this.applyXRayVisuals();
      }
      return;
    }

    const clampedDt = dt > 0 ? Math.min(dt, 0.05) : 1 / 60;
    this.xrayMix += diff * (1 - Math.exp(-clampedDt * 12));
    if (Math.abs(this.xrayTarget - this.xrayMix) < 0.004) {
      this.xrayMix = this.xrayTarget;
    }
    this.applyXRayVisuals();
  }

  private applyXRayVisuals(): void {
    const mix = this.xrayMix;
    const isTranslucent = mix > 0.005;
    const castShadow = mix < 0.5;

    for (let i = 0; i < this.managedMaterials.length; i++) {
      const entry = this.managedMaterials[i];
      const mat = entry.material;

      if (isTranslucent) {
        if (!mat.transparent) {
          mat.transparent = true;
          mat.needsUpdate = true;
        }
        mat.depthWrite = false;
        mat.opacity =
          entry.baseOpacity + (this.xrayOpacity - entry.baseOpacity) * mix;
        mat.emissive.copy(entry.baseEmissive).lerp(this.xrayAccent, mix * 0.65);
        mat.emissiveIntensity =
          entry.baseEmissiveIntensity + (0.28 - entry.baseEmissiveIntensity) * mix;
      } else {
        if (mat.transparent !== entry.baseTransparent) {
          mat.transparent = entry.baseTransparent;
          mat.needsUpdate = true;
        }
        mat.depthWrite = entry.baseDepthWrite;
        mat.opacity = entry.baseOpacity;
        mat.emissive.copy(entry.baseEmissive);
        mat.emissiveIntensity = entry.baseEmissiveIntensity;
      }
    }

    for (let i = 0; i < this.allMeshes.length; i++) {
      const mesh = this.allMeshes[i];
      mesh.castShadow = castShadow;
      mesh.renderOrder = isTranslucent ? 2 : 0;
    }
  }

  /**
   * Apply a live SOMA pose to the skinned character with zero per-frame heap allocations.
   *
   * @param boneNames      Ordered joint names matching `boneLocalQuats` (e.g. 30 ONNX joints or 77 SOMA joints).
   * @param hipsWorldPos   Hips world position `[x, y, z]` (or flat `Float32Array` where `[0..2]` is Hips) in meters.
   * @param boneLocalQuats Flat `Float32Array(boneNames.length * 4)` of local XYZW quaternions (`Hips` in world space).
   */
  public applySomaPose(
    boneNames: string[],
    hipsWorldPos: readonly [number, number, number] | Float32Array,
    boneLocalQuats: Float32Array
  ): void {
    // Auto-step X-Ray fade and hand fist curl smoothly even if caller does not call `update(dt)` separately
    const nowMs = performance.now();
    if (!this.updatedExternally) {
      const dt =
        this.lastXRayTickMs > 0 ? (nowMs - this.lastXRayTickMs) / 1000 : 1 / 60;
      if (this.xrayMix !== this.xrayTarget) this.stepXRayTransition(dt);
      if (this.fistMix !== this.fistTarget) this.stepHandFist(dt);
    }
    this.lastXRayTickMs = nowMs;
    this.updatedExternally = false;

    this.resolveBoneCache(boneNames);

    // 1. Apply calibrated world position to Hips (converted into Hips.parent local frame)
    if (this.hipsBone && hipsWorldPos.length >= 3) {
      _vPos
        .set(
          hipsWorldPos[0],
          hipsWorldPos[1] + this.hipsYOffset,
          hipsWorldPos[2]
        )
        .applyMatrix4(this.hipsParentInvMatrix);
      this.hipsBone.position.copy(_vPos);
    }

    // 2. Apply root and child bone local quaternions (XYZW)
    const count = boneNames.length;
    const maxBones = boneLocalQuats.length >>> 2;
    const limit = count < maxBones ? count : maxBones;
    const mode = this.restPoseMode;

    for (let i = 0; i < limit; i++) {
      const bone = this.cachedBones[i];
      if (!bone) continue;

      const o = i * 4;
      const qx = boneLocalQuats[o];
      const qy = boneLocalQuats[o + 1];
      const qz = boneLocalQuats[o + 2];
      const qw = boneLocalQuats[o + 3];

      const lenSq = qx * qx + qy * qy + qz * qz + qw * qw;
      if (lenSq > 1e-12) {
        const inv = 1 / Math.sqrt(lenSq);
        _qSoma.set(qx * inv, qy * inv, qz * inv, qw * inv);
      } else {
        _qSoma.identity();
      }

      const restDelta = this.cachedRestDeltas[i];
      if (restDelta) {
        if (mode === 'premultiply') {
          _qWork.copy(restDelta).multiply(_qSoma);
          _qSoma.copy(_qWork);
        } else {
          // Default delta convention: post-multiply rest frame alignment
          _qSoma.multiply(restDelta);
        }
      }

      if (i === this.cachedHipsIndex) {
        // Transform root world orientation into Hips.parent's local frame
        bone.quaternion.copy(this.hipsParentInvQuat).multiply(_qSoma);
      } else {
        bone.quaternion.copy(_qSoma);
      }
    }
  }

  /** Restore all indexed bones to their original GLB rest transforms. */
  public resetToRestPose(): void {
    for (const [name, bone] of this.boneMap) {
      const p = this.restLocalPositions.get(name);
      const q = this.restLocalQuats.get(name);
      if (p) bone.position.copy(p);
      if (q) bone.quaternion.copy(q);
    }
  }

  /** Dispose GPU geometries and materials owned by this skinned character. */
  public dispose(): void {
    const disposedGeo = new Set<THREE.BufferGeometry>();
    for (const mesh of this.allMeshes) {
      if (!disposedGeo.has(mesh.geometry)) {
        disposedGeo.add(mesh.geometry);
        mesh.geometry.dispose();
      }
    }
    for (const entry of this.managedMaterials) {
      entry.material.dispose();
    }
  }
}
