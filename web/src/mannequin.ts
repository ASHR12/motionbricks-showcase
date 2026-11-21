import * as THREE from 'three';

export type Vec3Tuple = [number, number, number];
export type QuatTuple = [number, number, number, number];

// Joint indices matching RENDER_JOINT_NAMES in server/api.py
export const J = {
  Hips: 0,
  Spine1: 1,
  Spine2: 2,
  Chest: 3,
  Neck1: 4,
  Neck2: 5,
  Head: 6,
  HeadEnd: 7,
  LeftShoulder: 8,
  LeftArm: 9,
  LeftForeArm: 10,
  LeftHand: 11,
  RightShoulder: 12,
  RightArm: 13,
  RightForeArm: 14,
  RightHand: 15,
  LeftLeg: 16,
  LeftShin: 17,
  LeftFoot: 18,
  LeftToeBase: 19,
  RightLeg: 20,
  RightShin: 21,
  RightFoot: 22,
  RightToeBase: 23,
} as const;

interface LimbDef {
  a: number;
  b: number;
  rTop: number;
  rBot: number;
  material: 'shell' | 'carbon' | 'accent';
}

const LIMBS: LimbDef[] = [
  // Spine & Neck (sleek carbon inner waist under athletic chest carapace)
  { a: J.Hips, b: J.Spine1, rTop: 0.082, rBot: 0.088, material: 'carbon' },
  { a: J.Spine1, b: J.Spine2, rTop: 0.084, rBot: 0.082, material: 'carbon' },
  { a: J.Spine2, b: J.Chest, rTop: 0.092, rBot: 0.084, material: 'carbon' },
  { a: J.Chest, b: J.Neck1, rTop: 0.054, rBot: 0.088, material: 'carbon' },
  { a: J.Neck1, b: J.Neck2, rTop: 0.044, rBot: 0.048, material: 'carbon' },
  { a: J.Neck2, b: J.Head, rTop: 0.044, rBot: 0.042, material: 'carbon' },

  // Shoulders & Arms
  { a: J.Chest, b: J.LeftShoulder, rTop: 0.044, rBot: 0.056, material: 'carbon' },
  { a: J.LeftShoulder, b: J.LeftArm, rTop: 0.048, rBot: 0.046, material: 'shell' },
  { a: J.LeftArm, b: J.LeftForeArm, rTop: 0.038, rBot: 0.049, material: 'shell' },
  { a: J.LeftForeArm, b: J.LeftHand, rTop: 0.028, rBot: 0.037, material: 'shell' },

  { a: J.Chest, b: J.RightShoulder, rTop: 0.044, rBot: 0.056, material: 'carbon' },
  { a: J.RightShoulder, b: J.RightArm, rTop: 0.048, rBot: 0.046, material: 'shell' },
  { a: J.RightArm, b: J.RightForeArm, rTop: 0.038, rBot: 0.049, material: 'shell' },
  { a: J.RightForeArm, b: J.RightHand, rTop: 0.028, rBot: 0.037, material: 'shell' },

  // Legs
  { a: J.Hips, b: J.LeftLeg, rTop: 0.062, rBot: 0.072, material: 'carbon' },
  { a: J.LeftLeg, b: J.LeftShin, rTop: 0.052, rBot: 0.072, material: 'shell' },
  { a: J.LeftShin, b: J.LeftFoot, rTop: 0.035, rBot: 0.050, material: 'shell' },

  { a: J.Hips, b: J.RightLeg, rTop: 0.062, rBot: 0.072, material: 'carbon' },
  { a: J.RightLeg, b: J.RightShin, rTop: 0.052, rBot: 0.072, material: 'shell' },
  { a: J.RightShin, b: J.RightFoot, rTop: 0.035, rBot: 0.050, material: 'shell' },
];

const JOINT_SPHERES: Array<{ idx: number; r: number; kind: 'metal' | 'shell' | 'accent' }> = [
  { idx: J.LeftArm, r: 0.054, kind: 'metal' },
  { idx: J.RightArm, r: 0.054, kind: 'metal' },
  { idx: J.LeftForeArm, r: 0.041, kind: 'metal' },
  { idx: J.RightForeArm, r: 0.041, kind: 'metal' },
  { idx: J.LeftHand, r: 0.038, kind: 'accent' },
  { idx: J.RightHand, r: 0.038, kind: 'accent' },
  { idx: J.LeftLeg, r: 0.068, kind: 'metal' },
  { idx: J.RightLeg, r: 0.068, kind: 'metal' },
  { idx: J.LeftShin, r: 0.052, kind: 'metal' },
  { idx: J.RightShin, r: 0.052, kind: 'metal' },
  { idx: J.LeftFoot, r: 0.038, kind: 'metal' },
  { idx: J.RightFoot, r: 0.038, kind: 'metal' },
];

const UP = new THREE.Vector3(0, 1, 0);
const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _qTmp = new THREE.Quaternion();
const _xAx = new THREE.Vector3();
const _yAx = new THREE.Vector3();
const _zAx = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();

function orientSegment(mesh: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3, radiusScale = 1): void {
  _dir.subVectors(b, a);
  const len = _dir.length();
  if (len < 1e-4) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  _mid.addVectors(a, b).multiplyScalar(0.5);
  mesh.position.copy(_mid);
  _dir.divideScalar(len);
  _qTmp.setFromUnitVectors(UP, _dir);
  mesh.quaternion.copy(_qTmp);
  mesh.scale.set(radiusScale, len, radiusScale);
}

/**
 * Sculpted robotics/studio mannequin driven directly by 24 SOMA 3D world joints,
 * complete with visor head, contoured chest & pelvis plates, athletic sneakers,
 * and live neural foot-contact glow rings.
 */
export class ArticulatedMannequin {
  public readonly group = new THREE.Group();

  private readonly shellMat: THREE.MeshStandardMaterial;
  private readonly carbonMat: THREE.MeshStandardMaterial;
  private readonly metalMat: THREE.MeshStandardMaterial;
  private readonly accentMat: THREE.MeshStandardMaterial;

  private readonly limbMeshes: THREE.Mesh[] = [];
  private readonly jointMeshes: THREE.Mesh[] = [];
  private readonly chestGroup: THREE.Group;
  private readonly pelvisPlate: THREE.Mesh;
  private readonly headGroup: THREE.Group;
  private readonly leftShoe: THREE.Group;
  private readonly rightShoe: THREE.Group;

  private readonly leftContactRing: THREE.Mesh;
  private readonly rightContactRing: THREE.Mesh;
  private readonly leftContactMat: THREE.MeshBasicMaterial;
  private readonly rightContactMat: THREE.MeshBasicMaterial;

  private readonly pts: THREE.Vector3[] = Array.from({ length: 24 }, () => new THREE.Vector3());

  constructor() {
    this.shellMat = new THREE.MeshStandardMaterial({
      color: 0xeaf0f8,
      roughness: 0.34,
      metalness: 0.14,
    });
    this.carbonMat = new THREE.MeshStandardMaterial({
      color: 0x1d2534,
      roughness: 0.52,
      metalness: 0.38,
    });
    this.metalMat = new THREE.MeshStandardMaterial({
      color: 0x3b4860,
      roughness: 0.28,
      metalness: 0.78,
    });
    this.accentMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.72,
      roughness: 0.2,
      metalness: 0.3,
    });

    const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 14, 1, false);
    for (const def of LIMBS) {
      const geo = new THREE.CylinderGeometry(def.rTop, def.rBot, 1.0, 14, 1);
      const mat =
        def.material === 'shell'
          ? this.shellMat
          : def.material === 'accent'
            ? this.accentMat
            : this.carbonMat;
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
      this.limbMeshes.push(mesh);
    }
    unitCyl.dispose();

    const sphereGeo = new THREE.SphereGeometry(1, 14, 12);
    for (const jdef of JOINT_SPHERES) {
      const mat =
        jdef.kind === 'metal'
          ? this.metalMat
          : jdef.kind === 'accent'
            ? this.accentMat
            : this.shellMat;
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.setScalar(jdef.r);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
      this.jointMeshes.push(mesh);
    }

    // Sculpted V-tapered athletic chest carapace + subtle sternum light
    this.chestGroup = new THREE.Group();
    const chestGeo = new THREE.CylinderGeometry(0.152, 0.104, 0.285, 18, 1);
    chestGeo.scale(1.0, 1.0, 0.66);
    const chestMesh = new THREE.Mesh(chestGeo, this.shellMat);
    chestMesh.castShadow = true;
    chestMesh.receiveShadow = true;
    this.chestGroup.add(chestMesh);

    const collarGeo = new THREE.CylinderGeometry(0.125, 0.152, 0.055, 18, 1);
    collarGeo.scale(1.0, 1.0, 0.66);
    const collarMesh = new THREE.Mesh(collarGeo, this.carbonMat);
    collarMesh.position.y = 0.165;
    collarMesh.castShadow = true;
    this.chestGroup.add(collarMesh);

    const coreStrip = new THREE.Mesh(new THREE.BoxGeometry(0.056, 0.016, 0.014), this.accentMat);
    coreStrip.position.set(0, 0.055, 0.098);
    this.chestGroup.add(coreStrip);
    this.group.add(this.chestGroup);

    // Sculpted pelvis housing
    const pelvisGeo = new THREE.CylinderGeometry(0.118, 0.102, 0.125, 16, 1);
    pelvisGeo.scale(1.05, 1.0, 0.72);
    this.pelvisPlate = new THREE.Mesh(pelvisGeo, this.shellMat);
    this.pelvisPlate.castShadow = true;
    this.pelvisPlate.receiveShadow = true;
    this.group.add(this.pelvisPlate);

    // Sculpted Head + Glowing Forward Visor
    this.headGroup = new THREE.Group();
    const craniumGeo = new THREE.SphereGeometry(0.105, 20, 16);
    craniumGeo.scale(0.92, 1.12, 1.02);
    const cranium = new THREE.Mesh(craniumGeo, this.shellMat);
    cranium.castShadow = true;
    this.headGroup.add(cranium);

    const visorGeo = new THREE.BoxGeometry(0.145, 0.034, 0.045);
    const visor = new THREE.Mesh(visorGeo, this.accentMat);
    visor.position.set(0, 0.014, 0.088);
    this.headGroup.add(visor);

    const chinBandGeo = new THREE.CylinderGeometry(0.072, 0.062, 0.04, 14);
    const chinBand = new THREE.Mesh(chinBandGeo, this.carbonMat);
    chinBand.position.set(0, -0.082, 0.01);
    this.headGroup.add(chinBand);
    this.group.add(this.headGroup);

    // Sculpted Sneakers
    this.leftShoe = this.buildShoe();
    this.rightShoe = this.buildShoe();
    this.group.add(this.leftShoe, this.rightShoe);

    // Live Neural Foot Contact Glow Rings on floor (y = 0.009)
    const ringGeo = new THREE.RingGeometry(0.08, 0.155, 28);
    ringGeo.rotateX(-Math.PI / 2);
    this.leftContactMat = new THREE.MeshBasicMaterial({
      color: 0xfbbf24,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.rightContactMat = new THREE.MeshBasicMaterial({
      color: 0xfbbf24,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.leftContactRing = new THREE.Mesh(ringGeo, this.leftContactMat);
    this.rightContactRing = new THREE.Mesh(ringGeo, this.rightContactMat);
    this.leftContactRing.position.y = 0.009;
    this.rightContactRing.position.y = 0.009;
    this.group.add(this.leftContactRing, this.rightContactRing);
  }

  private buildShoe(): THREE.Group {
    const g = new THREE.Group();
    const upper = new THREE.Mesh(new THREE.BoxGeometry(0.088, 0.056, 0.215), this.carbonMat);
    upper.position.set(0, 0.015, 0.032);
    upper.castShadow = true;
    const sole = new THREE.Mesh(new THREE.BoxGeometry(0.094, 0.022, 0.228), this.shellMat);
    sole.position.set(0, -0.022, 0.034);
    sole.castShadow = true;
    const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.096, 0.012, 0.085), this.accentMat);
    stripe.position.set(0, 0.002, 0.085);
    g.add(upper, sole, stripe);
    return g;
  }

  public setAccentColor(hex: string): void {
    this.accentMat.color.set(hex);
    this.accentMat.emissive.set(hex);
  }

  public setShowContacts(show: boolean): void {
    this.leftContactRing.visible = show;
    this.rightContactRing.visible = show;
  }

  public updatePose(
    positions: Vec3Tuple[],
    _quats: QuatTuple[],
    footContacts?: [number, number, number, number]
  ): void {
    for (let i = 0; i < 24; i++) {
      const p = positions[i];
      this.pts[i].set(p[0], p[1], p[2]);
    }

    // Update all limb segments
    for (let i = 0; i < LIMBS.length; i++) {
      const def = LIMBS[i];
      orientSegment(this.limbMeshes[i], this.pts[def.a], this.pts[def.b]);
    }

    // Update joint articulation spheres
    for (let i = 0; i < JOINT_SPHERES.length; i++) {
      this.jointMeshes[i].position.copy(this.pts[JOINT_SPHERES[i].idx]);
    }

    // Torso orthonormal basis from Spine2->Neck1 (Y) and RightArm->LeftArm (X)
    _yAx.subVectors(this.pts[J.Neck1], this.pts[J.Spine2]).normalize();
    _xAx.subVectors(this.pts[J.LeftArm], this.pts[J.RightArm]).normalize();
    // Forward Z = cross(X, Y): Left = +X, Up = +Y => X x Y = +Z (forward)
    _zAx.crossVectors(_xAx, _yAx).normalize();
    _xAx.crossVectors(_yAx, _zAx).normalize();
    _mat4.makeBasis(_xAx, _yAx, _zAx);
    _qTmp.setFromRotationMatrix(_mat4);

    // Athletic upper torso placed cleanly across upper ribcage (Chest -> Neck1)
    this.chestGroup.position.lerpVectors(this.pts[J.Chest], this.pts[J.Neck1], 0.48);
    this.chestGroup.quaternion.copy(_qTmp);

    // Pelvis plate oriented by hips (RightLeg->LeftLeg)
    _xAx.subVectors(this.pts[J.LeftLeg], this.pts[J.RightLeg]).normalize();
    _zAx.crossVectors(_xAx, _yAx).normalize();
    _xAx.crossVectors(_yAx, _zAx).normalize();
    _mat4.makeBasis(_xAx, _yAx, _zAx);
    this.pelvisPlate.position.copy(this.pts[J.Hips]);
    this.pelvisPlate.quaternion.setFromRotationMatrix(_mat4);

    // Head placed midway between Head & HeadEnd, aligned along neck->headEnd up axis & chest forward
    _yAx.subVectors(this.pts[J.HeadEnd], this.pts[J.Head]).normalize();
    _xAx.subVectors(this.pts[J.LeftArm], this.pts[J.RightArm]).normalize();
    _zAx.crossVectors(_xAx, _yAx).normalize();
    _xAx.crossVectors(_yAx, _zAx).normalize();
    _mat4.makeBasis(_xAx, _yAx, _zAx);
    this.headGroup.position.lerpVectors(this.pts[J.Head], this.pts[J.HeadEnd], 0.45);
    this.headGroup.quaternion.setFromRotationMatrix(_mat4);

    // Orient left & right sneakers from Foot -> ToeBase
    this.updateShoe(this.leftShoe, this.pts[J.LeftFoot], this.pts[J.LeftToeBase], this.pts[J.LeftShin]);
    this.updateShoe(this.rightShoe, this.pts[J.RightFoot], this.pts[J.RightToeBase], this.pts[J.RightShin]);

    // Update live ONNX foot-contact glow rings
    if (footContacts) {
      const lContact = Math.max(footContacts[0], footContacts[1]);
      const rContact = Math.max(footContacts[2], footContacts[3]);

      this.leftContactRing.position.set(this.pts[J.LeftFoot].x, 0.009, this.pts[J.LeftFoot].z);
      this.leftContactMat.opacity = lContact * 0.85;
      const ls = 0.72 + lContact * 0.42;
      this.leftContactRing.scale.set(ls, ls, ls);

      this.rightContactRing.position.set(this.pts[J.RightFoot].x, 0.009, this.pts[J.RightFoot].z);
      this.rightContactMat.opacity = rContact * 0.85;
      const rs = 0.72 + rContact * 0.42;
      this.rightContactRing.scale.set(rs, rs, rs);
    }
  }

  private updateShoe(
    shoe: THREE.Group,
    foot: THREE.Vector3,
    toe: THREE.Vector3,
    shin: THREE.Vector3
  ): void {
    _zAx.subVectors(toe, foot);
    if (_zAx.lengthSq() < 1e-5) return;
    _zAx.normalize();
    _yAx.subVectors(shin, foot).normalize();
    // Orthonormalize so sole stays flat to foot direction
    _xAx.crossVectors(_yAx, _zAx).normalize();
    _yAx.crossVectors(_zAx, _xAx).normalize();
    _mat4.makeBasis(_xAx, _yAx, _zAx);
    shoe.position.copy(foot);
    shoe.position.y = Math.max(0.032, foot.y - 0.025);
    shoe.quaternion.setFromRotationMatrix(_mat4);
  }
}

/**
 * Clean precision mocap / Maya-style anatomical bone skeleton (slim glowing bone links
 * + crisp joint nodes) rendered inside the translucent SOMA human skin in X-Ray mode.
 */
const SKELETON_BONES: Array<[number, number]> = [
  // Spine & Head
  [J.Hips, J.Spine1],
  [J.Spine1, J.Spine2],
  [J.Spine2, J.Chest],
  [J.Chest, J.Neck1],
  [J.Neck1, J.Neck2],
  [J.Neck2, J.Head],
  [J.Head, J.HeadEnd],
  // Left Arm
  [J.Chest, J.LeftShoulder],
  [J.LeftShoulder, J.LeftArm],
  [J.LeftArm, J.LeftForeArm],
  [J.LeftForeArm, J.LeftHand],
  // Right Arm
  [J.Chest, J.RightShoulder],
  [J.RightShoulder, J.RightArm],
  [J.RightArm, J.RightForeArm],
  [J.RightForeArm, J.RightHand],
  // Left Leg
  [J.Hips, J.LeftLeg],
  [J.LeftLeg, J.LeftShin],
  [J.LeftShin, J.LeftFoot],
  [J.LeftFoot, J.LeftToeBase],
  // Right Leg
  [J.Hips, J.RightLeg],
  [J.RightLeg, J.RightShin],
  [J.RightShin, J.RightFoot],
  [J.RightFoot, J.RightToeBase],
];

export class NeuralSkeletonRig {
  public readonly group = new THREE.Group();
  private readonly boneMat: THREE.MeshBasicMaterial;
  private readonly jointMat: THREE.MeshBasicMaterial;
  private readonly lineMat: THREE.LineBasicMaterial;
  private readonly boneMeshes: THREE.Mesh[] = [];
  private readonly jointMeshes: THREE.Mesh[] = [];
  private readonly lineGeo: THREE.BufferGeometry;
  private readonly linePos: Float32Array;
  private readonly pts: THREE.Vector3[] = Array.from({ length: 24 }, () => new THREE.Vector3());

  constructor(colorHex = 0x38bdf8) {
    // Subtle tapered octahedral bone prisms (classic Maya / MotionBuilder / Newton look)
    this.boneMat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.52,
      wireframe: true,
      depthWrite: false,
    });
    this.jointMat = new THREE.MeshBasicMaterial({
      color: 0xe0f2fe,
      transparent: true,
      opacity: 0.95,
    });
    this.lineMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.95,
    });

    // Double-pyramid bone geometry aligned along +Y from -0.5 to +0.5
    const boneGeo = new THREE.CylinderGeometry(0.004, 0.014, 1.0, 4, 1);
    for (let i = 0; i < SKELETON_BONES.length; i++) {
      const m = new THREE.Mesh(boneGeo, this.boneMat);
      m.renderOrder = 3;
      this.group.add(m);
      this.boneMeshes.push(m);
    }

    const nodeGeo = new THREE.SphereGeometry(0.017, 10, 8);
    for (let i = 0; i < 24; i++) {
      const n = new THREE.Mesh(nodeGeo, this.jointMat);
      const scale = i === J.Hips || i === J.Head ? 1.35 : 0.95;
      n.scale.setScalar(scale);
      n.renderOrder = 4;
      this.group.add(n);
      this.jointMeshes.push(n);
    }

    this.linePos = new Float32Array(SKELETON_BONES.length * 2 * 3);
    this.lineGeo = new THREE.BufferGeometry();
    this.lineGeo.setAttribute('position', new THREE.BufferAttribute(this.linePos, 3));
    const lines = new THREE.LineSegments(this.lineGeo, this.lineMat);
    lines.renderOrder = 4;
    this.group.add(lines);
  }

  public setAccentColor(hex: string): void {
    this.boneMat.color.set(hex);
    this.lineMat.color.set(hex);
  }

  public setShowContacts(_show: boolean): void {
    // Foot contacts are handled by dedicated floor rings in main.ts
  }

  public updatePose(positions: Vec3Tuple[], _quats?: QuatTuple[], _contacts?: [number, number, number, number]): void {
    for (let i = 0; i < 24; i++) {
      const p = positions[i];
      this.pts[i].set(p[0], p[1], p[2]);
      this.jointMeshes[i].position.copy(this.pts[i]);
    }
    for (let i = 0; i < SKELETON_BONES.length; i++) {
      const [aIdx, bIdx] = SKELETON_BONES[i];
      const a = this.pts[aIdx];
      const b = this.pts[bIdx];
      orientSegment(this.boneMeshes[i], a, b, 1.0);

      const o = i * 6;
      this.linePos[o] = a.x;
      this.linePos[o + 1] = a.y;
      this.linePos[o + 2] = a.z;
      this.linePos[o + 3] = b.x;
      this.linePos[o + 4] = b.y;
      this.linePos[o + 5] = b.z;
    }
    this.lineGeo.attributes.position.needsUpdate = true;
  }
}

/**
 * Translucent holographic ghost overlay showing MotionBricks Green Context
 * or Blue Target poses in real time.
 */
export class GhostMannequin {
  public readonly group = new THREE.Group();
  private readonly mat: THREE.MeshBasicMaterial;
  private readonly lineMat: THREE.LineBasicMaterial;
  private readonly boneMeshes: THREE.Mesh[] = [];
  private readonly jointMeshes: THREE.Mesh[] = [];
  private readonly lineGeo: THREE.BufferGeometry;
  private readonly linePos: Float32Array;
  private readonly ringMesh: THREE.Mesh;
  private readonly pts: THREE.Vector3[] = Array.from({ length: 24 }, () => new THREE.Vector3());

  constructor(colorHex: number, opacity = 0.48, seeThrough = false) {
    this.mat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: opacity * 0.78,
      wireframe: true,
      depthWrite: false,
      depthTest: !seeThrough,
    });
    this.lineMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: Math.min(0.95, opacity * 1.85),
      depthWrite: false,
      depthTest: !seeThrough,
    });

    const boneGeo = new THREE.CylinderGeometry(0.005, 0.015, 1.0, 4, 1);
    for (let i = 0; i < SKELETON_BONES.length; i++) {
      const m = new THREE.Mesh(boneGeo, this.mat);
      if (seeThrough) m.renderOrder = 6;
      this.group.add(m);
      this.boneMeshes.push(m);
    }

    const nodeGeo = new THREE.SphereGeometry(0.017, 8, 6);
    const nodeMat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: Math.min(0.96, opacity * 1.9),
      depthWrite: false,
      depthTest: !seeThrough,
    });
    for (let i = 0; i < 24; i++) {
      const n = new THREE.Mesh(nodeGeo, nodeMat);
      if (i === J.Head) n.scale.setScalar(1.35);
      if (seeThrough) n.renderOrder = 7;
      this.group.add(n);
      this.jointMeshes.push(n);
    }

    this.linePos = new Float32Array(SKELETON_BONES.length * 2 * 3);
    this.lineGeo = new THREE.BufferGeometry();
    this.lineGeo.setAttribute('position', new THREE.BufferAttribute(this.linePos, 3));
    const lines = new THREE.LineSegments(this.lineGeo, this.lineMat);
    if (seeThrough) lines.renderOrder = 7;
    this.group.add(lines);

    const ringInner = seeThrough ? 0.30 : 0.24;
    const ringOuter = seeThrough ? 0.365 : 0.285;
    const ringGeo = new THREE.RingGeometry(ringInner, ringOuter, 36);
    ringGeo.rotateX(-Math.PI / 2);
    this.ringMesh = new THREE.Mesh(
      ringGeo,
      new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: Math.min(0.88, opacity * 1.65),
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.ringMesh.position.y = 0.016;
    this.ringMesh.renderOrder = 5;
    this.group.add(this.ringMesh);
  }

  public updatePose(positions: Vec3Tuple[]): void {
    for (let i = 0; i < 24; i++) {
      const p = positions[i];
      this.pts[i].set(p[0], p[1], p[2]);
      this.jointMeshes[i].position.copy(this.pts[i]);
    }
    for (let i = 0; i < SKELETON_BONES.length; i++) {
      const [aIdx, bIdx] = SKELETON_BONES[i];
      const a = this.pts[aIdx];
      const b = this.pts[bIdx];
      orientSegment(this.boneMeshes[i], a, b, 1.0);
      const o = i * 6;
      this.linePos[o] = a.x;
      this.linePos[o + 1] = a.y;
      this.linePos[o + 2] = a.z;
      this.linePos[o + 3] = b.x;
      this.linePos[o + 4] = b.y;
      this.linePos[o + 5] = b.z;
    }
    this.lineGeo.attributes.position.needsUpdate = true;
    this.ringMesh.position.set(this.pts[J.Hips].x, 0.016, this.pts[J.Hips].z);
  }
}

/**
 * Helper to smoothly interpolate two 24-joint poses at 60+ fps between 30 Hz ONNX frames.
 */
export function interpolatePose(
  posA: Vec3Tuple[],
  posB: Vec3Tuple[],
  alpha: number,
  outPos: Vec3Tuple[]
): void {
  const inv = 1 - alpha;
  for (let i = 0; i < 24; i++) {
    const a = posA[i];
    const b = posB[i];
    outPos[i][0] = a[0] * inv + b[0] * alpha;
    outPos[i][1] = a[1] * inv + b[1] * alpha;
    outPos[i][2] = a[2] * inv + b[2] * alpha;
  }
}

export function interpolateContacts(
  cA: [number, number, number, number],
  cB: [number, number, number, number],
  alpha: number,
  out: [number, number, number, number]
): void {
  const inv = 1 - alpha;
  for (let i = 0; i < 4; i++) {
    out[i] = cA[i] * inv + cB[i] * alpha;
  }
}

export { _vA, _vB };
