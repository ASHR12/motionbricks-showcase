import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import './styles.css';
import { createStudioFloor } from './grid';
import {
  GhostMannequin,
  J,
  NeuralSkeletonRig,
  type QuatTuple,
  type Vec3Tuple,
} from './mannequin';
import {
  FootLockStabilizer,
  MotionInertializer,
  copyFlatPositionsToTuples,
  decodeBinaryChunk,
  sampleChunkCatmullRom,
  type BinaryChunk,
} from './motionBuffer';
import { MotionBankEngine } from './motionBank';
import { createMocapPavilion } from './pavilion';
import { SomaSkinnedCharacter } from './somaSkin';
import {
  DualSenseBridge,
  formatPS5HudBadge,
  formatPS5TopBarLabel,
  type PS5InputSource,
} from './ps5Input';
import { PS5ControllerHUD } from './ps5Hud';
import { createStudioObstacleManager } from './studioProps';
import { PAVILION_SPEC, ROOM_BOUNDS, STUDIO_CAMERAS, STUDIO_THEMES } from './studioSpec';

const API_BASE = 'http://127.0.0.1:8765';

interface StyleInfo {
  id: string;
  label: string;
  speed: number;
  accent: string;
}

interface ObstacleInfo {
  id: string;
  type: 'vault' | 'sit' | 'duck' | 'hurdle' | 'block';
  label: string;
  pos: [number, number, number];
  size: [number, number, number];
  clearance?: number;
  facing?: [number, number];
  color: string;
}

interface InitPayload {
  session_id: string;
  model: {
    name: string;
    precision: string;
    provider: string;
    fps: number;
    render_joints: number;
    bone_joints: number;
  };
  bone_names: string[];
  styles: StyleInfo[];
  obstacles: ObstacleInfo[];
  initial_pose: {
    pos: Vec3Tuple[];
    quat: QuatTuple[];
    bone_quat?: QuatTuple[];
  };
}

type ControllerMode =
  | 'idle'
  | 'steer'
  | 'goto'
  | 'jump_leap'
  | 'jump_land'
  | 'roll'
  | 'punch_left'
  | 'punch_right'
  | 'combat_recover'
  | 'vault_entry'
  | 'vault_action'
  | 'vault_exit'
  | 'duck_entry'
  | 'duck_action'
  | 'duck_exit'
  | 'emote_look'
  | 'emote_look_2'
  | 'emote_warmup'
  | 'emote_warmup_2'
  | 'sit_approach'
  | 'sit_down'
  | 'sit_hold'
  | 'stand_up';

declare global {
  interface Window {
    __MOTIONBRICKS__?: {
      triggerStyle: (id: string) => void;
      triggerPunch: (side?: 'left' | 'right') => void;
      triggerVault: () => void;
      triggerDuck: () => void;
      triggerRoll: () => void;
      triggerSit: () => void;
      triggerEmote: (kind: 'look' | 'warmup') => void;
      triggerGoto: (x: number, z: number) => void;
      toggleGuides: () => void;
      toggleXRay: () => void;
      toggleSlowMo: () => void;
      togglePS5Hud: () => void;
      toggleTheme: () => void;
      toggleHelp?: () => void;
      toggleEditGym: () => void;
      setEditGymMode: (active: boolean) => void;
      moveObstacle: (id: string, x: number, z: number) => boolean;
      rotateObstacle: (id: string) => boolean;
      shuffleObstacles: () => void;
      resetObstacles: () => void;
      getObstacles: () => ObstacleInfo[];
      setCameraBookmark: (id: string) => void;
      setCameraOrbit: (yaw?: number, pitch?: number, dist?: number) => void;
      simulatePS5: (override: Parameters<DualSenseBridge['setSimulation']>[0]) => void;
      setSteer: (dx: number, dz: number) => void;
      getState: () => Record<string, unknown>;
    };
  }
}

async function boot(): Promise<void> {
  const canvas = document.getElementById('stage') as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.96;

  const initialTheme = STUDIO_THEMES.dark;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(initialTheme.bg);
  scene.fog = new THREE.FogExp2(initialTheme.fog, initialTheme.fogDensity);

  // Soft studio PMREM environment for crisp fabric/skin/metal shading
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = envTex;
  scene.environmentIntensity = initialTheme.envIntensity;
  pmrem.dispose();

  // Initialize hero camera directly from Single Shared Spec (`cam-01`)
  const defaultBookmark = STUDIO_CAMERAS['cam-01'];
  const camera = new THREE.PerspectiveCamera(
    defaultBookmark.fov,
    window.innerWidth / window.innerHeight,
    0.1,
    160
  );

  // Studio 3-point lighting crafted from STUDIO_THEMES to sculpt the dressed SOMA athlete
  const hemi = new THREE.HemisphereLight(
    initialTheme.hemiSky,
    initialTheme.hemiGround,
    initialTheme.hemiIntensity
  );
  scene.add(hemi);

  const keyLight = new THREE.DirectionalLight(initialTheme.keyColor, initialTheme.keyIntensity);
  keyLight.position.set(-1.2, 6.2, 2.4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 32;
  const sSpan = 7.5;
  keyLight.shadow.camera.left = -sSpan;
  keyLight.shadow.camera.right = sSpan;
  keyLight.shadow.camera.top = sSpan;
  keyLight.shadow.camera.bottom = -sSpan;
  keyLight.shadow.bias = -0.0004;
  scene.add(keyLight);
  scene.add(keyLight.target);

  const rimLight = new THREE.DirectionalLight(initialTheme.rimColor, initialTheme.rimIntensity);
  rimLight.position.set(3.2, 3.6, -3.8);
  scene.add(rimLight);
  scene.add(rimLight.target);

  const fillLight = new THREE.DirectionalLight(0xfde68a, 0.14);
  fillLight.position.set(-3.5, 2.2, 4.5);
  scene.add(fillLight);

  // Architectural Mocap Pavilion & Grounded Studio Floor
  const floor = createStudioFloor();
  scene.add(floor.group);

  const pavilion = createMocapPavilion();
  scene.add(pavilion.group);

  // Completely internal engine resolution (ZERO URL query parameters required!):
  // - Hosted static builds (`VITE_ENGINE_LOCKED=true`) or any non-localhost domain:
  //   always boots directly into Locked Pre-Baked mode (zero localhost probe).
  // - Local dev (`localhost`/`127.0.0.1` with `server/api.py` live):
  //   boots into Live FP16 ONNX AND preloads the 0.38 MB Pre-Baked bank so the slider toggles live!
  const viteEnv = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
  const IS_HOSTED_LOCKED = viteEnv?.VITE_ENGINE_LOCKED === 'true';

  async function bootMotionEngines(): Promise<{
    initData: InitPayload;
    bank: MotionBankEngine | null;
    liveAvailable: boolean;
    initialMode: 'live-onnx' | 'baked-bank';
  }> {
    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1' || host === '';

    if (!IS_HOSTED_LOCKED && isLocalHost) {
      const ctrl = new AbortController();
      const timer = window.setTimeout(() => ctrl.abort(), 700);
      try {
        const res = await fetch(`${API_BASE}/api/init`, { signal: ctrl.signal });
        window.clearTimeout(timer);
        if (res.ok) {
          const liveData = (await res.json()) as InitPayload;
          // Silently preload the tiny 0.38 MB bank too so the local slider toggles seamlessly!
          const bank = await MotionBankEngine.load('./models/motion-bank.bin').catch(() => null);
          return {
            initData: liveData,
            bank,
            liveAvailable: true,
            initialMode: 'live-onnx',
          };
        }
      } catch {
        window.clearTimeout(timer);
      }
    }

    try {
      const bank = await MotionBankEngine.load('./models/motion-bank.bin');
      return {
        initData: bank.initData as unknown as InitPayload,
        bank,
        liveAvailable: false,
        initialMode: 'baked-bank',
      };
    } catch {
      const toast = document.getElementById('engine-toast');
      if (toast) {
        toast.textContent =
          'No local ONNX server detected on :8765 and no local motion-bank.bin found. Start `python3 server/api.py` (or export a local bank with `python3 tools/export_motion_bank.py`) and reload.';
        toast.classList.remove('hidden');
      }
      throw new Error(
        'Neither local ONNX server (http://127.0.0.1:8765) nor web/public/models/motion-bank.bin is available.'
      );
    }
  }

  // Load engine(s) + NVIDIA's official SOMA human GLB in parallel!
  const [sessionBoot, somaSkin] = await Promise.all([
    bootMotionEngines(),
    SomaSkinnedCharacter.load('./models/soma_human.glb', {
      xrayOpacity: 0.22,
      xrayAccentHex: '#38bdf8',
      restPoseCorrection: 'none',
    }),
  ]);
  const { initData, bank: motionBank, liveAvailable } = sessionBoot;
  let engineMode: 'live-onnx' | 'baked-bank' = sessionBoot.initialMode;
  const sessionId = initData.session_id;
  const boneNames = initData.bone_names;
  const numRenderJoints = initData.model.render_joints || 24;
  const numBoneJoints = boneNames.length || 26;

  function cloneObstacles(list: ObstacleInfo[]): ObstacleInfo[] {
    return list.map((o) => ({
      ...o,
      pos: [o.pos[0], o.pos[1], o.pos[2]] as [number, number, number],
      size: [o.size[0], o.size[1], o.size[2]] as [number, number, number],
      facing: o.facing ? ([o.facing[0], o.facing[1]] as [number, number]) : undefined,
    }));
  }
  const defaultObstacles = cloneObstacles(initData.obstacles);

  // Wire top-left Engine Mode Slider (Pre-Baked <-> Live ONNX)
  const engineSliderEl = document.getElementById('engine-slider');
  const engineOptBakedEl = document.getElementById('engine-opt-baked');
  const engineOptLiveEl = document.getElementById('engine-opt-live');
  const engineToastEl = document.getElementById('engine-toast');
  const statusDotEarly = document.getElementById('status-dot');
  let engineToastTimer: number | null = null;
  let engineBounceTimer: number | null = null;

  function showEngineToast(msg: string, durationMs = 4200): void {
    if (!engineToastEl) return;
    engineToastEl.textContent = msg;
    engineToastEl.classList.remove('hidden');
    if (engineToastTimer !== null) window.clearTimeout(engineToastTimer);
    engineToastTimer = window.setTimeout(() => {
      engineToastEl.classList.add('hidden');
      engineToastTimer = null;
    }, durationMs);
  }

  function syncEngineSliderUI(): void {
    const isLive = engineMode === 'live-onnx';
    if (engineSliderEl) {
      engineSliderEl.dataset.mode = isLive ? 'live' : 'baked';
      engineSliderEl.classList.toggle('locked', !liveAvailable);
      engineSliderEl.setAttribute('aria-checked', String(isLive));
    }
    engineOptBakedEl?.classList.toggle('active', !isLive);
    engineOptLiveEl?.classList.toggle('active', isLive);
    statusDotEarly?.classList.toggle('baked', !isLive);
  }
  syncEngineSliderUI();

  // Hero visuals: official NVIDIA SOMA SkinnedMesh + clean precision bone skeleton for X-Ray mode
  const skeletonRig = new NeuralSkeletonRig(0x38bdf8);
  scene.add(skeletonRig.group);

  let xrayMode = false;
  if (somaSkin) {
    scene.add(somaSkin.group);
    skeletonRig.group.visible = false; // clean full-body SOMA skin by default!
  }

  // Dedicated live foot-contact rings on the floor (work with BOTH SOMA Skin and Skeleton!)
  const ringGeo = new THREE.RingGeometry(0.085, 0.155, 28).rotateX(-Math.PI / 2);
  const leftContactMat = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const rightContactMat = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const leftContactRing = new THREE.Mesh(ringGeo, leftContactMat);
  const rightContactRing = new THREE.Mesh(ringGeo, rightContactMat);
  leftContactRing.position.y = 0.009;
  rightContactRing.position.y = 0.009;
  scene.add(leftContactRing, rightContactRing);

  // Green Context Ghost (seeThrough so past-frame anchor stays crisp even at idle/overlap) & Blue Target Ghost
  const contextGhost = new GhostMannequin(0x22c55e, 0.54, true);
  const targetGhost = new GhostMannequin(0x38bdf8, 0.48, false);
  contextGhost.group.visible = false;
  targetGhost.group.visible = false;
  scene.add(contextGhost.group, targetGhost.group);

  // Generated hip trajectory curve
  const trajMaxPts = 64;
  const trajPositions = new Float32Array(trajMaxPts * 3);
  const trajGeo = new THREE.BufferGeometry();
  trajGeo.setAttribute('position', new THREE.BufferAttribute(trajPositions, 3));
  trajGeo.setDrawRange(0, 0);
  const trajLine = new THREE.Line(
    trajGeo,
    new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.78 })
  );
  scene.add(trajLine);

  // Click-to-Move ground beacon
  const beaconGroup = new THREE.Group();
  const beaconRing = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.31, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    })
  );
  const beaconCore = new THREE.Mesh(
    new THREE.CircleGeometry(0.08, 20).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: 0x7dd3fc,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    })
  );
  beaconGroup.add(beaconRing, beaconCore);
  beaconGroup.position.y = 0.014;
  beaconGroup.visible = false;
  scene.add(beaconGroup);

  // Build crisp architectural studio parkour & interaction props + sensor pads
  const sensorZoneGroup = new THREE.Group();
  scene.add(sensorZoneGroup);
  const obstacleManager = createStudioObstacleManager(initData.obstacles, scene, sensorZoneGroup);

  // AAA Smoothness Engine: Zero-GC Flat Buffers + Catmull-Rom + 120 ms Inertializer + Foot-Lock
  const inertializer = new MotionInertializer(numRenderJoints, numBoneJoints);
  const footLocker = new FootLockStabilizer();

  const rawPos = new Float32Array(numRenderJoints * 3);
  const rawBoneQuat = new Float32Array(numBoneJoints * 4);
  const dispPos = new Float32Array(numRenderJoints * 3);
  const dispBoneQuat = new Float32Array(numBoneJoints * 4);
  const nextStartPos = new Float32Array(numRenderJoints * 3);
  const nextStartBoneQuat = new Float32Array(numBoneJoints * 4);
  const rawContacts: [number, number, number, number] = [1, 1, 1, 1];
  const dummyContacts: [number, number, number, number] = [1, 1, 1, 1];

  const tuplePos: Vec3Tuple[] = Array.from({ length: numRenderJoints }, () => [0, 0, 0]);
  const ghostCtxTuples: Vec3Tuple[] = Array.from({ length: numRenderJoints }, () => [0, 0, 0]);
  const ghostTgtTuples: Vec3Tuple[] = Array.from({ length: numRenderJoints }, () => [0, 0, 0]);
  const dummyQuats: QuatTuple[] = Array.from({ length: numRenderJoints }, () => [0, 0, 0, 1]);

  function seedInitialPose(posList: Vec3Tuple[], boneQuatList?: QuatTuple[]): void {
    for (let i = 0; i < numRenderJoints; i++) {
      rawPos[i * 3] = dispPos[i * 3] = posList[i][0];
      rawPos[i * 3 + 1] = dispPos[i * 3 + 1] = posList[i][1];
      rawPos[i * 3 + 2] = dispPos[i * 3 + 2] = posList[i][2];
    }
    if (boneQuatList && boneQuatList.length === numBoneJoints) {
      for (let i = 0; i < numBoneJoints; i++) {
        rawBoneQuat[i * 4] = dispBoneQuat[i * 4] = boneQuatList[i][0];
        rawBoneQuat[i * 4 + 1] = dispBoneQuat[i * 4 + 1] = boneQuatList[i][1];
        rawBoneQuat[i * 4 + 2] = dispBoneQuat[i * 4 + 2] = boneQuatList[i][2];
        rawBoneQuat[i * 4 + 3] = dispBoneQuat[i * 4 + 3] = boneQuatList[i][3];
      }
    }
    applyDisplayedPose(1 / 60);
  }

  const _hipsGroundTuple: [number, number, number] = [0, 0, 0];

  function applyDisplayedPose(dt: number): void {
    // Final hard wall-collision safety clamp on displayed pose (0.000 m wall penetration across all 24 joints + skin!)
    const rawHx = dispPos[0];
    const rawHz = dispPos[2];
    const clampedHx = THREE.MathUtils.clamp(rawHx, ROOM_BOUNDS.minX, ROOM_BOUNDS.maxX);
    const clampedHz = THREE.MathUtils.clamp(rawHz, ROOM_BOUNDS.minZ, ROOM_BOUNDS.maxZ);
    const fixX = clampedHx - rawHx;
    const fixZ = clampedHz - rawHz;
    if (Math.abs(fixX) > 1e-5 || Math.abs(fixZ) > 1e-5) {
      for (let j = 0; j < numRenderJoints; j++) {
        dispPos[j * 3] += fixX;
        dispPos[j * 3 + 2] += fixZ;
      }
    }

    if (somaSkin) {
      // Zero-GC floor-plane grounding clamp: strictly for upright moves & crawl
      // (during 'roll', server wheel-envelope FK grounding already places the rolling contour flush on y = 0.00 m)
      const isRolling = mode === 'roll' || currentChunk?.mode === 'roll';
      let groundLift = 0;
      if (!isRolling) {
        let minClearance = Math.min(
          dispPos[J.LeftFoot * 3 + 1] - 0.068,
          dispPos[J.RightFoot * 3 + 1] - 0.068,
          dispPos[J.LeftToeBase * 3 + 1] - 0.032,
          dispPos[J.RightToeBase * 3 + 1] - 0.032
        );
        if (activeStyle === 'crawl') {
          minClearance = Math.min(
            minClearance,
            dispPos[J.Head * 3 + 1] - 0.12,
            dispPos[J.Chest * 3 + 1] - 0.11,
            dispPos[J.LeftHand * 3 + 1] - 0.035,
            dispPos[J.RightHand * 3 + 1] - 0.035
          );
        }
        groundLift = minClearance < 0 ? Math.min(0.22, -minClearance) : 0;
      }
      _hipsGroundTuple[0] = dispPos[0];
      _hipsGroundTuple[1] = dispPos[1] + groundLift;
      _hipsGroundTuple[2] = dispPos[2];
      const fistTarget = isRolling
        ? 0.55
        : activeStyle === 'combat' ||
            mode === 'punch_left' ||
            mode === 'punch_right' ||
            mode === 'combat_recover'
          ? 1.0
          : activeStyle === 'crawl'
            ? 0.04
            : 0.18;
      somaSkin.setHandFistBlend(fistTarget);
      somaSkin.applySomaPose(boneNames, _hipsGroundTuple, dispBoneQuat);
      somaSkin.update(dt);
    }
    if (skeletonRig.group.visible) {
      copyFlatPositionsToTuples(dispPos, tuplePos);
      skeletonRig.updatePose(tuplePos, dummyQuats, rawContacts);
    }
    if (showAllGuides && showContacts) {
      const lC = Math.max(rawContacts[0], rawContacts[1]);
      const rC = Math.max(rawContacts[2], rawContacts[3]);
      leftContactRing.position.set(dispPos[J.LeftFoot * 3], 0.009, dispPos[J.LeftFoot * 3 + 2]);
      leftContactMat.opacity = lC * 0.85;
      leftContactRing.scale.setScalar(0.72 + lC * 0.4);

      rightContactRing.position.set(dispPos[J.RightFoot * 3], 0.009, dispPos[J.RightFoot * 3 + 2]);
      rightContactMat.opacity = rC * 0.85;
      rightContactRing.scale.setScalar(0.72 + rC * 0.4);
    }
  }

  // Runtime state
  let activeStyle = 'jog';
  let prevNonCrouchStyle = 'jog';
  const styleMap = new Map<string, StyleInfo>(initData.styles.map((s) => [s.id, s]));
  const styleOrder = initData.styles.map((s) => s.id);
  let mode: ControllerMode = 'idle';
  let slowMo = false;

  // Dual-buffer lookahead pipeline (ZERO snap-back!)
  let currentChunk: BinaryChunk | null = null;
  let nextChunk: BinaryChunk | null = null;
  let playFrame = 0;
  let inFlight = false;
  let requestSeq = 0;
  let pendingInterrupt: { mode: ControllerMode; steer?: THREE.Vector2 } | null = null;

  let lastSteerDir = new THREE.Vector2(0, 1);
  let actionDir = new THREE.Vector2(0, 1);
  let nextPunchSide: 'left' | 'right' = 'left';
  let gotoTarget: { x: number; z: number } | null = null;
  let activeVaultObsId: string | null = null;
  let activeVaultLane = 0;
  let vaultDir = 1;
  let vaultZ = 0;
  let activeDuckObsId: string | null = null;
  let duckDir = -1;
  let duckX = 0;
  let duckLane = 0;
  const obstacleCooldowns = new Map<string, number>();
  let syntheticSteer: THREE.Vector2 | null = null;
  let lastLatencyMs = 0;
  let lastInferMs = 0;
  let generationCount = 0;

  // Clean Studio default: AI Guides OFF (1 click or H turns them ON anytime!)
  let showAllGuides = false;
  let autoVaultEnabled = true;
  let showTargetGhost = true;
  let showContextGhost = true;
  let showContacts = true;
  let showTraj = true;

  skeletonRig.setAccentColor(styleMap.get(activeStyle)!.accent);
  seedInitialPose(initData.initial_pose.pos, initData.initial_pose.bone_quat);

  function updateSculpturalLights(hx: number, hz: number): void {
    keyLight.position.set(hx - 1.2, 6.2, hz + 2.4);
    keyLight.target.position.set(hx, 0.85, hz);
    rimLight.position.set(hx + 3.2, 3.6, hz - 3.8);
    rimLight.target.position.set(hx, 0.85, hz);
  }
  updateSculpturalLights(dispPos[0], dispPos[2]);

  // Smooth third-person orbit camera initialized from STUDIO_CAMERAS['cam-01']
  let activeCameraBookmark = defaultBookmark.id;
  let camYaw = defaultBookmark.yaw;
  let camPitch = defaultBookmark.pitch;
  let camDist = defaultBookmark.dist;
  let wideCamMode = false;
  let smoothDistBonus = 0;
  let lastZoomCeiling = 10.5;
  let camRecenterYaw: number | null = null;
  const camTargetOffset = new THREE.Vector3(
    defaultBookmark.targetOffset[0],
    defaultBookmark.targetOffset[1],
    defaultBookmark.targetOffset[2]
  );
  const camTarget = new THREE.Vector3();
  const desiredCamTarget = new THREE.Vector3();

  function computeDesiredCamTarget(out: THREE.Vector3): void {
    const hx = dispPos[0];
    const hy = dispPos[1];
    const hz = dispPos[2];
    const offsetWeight = activeCameraBookmark === 'cam-01' ? Math.exp(-camYaw * camYaw * 3.2) : 1.0;
    const effOffsetX = camTargetOffset.x * offsetWeight;
    const rgtX = Math.cos(camYaw);
    const rgtZ = -Math.sin(camYaw);
    const fwdX = -Math.sin(camYaw);
    const fwdZ = -Math.cos(camYaw);
    out.set(
      hx + rgtX * effOffsetX + fwdX * camTargetOffset.z,
      Math.max(0.72, hy * 0.94) + camTargetOffset.y,
      hz + rgtZ * effOffsetX + fwdZ * camTargetOffset.z
    );
  }

  function applyCameraRig(snapImmediate: boolean, rawDt: number): void {
    const isDriving = mode === 'steer' || mode === 'goto';
    const targetDistBonus = (isDriving ? 0.95 : 0) + (wideCamMode ? 0.95 : 0);

    if (snapImmediate) {
      smoothDistBonus = targetDistBonus;
      camRecenterYaw = null;
    } else {
      smoothDistBonus += (targetDistBonus - smoothDistBonus) * (1 - Math.exp(-rawDt * 5.5));
      if (camRecenterYaw !== null) {
        const diff = Math.atan2(
          Math.sin(camRecenterYaw - camYaw),
          Math.cos(camRecenterYaw - camYaw)
        );
        camYaw += diff * (1 - Math.exp(-rawDt * 7.5));
        if (Math.abs(diff) < 0.008) {
          camYaw = camRecenterYaw;
          camRecenterYaw = null;
        }
      }
    }

    computeDesiredCamTarget(desiredCamTarget);
    if (snapImmediate) {
      camTarget.copy(desiredCamTarget);
    } else {
      camTarget.lerp(desiredCamTarget, 1 - Math.exp(-rawDt * 8.5));
    }

    const effDist = camDist + smoothDistBonus;
    const horiz = Math.cos(camPitch) * effDist;
    const offX = Math.sin(camYaw) * horiz;
    const offY = Math.sin(camPitch) * effDist;
    const offZ = Math.cos(camYaw) * horiz;

    // Architectural 4-wall pavilion camera-collision envelope (never clips behind ANY wall, colonnade, or ceiling)
    const minX = PAVILION_SPEC.leftWallX + 0.45;
    const maxX = PAVILION_SPEC.rightColonnadeX - 0.55;
    const minZ = PAVILION_SPEC.backWallZ + 0.45;
    const maxZ = PAVILION_SPEC.frontWallZ - 0.55;
    const minY = 0.35;
    const maxY = PAVILION_SPEC.ceilingHeight - 0.55;

    let t = 1.0;
    if (camTarget.x + offX < minX && offX < -1e-4) t = Math.min(t, (minX - camTarget.x) / offX);
    if (camTarget.x + offX > maxX && offX > 1e-4) t = Math.min(t, (maxX - camTarget.x) / offX);
    if (camTarget.z + offZ < minZ && offZ < -1e-4) t = Math.min(t, (minZ - camTarget.z) / offZ);
    if (camTarget.z + offZ > maxZ && offZ > 1e-4) t = Math.min(t, (maxZ - camTarget.z) / offZ);
    if (camTarget.y + offY < minY && offY < -1e-4) t = Math.min(t, (minY - camTarget.y) / offY);
    if (camTarget.y + offY > maxY && offY > 1e-4) t = Math.min(t, (maxY - camTarget.y) / offY);
    t = THREE.MathUtils.clamp(t, 0.25, 1.0);
    lastZoomCeiling = Math.max(2.0, effDist * t - smoothDistBonus);

    camera.position.set(
      THREE.MathUtils.clamp(camTarget.x + offX * t, minX, maxX),
      THREE.MathUtils.clamp(camTarget.y + offY * t, minY, maxY),
      THREE.MathUtils.clamp(camTarget.z + offZ * t, minZ, maxZ)
    );
    camera.lookAt(camTarget);
  }

  function setCameraBookmark(id: string): void {
    const bm = STUDIO_CAMERAS[id];
    if (!bm) return;
    activeCameraBookmark = bm.id;
    wideCamMode = false;
    camRecenterYaw = null;
    camYaw = bm.yaw;
    camPitch = bm.pitch;
    camDist = bm.dist;
    camTargetOffset.set(bm.targetOffset[0], bm.targetOffset[1], bm.targetOffset[2]);
    camera.fov = bm.fov;
    camera.updateProjectionMatrix();
    applyCameraRig(true, 1 / 60);
  }

  applyCameraRig(true, 1 / 60);

  // DOM elements
  const statMs = document.getElementById('stat-ms')!;
  const statFps = document.getElementById('stat-fps')!;
  const statusDot = document.getElementById('status-dot')!;
  const modePill = document.getElementById('mode-pill')!;
  const styleBar = document.getElementById('style-bar')!;
  const btnMoreStyles = document.getElementById('btn-more-styles') as HTMLButtonElement;
  const moreStylesPopover = document.getElementById('more-styles-popover')!;
  const moreStylesGrid = document.getElementById('more-styles-grid')!;
  const moreEmotesGrid = document.getElementById('more-emotes-grid')!;
  const moreStylesWrap = btnMoreStyles.parentElement!;
  const subToggles = document.getElementById('sub-toggles')!;
  const toggleEditGymBtn = document.getElementById('toggle-edit-gym') as HTMLButtonElement;
  const editGymBar = document.getElementById('edit-gym-bar')!;
  const editGymLabel = document.getElementById('edit-gym-label')!;
  const btnEditRotate = document.getElementById('btn-edit-rotate') as HTMLButtonElement;
  const btnEditShuffle = document.getElementById('btn-edit-shuffle') as HTMLButtonElement;
  const btnEditReset = document.getElementById('btn-edit-reset') as HTMLButtonElement;
  const btnEditDone = document.getElementById('btn-edit-done') as HTMLButtonElement;
  const toggleGuidesBtn = document.getElementById('toggle-guides')!;
  const toggleXRayBtn = document.getElementById('toggle-xray')!;
  const toggleAutoVaultBtn = document.getElementById('toggle-autovault')!;
  const toggleTargetBtn = document.getElementById('toggle-target')!;
  const toggleContextBtn = document.getElementById('toggle-context')!;
  const toggleContactsBtn = document.getElementById('toggle-contacts')!;
  const toggleTrajBtn = document.getElementById('toggle-traj')!;
  const btnPunch = document.getElementById('btn-punch')!;
  const btnVault = document.getElementById('btn-vault')!;
  const btnDuck = document.getElementById('btn-duck')!;
  const btnRoll = document.getElementById('btn-roll')!;
  const btnSit = document.getElementById('btn-sit')!;
  const sitLabel = document.getElementById('sit-label')!;
  const btnSlowMo = document.getElementById('btn-slowmo')!;
  const btnPs5 = document.getElementById('btn-ps5')!;
  const ps5Label = document.getElementById('ps5-label')!;
  const togglePs5HudBtn = document.getElementById('toggle-ps5-hud')!;
  const ps5MapLabel = document.getElementById('ps5-map-label')!;
  const toggleThemeBtn = document.getElementById('toggle-theme')!;
  const bottomLeftDock = document.getElementById('bottom-left-dock')!;
  const keycapCluster = document.getElementById('keycap-cluster')!;
  const btnReset = document.getElementById('btn-reset')!;
  const btnHelp = document.getElementById('btn-help') as HTMLButtonElement;
  const helpModal = document.getElementById('help-modal')!;
  const btnHelpClose = document.getElementById('btn-help-close') as HTMLButtonElement;
  const colorSchemeMeta = document.querySelector<HTMLMetaElement>('meta[name="color-scheme"]');
  const inputBadges = Array.from(document.querySelectorAll<HTMLElement>('.input-badge[data-ps5]'));

  let ps5SprintActive = false;
  let ps5SlowMoHold = false;
  let emoteToggle = false;
  let ps5HudVisible = false;
  let lightTheme = false;
  let morePopoverOpen = false;
  let helpModalOpen = false;
  let editGymMode = false;
  let selectedObsId: string | null = null;
  let hoveredObsId: string | null = null;
  let draggingObs: ObstacleInfo | null = null;
  const dragOffset = new THREE.Vector3();
  const floorDragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const floorHitPoint = new THREE.Vector3();
  const emoteBtns = new Map<'look' | 'warmup', HTMLButtonElement>();

  const keys = new Set<string>();
  const keycapEls = new Map<string, HTMLElement>();
  document.querySelectorAll<HTMLElement>('.kcap[data-key]').forEach((el) => {
    keycapEls.set(el.dataset.key!, el);
  });

  function syncKeycapLights(): void {
    const locked = ps5.isConnected();
    keycapEls.forEach((el, code) => {
      const lit =
        !locked &&
        (keys.has(code) ||
          (code === 'ShiftLeft' && keys.has('ShiftRight')) ||
          (code === 'KeyC' && activeStyle === 'crouch'));
      el.classList.toggle('pressed', lit);
    });
  }

  function syncDockInputBadges(ps5Connected: boolean): void {
    document.body.classList.toggle('ps5-connected', ps5Connected);
    for (const badge of inputBadges) {
      badge.textContent = ps5Connected ? (badge.dataset.ps5 ?? '') : (badge.dataset.kb ?? '');
    }
    keycapCluster.classList.toggle('hidden', ps5Connected);
    ps5MapLabel.textContent = ps5Connected ? 'PS5 HUD · Keys Locked' : 'PS5 Map';
  }

  const appRoot = document.getElementById('app')!;
  const ps5Hud = new PS5ControllerHUD(appRoot, {
    onConnectClick: () => {
      void handlePs5Connect();
    },
    onTriggerAction: (key: string) => {
      switch (key) {
        case 'cross':
          startSmartJump();
          break;
        case 'circle':
          toggleCrouch();
          break;
        case 'square':
          ps5.triggerPunchHaptic('left');
          startPunch();
          break;
        case 'r2':
          ps5.triggerPunchHaptic('right');
          startPunch('right');
          break;
        case 'triangle':
          toggleSit();
          break;
        case 'dpadLR': {
          const idx = styleOrder.indexOf(activeStyle);
          selectStyle(styleOrder[(idx + 1) % styleOrder.length], false);
          break;
        }
        case 'dpadDown':
          startDuck(false);
          break;
        case 'dpadUp':
          emoteToggle = !emoteToggle;
          startEmote(emoteToggle ? 'look' : 'warmup');
          break;
        case 'l1':
          toggleWideCam();
          break;
        case 'l2':
          toggleSlowMo();
          break;
        case 'r1':
          selectStyle(activeStyle === 'run' ? 'jog' : 'run', false);
          break;
        case 'r3':
          startRoll();
          break;
        case 'touchpad':
          toggleXRay();
          break;
        case 'options':
          toggleGuides();
          break;
        case 'create':
          void resetCharacter();
          break;
        default:
          break;
      }
    },
  });

  function setPS5HudVisible(show: boolean): void {
    ps5HudVisible = show;
    ps5Hud.setVisible(show);
    bottomLeftDock.classList.toggle('hidden', show);
    togglePs5HudBtn.setAttribute('aria-expanded', String(show));
    btnPs5.setAttribute('aria-expanded', String(show));
  }

  function togglePS5Hud(): void {
    setPS5HudVisible(!ps5HudVisible);
  }

  function setStudioTheme(isLight: boolean): void {
    lightTheme = isLight;
    const themeMode: 'dark' | 'light' = isLight ? 'light' : 'dark';
    const palette = STUDIO_THEMES[themeMode];

    document.body.classList.toggle('theme-light', isLight);
    if (colorSchemeMeta) {
      colorSchemeMeta.content = themeMode;
    }

    scene.background = new THREE.Color(palette.bg);
    scene.fog = new THREE.FogExp2(palette.fog, palette.fogDensity);
    hemi.color.setHex(palette.hemiSky);
    hemi.groundColor.setHex(palette.hemiGround);
    hemi.intensity = palette.hemiIntensity;
    keyLight.color.setHex(palette.keyColor);
    keyLight.intensity = palette.keyIntensity;
    rimLight.color.setHex(palette.rimColor);
    rimLight.intensity = palette.rimIntensity;
    scene.environmentIntensity = palette.envIntensity;

    floor.setTheme(isLight);
    pavilion.setTheme(themeMode);

    toggleThemeBtn.textContent = isLight ? 'Dark' : 'Light';
    toggleThemeBtn.classList.toggle('active', isLight);
    toggleThemeBtn.setAttribute('aria-pressed', String(isLight));
  }

  function toggleTheme(): void {
    setStudioTheme(!lightTheme);
  }

  const ps5 = new DualSenseBridge({
    onJump: () => startSmartJump(),
    onCrouch: () => toggleCrouch(),
    onPunchLeft: () => startPunch(),
    onPunchRight: () => startPunch('right'),
    onSit: () => toggleSit(),
    onDuck: () => startDuck(false),
    onRoll: () => startRoll(),
    onEmote: () => {
      emoteToggle = !emoteToggle;
      startEmote(emoteToggle ? 'look' : 'warmup');
    },
    onCycleStyle: (delta) => {
      const idx = styleOrder.indexOf(activeStyle);
      const nextIdx = (idx + delta + styleOrder.length) % styleOrder.length;
      const nextId = styleOrder[nextIdx];
      selectStyle(nextId, false);
      ps5Hud.flashAction(
        delta < 0 ? 'dpadLeft' : 'dpadRight',
        `Style · ${styleMap.get(nextId)?.label ?? nextId}`
      );
    },
    onToggleWideCam: () => toggleWideCam(),
    onToggleSlowMo: () => toggleSlowMo(),
    onToggleXRay: () => toggleXRay(),
    onToggleGuides: () => toggleGuides(),
    onReset: () => void resetCharacter(),
    onStatusChange: (connected, _label, snapshot, source?: PS5InputSource) => {
      const resolvedSource: PS5InputSource = source ?? ps5.getSource();
      const topBarLabel = formatPS5TopBarLabel(connected, resolvedSource, snapshot);
      const hudBadge = formatPS5HudBadge(connected, resolvedSource, snapshot);

      // STRICT PS5 EXCLUSIVE LOCKOUT: clear all held keys & blur focused DOM controls on connect!
      if (connected) {
        keys.clear();
        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur();
        }
      }
      syncKeycapLights();

      btnPs5.classList.toggle('connected', connected);
      ps5Label.textContent = topBarLabel;
      ps5Hud.setConnectionStatus(connected, hudBadge, snapshot);
      syncDockInputBadges(connected);
      updateModeBanner();

      // Automatically open horizontal PS5 DualSense HUD on connect; swap cleanly back on disconnect!
      setPS5HudVisible(connected);
    },
    onActionTriggered: (buttonKey, actionLabel) => {
      ps5Hud.flashAction(buttonKey, actionLabel);
    },
    onHapticPulse: (left, right, reason) => {
      ps5Hud.pulseHapticVisual(left, right, reason);
    },
  });

  // Apply Dark Brutalist Pavilion palette on boot
  setStudioTheme(false);
  syncDockInputBadges(false);

  async function handlePs5Connect(): Promise<void> {
    setPS5HudVisible(true);
    const res = await ps5.connect();
    if (!res.ok && res.error) {
      ps5Hud.flashAction('options', res.error);
    }
  }

  btnPs5.addEventListener('click', () => {
    if (ps5.isConnected()) {
      togglePS5Hud();
    } else {
      void handlePs5Connect();
    }
  });
  togglePs5HudBtn.addEventListener('click', togglePS5Hud);
  toggleThemeBtn.addEventListener('click', toggleTheme);

  function syncOverlayVisibility(): void {
    const hasChunk = currentChunk !== null;
    targetGhost.group.visible = showAllGuides && showTargetGhost && hasChunk;
    contextGhost.group.visible = showAllGuides && showContextGhost && hasChunk;
    trajLine.visible = showAllGuides && showTraj;
    beaconGroup.visible = showAllGuides && showTraj && gotoTarget !== null;
    leftContactRing.visible = showAllGuides && showContacts;
    rightContactRing.visible = showAllGuides && showContacts;
    skeletonRig.setShowContacts(false);
    sensorZoneGroup.visible = showAllGuides && autoVaultEnabled;

    if (somaSkin) {
      somaSkin.setXRayMode(xrayMode);
      skeletonRig.group.visible = xrayMode;
    } else {
      skeletonRig.group.visible = true;
    }

    toggleGuidesBtn.classList.toggle('active', showAllGuides);
    toggleGuidesBtn.setAttribute('aria-pressed', String(showAllGuides));
    toggleGuidesBtn.setAttribute('aria-expanded', String(showAllGuides));
    subToggles.classList.toggle('hidden', !showAllGuides);

    toggleXRayBtn.classList.toggle('active', xrayMode);
    toggleXRayBtn.setAttribute('aria-pressed', String(xrayMode));

    toggleAutoVaultBtn.classList.toggle('active', autoVaultEnabled);
    toggleAutoVaultBtn.setAttribute('aria-pressed', String(autoVaultEnabled));
    toggleTargetBtn.classList.toggle('active', showTargetGhost);
    toggleTargetBtn.setAttribute('aria-pressed', String(showTargetGhost));
    toggleContextBtn.classList.toggle('active', showContextGhost);
    toggleContextBtn.setAttribute('aria-pressed', String(showContextGhost));
    toggleTrajBtn.classList.toggle('active', showTraj);
    toggleTrajBtn.setAttribute('aria-pressed', String(showTraj));
    toggleContactsBtn.classList.toggle('active', showContacts);
    toggleContactsBtn.setAttribute('aria-pressed', String(showContacts));
  }

  function updateModeBanner(): void {
    const connected = ps5.isConnected();
    const parts: string[] = [];
    if (mode === 'punch_left') parts.push('Combat · Left Jab');
    else if (mode === 'punch_right') parts.push('Combat · Right Cross');
    else if (mode === 'combat_recover') parts.push('Combat Guard');
    else if (mode.startsWith('vault')) parts.push('Parkour Vault');
    else if (mode.startsWith('duck')) parts.push('Overhead Duck');
    else if (mode === 'roll') parts.push('360° Combat Roll');
    else if (mode.startsWith('jump')) parts.push('Parkour Hurdle');
    else if (mode.startsWith('emote_look')) parts.push('Look Around');
    else if (mode.startsWith('emote_warmup')) parts.push('Warm-Up Stretch');
    else if (mode === 'sit_approach' || mode === 'sit_down') parts.push('Sitting Down');
    else if (mode === 'sit_hold') {
      parts.push(connected ? 'Seated · L-Stick or △ to Stand' : 'Seated · WASD or E to Stand');
    } else if (mode === 'stand_up') parts.push('Standing Up');
    else if (mode === 'goto') parts.push('Neural Path');

    if (activeStyle === 'crouch' && (mode === 'idle' || mode === 'steer')) {
      parts.push(connected ? 'Crouch · ○ to Stand' : 'Crouch · C to Stand');
    }
    if (xrayMode) parts.push('Neural X-Ray');
    if (wideCamMode) parts.push('Wide Cam');
    if (slowMo || ps5SlowMoHold) parts.push('0.25× Slow-Mo');

    if (parts.length > 0 && !editGymMode) {
      modePill.textContent = parts.join('  ·  ');
      modePill.classList.remove('hidden');
    } else {
      modePill.classList.add('hidden');
    }

    btnPunch.classList.toggle('active', mode === 'punch_left' || mode === 'punch_right');
    btnVault.classList.toggle('active', mode.startsWith('vault') || mode.startsWith('jump'));
    btnDuck.classList.toggle('active', mode.startsWith('duck'));
    btnRoll.classList.toggle('active', mode === 'roll');
    btnSit.classList.toggle('active', mode.startsWith('sit') || mode === 'stand_up');
    sitLabel.textContent = mode === 'sit_hold' ? 'Stand' : 'Sit';

    const effectiveSlow = slowMo || ps5SlowMoHold;
    btnSlowMo.classList.toggle('active', effectiveSlow);
    btnSlowMo.setAttribute('aria-pressed', String(effectiveSlow));

    emoteBtns.get('look')?.classList.toggle('active', mode.startsWith('emote_look'));
    emoteBtns.get('warmup')?.classList.toggle('active', mode.startsWith('emote_warmup'));
  }

  function updateOverlayGeometryFromChunk(chunk: BinaryChunk): void {
    // Use frame 0 of the 4-frame context window (trails cleanly behind the character at the start of the Path!)
    copyFlatPositionsToTuples(chunk.contextPos, ghostCtxTuples, 0);
    copyFlatPositionsToTuples(chunk.targetPos, ghostTgtTuples, 0);
    contextGhost.updatePose(ghostCtxTuples);
    targetGhost.updatePose(ghostTgtTuples);

    const nPts = Math.min(chunk.num_frames, trajMaxPts);
    for (let i = 0; i < nPts; i++) {
      const base = i * numRenderJoints * 3;
      trajPositions[i * 3] = chunk.framesPos[base];
      trajPositions[i * 3 + 1] = 0.03;
      trajPositions[i * 3 + 2] = chunk.framesPos[base + 2];
    }
    trajGeo.attributes.position.needsUpdate = true;
    trajGeo.setDrawRange(0, nPts);
  }

  // Populate compact bottom-right style switcher (6 inline + 4 specialty & 2 emotes in More ▾ popover)
  const PRIMARY_STYLE_IDS = new Set(['walk', 'jog', 'run', 'crouch', 'combat', 'zombie']);
  const styleBtns = new Map<string, HTMLButtonElement>();

  function setMorePopoverOpen(open: boolean): void {
    morePopoverOpen = open;
    moreStylesPopover.classList.toggle('hidden', !open);
    btnMoreStyles.setAttribute('aria-expanded', String(open));
  }

  function syncMoreStylesButton(): void {
    const isSpecialty = !PRIMARY_STYLE_IDS.has(activeStyle);
    const info = styleMap.get(activeStyle);
    btnMoreStyles.classList.toggle('active', isSpecialty);
    if (isSpecialty && info) {
      btnMoreStyles.style.setProperty('--style-accent', info.accent);
      btnMoreStyles.textContent = `${info.label} ▾`;
    } else {
      btnMoreStyles.style.removeProperty('--style-accent');
      btnMoreStyles.textContent = 'More ▾';
    }
  }

  initData.styles.forEach((s) => {
    const isPrimary = PRIMARY_STYLE_IDS.has(s.id);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'style-btn' + (s.id === activeStyle ? ' active' : '');
    btn.style.setProperty('--style-accent', s.accent);
    btn.setAttribute('aria-pressed', String(s.id === activeStyle));
    if (!isPrimary) btn.setAttribute('role', 'menuitem');
    btn.textContent = s.label;
    btn.addEventListener('click', () => {
      selectStyle(s.id, true);
      if (!isPrimary) setMorePopoverOpen(false);
    });
    if (isPrimary) {
      styleBar.appendChild(btn);
    } else {
      moreStylesGrid.appendChild(btn);
    }
    styleBtns.set(s.id, btn);
  });

  const EMOTE_SPECS: Array<{ kind: 'look' | 'warmup'; label: string }> = [
    { kind: 'look', label: 'Scan' },
    { kind: 'warmup', label: 'Stretch' },
  ];
  EMOTE_SPECS.forEach(({ kind, label }) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'style-btn emote-btn';
    btn.setAttribute('role', 'menuitem');
    btn.textContent = label;
    btn.addEventListener('click', () => {
      startEmote(kind);
      setMorePopoverOpen(false);
    });
    moreEmotesGrid.appendChild(btn);
    emoteBtns.set(kind, btn);
  });

  btnMoreStyles.addEventListener('click', (e) => {
    e.stopPropagation();
    setMorePopoverOpen(!morePopoverOpen);
  });

  document.addEventListener('pointerdown', (e) => {
    if (morePopoverOpen && !moreStylesWrap.contains(e.target as Node)) {
      setMorePopoverOpen(false);
    }
  });

  syncMoreStylesButton();

  function selectStyle(id: string, demoStrideIfIdle = false): void {
    if (!styleMap.has(id)) return;
    if (id !== 'crouch') prevNonCrouchStyle = id;
    activeStyle = id;
    const info = styleMap.get(id)!;
    skeletonRig.setAccentColor(info.accent);
    somaSkin?.setAccentColor(info.accent);
    for (const [k, b] of styleBtns) {
      const isCur = k === id;
      b.classList.toggle('active', isCur);
      b.setAttribute('aria-pressed', String(isCur));
    }
    syncMoreStylesButton();
    updateModeBanner();

    const steer = getSteerInput();
    if (steer || mode === 'steer') {
      mode = 'steer';
      triggerInterrupt('steer', steer ?? lastSteerDir);
    } else if (mode === 'goto' && gotoTarget) {
      triggerInterrupt('goto');
    } else if (mode === 'idle' || mode === 'combat_recover') {
      if (id === 'crouch' || id === 'combat' || !demoStrideIfIdle) {
        mode = id === 'combat' ? 'combat_recover' : 'idle';
        triggerInterrupt(mode);
      } else {
        const hx = dispPos[0];
        const hz = dispPos[2];
        const dx = currentChunk ? currentChunk.heading_end[0] : -0.28;
        const dz = currentChunk ? currentChunk.heading_end[1] : 0.96;
        let tx = hx + dx * 2.2;
        let tz = hz + dz * 2.2;
        if (Math.hypot(tx, tz) > 5.8) {
          const backLen = Math.hypot(hx, hz) || 1;
          tx = hx - (hx / backLen) * 2.2;
          tz = hz - (hz / backLen) * 2.2;
        }
        startGoto(tx, tz);
      }
    }
  }

  function toggleCrouch(): void {
    selectStyle(activeStyle === 'crouch' ? prevNonCrouchStyle : 'crouch', false);
  }

  function startPunch(side?: 'left' | 'right'): void {
    const chosen = side ?? nextPunchSide;
    nextPunchSide = chosen === 'left' ? 'right' : 'left';
    const steer = getSteerInput();
    if (steer) {
      actionDir.copy(steer);
    } else if (currentChunk) {
      actionDir.set(currentChunk.heading_end[0], currentChunk.heading_end[1]).normalize();
    } else {
      actionDir.set(-Math.sin(camYaw), -Math.cos(camYaw)).normalize();
    }
    gotoTarget = null;
    mode = chosen === 'left' ? 'punch_left' : 'punch_right';
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt(mode, actionDir);
  }

  function startGoto(x: number, z: number): void {
    const cx = THREE.MathUtils.clamp(x, ROOM_BOUNDS.minX, ROOM_BOUNDS.maxX);
    const cz = THREE.MathUtils.clamp(z, ROOM_BOUNDS.minZ, ROOM_BOUNDS.maxZ);
    gotoTarget = { x: cx, z: cz };
    beaconGroup.position.set(cx, 0.014, cz);
    mode = 'goto';
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt('goto');
  }

  function findNearestObstacle(kind: ObstacleInfo['type']): ObstacleInfo | undefined {
    const hx = dispPos[0];
    const hz = dispPos[2];
    let best: ObstacleInfo | undefined;
    let bestDist = Infinity;
    for (const obs of initData.obstacles) {
      if (obs.type !== kind) continue;
      const dist = Math.hypot(obs.pos[0] - hx, obs.pos[2] - hz);
      if (dist < bestDist) {
        bestDist = dist;
        best = obs;
      }
    }
    return best;
  }

  function startVault(fromAuto = false, targetObs?: ObstacleInfo): void {
    const obs = targetObs ?? findNearestObstacle('vault') ?? initData.obstacles[0];
    if (!obs) return;
    const hx = dispPos[0];
    const hz = dispPos[2];
    activeVaultObsId = obs.id;
    const crossZ = obs.size[2] < obs.size[0];
    if (crossZ) {
      vaultDir = hz <= obs.pos[2] ? 1 : -1;
      activeVaultLane = THREE.MathUtils.clamp(hx - obs.pos[0], -0.7, 0.7) + obs.pos[0];
      vaultZ = obs.pos[2];
    } else {
      vaultDir = hx <= obs.pos[0] ? 1 : -1;
      vaultZ = THREE.MathUtils.clamp(hz - obs.pos[2], -0.7, 0.7) + obs.pos[2];
      activeVaultLane = vaultZ;
    }
    gotoTarget = null;
    mode = fromAuto ? 'vault_action' : 'vault_entry';
    ps5.triggerVaultHaptic();
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt(mode);
  }

  function startDuck(fromAuto = false, targetObs?: ObstacleInfo): void {
    const obs = targetObs ?? findNearestObstacle('duck') ?? initData.obstacles[2];
    if (!obs) return;
    const hx = dispPos[0];
    const hz = dispPos[2];
    activeDuckObsId = obs.id;
    const crossZ = obs.size[2] < obs.size[0];
    if (crossZ) {
      duckDir = hz <= obs.pos[2] ? 1 : -1;
      duckLane = THREE.MathUtils.clamp(hx - obs.pos[0], -0.65, 0.65) + obs.pos[0];
      duckX = duckLane;
    } else {
      duckDir = hx <= obs.pos[0] ? 1 : -1;
      duckLane = THREE.MathUtils.clamp(hz - obs.pos[2], -0.65, 0.65) + obs.pos[2];
      duckX = obs.pos[0];
    }
    gotoTarget = null;
    mode = fromAuto ? 'duck_action' : 'duck_entry';
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt(mode);
  }

  function startEmote(kind: 'look' | 'warmup'): void {
    gotoTarget = null;
    mode = kind === 'look' ? 'emote_look' : 'emote_warmup';
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt(mode);
  }

  function startRoll(): void {
    const hx = dispPos[0];
    const hz = dispPos[2];
    const steer = getSteerInput();
    if (steer) actionDir.copy(steer);
    else if (currentChunk) {
      actionDir.set(currentChunk.heading_end[0], currentChunk.heading_end[1]).normalize();
    } else actionDir.set(0, 1);

    // Wall-collision guard so combat roll never dives into a concrete wall
    const wallMargin = 1.35;
    let rx = actionDir.x;
    let rz = actionDir.y;
    if (hx <= ROOM_BOUNDS.minX + wallMargin && rx < 0) rx = 0;
    if (hx >= ROOM_BOUNDS.maxX - wallMargin && rx > 0) rx = 0;
    if (hz <= ROOM_BOUNDS.minZ + wallMargin && rz < 0) rz = 0;
    if (hz >= ROOM_BOUNDS.maxZ - wallMargin && rz > 0) rz = 0;
    const rLen = Math.hypot(rx, rz);
    if (rLen < 0.25) return;
    actionDir.set(rx / rLen, rz / rLen);

    gotoTarget = null;
    mode = 'roll';
    footLocker.reset();
    ps5.triggerVaultHaptic();
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt('roll', actionDir);
  }

  function startSmartJump(): void {
    const hx = dispPos[0];
    const hz = dispPos[2];
    const nearestVault = findNearestObstacle('vault');
    if (nearestVault) {
      const crossZ = nearestVault.size[2] < nearestVault.size[0];
      const crossDist = crossZ
        ? Math.abs(hz - nearestVault.pos[2])
        : Math.abs(hx - nearestVault.pos[0]);
      const lateralDist = crossZ
        ? Math.abs(hx - nearestVault.pos[0])
        : Math.abs(hz - nearestVault.pos[2]);
      const halfSpan = (crossZ ? nearestVault.size[0] : nearestVault.size[2]) * 0.5 + 0.45;
      if (crossDist < 2.6 && lateralDist < halfSpan) {
        startVault(crossDist < 2.15, nearestVault);
        return;
      }
    }
    const steer = getSteerInput();
    if (steer) actionDir.copy(steer);
    else if (currentChunk) {
      actionDir.set(currentChunk.heading_end[0], currentChunk.heading_end[1]).normalize();
    } else actionDir.set(0, 1);

    // Wall-collision jump guard: prevent leaping outward into a concrete wall right in front of the player!
    const wallMargin = 1.25;
    let jx = actionDir.x;
    let jz = actionDir.y;
    if (hx <= ROOM_BOUNDS.minX + wallMargin && jx < 0) jx = 0;
    if (hx >= ROOM_BOUNDS.maxX - wallMargin && jx > 0) jx = 0;
    if (hz <= ROOM_BOUNDS.minZ + wallMargin && jz < 0) jz = 0;
    if (hz >= ROOM_BOUNDS.maxZ - wallMargin && jz > 0) jz = 0;
    const jLen = Math.hypot(jx, jz);
    if (jLen < 0.25) {
      // Directly facing concrete wall with zero leap clearance: do not jump through the wall!
      return;
    }
    actionDir.set(jx / jLen, jz / jLen);

    gotoTarget = null;
    mode = 'jump_leap';
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt('jump_leap', actionDir);
  }

  function toggleSit(): void {
    gotoTarget = null;
    syncOverlayVisibility();
    if (mode === 'sit_hold' || mode === 'sit_down') {
      mode = 'stand_up';
      updateModeBanner();
      triggerInterrupt('stand_up');
    } else {
      mode = 'sit_approach';
      updateModeBanner();
      triggerInterrupt('sit_approach');
    }
  }

  function toggleGuides(): void {
    showAllGuides = !showAllGuides;
    syncOverlayVisibility();
  }

  function toggleXRay(): void {
    xrayMode = !xrayMode;
    syncOverlayVisibility();
    updateModeBanner();
  }

  function toggleWideCam(): void {
    wideCamMode = !wideCamMode;
    let hdX = -0.28;
    let hdZ = 0.96;
    if (mode === 'steer' && lastSteerDir.lengthSq() > 1e-4) {
      hdX = lastSteerDir.x;
      hdZ = lastSteerDir.y;
    } else if (currentChunk) {
      hdX = currentChunk.heading_end[0];
      hdZ = currentChunk.heading_end[1];
    }
    camRecenterYaw = Math.atan2(-hdX, -hdZ);
    ps5Hud.flashAction('l1', 'Wide / Center Cam');
    updateModeBanner();
  }

  function toggleSlowMo(): void {
    slowMo = !slowMo;
    updateModeBanner();
  }

  function setHelpModalOpen(open: boolean): void {
    helpModalOpen = open;
    helpModal.classList.toggle('hidden', !open);
    btnHelp.classList.toggle('active', open);
    btnHelp.setAttribute('aria-expanded', String(open));
    if (open) {
      if (morePopoverOpen) setMorePopoverOpen(false);
      keys.clear();
      syncKeycapLights();
      btnHelpClose.focus();
    } else if (document.activeElement && helpModal.contains(document.activeElement)) {
      btnHelp.focus();
    }
  }

  function toggleHelp(): void {
    setHelpModalOpen(!helpModalOpen);
  }

  // =========================================================================
  // EDIT GYM MODE: Live Drag / Rotate 90° / Shuffle / Reset Obstacle Layout
  // Proves zero hardcoded coordinates across BOTH Live ONNX & Pre-Baked modes!
  // =========================================================================
  function shortObstacleLabel(obs: ObstacleInfo): string {
    switch (obs.type) {
      case 'vault':
        return obs.id === 'stress_vault' ? 'Center Vault Wall' : 'Vault Plinth';
      case 'duck':
        return obs.id === 'stress_duck' ? 'Low Tunnel Gate' : 'Duck Gantry';
      case 'sit':
        return 'Gallery Bench';
      case 'hurdle':
        return obs.id === 'hurdle_rail_side' ? 'Hurdle Rail (Right)' : 'Hurdle Rail (Left)';
      default:
        return obs.label;
    }
  }

  function clampObstaclePos(obs: ObstacleInfo, rawX: number, rawZ: number): [number, number] {
    const halfX = obs.type === 'sit' ? Math.max(obs.size[0], obs.size[2]) * 0.5 : obs.size[0] * 0.5;
    const halfZ = obs.type === 'sit' ? Math.max(obs.size[0], obs.size[2]) * 0.5 : obs.size[2] * 0.5;
    const margin = 0.55;
    const x = THREE.MathUtils.clamp(rawX, ROOM_BOUNDS.minX + halfX + margin, ROOM_BOUNDS.maxX - halfX - margin);
    const z = THREE.MathUtils.clamp(rawZ, ROOM_BOUNDS.minZ + halfZ + margin, ROOM_BOUNDS.maxZ - halfZ - margin);
    return [Number(x.toFixed(3)), Number(z.toFixed(3))];
  }

  function updateEditGymUI(): void {
    toggleEditGymBtn.classList.toggle('active', editGymMode);
    toggleEditGymBtn.setAttribute('aria-pressed', String(editGymMode));
    editGymBar.classList.toggle('hidden', !editGymMode);
    canvas.classList.toggle('edit-mode', editGymMode);
    canvas.classList.toggle('edit-hover', editGymMode && hoveredObsId !== null && draggingObs === null);
    canvas.classList.toggle('edit-dragging', editGymMode && draggingObs !== null);
    obstacleManager.setEditVisuals(editGymMode, hoveredObsId, selectedObsId);

    const sel = selectedObsId ? initData.obstacles.find((o) => o.id === selectedObsId) : undefined;
    btnEditRotate.disabled = !sel;
    if (sel) {
      editGymLabel.textContent = `${shortObstacleLabel(sel)} · (${sel.pos[0].toFixed(1)}, ${sel.pos[2].toFixed(1)})`;
    } else {
      editGymLabel.textContent = 'Drag any prop on floor';
    }
    updateModeBanner();
  }

  function syncObstaclesToEngines(): void {
    obstacleCooldowns.clear();
    obstacleManager.syncAll(initData.obstacles);
    motionBank?.setObstacles(initData.obstacles);
    if (liveAvailable) {
      void fetch(`${API_BASE}/api/obstacles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          obstacles: initData.obstacles,
        }),
      }).catch((err) => {
        console.warn('[MotionBricks] Failed to sync obstacles to Live ONNX server:', err);
      });
    }
  }

  function setEditGymMode(active: boolean): void {
    if (editGymMode === active) return;
    editGymMode = active;
    if (active) {
      if (morePopoverOpen) setMorePopoverOpen(false);
      if (helpModalOpen) setHelpModalOpen(false);
    } else {
      draggingObs = null;
      hoveredObsId = null;
      selectedObsId = null;
    }
    updateEditGymUI();
  }

  function toggleEditGym(): void {
    setEditGymMode(!editGymMode);
  }

  function moveObstacleById(id: string, rawX: number, rawZ: number): boolean {
    const obs = initData.obstacles.find((o) => o.id === id);
    if (!obs) return false;
    const [cx, cz] = clampObstaclePos(obs, rawX, rawZ);
    obs.pos[0] = cx;
    obs.pos[2] = cz;
    selectedObsId = obs.id;
    syncObstaclesToEngines();
    updateEditGymUI();
    return true;
  }

  function rotateObstacleById(id: string): boolean {
    const obs = initData.obstacles.find((o) => o.id === id);
    if (!obs) return false;
    if (obs.type === 'sit') {
      const fx = obs.facing ? obs.facing[0] : -0.8192;
      const fz = obs.facing ? obs.facing[1] : 0.5736;
      const len = Math.hypot(fx, fz) || 1;
      // Rotate 90° around Y: (fx, fz) -> (-fz, fx)
      obs.facing = [Number((-fz / len).toFixed(4)), Number((fx / len).toFixed(4))];
    } else {
      const tmp = obs.size[0];
      obs.size[0] = obs.size[2];
      obs.size[2] = tmp;
      const [cx, cz] = clampObstaclePos(obs, obs.pos[0], obs.pos[2]);
      obs.pos[0] = cx;
      obs.pos[2] = cz;
    }
    selectedObsId = obs.id;
    syncObstaclesToEngines();
    updateEditGymUI();
    return true;
  }

  function rotateSelectedObstacle(): void {
    if (!selectedObsId) return;
    rotateObstacleById(selectedObsId);
  }

  function shuffleGymObstacles(): void {
    const runnerX = dispPos[0];
    const runnerZ = dispPos[2];
    const placed: Array<{ x: number; z: number; hx: number; hz: number }> = [];
    const benchFacings: Array<[number, number]> = [
      [-0.8192, 0.5736],
      [0.8192, 0.5736],
      [0.0, 1.0],
      [1.0, 0.0],
      [-1.0, 0.0],
      [0.0, -1.0],
    ];

    for (const obs of initData.obstacles) {
      // Randomly choose 0° or 90° orientation first
      if (obs.type === 'sit') {
        const pick = benchFacings[Math.floor(Math.random() * benchFacings.length)];
        obs.facing = [pick[0], pick[1]];
      } else if (Math.random() < 0.5) {
        const tmp = obs.size[0];
        obs.size[0] = obs.size[2];
        obs.size[2] = tmp;
      }

      const hx = obs.type === 'sit' ? Math.max(obs.size[0], obs.size[2]) * 0.5 : obs.size[0] * 0.5;
      const hz = obs.type === 'sit' ? Math.max(obs.size[0], obs.size[2]) * 0.5 : obs.size[2] * 0.5;
      const minX = ROOM_BOUNDS.minX + hx + 1.2;
      const maxX = ROOM_BOUNDS.maxX - hx - 1.2;
      const minZ = Math.max(ROOM_BOUNDS.minZ + hz + 1.4, -14.8);
      const maxZ = Math.min(ROOM_BOUNDS.maxZ - hz - 1.6, 4.2);

      let chosenX = obs.pos[0];
      let chosenZ = obs.pos[2];
      let found = false;
      for (let attempt = 0; attempt < 90; attempt++) {
        const cx = minX + Math.random() * Math.max(0.5, maxX - minX);
        const cz = minZ + Math.random() * Math.max(0.5, maxZ - minZ);
        if (Math.hypot(cx - runnerX, cz - runnerZ) < 2.55) continue;
        const pad = attempt < 55 ? 1.45 : 0.95;
        let overlaps = false;
        for (const p of placed) {
          if (Math.abs(cx - p.x) < hx + p.hx + pad && Math.abs(cz - p.z) < hz + p.hz + pad) {
            overlaps = true;
            break;
          }
        }
        if (!overlaps) {
          chosenX = cx;
          chosenZ = cz;
          found = true;
          break;
        }
      }
      const [fx, fz] = clampObstaclePos(obs, found ? chosenX : obs.pos[0], found ? chosenZ : obs.pos[2]);
      obs.pos[0] = fx;
      obs.pos[2] = fz;
      placed.push({ x: fx, z: fz, hx, hz });
    }

    syncObstaclesToEngines();
    selectedObsId = null;
    updateEditGymUI();
    editGymLabel.textContent = 'Shuffled 7 props';
  }

  function resetGymObstacles(): void {
    const fresh = cloneObstacles(defaultObstacles);
    for (let i = 0; i < initData.obstacles.length && i < fresh.length; i++) {
      initData.obstacles[i].pos = fresh[i].pos;
      initData.obstacles[i].size = fresh[i].size;
      initData.obstacles[i].facing = fresh[i].facing;
    }
    syncObstaclesToEngines();
    selectedObsId = null;
    updateEditGymUI();
    editGymLabel.textContent = 'Default layout restored';
  }

  async function resetCharacter(): Promise<void> {
    mode = 'idle';
    gotoTarget = null;
    activeVaultObsId = null;
    activeDuckObsId = null;
    obstacleCooldowns.clear();
    currentChunk = null;
    nextChunk = null;
    playFrame = 0;
    inFlight = false;
    pendingInterrupt = null;
    syntheticSteer = null;
    requestSeq += 1;
    inertializer.reset();
    footLocker.reset();

    let data: { pose: { pos: Vec3Tuple[]; bone_quat?: QuatTuple[] } };
    if (engineMode === 'baked-bank' && motionBank) {
      data = motionBank.reset() as { pose: { pos: Vec3Tuple[]; bone_quat?: QuatTuple[] } };
    } else {
      const res = await fetch(`${API_BASE}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      data = await res.json();
    }
    seedInitialPose(data.pose.pos, data.pose.bone_quat);
    trajGeo.setDrawRange(0, 0);
    syncOverlayVisibility();
    updateModeBanner();
    triggerInterrupt('idle');
  }

  // Wire Engine Mode Slider Switch (Pre-Baked <-> Live ONNX)
  function requestEngineMode(target: 'baked-bank' | 'live-onnx'): void {
    if (target === 'live-onnx' && !liveAvailable) {
      // Hosted / locked mode: briefly slide toward Live ONNX, show clear message, then spring back to Pre-Baked!
      if (engineSliderEl) {
        engineSliderEl.classList.add('bouncing');
        if (engineBounceTimer !== null) window.clearTimeout(engineBounceTimer);
        engineBounceTimer = window.setTimeout(() => {
          engineSliderEl.classList.remove('bouncing');
          engineBounceTimer = null;
        }, 460);
      }
      const mbLabel = motionBank ? (motionBank.byteSize / (1024 * 1024)).toFixed(2) : '0.38';
      showEngineToast(
        `Live FP16 ONNX (363 MB) runs on a local server — clone the GitHub repo to run live ~25 ms inference. Staying on Pre-Baked (${mbLabel} MB).`,
        4500
      );
      return;
    }

    if (target === 'baked-bank' && !motionBank) {
      showEngineToast('Pre-Baked motion bank not loaded.', 2800);
      return;
    }

    if (target === engineMode) return;

    engineMode = target;
    nextChunk = null;
    pendingInterrupt = null;
    if (engineMode === 'baked-bank' && motionBank) {
      motionBank.setObstacles(initData.obstacles);
      motionBank.syncAnchor(dispPos[0], dispPos[2], lastSteerDir.x, lastSteerDir.y);
      const mbLabel = (motionBank.byteSize / (1024 * 1024)).toFixed(2);
      statMs.textContent = 'Pre-Baked';
      showEngineToast(`Switched to Pre-Baked neural bank (${mbLabel} MB · zero-server playback)`, 2800);
    } else {
      syncObstaclesToEngines();
      showEngineToast('Switched to Live FP16 ONNX local server (~25 ms real-time inference)', 2800);
    }
    syncEngineSliderUI();
    triggerInterrupt(mode === 'steer' ? 'steer' : 'idle', mode === 'steer' ? lastSteerDir : undefined);
  }

  engineOptBakedEl?.addEventListener('click', (e) => {
    e.stopPropagation();
    requestEngineMode('baked-bank');
  });
  engineOptLiveEl?.addEventListener('click', (e) => {
    e.stopPropagation();
    requestEngineMode('live-onnx');
  });
  engineSliderEl?.addEventListener('click', () => {
    requestEngineMode(engineMode === 'baked-bank' ? 'live-onnx' : 'baked-bank');
  });
  engineSliderEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      requestEngineMode(engineMode === 'baked-bank' ? 'live-onnx' : 'baked-bank');
    }
  });

  // Wire HUD buttons
  toggleEditGymBtn.addEventListener('click', toggleEditGym);
  btnEditRotate.addEventListener('click', rotateSelectedObstacle);
  btnEditShuffle.addEventListener('click', shuffleGymObstacles);
  btnEditReset.addEventListener('click', resetGymObstacles);
  btnEditDone.addEventListener('click', () => setEditGymMode(false));
  toggleGuidesBtn.addEventListener('click', toggleGuides);
  toggleXRayBtn.addEventListener('click', toggleXRay);
  toggleAutoVaultBtn.addEventListener('click', () => {
    autoVaultEnabled = !autoVaultEnabled;
    syncOverlayVisibility();
  });
  toggleTargetBtn.addEventListener('click', () => {
    showTargetGhost = !showTargetGhost;
    syncOverlayVisibility();
  });
  toggleContextBtn.addEventListener('click', () => {
    showContextGhost = !showContextGhost;
    syncOverlayVisibility();
  });
  toggleContactsBtn.addEventListener('click', () => {
    showContacts = !showContacts;
    syncOverlayVisibility();
  });
  toggleTrajBtn.addEventListener('click', () => {
    showTraj = !showTraj;
    syncOverlayVisibility();
  });
  btnPunch.addEventListener('click', () => startPunch());
  btnVault.addEventListener('click', () => startVault(false));
  btnDuck.addEventListener('click', () => startDuck(false));
  btnRoll.addEventListener('click', () => startRoll());
  btnSit.addEventListener('click', toggleSit);
  btnSlowMo.addEventListener('click', toggleSlowMo);
  btnReset.addEventListener('click', () => void resetCharacter());
  btnHelp.addEventListener('click', toggleHelp);
  btnHelpClose.addEventListener('click', () => setHelpModalOpen(false));
  helpModal.addEventListener('click', (e) => {
    if (e.target === helpModal) setHelpModalOpen(false);
  });

  // =========================================================================
  // STRICT PS5 EXCLUSIVE KEYBOARD LOCKOUT (Capture-Phase Guard!)
  // When ANY PS5 controller is connected (WebHID, Gamepad, or Simulated),
  // 100% of keyboard inputs are disabled & ignored until PS5 disconnects.
  // =========================================================================
  window.addEventListener(
    'keydown',
    (e) => {
      if (helpModalOpen) {
        if (e.code === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          setHelpModalOpen(false);
          return;
        }
        if (ps5.isConnected()) {
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
          }
          e.stopPropagation();
          keys.clear();
          syncKeycapLights();
          return;
        }
        if (
          (e.code === 'Enter' || e.code === 'Space') &&
          document.activeElement === btnHelpClose
        ) {
          e.preventDefault();
          e.stopPropagation();
          setHelpModalOpen(false);
          return;
        }
        // Never leak keys to character while Help Modal is open (keep Tab/scroll keys usable)
        const allowModalKey =
          e.metaKey ||
          e.ctrlKey ||
          e.code === 'Tab' ||
          e.code === 'ArrowUp' ||
          e.code === 'ArrowDown' ||
          e.code === 'PageUp' ||
          e.code === 'PageDown';
        if (!allowModalKey) {
          e.preventDefault();
        }
        e.stopPropagation();
        keys.clear();
        syncKeycapLights();
        return;
      }

      if (ps5.isConnected()) {
        if (!e.metaKey && !e.ctrlKey) {
          e.preventDefault();
        }
        e.stopPropagation();
        keys.clear();
        syncKeycapLights();
        return;
      }

      const code = e.code;
      if (code === 'Escape' && editGymMode) {
        e.preventDefault();
        setEditGymMode(false);
        return;
      }
      if (code === 'Escape' && morePopoverOpen) {
        setMorePopoverOpen(false);
        btnMoreStyles.focus();
        return;
      }
      keys.add(code);
      syncKeycapLights();

      if (code === 'KeyG' && !e.repeat) {
        toggleEditGym();
        return;
      }

      if (code === 'Tab') {
        e.preventDefault();
        if (!e.repeat) toggleSlowMo();
        return;
      }
      if (code === 'Space') {
        e.preventDefault();
        if (!e.repeat) startSmartJump();
        return;
      }
      if (code === 'KeyC' && !e.repeat) {
        toggleCrouch();
        return;
      }
      if (code === 'KeyJ' && !e.repeat) {
        startPunch();
        return;
      }
      if (code === 'KeyF' && !e.repeat) {
        startRoll();
        return;
      }
      if (code === 'KeyK' && !e.repeat) {
        startPunch('right');
        return;
      }
      if (code === 'KeyH' && !e.repeat) {
        toggleGuides();
        return;
      }
      if (code === 'KeyX' && !e.repeat) {
        toggleXRay();
        return;
      }
      if (code === 'KeyV' && !e.repeat) {
        startVault(false);
        return;
      }
      if (code === 'KeyB' && !e.repeat) {
        startDuck(false);
        return;
      }
      if (code === 'KeyQ' && !e.repeat) {
        startEmote('look');
        return;
      }
      if (code === 'KeyU' && !e.repeat) {
        startEmote('warmup');
        return;
      }
      if (code === 'KeyE' && !e.repeat) {
        toggleSit();
        return;
      }
      if (code === 'KeyP' && !e.repeat) {
        togglePS5Hud();
        return;
      }
      if (code === 'KeyL' && !e.repeat) {
        toggleTheme();
        return;
      }
      if (code === 'KeyR' && !e.repeat) {
        if (editGymMode) {
          rotateSelectedObstacle();
        } else {
          void resetCharacter();
        }
        return;
      }
      if (code.startsWith('Digit') && !e.repeat) {
        const num = Number(code.slice(5));
        const idx = num === 0 ? 9 : num - 1;
        if (idx >= 0 && idx < styleOrder.length) {
          selectStyle(styleOrder[idx], false);
        }
      }
    },
    { capture: true }
  );

  window.addEventListener(
    'keyup',
    (e) => {
      if (helpModalOpen || ps5.isConnected()) {
        if (!e.metaKey && !e.ctrlKey) {
          e.preventDefault();
        }
        e.stopPropagation();
        keys.clear();
        syncKeycapLights();
        return;
      }
      keys.delete(e.code);
      syncKeycapLights();
    },
    { capture: true }
  );

  window.addEventListener('blur', () => {
    keys.clear();
    syncKeycapLights();
  });

  function applyWallSlideToSteer(raw: THREE.Vector2): THREE.Vector2 | null {
    const hx = dispPos[0];
    const hz = dispPos[2];
    const buf = 0.36;
    let sx = raw.x;
    let sz = raw.y;
    if (hx <= ROOM_BOUNDS.minX + buf && sx < 0) sx = 0;
    if (hx >= ROOM_BOUNDS.maxX - buf && sx > 0) sx = 0;
    if (hz <= ROOM_BOUNDS.minZ + buf && sz < 0) sz = 0;
    if (hz >= ROOM_BOUNDS.maxZ - buf && sz > 0) sz = 0;
    const mag = Math.hypot(sx, sz);
    if (mag < 0.2) return null; // Pushing straight head-on into concrete wall -> stop smoothly!
    return new THREE.Vector2(sx / mag, sz / mag);
  }

  function getSteerInput(gpSteer?: THREE.Vector2 | null): THREE.Vector2 | null {
    if (syntheticSteer) return applyWallSlideToSteer(syntheticSteer);
    // STRICT PS5 EXCLUSIVE: never read WASD/Arrow keys while PS5 is connected!
    if (ps5.isConnected()) {
      return gpSteer ? applyWallSlideToSteer(gpSteer) : null;
    }
    if (gpSteer) return applyWallSlideToSteer(gpSteer);

    let forward = 0;
    let right = 0;
    if (keys.has('KeyW') || keys.has('ArrowUp')) forward += 1;
    if (keys.has('KeyS') || keys.has('ArrowDown')) forward -= 1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) right += 1;
    if (keys.has('KeyA') || keys.has('ArrowLeft')) right -= 1;
    if (forward === 0 && right === 0) return null;

    const fwdX = -Math.sin(camYaw);
    const fwdZ = -Math.cos(camYaw);
    const rgtX = Math.cos(camYaw);
    const rgtZ = -Math.sin(camYaw);
    const dx = fwdX * forward + rgtX * right;
    const dz = fwdZ * forward + rgtZ * right;
    const len = Math.hypot(dx, dz) || 1;
    return applyWallSlideToSteer(new THREE.Vector2(dx / len, dz / len));
  }

  // Pointer orbit + click floor / click obstacle + Edit Gym floor-plane drag
  let dragging = false;
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerMoved = false;
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  function setRayFromPointer(e: PointerEvent): void {
    const rect = canvas.getBoundingClientRect();
    ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
  }

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener('pointerdown', (e) => {
    pointerMoved = false;
    pointerDownX = e.clientX;
    pointerDownY = e.clientY;
    canvas.setPointerCapture(e.pointerId);

    if (editGymMode && e.button === 0) {
      setRayFromPointer(e);
      const obsHits = raycaster.intersectObjects(obstacleManager.getEditTargets(), false);
      if (obsHits.length > 0) {
        const obs = obsHits[0].object.userData.obstacle as ObstacleInfo;
        selectedObsId = obs.id;
        hoveredObsId = obs.id;
        draggingObs = obs;
        if (raycaster.ray.intersectPlane(floorDragPlane, floorHitPoint)) {
          dragOffset.set(obs.pos[0] - floorHitPoint.x, 0, obs.pos[2] - floorHitPoint.z);
        } else {
          dragOffset.set(0, 0, 0);
        }
        updateEditGymUI();
        return;
      }
    }

    dragging = true;
  });

  canvas.addEventListener('pointermove', (e) => {
    if (draggingObs) {
      if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 2) pointerMoved = true;
      setRayFromPointer(e);
      if (raycaster.ray.intersectPlane(floorDragPlane, floorHitPoint)) {
        const [cx, cz] = clampObstaclePos(
          draggingObs,
          floorHitPoint.x + dragOffset.x,
          floorHitPoint.z + dragOffset.z
        );
        draggingObs.pos[0] = cx;
        draggingObs.pos[2] = cz;
        obstacleManager.syncObstacle(draggingObs);
        editGymLabel.textContent = `${shortObstacleLabel(draggingObs)} · (${cx.toFixed(1)}, ${cz.toFixed(1)})`;
      }
      return;
    }

    if (dragging) {
      if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 4) pointerMoved = true;
      camRecenterYaw = null;
      camYaw -= e.movementX * 0.0040;
      camPitch = THREE.MathUtils.clamp(camPitch + e.movementY * 0.0045, -0.14, 1.05);
      return;
    }

    if (editGymMode) {
      setRayFromPointer(e);
      const obsHits = raycaster.intersectObjects(obstacleManager.getEditTargets(), false);
      const nextHover =
        obsHits.length > 0 ? (obsHits[0].object.userData.obstacle as ObstacleInfo).id : null;
      if (nextHover !== hoveredObsId) {
        hoveredObsId = nextHover;
        updateEditGymUI();
      }
    }
  });

  canvas.addEventListener('pointerup', (e) => {
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }

    if (draggingObs) {
      const moved = pointerMoved;
      draggingObs = null;
      if (moved) {
        syncObstaclesToEngines();
      }
      updateEditGymUI();
      return;
    }

    dragging = false;

    if (editGymMode) {
      if (!pointerMoved && e.button === 0) {
        selectedObsId = null;
        updateEditGymUI();
      }
      return;
    }

    if (pointerMoved || e.button !== 0) return;
    // STRICT PS5 EXCLUSIVE: block floor click-to-move & obstacle click while PS5 is connected!
    if (ps5.isConnected()) return;

    setRayFromPointer(e);

    const obsHits = raycaster.intersectObjects(obstacleManager.getClickTargets(), false);
    if (obsHits.length > 0) {
      const obs = obsHits[0].object.userData.obstacle as ObstacleInfo;
      if (obs.type === 'vault') startVault(false, obs);
      else if (obs.type === 'duck') startDuck(false, obs);
      else if (obs.type === 'sit') toggleSit();
      else if (obs.type === 'hurdle' || obs.type === 'block') startGoto(obs.pos[0], obs.pos[2]);
      return;
    }

    const floorHits = raycaster.intersectObject(floor.hitPlane, false);
    if (floorHits.length > 0) {
      const pt = floorHits[0].point;
      startGoto(pt.x, pt.z);
    }
  });

  canvas.addEventListener('pointerleave', () => {
    if (editGymMode && !draggingObs && hoveredObsId !== null) {
      hoveredObsId = null;
      updateEditGymUI();
    }
  });

  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (e.deltaY < 0 && camDist > lastZoomCeiling) {
        camDist = lastZoomCeiling;
      }
      const maxZoom = e.deltaY > 0 ? Math.min(10.5, lastZoomCeiling + 0.35) : 10.5;
      camDist = THREE.MathUtils.clamp(camDist + e.deltaY * 0.0045, 1.85, maxZoom);
    },
    { passive: false }
  );

  // =========================================================================
  // ZERO-SNAP-BACK DUAL-BUFFER BINARY STREAMER (Prefetch + Instant Interrupt)
  // =========================================================================
  function buildGenerateBody(
    reqMode: ControllerMode,
    baseChunkId: number,
    anchorFrame: number,
    dirOverride?: THREE.Vector2
  ): Record<string, unknown> {
    const sprintActive = ps5.isConnected()
      ? ps5SprintActive
      : keys.has('ShiftLeft') || keys.has('ShiftRight') || ps5SprintActive;

    const body: Record<string, unknown> = {
      session_id: sessionId,
      base_chunk_id: baseChunkId,
      anchor_frame: anchorFrame,
      mode: reqMode,
      style: activeStyle,
      sprint: sprintActive,
    };
    if (
      (reqMode === 'steer' ||
        reqMode === 'jump_leap' ||
        reqMode === 'jump_land' ||
        reqMode === 'roll' ||
        reqMode === 'punch_left' ||
        reqMode === 'punch_right') &&
      dirOverride
    ) {
      body.dir_x = dirOverride.x;
      body.dir_z = dirOverride.y;
      if (reqMode === 'steer') lastSteerDir.copy(dirOverride);
    } else if (reqMode === 'goto' && gotoTarget) {
      body.target_x = gotoTarget.x;
      body.target_z = gotoTarget.z;
    } else if (reqMode.startsWith('vault')) {
      body.obstacle_id = activeVaultObsId;
      body.vault_dir = vaultDir;
      body.vault_z = vaultZ;
      body.vault_lane = activeVaultLane;
    } else if (reqMode.startsWith('duck')) {
      body.obstacle_id = activeDuckObsId;
      body.duck_dir = duckDir;
      body.duck_x = duckX;
      body.duck_lane = duckLane;
    }
    return body;
  }

  function commitChunkWithInertialization(
    chunk: BinaryChunk,
    startFrame: number,
    blendDurationSec: number
  ): void {
    const prevChunkMode = currentChunk?.mode;
    // Capture smooth 120 ms Hermite inertialization residual from currently displayed pose!
    if (currentChunk !== null) {
      sampleChunkCatmullRom(chunk, startFrame, nextStartPos, nextStartBoneQuat, dummyContacts);
      inertializer.captureTransition(
        dispPos,
        dispBoneQuat,
        nextStartPos,
        nextStartBoneQuat,
        blendDurationSec
      );
    }
    currentChunk = chunk;
    if (chunk.mode === 'roll' || prevChunkMode === 'roll') {
      footLocker.reset();
    }
    playFrame = startFrame;
    lastLatencyMs = chunk.latency_ms;
    lastInferMs = chunk.infer_ms;
    generationCount += 1;

    if (chunk.mode === 'vault_action' && mode === 'vault_entry') {
      mode = 'vault_action';
      updateModeBanner();
    }
    if (chunk.mode === 'duck_action' && mode === 'duck_entry') {
      mode = 'duck_action';
      updateModeBanner();
    }
    if (chunk.mode === 'goto' && chunk.phase_done) {
      gotoTarget = null;
    }

    statMs.textContent =
      engineMode === 'baked-bank' ? 'Pre-Baked' : `${Math.round(chunk.infer_ms)} ms`;
    updateOverlayGeometryFromChunk(chunk);
    syncOverlayVisibility();
  }

  function triggerInterrupt(reqMode: ControllerMode, dirOverride?: THREE.Vector2): void {
    // Invalidate any prefetched chunk since user changed intent
    nextChunk = null;
    if (inFlight) {
      pendingInterrupt = { mode: reqMode, steer: dirOverride?.clone() };
      return;
    }
    const baseId = currentChunk ? currentChunk.chunk_id : 0;
    const lookahead = currentChunk
      ? Math.min(playFrame + (slowMo ? 0.25 : 1.05), currentChunk.num_frames - 1)
      : 0;
    void fetchBinaryChunk(reqMode, baseId, lookahead, false, dirOverride);
  }

  function triggerLookaheadPrefetch(
    reqMode: ControllerMode,
    anchorFrame: number,
    dirOverride?: THREE.Vector2
  ): void {
    if (inFlight || nextChunk || !currentChunk) return;
    void fetchBinaryChunk(reqMode, currentChunk.chunk_id, anchorFrame, true, dirOverride);
  }

  async function fetchBinaryChunk(
    reqMode: ControllerMode,
    baseChunkId: number,
    anchorFrame: number,
    isPrefetch: boolean,
    dirOverride?: THREE.Vector2
  ): Promise<void> {
    inFlight = true;
    const seq = ++requestSeq;
    statusDot.classList.add('busy');

    try {
      const reqBody = buildGenerateBody(reqMode, baseChunkId, anchorFrame, dirOverride);
      let chunk: BinaryChunk;
      if (engineMode === 'baked-bank' && motionBank) {
        await Promise.resolve();
        chunk = motionBank.generateChunk(reqBody);
      } else {
        const res = await fetch(`${API_BASE}/api/generate_bin`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reqBody),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        chunk = decodeBinaryChunk(buf);
      }

      if (seq !== requestSeq) return;

      if (isPrefetch) {
        if (currentChunk && chunk.base_chunk_id === currentChunk.chunk_id) {
          nextChunk = chunk;
        }
      } else {
        const overshoot = currentChunk ? Math.max(0, playFrame - chunk.anchor_frame) : 0;
        const startF = Math.min(overshoot, Math.max(0, chunk.num_frames - 2));
        commitChunkWithInertialization(chunk, startF, 0.115);
      }
    } catch (err) {
      console.error('[MotionBricks] Binary generate error:', err);
    } finally {
      inFlight = false;
      statusDot.classList.remove('busy');
      if (pendingInterrupt) {
        const next = pendingInterrupt;
        pendingInterrupt = null;
        triggerInterrupt(next.mode, next.steer);
      }
    }
  }

  function nextPhaseFor(
    doneChunk: BinaryChunk,
    nowMs: number,
    steer: THREE.Vector2 | null
  ): ControllerMode {
    switch (doneChunk.mode) {
      case 'vault_entry':
        return doneChunk.phase_done ? 'vault_action' : 'vault_entry';
      case 'vault_action':
        return 'vault_exit';
      case 'vault_exit':
        if (activeVaultObsId) obstacleCooldowns.set(activeVaultObsId, nowMs + 1400);
        return steer ? 'steer' : 'idle';
      case 'duck_entry':
        return doneChunk.phase_done ? 'duck_action' : 'duck_entry';
      case 'duck_action':
        return 'duck_exit';
      case 'duck_exit':
        if (activeDuckObsId) obstacleCooldowns.set(activeDuckObsId, nowMs + 1400);
        return steer ? 'steer' : 'idle';
      case 'emote_look':
        return steer ? 'steer' : 'emote_look_2';
      case 'emote_look_2':
        return steer ? 'steer' : 'idle';
      case 'emote_warmup':
        return steer ? 'steer' : 'emote_warmup_2';
      case 'emote_warmup_2':
        return steer ? 'steer' : 'idle';
      case 'jump_leap':
        return 'jump_land';
      case 'jump_land':
        return steer ? 'steer' : 'idle';
      case 'roll':
        return steer ? 'steer' : activeStyle === 'combat' ? 'combat_recover' : 'idle';
      case 'punch_left':
      case 'punch_right':
        return steer ? 'steer' : 'combat_recover';
      case 'combat_recover':
        return steer ? 'steer' : 'combat_recover';
      case 'sit_approach':
        return doneChunk.phase_done ? 'sit_down' : 'sit_approach';
      case 'sit_down':
        return 'sit_hold';
      case 'sit_hold':
        return steer ? 'stand_up' : 'sit_hold';
      case 'stand_up':
        return steer ? 'steer' : 'idle';
      case 'goto':
        if (steer) return 'steer';
        return doneChunk.phase_done || !gotoTarget ? 'idle' : 'goto';
      default:
        return steer ? 'steer' : activeStyle === 'combat' ? 'combat_recover' : 'idle';
    }
  }

  // Kick off initial breathing idle chunk
  syncOverlayVisibility();
  triggerInterrupt('idle');

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  window.__MOTIONBRICKS__ = {
    triggerStyle: (id: string) => selectStyle(id, true),
    triggerPunch: startPunch,
    triggerVault: () => startVault(false),
    triggerDuck: () => startDuck(false),
    triggerRoll: () => startRoll(),
    triggerSit: toggleSit,
    triggerEmote: startEmote,
    triggerGoto: startGoto,
    toggleGuides,
    toggleXRay,
    toggleSlowMo,
    togglePS5Hud,
    toggleTheme,
    toggleHelp,
    toggleEditGym,
    setEditGymMode,
    moveObstacle: moveObstacleById,
    rotateObstacle: rotateObstacleById,
    shuffleObstacles: shuffleGymObstacles,
    resetObstacles: resetGymObstacles,
    getObstacles: () => cloneObstacles(initData.obstacles),
    setCameraBookmark,
    setCameraOrbit: (yaw?: number, pitch?: number, dist?: number) => {
      camRecenterYaw = null;
      if (typeof yaw === 'number') camYaw = yaw;
      if (typeof pitch === 'number') camPitch = THREE.MathUtils.clamp(pitch, -0.14, 1.05);
      if (typeof dist === 'number') camDist = THREE.MathUtils.clamp(dist, 2.2, 11.5);
    },
    simulatePS5: (override) => {
      ps5.setSimulation(override);
    },
    setSteer: (dx: number, dz: number) => {
      if (dx === 0 && dz === 0) syntheticSteer = null;
      else syntheticSteer = new THREE.Vector2(dx, dz).normalize();
    },
    getState: () => ({
      engineMode,
      liveAvailable,
      mode,
      activeStyle,
      slowMo,
      xrayMode,
      lightTheme,
      helpModalOpen,
      editGymMode,
      selectedObsId,
      ps5HudVisible,
      ps5Connected: ps5.isConnected(),
      keyboardLocked: ps5.isConnected(),
      activeCameraBookmark,
      wideCamMode,
      camYaw: Number(camYaw.toFixed(3)),
      camPitch: Number(camPitch.toFixed(3)),
      camDist: Number(camDist.toFixed(3)),
      hasSomaSkin: Boolean(somaSkin),
      somaMeshCount: somaSkin?.calibration.meshCount ?? 0,
      showAllGuides,
      autoVaultEnabled,
      generationCount,
      lastLatencyMs,
      lastInferMs,
      playFrame: Number(playFrame.toFixed(2)),
      numFrames: currentChunk?.num_frames ?? 0,
      hasPrefetch: Boolean(nextChunk),
      hips: [
        Number(dispPos[0].toFixed(3)),
        Number(dispPos[1].toFixed(3)),
        Number(dispPos[2].toFixed(3)),
      ],
      leftHand: [
        Number(dispPos[J.LeftHand * 3].toFixed(3)),
        Number(dispPos[J.LeftHand * 3 + 1].toFixed(3)),
        Number(dispPos[J.LeftHand * 3 + 2].toFixed(3)),
      ],
      rightHand: [
        Number(dispPos[J.RightHand * 3].toFixed(3)),
        Number(dispPos[J.RightHand * 3 + 1].toFixed(3)),
        Number(dispPos[J.RightHand * 3 + 2].toFixed(3)),
      ],
    }),
  };

  /**
   * 2D ray-AABB slab intersection in obstacle local coordinates (returns closest hit distance in [rayMin, rayMax], or null).
   */
  function intersectRayOBB2D(
    pX: number,
    pZ: number,
    dX: number,
    dZ: number,
    halfW: number,
    halfD: number,
    rayMin: number,
    rayMax: number
  ): number | null {
    let tEnter = rayMin;
    let tExit = rayMax;

    if (Math.abs(dX) < 1e-5) {
      if (pX < -halfW || pX > halfW) return null;
    } else {
      const invX = 1 / dX;
      let t1 = (-halfW - pX) * invX;
      let t2 = (halfW - pX) * invX;
      if (t1 > t2) {
        const tmp = t1;
        t1 = t2;
        t2 = tmp;
      }
      tEnter = Math.max(tEnter, t1);
      tExit = Math.min(tExit, t2);
      if (tEnter > tExit) return null;
    }

    if (Math.abs(dZ) < 1e-5) {
      if (pZ < -halfD || pZ > halfD) return null;
    } else {
      const invZ = 1 / dZ;
      let t1 = (-halfD - pZ) * invZ;
      let t2 = (halfD - pZ) * invZ;
      if (t1 > t2) {
        const tmp = t1;
        t1 = t2;
        t2 = tmp;
      }
      tEnter = Math.max(tEnter, t1);
      tExit = Math.min(tExit, t2);
      if (tEnter > tExit) return null;
    }

    return tEnter;
  }

  let prevTime = performance.now();
  let fpsAccum = 0;
  let fpsFrames = 0;

  function tick(now: number): void {
    requestAnimationFrame(tick);
    const rawDt = Math.min((now - prevTime) / 1000, 0.05);
    prevTime = now;

    fpsAccum += rawDt;
    fpsFrames += 1;
    if (fpsAccum >= 0.4) {
      statFps.textContent = `${Math.round(fpsFrames / fpsAccum)} fps`;
      fpsAccum = 0;
      fpsFrames = 0;
    }

    // Poll PS5 DualSense WebHID (with automatic standard Gamepad API fallback)
    const ps5Input = ps5.poll(rawDt);
    ps5Hud.update(ps5Input.snapshot, rawDt, now);
    ps5SprintActive = ps5Input.sprint;
    if (ps5SlowMoHold !== ps5Input.slowMoHold) {
      ps5SlowMoHold = ps5Input.slowMoHold;
      updateModeBanner();
    }
    let padSteer: THREE.Vector2 | null = null;
    if (ps5Input.connected) {
      if (ps5Input.camYawDelta !== 0 || ps5Input.camPitchDelta !== 0) {
        if (ps5Input.camYawDelta !== 0) camRecenterYaw = null;
        camYaw += ps5Input.camYawDelta;
        camPitch = THREE.MathUtils.clamp(camPitch + ps5Input.camPitchDelta, -0.14, 1.05);
      }
      if (ps5Input.steerX !== 0 || ps5Input.steerY !== 0) {
        const fwdX = -Math.sin(camYaw);
        const fwdZ = -Math.cos(camYaw);
        const rgtX = Math.cos(camYaw);
        const rgtZ = -Math.sin(camYaw);
        const dx = fwdX * ps5Input.steerY + rgtX * ps5Input.steerX;
        const dz = fwdZ * ps5Input.steerY + rgtZ * ps5Input.steerX;
        const len = Math.hypot(dx, dz) || 1;
        padSteer = new THREE.Vector2(dx / len, dz / len);
      }
    }

    const steer = getSteerInput(padSteer);
    const effectiveSlowMo = slowMo || ps5SlowMoHold;
    const dt = rawDt * (effectiveSlowMo ? 0.25 : 1.0);

    // 1. Universal Geometric 3D Spatial Obstacle Detector (Zero-Hardcoded OBB Forward-Ray Sensor)
    const canChainExit =
      Boolean(steer) &&
      (mode === 'jump_land' || mode === 'vault_exit' || mode === 'duck_exit') &&
      currentChunk !== null &&
      playFrame >= Math.max(8, currentChunk.num_frames - currentChunk.transition_early - 6);

    if (autoVaultEnabled && (mode === 'steer' || mode === 'goto' || canChainExit)) {
      const hx = dispPos[0];
      const hz = dispPos[2];
      let moveX = 0;
      let moveZ = 0;
      let distToGoal = Infinity;
      if (steer) {
        moveX = steer.x;
        moveZ = steer.y;
      } else if (gotoTarget) {
        const gx = gotoTarget.x - hx;
        const gz = gotoTarget.z - hz;
        distToGoal = Math.hypot(gx, gz);
        if (distToGoal > 1e-4) {
          moveX = gx / distToGoal;
          moveZ = gz / distToGoal;
        }
      }

      if (moveX !== 0 || moveZ !== 0) {
        const hits: Array<{ obs: ObstacleInfo; dist: number }> = [];

        for (const obs of initData.obstacles) {
          if ((obstacleCooldowns.get(obs.id) ?? 0) > now) continue;

          const relX = hx - obs.pos[0];
          const relZ = hz - obs.pos[2];

          // Local +X follows obs.facing (matching studioProps.ts rotation.y), Local +Z is orthogonal
          let axX = 1;
          let axZ = 0;
          if (obs.facing) {
            const fLen = Math.hypot(obs.facing[0], obs.facing[1]) || 1;
            axX = obs.facing[0] / fLen;
            axZ = obs.facing[1] / fLen;
          }

          const pX = relX * axX + relZ * axZ;
          const pZ = -relX * axZ + relZ * axX;
          const dX = moveX * axX + moveZ * axZ;
          const dZ = -moveX * axZ + moveZ * axX;

          // Only fire when moving toward the obstacle center, never when stepping away
          if (pX * dX + pZ * dZ >= -0.05) continue;

          const halfW = obs.size[0] * 0.5 + 0.28;
          const halfD = obs.size[2] * 0.5 + 0.28;
          const hitDist = intersectRayOBB2D(pX, pZ, dX, dZ, halfW, halfD, 0.2, 2.25);
          if (hitDist !== null && hitDist < distToGoal + 0.15) {
            hits.push({ obs, dist: hitDist });
          }
        }

        if (hits.length > 0) {
          hits.sort((a, b) => a.dist - b.dist);
          const nearest = hits[0];
          const obs = nearest.obs;
          const isOverheadBeam =
            obs.clearance !== undefined && obs.clearance >= 0.9 && obs.clearance <= 1.65;
          const isLowCrawlGap =
            obs.clearance !== undefined && obs.clearance >= 0.45 && obs.clearance < 0.9;
          const obsH = obs.size[1];

          if (isLowCrawlGap) {
            obstacleCooldowns.set(obs.id, now + 2400);
            if (activeStyle !== 'crawl') {
              selectStyle('crawl', false);
            }
          } else if (isOverheadBeam) {
            obstacleCooldowns.set(obs.id, now + 2200);
            startDuck(true, obs);
          } else if (obsH >= 0.6) {
            obstacleCooldowns.set(obs.id, now + 2200);
            startVault(true, obs);
          } else if (obsH >= 0.25 && obsH < 0.6 && nearest.dist <= 0.92) {
            obstacleCooldowns.set(obs.id, now + 1800);
            actionDir.set(moveX, moveZ).normalize();
            gotoTarget = null;
            mode = 'jump_leap';
            ps5.triggerVaultHaptic();
            syncOverlayVisibility();
            updateModeBanner();
            triggerInterrupt('jump_leap', actionDir);
          }
        }
      }
    }

    // 2. Live snappy WASD / PS5 steering override
    if (steer) {
      if (mode === 'sit_hold' || mode === 'sit_down') {
        mode = 'stand_up';
        updateModeBanner();
        triggerInterrupt('stand_up');
      } else if (
        !mode.startsWith('vault') &&
        !mode.startsWith('duck') &&
        !mode.startsWith('jump') &&
        mode !== 'roll' &&
        mode !== 'stand_up' &&
        mode !== 'punch_left' &&
        mode !== 'punch_right'
      ) {
        gotoTarget = null;
        syncOverlayVisibility();
        const turnAngle =
          Math.acos(THREE.MathUtils.clamp(steer.dot(lastSteerDir), -1, 1)) * (180 / Math.PI);
        if (mode !== 'steer') {
          mode = 'steer';
          updateModeBanner();
          triggerInterrupt('steer', steer);
        } else if (turnAngle > 14 && playFrame >= 3 && !inFlight) {
          triggerInterrupt('steer', steer);
        }
      }
    } else if (mode === 'steer' && currentChunk && playFrame >= 3 && !inFlight) {
      mode = activeStyle === 'combat' ? 'combat_recover' : 'idle';
      updateModeBanner();
      triggerInterrupt(mode);
    }

    // 3. Advance playback with ZERO-SNAP-BACK lookahead splice + Catmull-Rom + Inertialization + Foot-Lock!
    if (currentChunk && currentChunk.num_frames > 0) {
      // Ballistic boxing snap cadence: explosive 255 ms strike launch -> crisp readable impact beat
      let modeSpeedScale = 1.0;
      if (mode === 'punch_left' || mode === 'punch_right') {
        modeSpeedScale = playFrame < 13.5 ? 1.75 : 0.65;
      } else if (mode === 'combat_recover') {
        modeSpeedScale = 1.4;
      }
      playFrame += dt * currentChunk.fps * modeSpeedScale;

      if (nextChunk && playFrame >= nextChunk.anchor_frame) {
        const overflow = playFrame - nextChunk.anchor_frame;
        const ready = nextChunk;
        nextChunk = null;
        const prevMode = mode;
        mode = ready.mode as ControllerMode;
        if (prevMode === 'roll' || mode === 'roll') {
          footLocker.reset();
        }
        updateModeBanner();
        commitChunkWithInertialization(ready, Math.min(overflow, ready.num_frames - 1), 0.05);
      }

      const maxIdx = currentChunk.num_frames - 1;
      const clampedFrame = Math.min(playFrame, maxIdx);

      sampleChunkCatmullRom(currentChunk, clampedFrame, rawPos, rawBoneQuat, rawContacts);
      inertializer.apply(dt, rawPos, rawBoneQuat, dispPos, dispBoneQuat);
      if (mode === 'roll' || currentChunk.mode === 'roll') {
        footLocker.reset();
      } else {
        footLocker.apply(dispPos, rawContacts, dt);
      }

      ps5.syncNeuralFootContacts(
        rawContacts,
        mode !== 'idle' && mode !== 'sit_hold' && mode !== 'combat_recover',
        now
      );

      applyDisplayedPose(rawDt);

      const anchorTarget = Math.max(
        2,
        Math.min(currentChunk.num_frames - 2, currentChunk.num_frames - currentChunk.transition_early)
      );
      const prefetchLead = Math.max(1, anchorTarget - 6);
      if (playFrame >= prefetchLead && !nextChunk && !inFlight) {
        const upcomingMode = nextPhaseFor(currentChunk, now, steer);
        const dirForNext = upcomingMode === 'jump_land' ? actionDir : steer ?? undefined;
        triggerLookaheadPrefetch(upcomingMode, anchorTarget, dirForNext);
      }

      if (playFrame >= maxIdx - 0.1 && !nextChunk && !inFlight) {
        const upcomingMode = nextPhaseFor(currentChunk, now, steer);
        if (mode === 'roll' || upcomingMode === 'roll') {
          footLocker.reset();
        }
        mode = upcomingMode;
        updateModeBanner();
        triggerInterrupt(
          upcomingMode,
          upcomingMode === 'jump_land' ? actionDir : steer ?? undefined
        );
      }
    }

    // 4. Animate click-to-move beacon pulse & pavilion subtle skylight atmosphere
    if (beaconGroup.visible) {
      beaconRing.scale.setScalar(1.0 + Math.sin(now * 0.008) * 0.14);
    }
    pavilion.update(now * 0.001);

    // 5. Smooth third-person damped camera & sculptural studio lighting
    const hx = dispPos[0];
    const hz = dispPos[2];
    applyCameraRig(false, rawDt);

    updateSculpturalLights(hx, hz);
    floor.updateCenter(hx, hz);

    syncKeycapLights();
    renderer.render(scene, camera);
  }

  requestAnimationFrame(tick);
}

void boot();
