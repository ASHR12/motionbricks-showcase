/**
 * Single Shared Spec for SOMA Motion Lab (Forge Protocol Blueprint & Design Tokens).
 * Every 3D environment generator, prop builder, camera rig, theme switch, and UI overlay
 * reads dimensions, datums, palettes, and camera bookmarks from this file.
 *
 * Units: 1 unit = 1 meter, +Y is up.
 */

export interface CameraBookmark {
  id: string;
  label: string;
  yaw: number;
  pitch: number;
  dist: number;
  targetOffset: [number, number, number]; // camera-relative framing offset from hips
  fov: number;
}

export const PAVILION_SPEC = {
  /** Spacious 15.4 m x 25.3 m architectural brutalist hall enclosed cleanly on all 4 sides */
  floorSize: 64,
  backWallZ: -17.5,
  frontWallZ: 7.8,
  leftWallX: -7.8,
  rightColonnadeX: 7.6,
  ceilingHeight: 4.55,
  columnSpacingZ: 2.2,
  columnWidth: 0.54,
  columnDepth: 0.66,
  /** Inlaid deep matte charcoal rubber runway running through main gantry & tight rear stress-test gauntlet */
  trainingLane: {
    x: 0.0,
    z: -6.4,
    width: 2.25,
    length: 14.8,
  },
  /** Three recessed warm tungsten-amber overhead skylight bays framed cleanly behind front concrete soffit */
  skylightBays: [
    { x: -1.65, z: -10.2, width: 1.08, length: 12.2 },
    { x: 0.0, z: -10.2, width: 1.08, length: 12.2 },
    { x: 1.65, z: -10.2, width: 1.08, length: 12.2 },
  ],
} as const;

/** Solid playable inner room bounds (character & jumps NEVER cross any concrete wall or colonnade pillar) */
export const ROOM_BOUNDS = {
  minX: PAVILION_SPEC.leftWallX + 0.65, // -7.15 m
  maxX: PAVILION_SPEC.rightColonnadeX - 0.9, // +6.70 m (respects inner colonnade pillar face)
  minZ: PAVILION_SPEC.backWallZ + 0.65, // -16.85 m
  maxZ: PAVILION_SPEC.frontWallZ - 0.68, // +7.12 m
} as const;

/** Primary prop datums synced with server/api.py GYM_OBSTACLES */
export const PROP_DATUMS = {
  vault: { pos: [-2.35, 0, -2.5] as const, size: [0.34, 0.82, 2.2] as const },
  sit: {
    pos: [1.85, 0, -2.9] as const,
    size: [0.52, 0.44, 0.72] as const,
    facing: [-0.8192, 0.5736] as const,
  },
  duck: { pos: [0.0, 0, -5.6] as const, size: [2.2, 0.42, 0.36] as const, clearance: 1.35 },
} as const;

/** Hero & Verification Camera Bookmarks (axial symmetrical nave matching targets/t01.png) */
export const STUDIO_CAMERAS: Record<string, CameraBookmark> = {
  'cam-01': {
    id: 'cam-01',
    label: 'Axial Hero Nave (Target t01)',
    yaw: 0.0,
    pitch: -0.02,
    dist: 3.45,
    targetOffset: [-1.12, 0.14, 0.0], // camera sits at x=0 looking straight down nave; athlete stands in right third (~60% H)
    fov: 48,
  },
  'cam-02': {
    id: 'cam-02',
    label: 'Close-Up Character & Outfit',
    yaw: -0.26,
    pitch: 0.05,
    dist: 2.52,
    targetOffset: [0.0, 0.14, 0.0],
    fov: 38,
  },
  'cam-03': {
    id: 'cam-03',
    label: 'Parkour Vault & Bench Wide',
    yaw: -0.45,
    pitch: 0.18,
    dist: 4.5,
    targetOffset: [0.0, 0.05, 0.0],
    fov: 44,
  },
  'cam-04': {
    id: 'cam-04',
    label: 'Overhead Gantry Duck View',
    yaw: 0.42,
    pitch: 0.16,
    dist: 3.9,
    targetOffset: [0.0, 0.05, 0.0],
    fov: 44,
  },
};

export interface StudioThemePalette {
  bg: string;
  fog: string;
  fogDensity: number;
  floorBase: string;
  floorMinor: string;
  floorMajor: string;
  floorAxis: string;
  concreteWall: string;
  concreteColumn: string;
  rubberLane: string;
  skylightEmissive: string;
  skylightIntensity: number;
  beamOpacity: number;
  hemiSky: number;
  hemiGround: number;
  hemiIntensity: number;
  keyColor: number;
  keyIntensity: number;
  rimColor: number;
  rimIntensity: number;
  envIntensity: number;
  shadowOpacity: number;
}

export const STUDIO_THEMES: Record<'dark' | 'light', StudioThemePalette> = {
  dark: {
    bg: '#0c0e13',
    fog: '#0c0e13',
    fogDensity: 0.02,
    floorBase: '#151820',
    floorMinor: '#1d222c',
    floorMajor: '#252b38',
    floorAxis: '#323a4a',
    concreteWall: '#222429',
    concreteColumn: '#272930',
    rubberLane: '#0d0f13',
    skylightEmissive: '#ffd89b',
    skylightIntensity: 1.12,
    beamOpacity: 0.052,
    hemiSky: 0xffe6c2,
    hemiGround: 0x0c0e14,
    hemiIntensity: 0.36,
    keyColor: 0xffe8c8,
    keyIntensity: 2.35,
    rimColor: 0x8cc8ff,
    rimIntensity: 1.05,
    envIntensity: 0.36,
    shadowOpacity: 0.56,
  },
  light: {
    bg: '#d9d4cb',
    fog: '#d9d4cb',
    fogDensity: 0.014,
    floorBase: '#cbc5ba',
    floorMinor: '#bab3a7',
    floorMajor: '#9d9588',
    floorAxis: '#7f786c',
    concreteWall: '#c8c2b7',
    concreteColumn: '#bdb6aa',
    rubberLane: '#20232a',
    skylightEmissive: '#ffefd6',
    skylightIntensity: 0.92,
    beamOpacity: 0.032,
    hemiSky: 0xfff8ec,
    hemiGround: 0x9b9488,
    hemiIntensity: 0.72,
    keyColor: 0xfff2dc,
    keyIntensity: 2.15,
    rimColor: 0x93c5fd,
    rimIntensity: 0.55,
    envIntensity: 0.38,
    shadowOpacity: 0.36,
  },
};
