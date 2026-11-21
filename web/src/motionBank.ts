/**
 * Browser-Static Pre-Baked Neural Motion Bank Engine for MotionBrain (NVIDIA MotionBricks SOMA).
 *
 * Loads the compact Int16-quantized `motion-bank.bin` (~0.38 MB raw / ~0.32 MB gzipped)
 * exported directly from the unquantized local SOMA FP16 ONNX server and produces
 * drop-in `BinaryChunk` packets identical to POST `/api/generate_bin` — enabling instant
 * 60 fps zero-server browser gameplay with full Catmull-Rom smoothing, 120 ms Hermite
 * inertialization, neural foot-locking, 3D OBB Auto-Parkour, PS5 DualSense WebHID,
 * AI Guides, and pure line-art X-Ray.
 */

import type { BinaryChunk, BinaryChunkMeta } from './motionBuffer';
import { PAVILION_SPEC } from './studioSpec';

interface BankClipEntry {
  byteOffset: number;
  byteLength: number;
  num_frames: number;
  fps: number;
  transition_early: number;
  phase_done: boolean;
  mode: string;
  style: string;
  disp_xz: [number, number];
  heading_end: [number, number];
  start_root_xz: [number, number];
  extra?: Record<string, number>;
}

interface BankHeader {
  version: number;
  pos_scale: number;
  quat_scale: number;
  num_render_joints: number;
  num_bone_joints: number;
  init_data: Record<string, unknown>;
  clips: Record<string, BankClipEntry>;
}

interface DecodedClip {
  entry: BankClipEntry;
  framesPos: Float32Array;
  framesBoneQuat: Float32Array;
  footContacts: Float32Array;
  contextPos: Float32Array;
  targetPos: Float32Array;
}

interface StoredHistory {
  chunkId: number;
  numFrames: number;
  framesPos: Float32Array;
  headingX: number;
  headingZ: number;
}

const textDecoder = new TextDecoder('utf-8');

const MIN_X = PAVILION_SPEC.leftWallX + 0.55;
const MAX_X = PAVILION_SPEC.rightColonnadeX - 0.55;
const MIN_Z = PAVILION_SPEC.backWallZ + 0.55;
const MAX_Z = PAVILION_SPEC.frontWallZ - 0.55;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function norm2(x: number, z: number, fallbackX = 0, fallbackZ = -1): [number, number] {
  const len = Math.hypot(x, z);
  if (len < 1e-4) return [fallbackX, fallbackZ];
  return [x / len, z / len];
}

export class MotionBankEngine {
  public readonly initData: Record<string, unknown>;
  public readonly byteSize: number;
  private readonly numRenderJoints: number;
  private readonly numBoneJoints: number;
  private readonly clips = new Map<string, DecodedClip>();
  private readonly history = new Map<number, StoredHistory>();
  private nextChunkId = 1;
  private curRootX = 0;
  private curRootZ = 0;
  private curHeadingX = 0;
  private curHeadingZ = -1;

  private constructor(header: BankHeader, buf: ArrayBuffer, payloadStart: number) {
    this.initData = header.init_data;
    this.byteSize = buf.byteLength;
    this.numRenderJoints = header.num_render_joints;
    this.numBoneJoints = header.num_bone_joints;

    const invPos = 1 / header.pos_scale;
    const invQuat = 1 / header.quat_scale;
    const view = new DataView(buf);

    for (const [key, entry] of Object.entries(header.clips)) {
      const nf = entry.num_frames;
      const lenPos = nf * this.numRenderJoints * 3;
      const lenQuat = nf * this.numBoneJoints * 4;
      const lenContacts = nf * 4;
      const lenCtx = 4 * this.numRenderJoints * 3;
      const lenTgt = 4 * this.numRenderJoints * 3;

      let off = payloadStart + entry.byteOffset;

      const framesPos = new Float32Array(lenPos);
      for (let i = 0; i < lenPos; i++, off += 2) {
        framesPos[i] = view.getInt16(off, true) * invPos;
      }

      const framesBoneQuat = new Float32Array(lenQuat);
      for (let i = 0; i < lenQuat; i++, off += 2) {
        framesBoneQuat[i] = view.getInt16(off, true) * invQuat;
      }

      const footContacts = new Float32Array(lenContacts);
      for (let i = 0; i < lenContacts; i++, off += 1) {
        footContacts[i] = view.getUint8(off) / 255;
      }

      const contextPos = new Float32Array(lenCtx);
      for (let i = 0; i < lenCtx; i++, off += 2) {
        contextPos[i] = view.getInt16(off, true) * invPos;
      }

      const targetPos = new Float32Array(lenTgt);
      for (let i = 0; i < lenTgt; i++, off += 2) {
        targetPos[i] = view.getInt16(off, true) * invPos;
      }

      this.clips.set(key, {
        entry,
        framesPos,
        framesBoneQuat,
        footContacts,
        contextPos,
        targetPos,
      });
    }
  }

  public static async load(url = '/models/motion-bank.bin'): Promise<MotionBankEngine> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch motion bank (${res.status}): ${url}`);
    }
    const buf = await res.arrayBuffer();
    if (buf.byteLength < 12) {
      throw new Error('Motion bank buffer too short');
    }
    const magic = textDecoder.decode(new Uint8Array(buf, 0, 4));
    if (magic !== 'MBNK') {
      throw new Error(`Invalid motion bank magic: ${magic}`);
    }
    const view = new DataView(buf);
    const headerLen = view.getUint32(4, true);
    const headerJson = textDecoder.decode(new Uint8Array(buf, 8, headerLen)).trim();
    const header = JSON.parse(headerJson) as BankHeader;
    return new MotionBankEngine(header, buf, 8 + headerLen);
  }

  public reset(): { pose: Record<string, unknown> } {
    this.history.clear();
    this.curRootX = 0;
    this.curRootZ = 0;
    this.curHeadingX = 0;
    this.curHeadingZ = -1;
    const initPose = (this.initData.initial_pose ?? {}) as Record<string, unknown>;
    return { pose: initPose };
  }

  /** Sync root world anchor when toggling live from ONNX -> Pre-Baked mid-session. */
  public syncAnchor(x: number, z: number, hx = 0, hz = -1): void {
    this.history.clear();
    this.curRootX = clamp(x, MIN_X, MAX_X);
    this.curRootZ = clamp(z, MIN_Z, MAX_Z);
    const [nhx, nhz] = norm2(hx, hz, 0, -1);
    this.curHeadingX = nhx;
    this.curHeadingZ = nhz;
  }

  private findObstacle(id: unknown, fallbackType: string): {
    id: string;
    type: string;
    pos: [number, number, number];
    size: [number, number, number];
    facing?: [number, number];
  } | null {
    const list = (this.initData.obstacles ?? []) as Array<{
      id: string;
      type: string;
      pos: [number, number, number];
      size: [number, number, number];
      facing?: [number, number];
    }>;
    if (typeof id === 'string') {
      const exact = list.find((o) => o.id === id);
      if (exact) return exact;
    }
    return list.find((o) => o.type === fallbackType) ?? null;
  }

  private getClip(key: string, fallback = 'idle:walk'): DecodedClip {
    return this.clips.get(key) ?? this.clips.get(fallback) ?? this.clips.values().next().value!;
  }

  private resolveAnchor(baseChunkId: number, anchorFrame: number): {
    x: number;
    z: number;
    hx: number;
    hz: number;
    hist: StoredHistory | null;
    anchorIdx: number;
  } {
    const hist = this.history.get(baseChunkId) ?? null;
    if (!hist) {
      return {
        x: this.curRootX,
        z: this.curRootZ,
        hx: this.curHeadingX,
        hz: this.curHeadingZ,
        hist: null,
        anchorIdx: 0,
      };
    }
    const nf = hist.numFrames;
    const clamped = clamp(anchorFrame, 0, Math.max(0, nf - 1));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(nf - 1, i0 + 1);
    const frac = clamped - i0;
    const stride = this.numRenderJoints * 3;
    const x = hist.framesPos[i0 * stride] * (1 - frac) + hist.framesPos[i1 * stride] * frac;
    const z = hist.framesPos[i0 * stride + 2] * (1 - frac) + hist.framesPos[i1 * stride + 2] * frac;
    return {
      x: clamp(x, MIN_X, MAX_X),
      z: clamp(z, MIN_Z, MAX_Z),
      hx: hist.headingX,
      hz: hist.headingZ,
      hist,
      anchorIdx: i0,
    };
  }

  public generateChunk(req: Record<string, unknown>): BinaryChunk {
    const t0 = performance.now();
    const baseChunkId = Number(req.base_chunk_id ?? 0);
    const anchorFrame = Number(req.anchor_frame ?? 0);
    const mode = String(req.mode ?? 'idle');
    const style = String(req.style ?? 'walk');
    const sprint = Boolean(req.sprint ?? false);

    const anchor = this.resolveAnchor(baseChunkId, anchorFrame);
    let placeX = anchor.x;
    let placeZ = anchor.z;
    let headX = anchor.hx;
    let headZ = anchor.hz;
    let xzScale = 1.0;
    let phaseDone = true;
    let clip: DecodedClip;
    let outMode = mode;

    // Effective locomotion style (Shift / R1 promotes walk/jog to run sprint!)
    const locoStyle = sprint && (style === 'walk' || style === 'jog') ? 'run' : style;

    if (mode === 'idle') {
      clip = this.getClip(`idle:${style}`, 'idle:walk');
      phaseDone = false;
    } else if (mode === 'steer') {
      clip = this.getClip(`steer:${locoStyle}`, 'steer:walk');
      const [dx, dz] = norm2(Number(req.dir_x ?? headX), Number(req.dir_z ?? headZ), headX, headZ);
      headX = dx;
      headZ = dz;
      phaseDone = false;
    } else if (mode === 'goto') {
      const tx = Number(req.target_x ?? anchor.x);
      const tz = Number(req.target_z ?? anchor.z);
      const gx = tx - anchor.x;
      const gz = tz - anchor.z;
      const dist = Math.hypot(gx, gz);
      clip = this.getClip(`steer:${locoStyle}`, 'steer:jog');
      const clipDist = Math.hypot(clip.entry.disp_xz[0], clip.entry.disp_xz[1]) || 1.2;
      if (dist <= 0.32) {
        clip = this.getClip(`idle:${style}`, 'idle:walk');
        phaseDone = true;
      } else {
        const [dx, dz] = norm2(gx, gz, headX, headZ);
        headX = dx;
        headZ = dz;
        if (dist <= clipDist * 1.08) {
          xzScale = clamp(dist / clipDist, 0.25, 1.05);
          phaseDone = true;
        } else {
          phaseDone = false;
        }
      }
    } else if (mode === 'vault_entry') {
      const obs = this.findObstacle(req.obstacle_id, 'vault');
      const vClip = this.getClip('action:vault_action');
      const entryDist = vClip.entry.extra?.entry_dist ?? 1.85;
      if (obs) {
        const crossZ = obs.size[2] < obs.size[0];
        const vDir = Number(req.vault_dir ?? -1) >= 0 ? 1 : -1;
        const lane = Number(req.vault_lane ?? (crossZ ? obs.pos[0] : obs.pos[2]));
        const goalX = crossZ ? lane : obs.pos[0] - vDir * entryDist;
        const goalZ = crossZ ? obs.pos[2] - vDir * entryDist : lane;
        const gx = goalX - anchor.x;
        const gz = goalZ - anchor.z;
        const dist = Math.hypot(gx, gz);
        if (dist <= 0.65) {
          clip = vClip;
          outMode = 'vault_action';
          headX = crossZ ? 0 : vDir;
          headZ = crossZ ? vDir : 0;
          phaseDone = true;
        } else {
          clip = this.getClip('steer:jog', 'steer:walk');
          const clipDist = Math.hypot(clip.entry.disp_xz[0], clip.entry.disp_xz[1]) || 1.5;
          const [dx, dz] = norm2(gx, gz, headX, headZ);
          headX = dx;
          headZ = dz;
          if (dist <= clipDist * 1.15) {
            xzScale = clamp(dist / clipDist, 0.35, 1.05);
            phaseDone = true;
          } else {
            phaseDone = false;
          }
        }
      } else {
        clip = vClip;
        outMode = 'vault_action';
        phaseDone = true;
      }
    } else if (mode === 'vault_action' || mode === 'vault_exit') {
      clip = this.getClip(`action:${mode}`);
      const obs = this.findObstacle(req.obstacle_id, 'vault');
      if (obs) {
        const crossZ = obs.size[2] < obs.size[0];
        const vDir = Number(req.vault_dir ?? -1) >= 0 ? 1 : -1;
        headX = crossZ ? 0 : vDir;
        headZ = crossZ ? vDir : 0;
        if (mode === 'vault_action') {
          const lane = Number(req.vault_lane ?? (crossZ ? obs.pos[0] : obs.pos[2]));
          // Nudge gently onto the lane axis so hands plant cleanly on the copper rail!
          if (crossZ) placeX = placeX * 0.35 + lane * 0.65;
          else placeZ = placeZ * 0.35 + lane * 0.65;
        }
      }
      phaseDone = true;
    } else if (mode === 'duck_entry') {
      const obs = this.findObstacle(req.obstacle_id, 'duck');
      const dClip = this.getClip('action:duck_action');
      const entryDist = dClip.entry.extra?.entry_dist ?? 1.5;
      if (obs) {
        const dDir = Number(req.duck_dir ?? -1) >= 0 ? 1 : -1;
        const laneX = Number(req.duck_x ?? obs.pos[0]);
        const goalX = laneX;
        const goalZ = obs.pos[2] - dDir * entryDist;
        const gx = goalX - anchor.x;
        const gz = goalZ - anchor.z;
        const dist = Math.hypot(gx, gz);
        if (dist <= 0.65) {
          clip = dClip;
          outMode = 'duck_action';
          headX = 0;
          headZ = dDir;
          phaseDone = true;
        } else {
          clip = this.getClip('steer:jog', 'steer:walk');
          const clipDist = Math.hypot(clip.entry.disp_xz[0], clip.entry.disp_xz[1]) || 1.5;
          const [dx, dz] = norm2(gx, gz, headX, headZ);
          headX = dx;
          headZ = dz;
          if (dist <= clipDist * 1.15) {
            xzScale = clamp(dist / clipDist, 0.35, 1.05);
            phaseDone = true;
          } else {
            phaseDone = false;
          }
        }
      } else {
        clip = dClip;
        outMode = 'duck_action';
        phaseDone = true;
      }
    } else if (mode === 'duck_action' || mode === 'duck_exit') {
      clip = this.getClip(`action:${mode}`);
      const obs = this.findObstacle(req.obstacle_id, 'duck');
      if (obs) {
        const dDir = Number(req.duck_dir ?? -1) >= 0 ? 1 : -1;
        headX = 0;
        headZ = dDir;
        if (mode === 'duck_action') {
          const laneX = Number(req.duck_x ?? obs.pos[0]);
          placeX = placeX * 0.35 + laneX * 0.65;
        }
      }
      phaseDone = true;
    } else if (mode === 'sit_approach') {
      const obs = this.findObstacle(req.obstacle_id, 'sit');
      if (obs) {
        const [fx, fz] = norm2(obs.facing?.[0] ?? -0.8192, obs.facing?.[1] ?? 0.5736);
        const goalX = obs.pos[0] + fx * 0.58;
        const goalZ = obs.pos[2] + fz * 0.58;
        const gx = goalX - anchor.x;
        const gz = goalZ - anchor.z;
        const dist = Math.hypot(gx, gz);
        if (dist <= 0.55) {
          clip = this.getClip('action:sit_down');
          outMode = 'sit_down';
          placeX = obs.pos[0];
          placeZ = obs.pos[2];
          headX = fx;
          headZ = fz;
          phaseDone = true;
        } else {
          clip = this.getClip('steer:walk');
          const clipDist = Math.hypot(clip.entry.disp_xz[0], clip.entry.disp_xz[1]) || 1.1;
          const [dx, dz] = norm2(gx, gz, headX, headZ);
          headX = dx;
          headZ = dz;
          if (dist <= clipDist * 1.15) {
            xzScale = clamp(dist / clipDist, 0.35, 1.05);
            phaseDone = true;
          } else {
            phaseDone = false;
          }
        }
      } else {
        clip = this.getClip('action:sit_down');
        outMode = 'sit_down';
        phaseDone = true;
      }
    } else if (mode === 'sit_down' || mode === 'sit_hold' || mode === 'stand_up') {
      clip = this.getClip(`action:${mode}`);
      const obs = this.findObstacle(req.obstacle_id, 'sit');
      if (obs) {
        const [fx, fz] = norm2(obs.facing?.[0] ?? -0.8192, obs.facing?.[1] ?? 0.5736);
        placeX = obs.pos[0];
        placeZ = obs.pos[2];
        headX = fx;
        headZ = fz;
      }
      phaseDone = mode !== 'sit_hold';
    } else if (
      mode === 'jump_leap' ||
      mode === 'jump_land' ||
      mode === 'roll' ||
      mode === 'punch_left' ||
      mode === 'punch_right' ||
      mode === 'combat_recover' ||
      mode === 'emote_look' ||
      mode === 'emote_look_2' ||
      mode === 'emote_warmup' ||
      mode === 'emote_warmup_2'
    ) {
      clip = this.getClip(`action:${mode}`);
      if (req.dir_x !== undefined && req.dir_z !== undefined) {
        const [dx, dz] = norm2(Number(req.dir_x), Number(req.dir_z), headX, headZ);
        headX = dx;
        headZ = dz;
      }
      phaseDone = mode !== 'combat_recover';
    } else {
      clip = this.getClip(`idle:${style}`, 'idle:walk');
      phaseDone = false;
    }

    // Transform canonical clip (aligned to (0, -1)) into world pose at (placeX, placeZ) facing (headX, headZ)
    const nf = clip.entry.num_frames;
    const nr = this.numRenderJoints;
    const nb = this.numBoneJoints;

    // Right-handed Y-up yaw rotating canonical (0, -1) -> (headX, headZ)
    let alpha = Math.atan2(headX, headZ) - Math.PI;
    alpha = ((alpha + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    const cosA = Math.cos(alpha);
    const sinA = Math.sin(alpha);
    const qy = Math.sin(alpha * 0.5);
    const qw = Math.cos(alpha * 0.5);

    const isBenchRelative = outMode === 'sit_down' || outMode === 'sit_hold' || outMode === 'stand_up';
    const root0X = isBenchRelative ? 0 : clip.framesPos[0];
    const root0Z = isBenchRelative ? 0 : clip.framesPos[2];

    const outPos = new Float32Array(nf * nr * 3);
    const stridePos = nr * 3;
    for (let f = 0; f < nf; f++) {
      const fBase = f * stridePos;
      // Frame root translation scaled by xzScale (preserves exact internal limb lengths!)
      const relRootX = (clip.framesPos[fBase] - root0X) * xzScale;
      const relRootZ = (clip.framesPos[fBase + 2] - root0Z) * xzScale;
      const wRootX = clamp(placeX + (cosA * relRootX + sinA * relRootZ), MIN_X, MAX_X);
      const wRootZ = clamp(placeZ + (-sinA * relRootX + cosA * relRootZ), MIN_Z, MAX_Z);

      for (let j = 0; j < nr; j++) {
        const p = fBase + j * 3;
        const jDx = clip.framesPos[p] - clip.framesPos[fBase];
        const jDz = clip.framesPos[p + 2] - clip.framesPos[fBase + 2];
        outPos[p] = wRootX + (cosA * jDx + sinA * jDz);
        outPos[p + 1] = clip.framesPos[p + 1];
        outPos[p + 2] = wRootZ + (-sinA * jDx + cosA * jDz);
      }
    }

    // Rotate ONLY root bone (Hips, joint 0) quaternion by q_yaw(alpha); child joints 1..25 are local!
    const outQuat = new Float32Array(clip.framesBoneQuat);
    const strideQuat = nb * 4;
    for (let f = 0; f < nf; f++) {
      const o = f * strideQuat;
      const x2 = outQuat[o];
      const y2 = outQuat[o + 1];
      const z2 = outQuat[o + 2];
      const w2 = outQuat[o + 3];
      // q_yaw = (0, qy, 0, qw) * (x2, y2, z2, w2)
      outQuat[o] = qw * x2 + qy * z2;
      outQuat[o + 1] = qw * y2 + qy * w2;
      outQuat[o + 2] = qw * z2 - qy * x2;
      outQuat[o + 3] = qw * w2 - qy * y2;
    }

    // Build authentic 4-frame Green Context ghost (from actual previous frames when available!)
    const outCtx = new Float32Array(4 * stridePos);
    if (anchor.hist) {
      const hFrames = anchor.hist.framesPos;
      for (let k = 0; k < 4; k++) {
        const srcIdx = clamp(anchor.anchorIdx - (3 - k), 0, anchor.hist.numFrames - 1);
        outCtx.set(hFrames.subarray(srcIdx * stridePos, (srcIdx + 1) * stridePos), k * stridePos);
      }
    } else {
      for (let k = 0; k < 4; k++) {
        const fBase = k * stridePos;
        for (let j = 0; j < nr; j++) {
          const p = fBase + j * 3;
          const dx = clip.contextPos[p] - root0X;
          const dz = clip.contextPos[p + 2] - root0Z;
          outCtx[p] = clamp(placeX + (cosA * dx + sinA * dz), MIN_X, MAX_X);
          outCtx[p + 1] = clip.contextPos[p + 1];
          outCtx[p + 2] = clamp(placeZ + (-sinA * dx + cosA * dz), MIN_Z, MAX_Z);
        }
      }
    }

    // Build 4-frame Blue Target ghost at chunk's planned destination
    const outTgt = new Float32Array(4 * stridePos);
    const lastBase = (nf - 1) * stridePos;
    const endRootX = outPos[lastBase];
    const endRootZ = outPos[lastBase + 2];
    const tgt0X = clip.targetPos[0];
    const tgt0Z = clip.targetPos[2];
    for (let k = 0; k < 4; k++) {
      const fBase = k * stridePos;
      for (let j = 0; j < nr; j++) {
        const p = fBase + j * 3;
        const jDx = clip.targetPos[p] - tgt0X;
        const jDz = clip.targetPos[p + 2] - tgt0Z;
        outTgt[p] = endRootX + (cosA * jDx + sinA * jDz);
        outTgt[p + 1] = clip.targetPos[p + 1];
        outTgt[p + 2] = endRootZ + (-sinA * jDx + cosA * jDz);
      }
    }

    const chunkId = this.nextChunkId++;
    this.curRootX = endRootX;
    this.curRootZ = endRootZ;
    this.curHeadingX = headX;
    this.curHeadingZ = headZ;

    // Maintain 5-chunk ring buffer just like server/api.py
    this.history.set(chunkId, {
      chunkId,
      numFrames: nf,
      framesPos: outPos,
      headingX: headX,
      headingZ: headZ,
    });
    if (this.history.size > 6) {
      const oldest = this.history.keys().next().value;
      if (oldest !== undefined) this.history.delete(oldest);
    }

    const elapsedMs = Math.max(0.1, performance.now() - t0);
    const effTransitionEarly =
      outMode === 'punch_left' || outMode === 'punch_right' ? 0 : clip.entry.transition_early;
    const meta: BinaryChunkMeta = {
      chunk_id: chunkId,
      base_chunk_id: baseChunkId,
      anchor_frame: anchorFrame,
      mode: outMode,
      style,
      phase_done: phaseDone,
      num_frames: nf,
      fps: clip.entry.fps,
      pred_num_tokens: 0,
      transition_early: effTransitionEarly,
      latency_ms: elapsedMs,
      infer_ms: elapsedMs,
      root_end: [endRootX, endRootZ],
      heading_end: [headX, headZ],
      num_render_joints: nr,
      num_bone_joints: nb,
    };

    return {
      ...meta,
      meta,
      framesPos: outPos,
      framesBoneQuat: outQuat,
      footContacts: new Float32Array(clip.footContacts),
      contextPos: outCtx,
      targetPos: outTgt,
    };
  }
}
