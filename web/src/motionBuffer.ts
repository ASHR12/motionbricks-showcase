/**
 * AAA Smoothness Motion Engine for NVIDIA MotionBricks SOMA ONNX Web Runtime.
 *
 * Provides:
 *  1. `decodeBinaryChunk`   - Zero-copy binary wire decoder for server Float32 motion chunks.
 *  2. `sampleChunkCatmullRom` - Zero-GC 4-point Catmull-Rom spline position sampler +
 *                               shortest-arc quaternion SLERP + linear foot-contact sampler.
 *  3. `MotionInertializer`  - 120 ms critically-damped C0/C1 inertialization cross-fader
 *                               for snap-free chunk handoffs and mid-chunk interrupts.
 *  4. `FootLockStabilizer`  - Neural ONNX foot-contact grounding stabilizer with strict
 *                               <= 4.5 cm limb-length preservation clamp.
 */

export interface BinaryChunkMeta {
  chunk_id: number;
  base_chunk_id: number;
  anchor_frame: number;
  mode: string;
  style: string;
  phase_done: boolean;
  num_frames: number;
  fps: number;
  pred_num_tokens: number;
  transition_early: number;
  latency_ms: number;
  infer_ms: number;
  root_end: [number, number];
  heading_end: [number, number];
  num_render_joints: number;
  num_bone_joints: number;
}

export interface BinaryChunk extends BinaryChunkMeta {
  /** Original parsed JSON metadata block. */
  meta: BinaryChunkMeta;
  /** World-space render joint positions: length `num_frames * num_render_joints * 3` (meters, Y-up). */
  framesPos: Float32Array;
  /** Local bone quaternions (XYZW): length `num_frames * num_bone_joints * 4`. */
  framesBoneQuat: Float32Array;
  /** ONNX predicted foot contacts [L_heel, L_toe, R_heel, R_toe]: length `num_frames * 4`. */
  footContacts: Float32Array;
  /** 4-frame Green Context ghost world joint positions: length `4 * num_render_joints * 3`. */
  contextPos: Float32Array;
  /** 4-frame Blue Target ghost world joint positions: length `4 * num_render_joints * 3`. */
  targetPos: Float32Array;
}

const textDecoder = new TextDecoder('utf-8');
const IS_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

function sliceFloat32LE(
  buf: ArrayBuffer,
  byteOffset: number,
  length: number
): Float32Array {
  if (IS_LITTLE_ENDIAN && (byteOffset & 3) === 0) {
    return new Float32Array(buf, byteOffset, length);
  }
  const view = new DataView(buf, byteOffset, length * 4);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = view.getFloat32(i * 4, true);
  }
  return out;
}

/**
 * Decode a binary MotionBricks chunk buffer into metadata + zero-copy Float32Array views.
 *
 * Wire layout:
 *  - Bytes `0..4`:         uint32 little-endian `metaLen` (multiple of 4)
 *  - Bytes `4..4+metaLen`: UTF-8 JSON metadata (optional trailing spaces / NUL padding)
 *  - Remaining bytes (contiguous little-endian float32 arrays):
 *      1. `framesPos`      (`F * num_render_joints * 3`)
 *      2. `framesBoneQuat` (`F * num_bone_joints * 4`)
 *      3. `footContacts`   (`F * 4`)
 *      4. `contextPos`     (`4 * num_render_joints * 3`)
 *      5. `targetPos`      (`4 * num_render_joints * 3`)
 */
export function decodeBinaryChunk(buf: ArrayBuffer): BinaryChunk {
  if (buf.byteLength < 4) {
    throw new Error(
      `Invalid MotionBricks binary chunk: buffer too short (${buf.byteLength} bytes)`
    );
  }

  const headerView = new DataView(buf, 0, 4);
  const metaLen = headerView.getUint32(0, true);
  const dataStart = 4 + metaLen;

  if (metaLen === 0 || dataStart > buf.byteLength) {
    throw new Error(
      `Invalid MotionBricks binary chunk: metaLen=${metaLen} exceeds buffer (${buf.byteLength} bytes)`
    );
  }

  const metaBytes = new Uint8Array(buf, 4, metaLen);
  const metaJson = textDecoder.decode(metaBytes).replace(/\0+$/, '');
  const rawMeta = JSON.parse(metaJson) as Partial<BinaryChunkMeta>;

  const numFrames = Number(rawMeta.num_frames ?? 0);
  const numRenderJoints = Number(rawMeta.num_render_joints ?? 24);
  const numBoneJoints = Number(rawMeta.num_bone_joints ?? 0);

  const meta: BinaryChunkMeta = {
    chunk_id: Number(rawMeta.chunk_id ?? 0),
    base_chunk_id: Number(rawMeta.base_chunk_id ?? 0),
    anchor_frame: Number(rawMeta.anchor_frame ?? 0),
    mode: String(rawMeta.mode ?? 'idle'),
    style: String(rawMeta.style ?? 'walk'),
    phase_done: Boolean(rawMeta.phase_done ?? false),
    num_frames: numFrames,
    fps: Number(rawMeta.fps ?? 30),
    pred_num_tokens: Number(rawMeta.pred_num_tokens ?? 0),
    transition_early: Number(rawMeta.transition_early ?? 4),
    latency_ms: Number(rawMeta.latency_ms ?? 0),
    infer_ms: Number(rawMeta.infer_ms ?? rawMeta.latency_ms ?? 0),
    root_end: rawMeta.root_end ?? [0, 0],
    heading_end: rawMeta.heading_end ?? [0, 1],
    num_render_joints: numRenderJoints,
    num_bone_joints: numBoneJoints,
  };

  const lenPos = numFrames * numRenderJoints * 3;
  const lenQuat = numFrames * numBoneJoints * 4;
  const lenContacts = numFrames * 4;
  const lenCtx = 4 * numRenderJoints * 3;
  const lenTgt = 4 * numRenderJoints * 3;
  const totalFloats = lenPos + lenQuat + lenContacts + lenCtx + lenTgt;
  const requiredBytes = dataStart + totalFloats * 4;

  if (buf.byteLength < requiredBytes) {
    throw new Error(
      `Truncated MotionBricks binary chunk: expected ${requiredBytes} bytes, got ${buf.byteLength}`
    );
  }

  let offset = dataStart;
  const framesPos = sliceFloat32LE(buf, offset, lenPos);
  offset += lenPos * 4;

  const framesBoneQuat = sliceFloat32LE(buf, offset, lenQuat);
  offset += lenQuat * 4;

  const footContacts = sliceFloat32LE(buf, offset, lenContacts);
  offset += lenContacts * 4;

  const contextPos = sliceFloat32LE(buf, offset, lenCtx);
  offset += lenCtx * 4;

  const targetPos = sliceFloat32LE(buf, offset, lenTgt);

  return {
    ...meta,
    meta,
    framesPos,
    framesBoneQuat,
    footContacts,
    contextPos,
    targetPos,
  };
}

/**
 * Sample a BinaryChunk at fractional `frameFloat` into caller-owned pre-allocated buffers.
 *
 * - Positions: 4-point uniform Catmull-Rom cubic spline (`f-1, f0, f1, f2`, clamped at ends)
 *   preserving C1 velocity continuity and exact parabolic jump/vault arcs.
 * - Bone quaternions: shortest-arc spherical SLERP (XYZW) with unit-length renormalization.
 * - Foot contacts: linear interpolation into `[c0, c1, c2, c3]`.
 *
 * Performs zero heap allocations.
 */
export function sampleChunkCatmullRom(
  chunk: BinaryChunk,
  frameFloat: number,
  outPos: Float32Array,
  outBoneQuat: Float32Array,
  outFootContacts: [number, number, number, number]
): void {
  const numFrames = chunk.num_frames;
  if (numFrames <= 0) return;

  const maxIdx = numFrames - 1;
  const clamped =
    frameFloat <= 0 ? 0 : frameFloat >= maxIdx ? maxIdx : frameFloat;
  const f0 = Math.floor(clamped);
  const f1 = f0 < maxIdx ? f0 + 1 : maxIdx;
  const fm1 = f0 > 0 ? f0 - 1 : 0;
  const f2 = f1 < maxIdx ? f1 + 1 : maxIdx;
  const t = clamped - f0;

  const posStride = chunk.num_render_joints * 3;
  const pos = chunk.framesPos;

  // 1. Positions via 4-point Catmull-Rom cubic spline
  if (posStride > 0 && outPos.length >= posStride) {
    const o0 = f0 * posStride;
    if (t <= 1e-6 || f0 === f1) {
      for (let i = 0; i < posStride; i++) {
        outPos[i] = pos[o0 + i];
      }
    } else if (t >= 1 - 1e-6) {
      const o1 = f1 * posStride;
      for (let i = 0; i < posStride; i++) {
        outPos[i] = pos[o1 + i];
      }
    } else {
      const t2 = t * t;
      const t3 = t2 * t;
      const w0 = 0.5 * (-t + 2 * t2 - t3);
      const w1 = 0.5 * (2 - 5 * t2 + 3 * t3);
      const w2 = 0.5 * (t + 4 * t2 - 3 * t3);
      const w3 = 0.5 * (-t2 + t3);

      const oM1 = fm1 * posStride;
      const o1 = f1 * posStride;
      const o2 = f2 * posStride;

      for (let i = 0; i < posStride; i++) {
        outPos[i] =
          w0 * pos[oM1 + i] +
          w1 * pos[o0 + i] +
          w2 * pos[o1 + i] +
          w3 * pos[o2 + i];
      }
    }
  }

  // 2. Bone quaternions via shortest-arc spherical SLERP (XYZW)
  const numBones = chunk.num_bone_joints;
  const quats = chunk.framesBoneQuat;
  if (numBones > 0 && outBoneQuat.length >= numBones * 4) {
    const qStride = numBones * 4;
    const base0 = f0 * qStride;
    const base1 = f1 * qStride;

    if (t <= 1e-6 || f0 === f1) {
      for (let i = 0; i < qStride; i++) {
        outBoneQuat[i] = quats[base0 + i];
      }
    } else if (t >= 1 - 1e-6) {
      for (let i = 0; i < qStride; i++) {
        outBoneQuat[i] = quats[base1 + i];
      }
    } else {
      const invT = 1 - t;
      for (let b = 0; b < numBones; b++) {
        const io = b * 4;
        const i0 = base0 + io;
        const i1 = base1 + io;

        const x0 = quats[i0];
        const y0 = quats[i0 + 1];
        const z0 = quats[i0 + 2];
        const w0 = quats[i0 + 3];

        let x1 = quats[i1];
        let y1 = quats[i1 + 1];
        let z1 = quats[i1 + 2];
        let w1 = quats[i1 + 3];

        let dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
        if (dot < 0) {
          x1 = -x1;
          y1 = -y1;
          z1 = -z1;
          w1 = -w1;
          dot = -dot;
        }

        let rx: number;
        let ry: number;
        let rz: number;
        let rw: number;

        if (dot > 0.9995) {
          rx = x0 + t * (x1 - x0);
          ry = y0 + t * (y1 - y0);
          rz = z0 + t * (z1 - z0);
          rw = w0 + t * (w1 - w0);
        } else {
          const clampedDot = dot > 1 ? 1 : dot;
          const theta = Math.acos(clampedDot);
          const invSin = 1 / Math.sin(theta);
          const s0 = Math.sin(invT * theta) * invSin;
          const s1 = Math.sin(t * theta) * invSin;
          rx = s0 * x0 + s1 * x1;
          ry = s0 * y0 + s1 * y1;
          rz = s0 * z0 + s1 * z1;
          rw = s0 * w0 + s1 * w1;
        }

        const lenSq = rx * rx + ry * ry + rz * rz + rw * rw;
        if (lenSq > 1e-12) {
          const invLen = 1 / Math.sqrt(lenSq);
          outBoneQuat[io] = rx * invLen;
          outBoneQuat[io + 1] = ry * invLen;
          outBoneQuat[io + 2] = rz * invLen;
          outBoneQuat[io + 3] = rw * invLen;
        } else {
          outBoneQuat[io] = 0;
          outBoneQuat[io + 1] = 0;
          outBoneQuat[io + 2] = 0;
          outBoneQuat[io + 3] = 1;
        }
      }
    }
  }

  // 3. Foot contacts via linear interpolation
  const fc = chunk.footContacts;
  if (fc.length >= (f1 + 1) * 4) {
    const c0 = f0 * 4;
    const c1 = f1 * 4;
    const invT = 1 - t;
    outFootContacts[0] = fc[c0] * invT + fc[c1] * t;
    outFootContacts[1] = fc[c0 + 1] * invT + fc[c1 + 1] * t;
    outFootContacts[2] = fc[c0 + 2] * invT + fc[c1 + 2] * t;
    outFootContacts[3] = fc[c0 + 3] * invT + fc[c1 + 3] * t;
  }
}

/**
 * Zero-GC helper to copy a flat `Float32Array(N * 3)` into pre-allocated `[x, y, z][]` tuples
 * (useful when driving tuple-based renderers like `ArticulatedMannequin` or `GhostMannequin`).
 */
export function copyFlatPositionsToTuples(
  flatPos: Float32Array,
  outTuples: [number, number, number][],
  frameOffset = 0
): void {
  const count = outTuples.length;
  const base = frameOffset * count * 3;
  if (flatPos.length < base + count * 3) return;
  for (let i = 0; i < count; i++) {
    const o = base + i * 3;
    const dst = outTuples[i];
    dst[0] = flatPos[o];
    dst[1] = flatPos[o + 1];
    dst[2] = flatPos[o + 2];
  }
}

/**
 * 120 ms Critically-Damped Inertialization Cross-Fader.
 *
 * Captures exact pose residuals at any chunk boundary or mid-chunk interrupt so
 * frame 0 of the incoming chunk reproduces the exact displayed pose (C0 continuous),
 * then decays the residual cleanly to zero over `durationSec` via a monotonic
 * critically-damped Hermite curve (`w(0)=1, w'(0)=0, w(1)=0, w'(1)=0`).
 *
 * Performs zero heap allocations during `captureTransition` and `apply`.
 */
export class MotionInertializer {
  private posDelta: Float32Array;
  private quatDelta: Float32Array;
  private posLen = 0;
  private numBones = 0;
  private elapsed = 0;
  private duration = 0.12;
  private active = false;

  constructor(numRenderJoints = 24, numBoneJoints = 30) {
    this.posDelta = new Float32Array(numRenderJoints * 3);
    this.quatDelta = new Float32Array(numBoneJoints * 4);
    this.resetQuatIdentity();
  }

  private resetQuatIdentity(): void {
    const bones = this.quatDelta.length >>> 2;
    for (let b = 0; b < bones; b++) {
      const o = b * 4;
      this.quatDelta[o] = 0;
      this.quatDelta[o + 1] = 0;
      this.quatDelta[o + 2] = 0;
      this.quatDelta[o + 3] = 1;
    }
  }

  public isActive(): boolean {
    return this.active;
  }

  /** Normalized blend progress in `[0, 1]` (1 when inactive/complete). */
  public getProgress(): number {
    if (!this.active || this.duration <= 0) return 1;
    const u = this.elapsed / this.duration;
    return u >= 1 ? 1 : u <= 0 ? 0 : u;
  }

  public reset(): void {
    this.active = false;
    this.elapsed = 0;
    this.posDelta.fill(0);
    this.resetQuatIdentity();
  }

  /**
   * Capture exact residual `(prev - next)` for positions and `prev * conj(next)`
   * for bone quaternions (canonicalized to the shortest-arc `w >= 0` hemisphere).
   *
   * Safe to call even while an earlier transition is mid-flight, provided
   * `prevPos` / `prevBoneQuat` hold the currently displayed pose.
   */
  public captureTransition(
    prevPos: Float32Array,
    prevBoneQuat: Float32Array,
    nextPos: Float32Array,
    nextBoneQuat: Float32Array,
    durationSec = 0.12
  ): void {
    if (durationSec <= 1e-5) {
      this.active = false;
      this.elapsed = 0;
      return;
    }

    const posLen = Math.min(prevPos.length, nextPos.length);
    if (posLen > this.posDelta.length) {
      this.posDelta = new Float32Array(posLen);
    }
    this.posLen = posLen;
    for (let i = 0; i < posLen; i++) {
      this.posDelta[i] = prevPos[i] - nextPos[i];
    }

    const quatLen = Math.min(prevBoneQuat.length, nextBoneQuat.length);
    const numBones = quatLen >>> 2;
    if (numBones * 4 > this.quatDelta.length) {
      this.quatDelta = new Float32Array(numBones * 4);
    }
    this.numBones = numBones;

    for (let b = 0; b < numBones; b++) {
      const o = b * 4;

      // Normalize prev
      let px = prevBoneQuat[o];
      let py = prevBoneQuat[o + 1];
      let pz = prevBoneQuat[o + 2];
      let pw = prevBoneQuat[o + 3];
      const pLenSq = px * px + py * py + pz * pz + pw * pw;
      if (pLenSq > 1e-12) {
        const inv = 1 / Math.sqrt(pLenSq);
        px *= inv;
        py *= inv;
        pz *= inv;
        pw *= inv;
      } else {
        px = 0;
        py = 0;
        pz = 0;
        pw = 1;
      }

      // Normalize next and take conjugate (-x, -y, -z, w)
      let nx = nextBoneQuat[o];
      let ny = nextBoneQuat[o + 1];
      let nz = nextBoneQuat[o + 2];
      let nw = nextBoneQuat[o + 3];
      const nLenSq = nx * nx + ny * ny + nz * nz + nw * nw;
      if (nLenSq > 1e-12) {
        const inv = 1 / Math.sqrt(nLenSq);
        nx = -nx * inv;
        ny = -ny * inv;
        nz = -nz * inv;
        nw = nw * inv;
      } else {
        nx = 0;
        ny = 0;
        nz = 0;
        nw = 1;
      }

      // qDelta = prev * conj(next)
      let dx = pw * nx + px * nw + py * nz - pz * ny;
      let dy = pw * ny - px * nz + py * nw + pz * nx;
      let dz = pw * nz + px * ny - py * nx + pz * nw;
      let dw = pw * nw - px * nx - py * ny - pz * nz;

      const dLenSq = dx * dx + dy * dy + dz * dz + dw * dw;
      if (dLenSq > 1e-12) {
        const inv = 1 / Math.sqrt(dLenSq);
        dx *= inv;
        dy *= inv;
        dz *= inv;
        dw *= inv;
      } else {
        dx = 0;
        dy = 0;
        dz = 0;
        dw = 1;
      }

      // Canonicalize to shortest-arc hemisphere (w >= 0) so slerp(I, qDelta, w) never flips
      if (dw < 0) {
        dx = -dx;
        dy = -dy;
        dz = -dz;
        dw = -dw;
      }

      this.quatDelta[o] = dx;
      this.quatDelta[o + 1] = dy;
      this.quatDelta[o + 2] = dz;
      this.quatDelta[o + 3] = dw;
    }

    this.elapsed = 0;
    this.duration = durationSec;
    this.active = true;
  }

  /**
   * Advance the inertializer by `dt` seconds and write the smoothed pose into `outPos`
   * and `outBoneQuat`. Safe for in-place usage (`outPos === rawPos`, `outBoneQuat === rawBoneQuat`).
   */
  public apply(
    dt: number,
    rawPos: Float32Array,
    rawBoneQuat: Float32Array,
    outPos: Float32Array,
    outBoneQuat: Float32Array
  ): void {
    if (!this.active) {
      if (outPos !== rawPos) outPos.set(rawPos);
      if (outBoneQuat !== rawBoneQuat) outBoneQuat.set(rawBoneQuat);
      return;
    }

    if (dt > 0) {
      this.elapsed += dt;
    }

    if (this.elapsed >= this.duration) {
      this.active = false;
      if (outPos !== rawPos) outPos.set(rawPos);
      if (outBoneQuat !== rawBoneQuat) outBoneQuat.set(rawBoneQuat);
      return;
    }

    // Critically-damped spring envelope tapered cleanly to 0 with zero boundary slope:
    // w(0) = 1, w'(0) = 0, w(1) = 0, w'(1) = 0, strictly monotonic on [0, 1].
    const u = this.elapsed / this.duration;
    const oneMinusU = 1 - u;
    const w =
      (1 + 4 * u) * Math.exp(-4 * u) * oneMinusU * oneMinusU * (1 + 2 * u);

    if (w <= 1e-5) {
      this.active = false;
      if (outPos !== rawPos) outPos.set(rawPos);
      if (outBoneQuat !== rawBoneQuat) outBoneQuat.set(rawBoneQuat);
      return;
    }

    // Apply decaying position residual
    const posLen = Math.min(this.posLen, rawPos.length, outPos.length);
    for (let i = 0; i < posLen; i++) {
      outPos[i] = rawPos[i] + w * this.posDelta[i];
    }
    if (outPos !== rawPos && rawPos.length > posLen) {
      for (let i = posLen; i < rawPos.length && i < outPos.length; i++) {
        outPos[i] = rawPos[i];
      }
    }

    // Apply decaying quaternion residual: out = slerp(I, qDelta, w) * raw
    const numBones = Math.min(
      this.numBones,
      rawBoneQuat.length >>> 2,
      outBoneQuat.length >>> 2
    );
    for (let b = 0; b < numBones; b++) {
      const o = b * 4;
      const dx = this.quatDelta[o];
      const dy = this.quatDelta[o + 1];
      const dz = this.quatDelta[o + 2];
      const dw = this.quatDelta[o + 3];

      const rx = rawBoneQuat[o];
      const ry = rawBoneQuat[o + 1];
      const rz = rawBoneQuat[o + 2];
      const rw = rawBoneQuat[o + 3];

      let sx: number;
      let sy: number;
      let sz: number;
      let sw: number;

      const cosHalf = dw > 1 ? 1 : dw < 0 ? 0 : dw;
      if (cosHalf > 0.99999) {
        sx = dx * w;
        sy = dy * w;
        sz = dz * w;
        sw = 1;
        const sInv = 1 / Math.sqrt(sx * sx + sy * sy + sz * sz + 1);
        sx *= sInv;
        sy *= sInv;
        sz *= sInv;
        sw *= sInv;
      } else {
        const halfAngle = Math.acos(cosHalf);
        const sinHalf = Math.sin(halfAngle);
        const newHalf = halfAngle * w;
        const scale = Math.sin(newHalf) / sinHalf;
        sx = dx * scale;
        sy = dy * scale;
        sz = dz * scale;
        sw = Math.cos(newHalf);
      }

      // Hamilton product: s * raw
      const ox = sw * rx + sx * rw + sy * rz - sz * ry;
      const oy = sw * ry - sx * rz + sy * rw + sz * rx;
      const oz = sw * rz + sx * ry - sy * rx + sz * rw;
      const ow = sw * rw - sx * rx - sy * ry - sz * rz;

      const oLenSq = ox * ox + oy * oy + oz * oz + ow * ow;
      if (oLenSq > 1e-12) {
        const inv = 1 / Math.sqrt(oLenSq);
        outBoneQuat[o] = ox * inv;
        outBoneQuat[o + 1] = oy * inv;
        outBoneQuat[o + 2] = oz * inv;
        outBoneQuat[o + 3] = ow * inv;
      } else {
        outBoneQuat[o] = rx;
        outBoneQuat[o + 1] = ry;
        outBoneQuat[o + 2] = rz;
        outBoneQuat[o + 3] = rw;
      }
    }

    if (outBoneQuat !== rawBoneQuat && rawBoneQuat.length > numBones * 4) {
      for (
        let i = numBones * 4;
        i < rawBoneQuat.length && i < outBoneQuat.length;
        i++
      ) {
        outBoneQuat[i] = rawBoneQuat[i];
      }
    }
  }
}

export interface FootLockOptions {
  /** Render-joint index for LeftFoot (default 18). */
  leftFootIndex?: number;
  /** Render-joint index for LeftToeBase (default 19). */
  leftToeIndex?: number;
  /** Render-joint index for RightFoot (default 22). */
  rightFootIndex?: number;
  /** Render-joint index for RightToeBase (default 23). */
  rightToeIndex?: number;
  /** Contact confidence required to engage anchor lock (default 0.72). */
  lockThreshold?: number;
  /** Maximum 3D stabilization shift in meters so limbs never visibly stretch (default 0.045). */
  maxOffsetMeters?: number;
  /** Floor Y plane in meters (default 0.0). */
  floorY?: number;
}

interface FootAnchorState {
  locked: boolean;
  anchorX: number;
  anchorY: number;
  anchorZ: number;
  offsetX: number;
  offsetY: number;
  offsetZ: number;
}

/**
 * Neural Foot-Lock Grounding Stabilizer driven by ONNX `footContacts[0..3]`.
 *
 * Eliminates subtle micro-skate and vertical jitter during stance phases while
 * strictly clamping total shift `<= 0.045 m` so leg segment lengths stay faithful
 * and never over-extend. Performs zero heap allocations per frame.
 */
export class FootLockStabilizer {
  private readonly leftFootIdx: number;
  private readonly leftToeIdx: number;
  private readonly rightFootIdx: number;
  private readonly rightToeIdx: number;
  private readonly lockThreshold: number;
  private readonly maxOffset: number;
  private readonly maxYOffset: number;
  private readonly floorY: number;

  private readonly left: FootAnchorState = {
    locked: false,
    anchorX: 0,
    anchorY: 0,
    anchorZ: 0,
    offsetX: 0,
    offsetY: 0,
    offsetZ: 0,
  };

  private readonly right: FootAnchorState = {
    locked: false,
    anchorX: 0,
    anchorY: 0,
    anchorZ: 0,
    offsetX: 0,
    offsetY: 0,
    offsetZ: 0,
  };

  constructor(options: FootLockOptions = {}) {
    this.leftFootIdx = options.leftFootIndex ?? 18;
    this.leftToeIdx = options.leftToeIndex ?? 19;
    this.rightFootIdx = options.rightFootIndex ?? 22;
    this.rightToeIdx = options.rightToeIndex ?? 23;
    this.lockThreshold = options.lockThreshold ?? 0.72;
    this.maxOffset = Math.min(options.maxOffsetMeters ?? 0.045, 0.045);
    this.maxYOffset = Math.min(0.025, this.maxOffset * 0.6);
    this.floorY = options.floorY ?? 0.0;
  }

  /** Clear planted anchors (call on scene reset or large teleport). */
  public reset(): void {
    this.left.locked = false;
    this.left.offsetX = 0;
    this.left.offsetY = 0;
    this.left.offsetZ = 0;
    this.right.locked = false;
    this.right.offsetX = 0;
    this.right.offsetY = 0;
    this.right.offsetZ = 0;
  }

  /**
   * Apply neural foot-lock stabilization in-place onto `pos` (`Float32Array(numRenderJoints * 3)`).
   */
  public apply(
    pos: Float32Array,
    footContacts: readonly [number, number, number, number] | Float32Array,
    dt = 1 / 60
  ): void {
    if (footContacts.length < 4) return;
    const leftContact =
      footContacts[0] > footContacts[1] ? footContacts[0] : footContacts[1];
    const rightContact =
      footContacts[2] > footContacts[3] ? footContacts[2] : footContacts[3];

    this.stabilizeFoot(
      this.left,
      pos,
      this.leftFootIdx,
      this.leftToeIdx,
      leftContact,
      dt
    );
    this.stabilizeFoot(
      this.right,
      pos,
      this.rightFootIdx,
      this.rightToeIdx,
      rightContact,
      dt
    );
  }

  /** Read-only diagnostic state for debug overlays. */
  public getLockState(): {
    leftLocked: boolean;
    rightLocked: boolean;
    leftOffsetMeters: number;
    rightOffsetMeters: number;
  } {
    return {
      leftLocked: this.left.locked,
      rightLocked: this.right.locked,
      leftOffsetMeters: Math.hypot(
        this.left.offsetX,
        this.left.offsetY,
        this.left.offsetZ
      ),
      rightOffsetMeters: Math.hypot(
        this.right.offsetX,
        this.right.offsetY,
        this.right.offsetZ
      ),
    };
  }

  private stabilizeFoot(
    state: FootAnchorState,
    pos: Float32Array,
    footIdx: number,
    toeIdx: number,
    contact: number,
    dt: number
  ): void {
    const fOff = footIdx * 3;
    const tOff = toeIdx * 3;
    if (pos.length < fOff + 3 || pos.length < tOff + 3) return;

    const fx = pos[fOff];
    const fy = pos[fOff + 1];
    const fz = pos[fOff + 2];

    // Smooth blend factor: clamp((contact - 0.55) / 0.3, 0, 1)
    const rawW = (contact - 0.55) / 0.3;
    const w = rawW <= 0 ? 0 : rawW >= 1 ? 1 : rawW;

    // Floor-contact vertical band (SOMA ankle sits ~0.075 m above floor when planted)
    const groundBandMaxY = this.floorY + 0.22;
    const ankleMinY = this.floorY + 0.048;
    const ankleCapY = this.floorY + 0.088;

    // 1. Engage or release lock with teleport safety guard
    if (!state.locked) {
      if (contact > this.lockThreshold) {
        state.locked = true;
        state.anchorX = fx;
        state.anchorZ = fz;
        state.anchorY = fy < groundBandMaxY ? Math.min(fy, ankleCapY) : fy;
      }
    } else {
      const distXZ = Math.hypot(state.anchorX - fx, state.anchorZ - fz);
      if (w <= 0 || distXZ > 0.35) {
        state.locked = false;
      }
    }

    // 2. Compute target correction vector (clamped strictly <= maxOffset)
    let targetX = 0;
    let targetY = 0;
    let targetZ = 0;

    if (state.locked && w > 0) {
      let dx = (state.anchorX - fx) * w;
      let dz = (state.anchorZ - fz) * w;
      let dy = 0;

      if (fy < groundBandMaxY) {
        // Gently anchor lower jitter envelope toward floor contact height
        if (fy < state.anchorY && fy >= ankleMinY) {
          state.anchorY = fy;
        }
        dy = (state.anchorY - fy) * (w * 0.72);
        if (fy + dy < ankleMinY) {
          dy = ankleMinY - fy;
        }
        if (dy > this.maxYOffset) dy = this.maxYOffset;
        else if (dy < -this.maxYOffset) dy = -this.maxYOffset;
      }

      // Enforce strict 3D clamp <= 0.045 m so leg segment length never stretches
      const mag = Math.hypot(dx, dy, dz);
      if (mag > this.maxOffset) {
        const scale = this.maxOffset / mag;
        dx *= scale;
        dy *= scale;
        dz *= scale;
      }

      targetX = dx;
      targetY = dy;
      targetZ = dz;
    }

    // 3. Smoothly track target when locked, or ease cleanly back to 0 on swing release
    const clampedDt = dt > 0 ? Math.min(dt, 0.05) : 1 / 60;
    const rate = state.locked ? 30 : 24;
    const alpha = 1 - Math.exp(-clampedDt * rate);

    state.offsetX += (targetX - state.offsetX) * alpha;
    state.offsetY += (targetY - state.offsetY) * alpha;
    state.offsetZ += (targetZ - state.offsetZ) * alpha;

    // Re-clamp smoothed offset as an absolute invariant
    const curMag = Math.hypot(state.offsetX, state.offsetY, state.offsetZ);
    if (curMag > this.maxOffset) {
      const scale = this.maxOffset / curMag;
      state.offsetX *= scale;
      state.offsetY *= scale;
      state.offsetZ *= scale;
    } else if (!state.locked && curMag < 1e-4) {
      state.offsetX = 0;
      state.offsetY = 0;
      state.offsetZ = 0;
    }

    // 4. Shift Foot and ToeBase together cleanly
    if (
      state.offsetX !== 0 ||
      state.offsetY !== 0 ||
      state.offsetZ !== 0
    ) {
      pos[fOff] = fx + state.offsetX;
      pos[fOff + 1] = fy + state.offsetY;
      pos[fOff + 2] = fz + state.offsetZ;

      pos[tOff] += state.offsetX;
      const newToeY = pos[tOff + 1] + state.offsetY;
      pos[tOff + 1] =
        state.locked && newToeY < this.floorY + 0.014
          ? this.floorY + 0.014
          : newToeY;
      pos[tOff + 2] += state.offsetZ;
    }
  }
}
