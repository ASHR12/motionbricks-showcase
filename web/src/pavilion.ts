import * as THREE from 'three';
import { PAVILION_SPEC, STUDIO_THEMES } from './studioSpec';

export interface MocapPavilionHandle {
  group: THREE.Group;
  setTheme: (mode: 'dark' | 'light') => void;
  update: (timeSec: number) => void;
}

export interface ConcreteTextureSet {
  map: THREE.CanvasTexture;
  roughnessMap: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
}

/** Deterministic PRNG so procedural architectural textures are art-directed & reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let cachedConcreteTextures: ConcreteTextureSet | null = null;

/**
 * Generates museum-grade Tadao Ando poured architectural concrete textures (512x512):
 * subtle micro-cement hydration patina, crisp formwork panel joints, and recessed tie-rod
 * cone dots with matching linear roughness & micro-relief bump maps.
 */
export function getArchitecturalConcreteTextures(): ConcreteTextureSet {
  if (cachedConcreteTextures) return cachedConcreteTextures;

  const size = 512;
  const rand = mulberry32(41092);

  // 1. Albedo canvas (high-value neutral grey so material.color cleanly drives dark/travertine themes)
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = size;
  colorCanvas.height = size;
  const cCtx = colorCanvas.getContext('2d')!;

  // 2. Roughness canvas (linear: 188..244 => roughness 0.74..0.96)
  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = size;
  roughCanvas.height = size;
  const rCtx = roughCanvas.getContext('2d')!;

  // 3. Bump canvas (linear neutral 128, depressed seams & tie-rod cones)
  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  // Base coats
  cCtx.fillStyle = '#e5e3df';
  cCtx.fillRect(0, 0, size, size);

  rCtx.fillStyle = 'rgb(214, 214, 214)';
  rCtx.fillRect(0, 0, size, size);

  bCtx.fillStyle = 'rgb(128, 128, 128)';
  bCtx.fillRect(0, 0, size, size);

  // Soft large-scale poured-cement hydration mottling (wrap-aware 3x3 tiling)
  for (let i = 0; i < 42; i++) {
    const cx = rand() * size;
    const cy = rand() * size;
    const radius = 58 + rand() * 115;
    const delta = (rand() - 0.5) * 14; // gentle ±2.7% luminance shift
    const shade = Math.round(229 + delta);
    const roughVal = Math.round(212 - delta * 1.1);

    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        const px = cx + ox * size;
        const py = cy + oy * size;
        const cg = cCtx.createRadialGradient(px, py, 0, px, py, radius);
        cg.addColorStop(0, `rgba(${shade}, ${shade - 1}, ${shade - 3}, 0.24)`);
        cg.addColorStop(1, `rgba(${shade}, ${shade - 1}, ${shade - 3}, 0)`);
        cCtx.fillStyle = cg;
        cCtx.fillRect(px - radius, py - radius, radius * 2, radius * 2);

        const rg = rCtx.createRadialGradient(px, py, 0, px, py, radius);
        rg.addColorStop(0, `rgba(${roughVal}, ${roughVal}, ${roughVal}, 0.28)`);
        rg.addColorStop(1, `rgba(${roughVal}, ${roughVal}, ${roughVal}, 0)`);
        rCtx.fillStyle = rg;
        rCtx.fillRect(px - radius, py - radius, radius * 2, radius * 2);
      }
    }
  }

  // Subtle horizontal board-formed troweling streaks
  for (let i = 0; i < 64; i++) {
    const y = rand() * size;
    const h = 2 + rand() * 7;
    const alpha = 0.015 + rand() * 0.025;
    const bright = rand() > 0.5;
    cCtx.fillStyle = bright ? `rgba(255, 253, 248, ${alpha})` : `rgba(45, 48, 55, ${alpha})`;
    cCtx.fillRect(0, y, size, h);

    rCtx.fillStyle = bright ? `rgba(196, 196, 196, ${alpha * 1.4})` : `rgba(232, 232, 232, ${alpha * 1.4})`;
    rCtx.fillRect(0, y, size, h);
  }

  // Architectural 2x2 formwork panel grid with subtle per-panel tonal variation
  const cols = 2;
  const rows = 2;
  const panelW = size / cols;
  const panelH = size / rows;
  const panelTints = [-4, 3, 2, -3];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const px = c * panelW;
      const py = r * panelH;
      const tint = panelTints[r * cols + c];
      cCtx.fillStyle = tint > 0 ? `rgba(255, 253, 248, ${tint * 0.008})` : `rgba(30, 34, 42, ${-tint * 0.009})`;
      cCtx.fillRect(px, py, panelW, panelH);

      // Subtle Tadao Ando tie-rod cone micro-dots (2x3 per formwork slab)
      const insetX = panelW * 0.17;
      const insetY = panelH * 0.22;
      const tiePoints: [number, number][] = [
        [px + insetX, py + insetY],
        [px + panelW * 0.5, py + insetY],
        [px + panelW - insetX, py + insetY],
        [px + insetX, py + panelH - insetY],
        [px + panelW * 0.5, py + panelH - insetY],
        [px + panelW - insetX, py + panelH - insetY],
      ];

      for (const [tx, ty] of tiePoints) {
        // Soft recessed cone halo
        const halo = cCtx.createRadialGradient(tx, ty, 1.2, tx, ty, 5.2);
        halo.addColorStop(0, 'rgba(32, 35, 40, 0.32)');
        halo.addColorStop(0.55, 'rgba(55, 58, 64, 0.14)');
        halo.addColorStop(1, 'rgba(55, 58, 64, 0)');
        cCtx.fillStyle = halo;
        cCtx.beginPath();
        cCtx.arc(tx, ty, 5.2, 0, Math.PI * 2);
        cCtx.fill();

        // Crisp inner tie-cone center
        cCtx.fillStyle = 'rgba(28, 30, 36, 0.42)';
        cCtx.beginPath();
        cCtx.arc(tx, ty, 2.2, 0, Math.PI * 2);
        cCtx.fill();

        // Subtle lower rim highlight catching overhead skylight
        cCtx.strokeStyle = 'rgba(255, 252, 245, 0.18)';
        cCtx.lineWidth = 0.9;
        cCtx.beginPath();
        cCtx.arc(tx, ty + 0.6, 2.7, 0.15 * Math.PI, 0.85 * Math.PI);
        cCtx.stroke();

        // Roughness & bump depression at tie cone
        rCtx.fillStyle = 'rgba(242, 242, 242, 0.8)';
        rCtx.beginPath();
        rCtx.arc(tx, ty, 3.8, 0, Math.PI * 2);
        rCtx.fill();

        bCtx.fillStyle = 'rgba(64, 64, 64, 0.85)';
        bCtx.beginPath();
        bCtx.arc(tx, ty, 3.1, 0, Math.PI * 2);
        bCtx.fill();
      }
    }
  }

  // Formwork panel seam grooves (crisp recessed architectural joints + delicate upper highlight)
  const drawSeamH = (y: number) => {
    cCtx.fillStyle = 'rgba(22, 25, 31, 0.36)';
    cCtx.fillRect(0, y, size, 2);
    cCtx.fillStyle = 'rgba(255, 252, 246, 0.14)';
    cCtx.fillRect(0, (y + 2) % size, size, 1);

    rCtx.fillStyle = 'rgba(245, 245, 245, 0.88)';
    rCtx.fillRect(0, y, size, 2.5);

    bCtx.fillStyle = 'rgba(54, 54, 54, 0.9)';
    bCtx.fillRect(0, y, size, 2);
  };
  const drawSeamV = (x: number) => {
    cCtx.fillStyle = 'rgba(22, 25, 31, 0.34)';
    cCtx.fillRect(x, 0, 2, size);
    cCtx.fillStyle = 'rgba(255, 252, 246, 0.12)';
    cCtx.fillRect((x + 2) % size, 0, 1, size);

    rCtx.fillStyle = 'rgba(245, 245, 245, 0.88)';
    rCtx.fillRect(x, 0, 2.5, size);

    bCtx.fillStyle = 'rgba(58, 58, 58, 0.88)';
    bCtx.fillRect(x, 0, 2, size);
  };

  drawSeamH(0);
  drawSeamH(panelH);
  drawSeamV(0);
  drawSeamV(panelW);

  // Whisper-fine micro-cement tactile grain (restrained ±1.5% — never noisy)
  const imgData = cCtx.getImageData(0, 0, size, size);
  const bumpData = bCtx.getImageData(0, 0, size, size);
  const d = imgData.data;
  const bd = bumpData.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (rand() - 0.5) * 7.0;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n));

    const bn = (rand() - 0.5) * 10.0;
    const bv = Math.max(0, Math.min(255, bd[i] + bn));
    bd[i] = bd[i + 1] = bd[i + 2] = bv;
  }
  cCtx.putImageData(imgData, 0, 0);
  bCtx.putImageData(bumpData, 0, 0);

  const configureTex = (canvas: HTMLCanvasElement, isSRGB: boolean): THREE.CanvasTexture => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = 4;
    tex.colorSpace = isSRGB ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  };

  cachedConcreteTextures = {
    map: configureTex(colorCanvas, true),
    roughnessMap: configureTex(roughCanvas, false),
    bumpMap: configureTex(bumpCanvas, false),
  };
  return cachedConcreteTextures;
}

/** Clones the shared concrete texture set with independent UV tiling for a specific wall/prop surface. */
export function cloneConcreteTextures(repeatX: number, repeatY: number): ConcreteTextureSet {
  const base = getArchitecturalConcreteTextures();
  const cloneWithRepeat = (tex: THREE.CanvasTexture): THREE.CanvasTexture => {
    const c = tex.clone();
    c.repeat.set(repeatX, repeatY);
    c.needsUpdate = true;
    return c;
  };
  return {
    map: cloneWithRepeat(base.map),
    roughnessMap: cloneWithRepeat(base.roughnessMap),
    bumpMap: cloneWithRepeat(base.bumpMap),
  };
}

// Official AI Compass vector mark paths (viewBox="0 0 64 64")
const AI_COMPASS_FACET_PATH = 'M42.98 16.84 36.3 38.08 18.77 52 26.32 30.98Z';
const AI_COMPASS_SOLID_PATH =
  'M37.59 38.97 33.4 42.3 37.92 52 46 52 46 49.8 38.85 34.95ZM4.4 52 13.55 52 25.73 25.85 26.95 28.47 33.19 23.18 28.72 13.88A3.32 3.32 0 0 0 22.74 13.88ZM48 52 48 15.37A3.37 3.37 0 0 1 51.37 12L56 12 56 52ZM42.98 16.84 36.3 38.08 30.97 34.29ZM18.77 52 26.32 30.98 30.97 34.29Z';

interface WallBrandingTextureSet {
  darkMap: THREE.CanvasTexture;
  lightMap: THREE.CanvasTexture;
  roughnessMap: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
  metalnessMap: THREE.CanvasTexture;
}

let cachedWallBrandingTextures: WallBrandingTextureSet | null = null;

/**
 * Generates high-resolution (1536x384) cast-in-concrete relief + flush brushed architectural metal
 * inlay PBR textures for the official AI Compass upper gym wall branding.
 * Soft perimeter alpha feathering blends the poured concrete surround seamlessly into the wall.
 */
function getWallBrandingTextures(): WallBrandingTextureSet {
  if (cachedWallBrandingTextures) return cachedWallBrandingTextures;

  const W = 1536;
  const H = 448;
  // Physical wall panels are ~4.09:1 (5.4x1.32 m and 4.6x1.12 m) vs 3.43:1 canvas; pre-compensate X so mark & type stay 1:1
  const xComp = (W / H) / (5.4 / 1.32);
  const facetPath = new Path2D(AI_COMPASS_FACET_PATH);
  const solidPath = new Path2D(AI_COMPASS_SOLID_PATH);

  const TITLE_TEXT = 'AI COMPASS';
  const SUB_TEXT = 'ashutoshai.com/ai-compass';
  const CREDIT_TEXT = 'Built using NVIDIA MotionBricks  ·  SOMA FP16 Neural Motion';
  const FONT_STACK =
    '"SF Pro Display", "Inter", -apple-system, BlinkMacSystemFont, "Helvetica Neue", system-ui, sans-serif';

  const measureTracked = (ctx: CanvasRenderingContext2D, text: string, tracking: number): number => {
    let total = 0;
    for (let i = 0; i < text.length; i++) {
      total += ctx.measureText(text[i]).width;
      if (i < text.length - 1) total += tracking;
    }
    return total;
  };

  const drawTracked = (
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    tracking: number,
    scaleX: number,
    strokeOnly = false
  ): void => {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(scaleX, 1);
    let cursor = 0;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (strokeOnly) {
        ctx.strokeText(ch, cursor, 0);
      } else {
        ctx.fillText(ch, cursor, 0);
      }
      cursor += ctx.measureText(ch).width + tracking;
    }
    ctx.restore();
  };

  // Compute mathematically centered lockup geometry once so albedo, bump, roughness & metalness align sub-pixel
  const probeCanvas = document.createElement('canvas');
  const probeCtx = probeCanvas.getContext('2d')!;

  let titleSize = 104;
  let titleTracking = 18;
  let subSize = 36;
  let subTracking = 6.0;
  let creditSize = 27;
  let creditTracking = 3.6;
  let markScaleY = 3.85;
  let markScaleX = markScaleY * xComp;
  let gapBeforeDivider = 40;
  let dividerW = 2.6;
  let gapAfterDivider = 42;

  const computeLockupMetrics = () => {
    const titleFont = `600 ${titleSize}px ${FONT_STACK}`;
    const subFont = `600 ${subSize}px ${FONT_STACK}`;
    const creditFont = `500 ${creditSize}px ${FONT_STACK}`;
    probeCtx.font = titleFont;
    const titleW = measureTracked(probeCtx, TITLE_TEXT, titleTracking) * xComp;
    probeCtx.font = subFont;
    const subW = measureTracked(probeCtx, SUB_TEXT, subTracking) * xComp;
    probeCtx.font = creditFont;
    const creditW = measureTracked(probeCtx, CREDIT_TEXT, creditTracking) * xComp;
    const textColW = Math.max(titleW, subW);
    const markW = 51.6 * markScaleX;
    const markH = 40.0 * markScaleY;
    const totalW = markW + gapBeforeDivider + dividerW + gapAfterDivider + textColW;
    return { titleFont, subFont, creditFont, titleW, subW, creditW, textColW, markW, markH, totalW };
  };

  let metrics = computeLockupMetrics();
  if (metrics.totalW > 1180) {
    const fit = 1180 / metrics.totalW;
    titleSize = Math.round(titleSize * fit);
    titleTracking *= fit;
    subSize = Math.round(subSize * fit);
    subTracking *= fit;
    markScaleY *= fit;
    markScaleX = markScaleY * xComp;
    gapBeforeDivider *= fit;
    gapAfterDivider *= fit;
    metrics = computeLockupMetrics();
  }
  if (metrics.creditW > 1220) {
    const cFit = 1220 / metrics.creditW;
    creditSize = Math.round(creditSize * cFit);
    creditTracking *= cFit;
    metrics = computeLockupMetrics();
  }

  const topRowCenterY = 154;
  const lockupLeft = (W - metrics.totalW) * 0.5;
  const markLeft = lockupLeft;
  const markTop = topRowCenterY - metrics.markH * 0.5;
  const markDrawX = markLeft - 4.4 * markScaleX;
  const markDrawY = markTop - 12.0 * markScaleY;
  const dividerX = markLeft + metrics.markW + gapBeforeDivider;
  const dividerH = metrics.markH * 0.84;
  const dividerY = topRowCenterY - dividerH * 0.5;
  const textX = dividerX + dividerW + gapAfterDivider;
  const titleX = textX + (metrics.textColW - metrics.titleW) * 0.5;
  const subX = textX + (metrics.textColW - metrics.subW) * 0.5;
  const titleY = topRowCenterY + 6;
  const subY = titleY + Math.round(subSize * 1.62);
  const hRuleW = Math.min(Math.max(metrics.totalW, metrics.creditW) * 0.92, 1140);
  const hRuleX = (W - hRuleW) * 0.5;
  const hRuleY = 270;
  const hRuleH = 2.0;
  const creditX = (W - metrics.creditW) * 0.5;
  const creditY = 334;

  const drawMarkPath = (
    ctx: CanvasRenderingContext2D,
    path: Path2D,
    dx = 0,
    dy = 0,
    strokePx = 0
  ): void => {
    ctx.save();
    ctx.translate(markDrawX + dx, markDrawY + dy);
    ctx.scale(markScaleX, markScaleY);
    if (strokePx > 0) {
      ctx.lineWidth = strokePx / markScaleY;
      ctx.lineJoin = 'round';
      ctx.stroke(path);
    } else {
      ctx.fill(path);
    }
    ctx.restore();
  };

  const drawAllTextAndDivider = (
    ctx: CanvasRenderingContext2D,
    dx = 0,
    dy = 0,
    strokePx = 0
  ): void => {
    ctx.save();
    ctx.textBaseline = 'alphabetic';
    if (strokePx > 0) {
      ctx.lineWidth = strokePx;
      ctx.lineJoin = 'round';
      ctx.strokeRect(dividerX + dx, dividerY + dy, dividerW, dividerH);
      ctx.strokeRect(hRuleX + dx, hRuleY + dy, hRuleW, hRuleH);
      ctx.font = metrics.titleFont;
      drawTracked(ctx, TITLE_TEXT, titleX + dx, titleY + dy, titleTracking, xComp, true);
      ctx.font = metrics.subFont;
      drawTracked(ctx, SUB_TEXT, subX + dx, subY + dy, subTracking, xComp, true);
      ctx.font = metrics.creditFont;
      drawTracked(ctx, CREDIT_TEXT, creditX + dx, creditY + dy, creditTracking, xComp, true);
    } else {
      ctx.fillRect(dividerX + dx, dividerY + dy, dividerW, dividerH);
      ctx.fillRect(hRuleX + dx, hRuleY + dy, hRuleW, hRuleH);
      ctx.font = metrics.titleFont;
      drawTracked(ctx, TITLE_TEXT, titleX + dx, titleY + dy, titleTracking, xComp, false);
      ctx.font = metrics.subFont;
      drawTracked(ctx, SUB_TEXT, subX + dx, subY + dy, subTracking, xComp, false);
      ctx.font = metrics.creditFont;
      drawTracked(ctx, CREDIT_TEXT, creditX + dx, creditY + dy, creditTracking, xComp, false);
    }
    ctx.restore();
  };

  // Precompute soft outer perimeter alpha mask so concrete surround blends invisibly into wall
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = W;
  maskCanvas.height = H;
  const maskCtx = maskCanvas.getContext('2d')!;
  const maskImg = maskCtx.createImageData(W, H);
  const fadeX = 104;
  const fadeY = 46;
  const smooth = (t: number): number => {
    const c = Math.max(0, Math.min(1, t));
    return c * c * (3 - 2 * c);
  };
  for (let y = 0; y < H; y++) {
    const fy = smooth(Math.min(y, H - 1 - y) / fadeY);
    for (let x = 0; x < W; x++) {
      const fx = smooth(Math.min(x, W - 1 - x) / fadeX);
      const idx = (y * W + x) * 4;
      maskImg.data[idx] = 255;
      maskImg.data[idx + 1] = 255;
      maskImg.data[idx + 2] = 255;
      maskImg.data[idx + 3] = Math.round(fx * fy * 255);
    }
  }
  maskCtx.putImageData(maskImg, 0, 0);

  // 1. ROUGHNESS MAP (linear: 0.94 matte poured concrete surround vs 0.26 smooth brushed metal inlay)
  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = W;
  roughCanvas.height = H;
  const rCtx = roughCanvas.getContext('2d')!;
  const rRand = mulberry32(77311);
  rCtx.fillStyle = 'rgb(240, 240, 240)'; // 0.941 matte concrete
  rCtx.fillRect(0, 0, W, H);
  for (let i = 0; i < 40; i++) {
    const y = rRand() * H;
    const h = 2 + rRand() * 6;
    rCtx.fillStyle = rRand() > 0.5 ? 'rgba(228, 228, 228, 0.25)' : 'rgba(248, 248, 248, 0.25)';
    rCtx.fillRect(0, y, W, h);
  }
  // Rough recessed concrete bevel rim around inlay
  rCtx.strokeStyle = 'rgb(248, 248, 248)';
  drawMarkPath(rCtx, facetPath, 0, 0, 5.5);
  drawMarkPath(rCtx, solidPath, 0, 0, 5.5);
  drawAllTextAndDivider(rCtx, 0, 0, 5.0);
  // Satin translucent compass facet (0.56 roughness)
  rCtx.fillStyle = 'rgb(142, 142, 142)';
  drawMarkPath(rCtx, facetPath);
  // Brushed architectural metal inlay (0.26 roughness = rgb(66,66,66)) with subtle brushed sheen streaks
  const roughInlay = document.createElement('canvas');
  roughInlay.width = W;
  roughInlay.height = H;
  const riCtx = roughInlay.getContext('2d')!;
  riCtx.fillStyle = 'rgb(66, 66, 66)';
  drawMarkPath(riCtx, solidPath);
  drawAllTextAndDivider(riCtx);
  riCtx.globalCompositeOperation = 'source-in';
  riCtx.fillStyle = 'rgb(66, 66, 66)';
  riCtx.fillRect(0, 0, W, H);
  for (let y = 0; y < H; y += 2) {
    const v = Math.round(62 + (rRand() - 0.5) * 14);
    riCtx.fillStyle = `rgb(${v}, ${v}, ${v})`;
    riCtx.fillRect(0, y, W, 1);
  }
  rCtx.drawImage(roughInlay, 0, 0);

  // 2. METALNESS MAP (linear: 0.04 concrete surround vs 0.78 brushed metal inlay)
  const metalCanvas = document.createElement('canvas');
  metalCanvas.width = W;
  metalCanvas.height = H;
  const mCtx = metalCanvas.getContext('2d')!;
  mCtx.fillStyle = 'rgb(10, 10, 10)'; // 0.039 matte concrete
  mCtx.fillRect(0, 0, W, H);
  mCtx.fillStyle = 'rgb(82, 82, 82)'; // 0.32 satin facet
  drawMarkPath(mCtx, facetPath);
  mCtx.fillStyle = 'rgb(199, 199, 199)'; // 0.780 brushed architectural metal
  drawMarkPath(mCtx, solidPath);
  drawAllTextAndDivider(mCtx);

  // 3. BUMP MAP (linear neutral 128 concrete, recessed cast-in-concrete groove + flush brushed inlay relief)
  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = W;
  bumpCanvas.height = H;
  const bCtx = bumpCanvas.getContext('2d')!;
  const bRand = mulberry32(55109);
  bCtx.fillStyle = 'rgb(128, 128, 128)';
  bCtx.fillRect(0, 0, W, H);

  // Subtle poured-concrete micro-grain matching wall surface
  const bBase = bCtx.getImageData(0, 0, W, H);
  for (let i = 0; i < bBase.data.length; i += 4) {
    const bv = Math.max(0, Math.min(255, 128 + (bRand() - 0.5) * 8));
    bBase.data[i] = bBase.data[i + 1] = bBase.data[i + 2] = bv;
  }
  bCtx.putImageData(bBase, 0, 0);

  // Outer chamfered cast-concrete bevel depression
  bCtx.save();
  bCtx.strokeStyle = 'rgba(84, 84, 84, 0.75)';
  bCtx.shadowColor = 'rgba(72, 72, 72, 0.55)';
  bCtx.shadowBlur = 4;
  drawMarkPath(bCtx, facetPath, 0, 0, 8.0);
  drawMarkPath(bCtx, solidPath, 0, 0, 8.0);
  drawAllTextAndDivider(bCtx, 0, 0, 6.5);
  bCtx.restore();

  // Crisp inner debossed concrete channel
  bCtx.strokeStyle = 'rgb(42, 42, 42)';
  drawMarkPath(bCtx, facetPath, 0, 0, 3.6);
  drawMarkPath(bCtx, solidPath, 0, 0, 3.6);
  drawAllTextAndDivider(bCtx, 0, 0, 3.2);

  // Recessed translucent facet pocket
  bCtx.fillStyle = 'rgb(86, 86, 86)';
  drawMarkPath(bCtx, facetPath);

  // Flush-recessed brushed metal plane with fine horizontal brushed grain
  const bumpInlay = document.createElement('canvas');
  bumpInlay.width = W;
  bumpInlay.height = H;
  const biCtx = bumpInlay.getContext('2d')!;
  biCtx.fillStyle = 'rgb(108, 108, 108)';
  drawMarkPath(biCtx, solidPath);
  drawAllTextAndDivider(biCtx);
  biCtx.globalCompositeOperation = 'source-in';
  biCtx.fillStyle = 'rgb(108, 108, 108)';
  biCtx.fillRect(0, 0, W, H);
  for (let y = 0; y < H; y += 2) {
    const v = Math.round(108 + (bRand() - 0.5) * 10);
    biCtx.fillStyle = `rgb(${v}, ${v}, ${v})`;
    biCtx.fillRect(0, y, W, 1);
  }
  bCtx.drawImage(bumpInlay, 0, 0);

  // 4. THEME-HARMONIZED ALBEDO MAPS (Dark Brutalist champagne-gunmetal vs Light Travertine dark bronze)
  const buildAlbedoMap = (mode: 'dark' | 'light'): THREE.CanvasTexture => {
    const isLight = mode === 'light';
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d')!;
    const rand = mulberry32(isLight ? 88421 : 31907);

    // Exact linear-matched poured concrete surround color (matches backWallMat = concreteWall * #e5e3df)
    const wallCol = new THREE.Color(STUDIO_THEMES[mode].concreteWall).multiply(
      new THREE.Color('#e5e3df')
    );
    ctx.fillStyle = `#${wallCol.getHexString()}`;
    ctx.fillRect(0, 0, W, H);

    // Subtle hydration mottling & horizontal board-form streaks so concrete feels cast in place
    for (let i = 0; i < 28; i++) {
      const cx = rand() * W;
      const cy = rand() * H;
      const r = 60 + rand() * 140;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      const bright = rand() > 0.5;
      const col = bright
        ? isLight
          ? 'rgba(255, 251, 244, 0.045)'
          : 'rgba(255, 248, 238, 0.024)'
        : isLight
          ? 'rgba(58, 52, 44, 0.04)'
          : 'rgba(6, 8, 12, 0.05)';
      g.addColorStop(0, col);
      g.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
    }

    // Whisper-fine micro-cement tactile grain
    const cImg = ctx.getImageData(0, 0, W, H);
    const cd = cImg.data;
    const grainAmp = isLight ? 5.5 : 4.2;
    for (let i = 0; i < cd.length; i += 4) {
      const n = (rand() - 0.5) * grainAmp;
      cd[i] = Math.max(0, Math.min(255, cd[i] + n));
      cd[i + 1] = Math.max(0, Math.min(255, cd[i + 1] + n));
      cd[i + 2] = Math.max(0, Math.min(255, cd[i + 2] + n));
    }
    ctx.putImageData(cImg, 0, 0);

    // Lower rim warm specular bevel highlight (overhead skylight grazing bottom cast-concrete lip)
    ctx.save();
    ctx.fillStyle = isLight ? 'rgba(255, 251, 242, 0.44)' : 'rgba(232, 210, 176, 0.16)';
    ctx.strokeStyle = isLight ? 'rgba(255, 251, 242, 0.44)' : 'rgba(232, 210, 176, 0.16)';
    drawMarkPath(ctx, facetPath, 0, 2.4);
    drawMarkPath(ctx, solidPath, 0, 2.4);
    drawAllTextAndDivider(ctx, 0, 2.2);
    ctx.restore();

    // Debossed cast-in-concrete shadow groove (upper-left recess catching overhead skylight shadow)
    ctx.save();
    const grooveShadow = isLight ? 'rgba(42, 36, 28, 0.52)' : 'rgba(5, 6, 9, 0.78)';
    ctx.fillStyle = grooveShadow;
    ctx.strokeStyle = grooveShadow;
    ctx.shadowColor = isLight ? 'rgba(34, 29, 22, 0.36)' : 'rgba(0, 0, 0, 0.65)';
    ctx.shadowBlur = 4.5;
    drawMarkPath(ctx, facetPath, -1.2, -2.4, 3.6);
    drawMarkPath(ctx, solidPath, -1.2, -2.4, 3.6);
    drawAllTextAndDivider(ctx, -1.0, -2.2, 3.0);
    drawMarkPath(ctx, facetPath, -1.0, -2.0);
    drawMarkPath(ctx, solidPath, -1.0, -2.0);
    drawAllTextAndDivider(ctx, -0.8, -1.8);
    ctx.restore();

    // Build brushed architectural metal fill layer (champagne-gunmetal in dark, oil-rubbed dark bronze in light)
    const metalLayer = document.createElement('canvas');
    metalLayer.width = W;
    metalLayer.height = H;
    const mlCtx = metalLayer.getContext('2d')!;

    // Helper to fill brushed metal clipped to current alpha mask
    const paintBrushedMetal = (
      targetCtx: CanvasRenderingContext2D,
      stops: [string, string, string],
      streakAlpha: number
    ) => {
      targetCtx.globalCompositeOperation = 'source-in';
      const grad = targetCtx.createLinearGradient(0, markTop - 10, 0, creditY + 24);
      grad.addColorStop(0, stops[0]);
      grad.addColorStop(0.48, stops[1]);
      grad.addColorStop(1, stops[2]);
      targetCtx.fillStyle = grad;
      targetCtx.fillRect(0, 0, W, H);
      for (let y = 0; y < H; y += 2) {
        const light = rand() > 0.5;
        targetCtx.fillStyle = light
          ? `rgba(255, 248, 236, ${streakAlpha})`
          : `rgba(12, 10, 8, ${streakAlpha * 1.15})`;
        targetCtx.fillRect(0, y, W, 1);
      }
      targetCtx.globalCompositeOperation = 'source-over';
    };

    // 4a. Official AI Compass translucent facet (opacity 0.35)
    mlCtx.clearRect(0, 0, W, H);
    mlCtx.fillStyle = '#ffffff';
    drawMarkPath(mlCtx, facetPath);
    paintBrushedMetal(
      mlCtx,
      isLight ? ['#5c5144', '#493f34', '#3c332a'] : ['#cfc2aa', '#b4a68e', '#948772'],
      isLight ? 0.04 : 0.055
    );
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.drawImage(metalLayer, 0, 0);
    ctx.restore();

    // 4b. Official AI Compass solid facets + primary architectural wordmark (`AI COMPASS`)
    mlCtx.clearRect(0, 0, W, H);
    mlCtx.fillStyle = '#ffffff';
    drawMarkPath(mlCtx, solidPath);
    mlCtx.textBaseline = 'alphabetic';
    mlCtx.font = metrics.titleFont;
    drawTracked(mlCtx, TITLE_TEXT, titleX, titleY, titleTracking, xComp, false);
    paintBrushedMetal(
      mlCtx,
      isLight ? ['#584d40', '#443b31', '#352e26'] : ['#d1c4ac', '#b6a890', '#968973'],
      isLight ? 0.045 : 0.06
    );
    ctx.save();
    ctx.globalAlpha = isLight ? 0.92 : 0.9;
    ctx.drawImage(metalLayer, 0, 0);
    ctx.restore();

    // 4c. Centered URL sub-line (`ashutoshai.com/ai-compass`)
    mlCtx.clearRect(0, 0, W, H);
    mlCtx.fillStyle = '#ffffff';
    mlCtx.textBaseline = 'alphabetic';
    mlCtx.font = metrics.subFont;
    drawTracked(mlCtx, SUB_TEXT, subX, subY, subTracking, xComp, false);
    paintBrushedMetal(
      mlCtx,
      isLight ? ['#65594c', '#52483c', '#433a30'] : ['#beb29e', '#a59985', '#8b806e'],
      isLight ? 0.035 : 0.045
    );
    ctx.save();
    ctx.globalAlpha = isLight ? 0.84 : 0.82;
    ctx.drawImage(metalLayer, 0, 0);
    ctx.restore();

    // 4d. Centered NVIDIA MotionBricks credit line
    mlCtx.clearRect(0, 0, W, H);
    mlCtx.fillStyle = '#ffffff';
    mlCtx.textBaseline = 'alphabetic';
    mlCtx.font = metrics.creditFont;
    drawTracked(mlCtx, CREDIT_TEXT, creditX, creditY, creditTracking, xComp, false);
    paintBrushedMetal(
      mlCtx,
      isLight ? ['#6d6153', '#5a5043', '#494035'] : ['#b6aa96', '#9d917e', '#827766'],
      isLight ? 0.03 : 0.04
    );
    ctx.save();
    ctx.globalAlpha = isLight ? 0.78 : 0.76;
    ctx.drawImage(metalLayer, 0, 0);
    ctx.restore();

    // Architectural hairline divider rules (vertical mark divider + subtle horizontal credit separator)
    ctx.save();
    ctx.fillStyle = isLight ? 'rgba(68, 59, 49, 0.38)' : 'rgba(186, 172, 148, 0.34)';
    ctx.fillRect(dividerX, dividerY, dividerW, dividerH);
    ctx.fillStyle = isLight ? 'rgba(68, 59, 49, 0.24)' : 'rgba(186, 172, 148, 0.22)';
    ctx.fillRect(hRuleX, hRuleY, hRuleW, hRuleH);
    ctx.restore();

    // Apply soft perimeter alpha feather so the slab melts seamlessly into the poured concrete wall
    ctx.save();
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.restore();

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.anisotropy = 8;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  };

  const configureLinear = (canvas: HTMLCanvasElement): THREE.CanvasTexture => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.anisotropy = 8;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
  };

  cachedWallBrandingTextures = {
    darkMap: buildAlbedoMap('dark'),
    lightMap: buildAlbedoMap('light'),
    roughnessMap: configureLinear(roughCanvas),
    bumpMap: configureLinear(bumpCanvas),
    metalnessMap: configureLinear(metalCanvas),
  };
  return cachedWallBrandingTextures;
}

/** Subtle vulcanized athletic-rubber texture for the inlaid floor training lane. */
function createRubberLaneTextures(laneLength: number): {
  map: THREE.CanvasTexture;
  roughnessMap: THREE.CanvasTexture;
} {
  const size = 256;
  const rand = mulberry32(99071);

  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = size;
  colorCanvas.height = size;
  const cCtx = colorCanvas.getContext('2d')!;

  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = size;
  roughCanvas.height = size;
  const rCtx = roughCanvas.getContext('2d')!;

  cCtx.fillStyle = '#787d87';
  cCtx.fillRect(0, 0, size, size);

  rCtx.fillStyle = 'rgb(238, 238, 238)';
  rCtx.fillRect(0, 0, size, size);

  // Subtle matte vulcanized micro-grain
  const cImg = cCtx.getImageData(0, 0, size, size);
  const rImg = rCtx.getImageData(0, 0, size, size);
  for (let i = 0; i < cImg.data.length; i += 4) {
    const g = (rand() - 0.5) * 12;
    const fleck = rand() > 0.99 ? 14 : 0; // ultra-subtle fine vulcanized crumb fleck
    cImg.data[i] = Math.max(0, Math.min(255, cImg.data[i] + g + fleck));
    cImg.data[i + 1] = Math.max(0, Math.min(255, cImg.data[i + 1] + g + fleck));
    cImg.data[i + 2] = Math.max(0, Math.min(255, cImg.data[i + 2] + g + fleck));

    const rv = Math.max(218, Math.min(252, 238 + (rand() - 0.5) * 14));
    rImg.data[i] = rImg.data[i + 1] = rImg.data[i + 2] = rv;
  }
  cCtx.putImageData(cImg, 0, 0);
  rCtx.putImageData(rImg, 0, 0);

  // Subtle transverse athletic mat seam lines
  cCtx.fillStyle = 'rgba(14, 16, 20, 0.22)';
  cCtx.fillRect(0, 0, size, 1.5);
  cCtx.fillRect(0, size * 0.5, size, 1);

  const repeatY = Math.max(4.5, Math.round(laneLength * 0.62 * 10) / 10);

  const map = new THREE.CanvasTexture(colorCanvas);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(1.5, repeatY);
  map.anisotropy = 4;
  map.colorSpace = THREE.SRGBColorSpace;

  const roughnessMap = new THREE.CanvasTexture(roughCanvas);
  roughnessMap.wrapS = THREE.RepeatWrapping;
  roughnessMap.wrapT = THREE.RepeatWrapping;
  roughnessMap.repeat.set(1.5, repeatY);
  roughnessMap.anisotropy = 4;
  roughnessMap.colorSpace = THREE.NoColorSpace;

  return { map, roughnessMap };
}

/** Builds a tall vertical architectural column with subtle chamfered arris edges that catch rim light. */
function createChamferedColumnGeometry(width: number, depth: number, height: number, bevel = 0.024): THREE.BufferGeometry {
  const hw = width * 0.5 - bevel;
  const hd = depth * 0.5 - bevel;
  const shape = new THREE.Shape();
  shape.moveTo(-hw, -hd);
  shape.lineTo(hw, -hd);
  shape.lineTo(hw, hd);
  shape.lineTo(-hw, hd);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: height - bevel * 2,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    curveSegments: 1,
  });
  // Extrude runs along +Z; rotate so column height runs cleanly along +Y from 0 to height
  geo.rotateX(-Math.PI / 2);
  geo.translate(0, bevel, 0);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds the monumental Nike/Apple-industrial Brutalist Architectural Mocap Pavilion
 * matching targets/t01.png and PAVILION_SPEC / STUDIO_THEMES.
 */
export function createMocapPavilion(): MocapPavilionHandle {
  const group = new THREE.Group();
  group.name = 'MocapPavilion';

  const initialTheme = STUDIO_THEMES.dark;

  // Derived spacious 4-wall nave bounds from PAVILION_SPEC (enclosed cleanly on all 4 sides!)
  const ceilH = PAVILION_SPEC.ceilingHeight;
  const wallH = ceilH + 0.6;
  const backZ = PAVILION_SPEC.backWallZ;
  const frontZ = PAVILION_SPEC.frontWallZ;
  const leftX = PAVILION_SPEC.leftWallX;
  const rightX = PAVILION_SPEC.rightColonnadeX;

  const naveMinZ = backZ - 0.4;
  const naveMaxZ = frontZ + 0.4;
  const naveDepthZ = naveMaxZ - naveMinZ;
  const naveCenterZ = (naveMinZ + naveMaxZ) * 0.5;
  const outerRightX = rightX + 2.8;

  const backMinX = leftX - 0.8;
  const backMaxX = outerRightX + 0.8;
  const backWallW = backMaxX - backMinX;
  const backWallCenterX = (backMinX + backMaxX) * 0.5;

  // Shared PBR material sets (world-proportional concrete & rubber UV repeat for expanded nave)
  const backWallTex = cloneConcreteTextures(Math.max(5.2, Math.round((backWallW / 3.0) * 10) / 10), 1.75);
  const leftWallTex = cloneConcreteTextures(Math.max(6.0, Math.round((naveDepthZ / 3.1) * 10) / 10), 1.75);
  const columnTex = cloneConcreteTextures(0.52, 0.52);
  const baffleTex = cloneConcreteTextures(
    Math.max(2.4, Math.round((PAVILION_SPEC.skylightBays[0].length + 0.9) * 0.34 * 10) / 10),
    0.78
  );
  const rubberTex = createRubberLaneTextures(PAVILION_SPEC.trainingLane.length);

  const backWallMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.concreteWall),
    map: backWallTex.map,
    roughnessMap: backWallTex.roughnessMap,
    bumpMap: backWallTex.bumpMap,
    bumpScale: 0.012,
    roughness: 0.92,
    metalness: 0.04,
  });

  const leftWallMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.concreteWall).multiplyScalar(0.84),
    map: leftWallTex.map,
    roughnessMap: leftWallTex.roughnessMap,
    bumpMap: leftWallTex.bumpMap,
    bumpScale: 0.012,
    roughness: 0.96,
    metalness: 0.03,
    envMapIntensity: 0.55,
  });

  const recessWallMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.concreteWall).multiplyScalar(0.36),
    map: backWallTex.map,
    roughnessMap: backWallTex.roughnessMap,
    bumpMap: backWallTex.bumpMap,
    bumpScale: 0.008,
    roughness: 0.97,
    metalness: 0.03,
    envMapIntensity: 0.35,
  });

  const columnMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.concreteColumn),
    map: columnTex.map,
    roughnessMap: columnTex.roughnessMap,
    bumpMap: columnTex.bumpMap,
    bumpScale: 0.01,
    roughness: 0.88,
    metalness: 0.06,
  });

  const baffleMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.concreteColumn).multiplyScalar(0.94),
    map: baffleTex.map,
    roughnessMap: baffleTex.roughnessMap,
    bumpMap: baffleTex.bumpMap,
    bumpScale: 0.008,
    roughness: 0.9,
    metalness: 0.05,
  });

  const shadowRevealMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#12161e'),
    roughness: 0.88,
    metalness: 0.14,
  });

  const laneEdgeMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#363e4b'),
    roughness: 0.56,
    metalness: 0.65,
    envMapIntensity: 0.45,
  });

  const rubberLaneMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.rubberLane),
    map: rubberTex.map,
    roughnessMap: rubberTex.roughnessMap,
    roughness: 0.96,
    metalness: 0.02,
    envMapIntensity: 0.22,
  });

  const skylightMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(initialTheme.skylightEmissive),
    emissive: new THREE.Color(initialTheme.skylightEmissive),
    emissiveIntensity: initialTheme.skylightIntensity,
    roughness: 0.22,
    metalness: 0.0,
  });

  // Cast-in-concrete relief + flush brushed architectural metal AI Compass wall branding materials
  const brandingTex = getWallBrandingTextures();
  const backBrandingMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#ffffff'),
    map: brandingTex.darkMap,
    roughnessMap: brandingTex.roughnessMap,
    bumpMap: brandingTex.bumpMap,
    bumpScale: 0.022,
    metalnessMap: brandingTex.metalnessMap,
    roughness: 1.0,
    metalness: 1.0,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
    envMapIntensity: 0.95,
  });

  const leftBrandingMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#ffffff').multiplyScalar(0.84),
    map: brandingTex.darkMap,
    roughnessMap: brandingTex.roughnessMap,
    bumpMap: brandingTex.bumpMap,
    bumpScale: 0.022,
    metalnessMap: brandingTex.metalnessMap,
    roughness: 1.0,
    metalness: 1.0,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
    envMapIntensity: 0.75,
  });

  // =========================================================================
  // 1. BACK & FRONT ARCHITECTURAL CONCRETE WALLS (enclosed 4-wall refined pavilion)
  // =========================================================================
  const backWallMesh = new THREE.Mesh(new THREE.BoxGeometry(backWallW, wallH, 0.36), backWallMat);
  backWallMesh.position.set(backWallCenterX, wallH * 0.5, backZ - 0.18);
  backWallMesh.receiveShadow = true;
  group.add(backWallMesh);

  // Subtle refined framing piers flanking the skylight nave width (px = ±2.95)
  const pierH = ceilH - 0.12;
  const pierW = 0.36;
  const pierGeo = new THREE.BoxGeometry(pierW, pierH, 0.11);
  for (const px of [-2.95, 2.95]) {
    const pier = new THREE.Mesh(pierGeo, backWallMat);
    pier.position.set(px, pierH * 0.5, backZ + 0.055);
    pier.receiveShadow = true;
    group.add(pier);
  }

  // Fine architectural floor shadow-gap reveal & crisp mid-height reglet along back wall
  const backBaseboard = new THREE.Mesh(new THREE.BoxGeometry(backWallW, 0.05, 0.025), shadowRevealMat);
  backBaseboard.position.set(backWallCenterX, 0.025, backZ + 0.01);
  group.add(backBaseboard);

  const backRegletSpan = 2.95 * 2 - pierW;
  const backReglet = new THREE.Mesh(new THREE.BoxGeometry(backRegletSpan, 0.012, 0.014), shadowRevealMat);
  backReglet.position.set(0, ceilH * 0.52, backZ + 0.006);
  group.add(backReglet);

  // Primary Upper Back Concrete Wall Architectural Branding (flush between px = ±2.95 piers under backWashLight)
  const backWallBranding = new THREE.Mesh(new THREE.PlaneGeometry(5.4, 1.32), backBrandingMat);
  backWallBranding.name = 'AICompassBackWallBranding';
  backWallBranding.position.set(0, 3.12, backZ + 0.012);
  backWallBranding.receiveShadow = true;
  group.add(backWallBranding);

  // Enclosed 4th Front Wall (z = frontWallZ) so 360° orbit & +Z movement stay inside a finished hall
  const frontWallMesh = new THREE.Mesh(new THREE.BoxGeometry(backWallW, wallH, 0.36), leftWallMat);
  frontWallMesh.position.set(backWallCenterX, wallH * 0.5, frontZ + 0.18);
  frontWallMesh.receiveShadow = true;
  group.add(frontWallMesh);

  const frontBaseboard = new THREE.Mesh(new THREE.BoxGeometry(backWallW, 0.05, 0.025), shadowRevealMat);
  frontBaseboard.position.set(backWallCenterX, 0.025, frontZ - 0.01);
  group.add(frontBaseboard);

  const frontReglet = new THREE.Mesh(new THREE.BoxGeometry(backWallW, 0.012, 0.014), shadowRevealMat);
  frontReglet.position.set(0, ceilH * 0.52, frontZ - 0.006);
  group.add(frontReglet);

  // =========================================================================
  // 2. LEFT MONOLITHIC CONCRETE WALL (finer architectural formwork panels)
  // =========================================================================
  const leftWallMesh = new THREE.Mesh(new THREE.BoxGeometry(0.36, wallH, naveDepthZ), leftWallMat);
  leftWallMesh.position.set(leftX - 0.18, wallH * 0.5, naveCenterZ);
  leftWallMesh.receiveShadow = true;
  group.add(leftWallMesh);

  // Refined flush architectural concrete panels with crisp hairline shadow reveals
  const leftPanelCount = Math.max(8, Math.round((frontZ - backZ) / 2.8));
  const leftBayPitch = (frontZ - backZ) / leftPanelCount;
  const panelGap = 0.022;
  const lowerPanelH = ceilH * 0.48;
  const upperPanelH = ceilH - lowerPanelH - 0.42;
  const lowerPanelGeo = new THREE.BoxGeometry(0.014, lowerPanelH, leftBayPitch - panelGap);
  const upperPanelGeo = new THREE.BoxGeometry(0.014, upperPanelH, leftBayPitch - panelGap);

  for (let i = 0; i < leftPanelCount; i++) {
    const midZ = backZ + (i + 0.5) * leftBayPitch;
    const lowerPanel = new THREE.Mesh(lowerPanelGeo, leftWallMat);
    lowerPanel.position.set(leftX + 0.007, 0.065 + lowerPanelH * 0.5, midZ);
    lowerPanel.receiveShadow = true;

    const upperPanel = new THREE.Mesh(upperPanelGeo, leftWallMat);
    upperPanel.position.set(leftX + 0.007, 0.065 + lowerPanelH + panelGap + upperPanelH * 0.5, midZ);
    upperPanel.receiveShadow = true;

    group.add(lowerPanel, upperPanel);
  }

  // Fine flush top lintel soffit + floor shadow-gap reveal along left wall
  const leftLintelH = 0.28;
  const leftLintel = new THREE.Mesh(new THREE.BoxGeometry(0.16, leftLintelH, naveDepthZ), leftWallMat);
  leftLintel.position.set(leftX + 0.08, ceilH - leftLintelH * 0.5, naveCenterZ);
  group.add(leftLintel);

  const leftBaseboard = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.05, naveDepthZ), shadowRevealMat);
  leftBaseboard.position.set(leftX + 0.01, 0.025, naveCenterZ);
  group.add(leftBaseboard);

  // Secondary Upper Left Concrete Wall Architectural Relief Bay (flush into upper formwork panel, facing +X)
  const leftWallBranding = new THREE.Mesh(new THREE.PlaneGeometry(4.6, 1.12), leftBrandingMat);
  leftWallBranding.name = 'AICompassLeftWallBranding';
  leftWallBranding.position.set(leftX + 0.018, 3.05, -5.25);
  leftWallBranding.rotation.y = Math.PI / 2;
  leftWallBranding.receiveShadow = true;
  group.add(leftWallBranding);

  // =========================================================================
  // 3. RIGHT ARCHITECTURAL COLONNADE PILLARS (finer full-length colonnade along rightColonnadeX)
  // =========================================================================
  const colW = PAVILION_SPEC.columnWidth;
  const colD = PAVILION_SPEC.columnDepth;
  const colSpacing = PAVILION_SPEC.columnSpacingZ;
  const colStartZ = backZ + 0.9;
  const colCount = Math.floor((frontZ - 0.8 - colStartZ) / colSpacing) + 1;

  const colPlinthH = 0.06;
  const colCapH = 0.09;
  const colBeamBottomY = ceilH - 0.19;
  const colShaftH = colBeamBottomY - colPlinthH - colCapH;

  const colShaftGeo = createChamferedColumnGeometry(colW, colD, colShaftH, 0.022);
  const colPlinthGeo = new THREE.BoxGeometry(colW - 0.05, colPlinthH, colD - 0.05);
  const colCapitalGeo = new THREE.BoxGeometry(colW + 0.06, colCapH, colD + 0.06);

  for (let i = 0; i < colCount; i++) {
    const cz = colStartZ + i * colSpacing;
    const colGroup = new THREE.Group();
    colGroup.position.set(rightX, 0, cz);

    const plinth = new THREE.Mesh(colPlinthGeo, shadowRevealMat);
    plinth.position.y = colPlinthH * 0.5;
    plinth.receiveShadow = true;

    const shaft = new THREE.Mesh(colShaftGeo, columnMat);
    shaft.position.y = colPlinthH;
    shaft.castShadow = true;
    shaft.receiveShadow = true;

    const capital = new THREE.Mesh(colCapitalGeo, baffleMat);
    capital.position.y = colBeamBottomY - colCapH * 0.5;
    capital.receiveShadow = true;

    colGroup.add(plinth, shaft, capital);
    group.add(colGroup);
  }

  // Continuous overhead colonnade entablature beam tying all pillars together across naveDepthZ
  const colBeamH = 0.64;
  const colonnadeBeam = new THREE.Mesh(
    new THREE.BoxGeometry(colW + 0.34, colBeamH, naveDepthZ),
    baffleMat
  );
  colonnadeBeam.position.set(rightX, colBeamBottomY + colBeamH * 0.5, naveCenterZ);
  group.add(colonnadeBeam);

  // Deep dark recessed outer gallery wall behind the colonnade (dramatic shadow bays between pillars)
  const outerRightWall = new THREE.Mesh(new THREE.BoxGeometry(0.6, wallH, naveDepthZ), recessWallMat);
  outerRightWall.position.set(outerRightX + 0.3, wallH * 0.5, naveCenterZ);
  outerRightWall.receiveShadow = true;
  group.add(outerRightWall);

  // Matching dark recessed back-gallery end cap behind the colonnade so rear pillar slots stay deeply shaded
  const galleryBackW = outerRightX - rightX + 0.4;
  const galleryBackWall = new THREE.Mesh(new THREE.BoxGeometry(galleryBackW, wallH, 0.04), recessWallMat);
  galleryBackWall.position.set(rightX + galleryBackW * 0.5 - 0.1, wallH * 0.5, backZ + 0.02);
  galleryBackWall.receiveShadow = true;
  group.add(galleryBackWall);

  // =========================================================================
  // 4. OVERHEAD CONCRETE SKYLIGHT BAFFLES, RECESSED SKYLIGHTS & LIGHT SHAFTS
  // =========================================================================
  const bays = PAVILION_SPEC.skylightBays;
  const bayCenterZ = bays[0].z;
  const bayLength = bays[0].length;
  const bayWidth = bays[0].width;
  const bayPitchX = bays.length >= 2 ? bays[1].x - bays[0].x : 1.65;
  const baffleWidth = bayPitchX - bayWidth; // 1.65 - 1.08 = 0.57
  const baffleHeight = 0.92;
  const baffleBottomY = ceilH - 0.5; // 4.05 m
  const baffleCenterY = baffleBottomY + baffleHeight * 0.5;
  const baffleLength = bayLength + 0.9;

  // Derive 4 concrete ceiling baffle X coordinates directly from PAVILION_SPEC.skylightBays
  const baffleXs: number[] = [bays[0].x - bayPitchX * 0.5];
  for (let i = 0; i < bays.length - 1; i++) {
    baffleXs.push((bays[i].x + bays[i + 1].x) * 0.5);
  }
  baffleXs.push(bays[bays.length - 1].x + bayPitchX * 0.5); // [-2.475, -0.825, 0.825, 2.475]

  const baffleGeo = new THREE.BoxGeometry(baffleWidth, baffleHeight, baffleLength);
  for (const bx of baffleXs) {
    const baffle = new THREE.Mesh(baffleGeo, baffleMat);
    baffle.position.set(bx, baffleCenterY, bayCenterZ);
    baffle.receiveShadow = true;
    group.add(baffle);
  }

  // Exact coffer perimeter bounds for airtight ceiling sealing (zero sky leaks)
  const cofferMinX = baffleXs[0] - baffleWidth * 0.5;
  const cofferMaxX = baffleXs[baffleXs.length - 1] + baffleWidth * 0.5;
  const skylightFrameW = cofferMaxX - cofferMinX;
  const bayBackZ = bayCenterZ - bayLength * 0.5;
  const bayFrontZ = bayCenterZ + bayLength * 0.5;
  const ceilingSlabH = 0.62;
  const ceilingBottomY = ceilH - 0.12;
  const ceilingCenterY = ceilingBottomY + ceilingSlabH * 0.5;

  // Rear & front transverse header soffits sealing both ends of the skylight coffers
  const backHeaderMinZ = naveMinZ;
  const backHeaderDepth = Math.max(0.2, bayBackZ - backHeaderMinZ);
  const backHeader = new THREE.Mesh(
    new THREE.BoxGeometry(skylightFrameW, ceilingSlabH, backHeaderDepth),
    baffleMat
  );
  backHeader.position.set(0, ceilingCenterY + 0.004, (backHeaderMinZ + bayBackZ) * 0.5);
  group.add(backHeader);

  const frontHeaderDepth = Math.max(0.5, naveMaxZ - bayFrontZ);
  const frontHeader = new THREE.Mesh(
    new THREE.BoxGeometry(skylightFrameW, ceilingSlabH, frontHeaderDepth),
    baffleMat
  );
  frontHeader.position.set(0, ceilingCenterY + 0.004, (bayFrontZ + naveMaxZ) * 0.5);
  group.add(frontHeader);

  // Flanking ceiling slabs spanning seamlessly from outer baffles to leftWall & outerRightWall
  const leftCeilMinX = leftX - 0.6;
  const leftCeilMaxX = cofferMinX;
  const leftCeilingSlab = new THREE.Mesh(
    new THREE.BoxGeometry(leftCeilMaxX - leftCeilMinX, ceilingSlabH, naveDepthZ),
    baffleMat
  );
  leftCeilingSlab.position.set((leftCeilMinX + leftCeilMaxX) * 0.5, ceilingCenterY, naveCenterZ);
  group.add(leftCeilingSlab);

  const rightCeilMinX = cofferMaxX;
  const rightCeilMaxX = outerRightX + 0.6;
  const rightCeilingSlab = new THREE.Mesh(
    new THREE.BoxGeometry(rightCeilMaxX - rightCeilMinX, ceilingSlabH, naveDepthZ),
    baffleMat
  );
  rightCeilingSlab.position.set((rightCeilMinX + rightCeilMaxX) * 0.5, ceilingCenterY, naveCenterZ);
  group.add(rightCeilingSlab);

  // 3 recessed glowing warm skylight diffuser panels (recessed at ceilH + 0.16)
  const skylightPanelY = ceilH + 0.16;
  const beamUniforms = {
    uColor: { value: new THREE.Color(initialTheme.skylightEmissive) },
    uOpacity: { value: initialTheme.beamOpacity },
    uTime: { value: 0 },
  };

  const beamMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
    uniforms: beamUniforms,
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;
      void main() {
        vUv = uv;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uTime;
      varying vec2 vUv;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;

      void main() {
        // Soft vertical falloff from ceiling skylight (vUv.y = 1.0) down toward floor (vUv.y = 0.0)
        float vertFade = pow(clamp(vUv.y, 0.0, 1.0), 1.65)
          * smoothstep(0.0, 0.24, vUv.y)
          * smoothstep(1.0, 0.88, vUv.y);
        // Smooth horizontal edge feathering so plane borders are completely invisible
        float horizFade = pow(sin(clamp(vUv.x, 0.0, 1.0) * 3.14159265), 1.9);
        // View-angle grazing fade prevents sharp card edges when orbiting edge-on
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        float facing = abs(dot(normalize(vWorldNormal), viewDir));
        float angleFade = smoothstep(0.08, 0.48, facing);
        // Ultra-subtle breathing modulation
        float breathe = 0.94 + 0.06 * sin(uTime * 0.45 + vWorldPos.x * 0.7);

        float alpha = uOpacity * vertFade * horizFade * angleFade * breathe;
        if (alpha < 0.0008) discard;
        gl_FragColor = vec4(uColor, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });

  const shaftPlaneH = 3.8;
  const shaftCenterY = 2.25;
  const shaftPlaneGeo = new THREE.PlaneGeometry(bayWidth * 1.48, shaftPlaneH);
  const shaftCrossGeo = new THREE.PlaneGeometry(bayLength * 0.82, shaftPlaneH);

  for (const bay of bays) {
    // Clean seamless glowing skylight diffuser panel overlapping coffer borders
    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(bay.width + 0.08, bay.length + 0.08),
      skylightMat
    );
    panel.rotation.x = Math.PI / 2;
    panel.position.set(bay.x, skylightPanelY, bay.z);
    group.add(panel);

    // Ultra-subtle volumetric skylight light-shaft gradient planes spanning foreground -> rear gauntlet
    const shaftFront = new THREE.Mesh(shaftPlaneGeo, beamMat);
    shaftFront.position.set(bay.x, shaftCenterY, bay.z + bay.length * 0.30);
    shaftFront.rotation.x = 0.12;
    group.add(shaftFront);

    const shaftMid = new THREE.Mesh(shaftPlaneGeo, beamMat);
    shaftMid.position.set(bay.x, shaftCenterY, bay.z - bay.length * 0.06);
    shaftMid.rotation.x = 0.12;
    group.add(shaftMid);

    const shaftRear = new THREE.Mesh(shaftPlaneGeo, beamMat);
    shaftRear.position.set(bay.x, shaftCenterY, bay.z - bay.length * 0.34);
    shaftRear.rotation.x = 0.12;
    group.add(shaftRear);

    // Complementary orthogonal shaft plane so volume holds smoothly while orbiting
    const shaftSide = new THREE.Mesh(shaftCrossGeo, beamMat);
    shaftSide.position.set(bay.x, shaftCenterY, bay.z);
    shaftSide.rotation.y = Math.PI / 2;
    shaftSide.rotation.z = -0.06;
    group.add(shaftSide);
  }

  // 4 zero-shadow warm amber skylight downlight pools (castShadow = false => zero shadow-map cost)
  const backWashLight = new THREE.PointLight(0xffd699, 26.0, 14.0, 1.55);
  backWashLight.castShadow = false;
  backWashLight.position.set(0.0, 3.75, backZ + 1.3);
  group.add(backWashLight);

  const rearGauntletLight = new THREE.PointLight(0xffe0ad, 18.0, 14.5, 1.6);
  rearGauntletLight.castShadow = false;
  rearGauntletLight.position.set(0.0, 3.85, -10.8);
  group.add(rearGauntletLight);

  const midSkylightLight = new THREE.PointLight(0xffe2b5, 13.5, 12.5, 1.65);
  midSkylightLight.castShadow = false;
  midSkylightLight.position.set(0.0, 3.85, -4.5);
  group.add(midSkylightLight);

  const colonnadeGrazeLight = new THREE.PointLight(0xffddb0, 15.5, 13.5, 1.55);
  colonnadeGrazeLight.castShadow = false;
  colonnadeGrazeLight.position.set(rightX - 2.1, 3.55, -3.5);
  group.add(colonnadeGrazeLight);

  // =========================================================================
  // 5. INLAID DEEP MATTE CHARCOAL RUBBER TRAINING LANE (PAVILION_SPEC.trainingLane)
  // =========================================================================
  const lane = PAVILION_SPEC.trainingLane;
  const laneGroup = new THREE.Group();
  laneGroup.position.set(lane.x, 0, lane.z);

  const laneMatGeo = new THREE.PlaneGeometry(lane.width, lane.length);
  laneMatGeo.rotateX(-Math.PI / 2);
  const laneMesh = new THREE.Mesh(laneMatGeo, rubberLaneMat);
  laneMesh.position.y = 0.0024;
  laneMesh.receiveShadow = true;
  laneGroup.add(laneMesh);

  // Subtle matte dark-steel recessed border edging strips framing all 4 edges of the runway
  const edgeW = 0.028;
  const edgeH = 0.004;
  const sideEdgeGeo = new THREE.BoxGeometry(edgeW, edgeH, lane.length + edgeW * 2);
  const endEdgeGeo = new THREE.BoxGeometry(lane.width, edgeH, edgeW);

  const leftEdge = new THREE.Mesh(sideEdgeGeo, laneEdgeMat);
  leftEdge.position.set(-(lane.width * 0.5 + edgeW * 0.5), edgeH * 0.5, 0);
  leftEdge.receiveShadow = true;

  const rightEdge = new THREE.Mesh(sideEdgeGeo, laneEdgeMat);
  rightEdge.position.set(lane.width * 0.5 + edgeW * 0.5, edgeH * 0.5, 0);
  rightEdge.receiveShadow = true;

  const frontEdge = new THREE.Mesh(endEdgeGeo, laneEdgeMat);
  frontEdge.position.set(0, edgeH * 0.5, lane.length * 0.5 + edgeW * 0.5);
  frontEdge.receiveShadow = true;

  const backEdge = new THREE.Mesh(endEdgeGeo, laneEdgeMat);
  backEdge.position.set(0, edgeH * 0.5, -(lane.length * 0.5 + edgeW * 0.5));
  backEdge.receiveShadow = true;

  laneGroup.add(leftEdge, rightEdge, frontEdge, backEdge);
  group.add(laneGroup);

  function setTheme(mode: 'dark' | 'light'): void {
    const theme = STUDIO_THEMES[mode];
    const isLight = mode === 'light';

    backWallMat.color.set(theme.concreteWall);
    leftWallMat.color.set(theme.concreteWall).multiplyScalar(isLight ? 0.98 : 0.84);
    recessWallMat.color.set(theme.concreteWall).multiplyScalar(isLight ? 0.78 : 0.36);
    columnMat.color.set(theme.concreteColumn);
    baffleMat.color.set(theme.concreteColumn).multiplyScalar(isLight ? 0.96 : 0.94);

    // Harmonize flush cast-in-concrete & brushed-metal wall branding across Dark Brutalist & Warm Travertine themes
    const brandAlbedo = isLight ? brandingTex.lightMap : brandingTex.darkMap;
    backBrandingMat.map = brandAlbedo;
    backBrandingMat.color.setRGB(1, 1, 1);
    backBrandingMat.envMapIntensity = isLight ? 0.75 : 0.95;
    backBrandingMat.needsUpdate = true;

    leftBrandingMat.map = brandAlbedo;
    leftBrandingMat.color.setRGB(1, 1, 1).multiplyScalar(isLight ? 0.98 : 0.84);
    leftBrandingMat.envMapIntensity = isLight ? 0.65 : 0.75;
    leftBrandingMat.needsUpdate = true;

    shadowRevealMat.color.set(isLight ? '#7d776d' : '#12161e');
    laneEdgeMat.color.set(isLight ? '#78746c' : '#363e4b');
    rubberLaneMat.color.set(theme.rubberLane);

    skylightMat.color.set(theme.skylightEmissive);
    skylightMat.emissive.set(theme.skylightEmissive);
    skylightMat.emissiveIntensity = theme.skylightIntensity;

    beamUniforms.uColor.value.set(theme.skylightEmissive);
    beamUniforms.uOpacity.value = theme.beamOpacity;

    // Warm skylight wall-wash, rear stress-test gauntlet pool, midground modeling, and right colonnade graze
    backWashLight.color.set(isLight ? 0xfff3de : 0xffd699);
    backWashLight.intensity = isLight ? 7.5 : 26.0;
    rearGauntletLight.color.set(isLight ? 0xfff5e2 : 0xffe0ad);
    rearGauntletLight.intensity = isLight ? 6.0 : 18.0;
    midSkylightLight.color.set(isLight ? 0xfff6e5 : 0xffe2b5);
    midSkylightLight.intensity = isLight ? 5.0 : 13.5;
    colonnadeGrazeLight.color.set(isLight ? 0xfff4e0 : 0xffddb0);
    colonnadeGrazeLight.intensity = isLight ? 5.5 : 14.5;
  }

  function update(timeSec: number): void {
    beamUniforms.uTime.value = timeSec;
  }

  return {
    group,
    setTheme,
    update,
  };
}
