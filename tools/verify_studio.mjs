#!/usr/bin/env node
/**
 * Forge Protocol Outcome Verification, --prove-broken Proof & Unified Capture Runner
 * for SOMA Motion Lab (MotionBricks Live Studio).
 *
 * Enforces a hard 110-second self-destruct watchdog (exit 124) per Forge Protocol.
 * Verifies all 8 outcome checks (including all 10 BAR.md C5 DualSense buttons + More popover)
 * and captures EVERY acceptance image into ONE single manifest directory from ONE live build.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

// Hard 110s watchdog
const WATCHDOG = setTimeout(() => {
  console.error('[verify_studio] WATCHDOG TIMEOUT (110s) — aborting!');
  process.exit(124);
}, 110_000);
WATCHDOG.unref();

const args = process.argv.slice(2);
const proveBroken = args.includes('--prove-broken');
const outIdx = args.indexOf('--out');
const outDir =
  outIdx >= 0 && args[outIdx + 1]
    ? path.resolve(args[outIdx + 1])
    : path.resolve(path.resolve(import.meta.dirname, '..', 'artifacts', 'captures', 'W04'));
const urlIdx = args.indexOf('--url');
const APP_URL = urlIdx >= 0 && args[urlIdx + 1] ? args[urlIdx + 1] : 'http://localhost:5174/';

function evaluateChecks(observed) {
  const results = [];
  const check = (id, pass, detail) => results.push({ id, pass: Boolean(pass), detail });

  check('CHK-01 Zero Console/Page/HTTP Errors', observed.errors.length === 0, `errors=${observed.errors.length}`);
  check(
    'CHK-02 Dressed Multi-Material SOMA GLB Loaded',
    observed.hasSomaSkin && observed.meshCount >= 2,
    `hasSomaSkin=${observed.hasSomaSkin}, meshCount=${observed.meshCount}`
  );
  check(
    'CHK-03 PS5 Exclusive Keyboard Lockout (Zero Leak)',
    observed.ps5KeyboardLocked && observed.keyboardLeaksWhilePS5 === 0,
    `keyboardLocked=${observed.ps5KeyboardLocked}, leaks=${observed.keyboardLeaksWhilePS5}`
  );
  check(
    'CHK-04 Keyboard Restored After PS5 Disconnect',
    observed.keyboardWorksWhenDisconnected,
    `restored=${observed.keyboardWorksWhenDisconnected}`
  );
  check(
    'CHK-05 PS5 Live SVG + Legend + Action Causality (All 10 BAR.md C5 Buttons)',
    observed.ps5ButtonsVerified >= 10 && observed.xrayClearVerified,
    `verifiedButtons=${observed.ps5ButtonsVerified}/10, xrayOffClean=${observed.xrayClearVerified} (${observed.ps5ButtonLog.join(', ')})`
  );
  check(
    'CHK-06 Zero UI Overlap, Clipping or Truncation (1440x900 & 1280x720)',
    observed.layoutIssues.length === 0,
    observed.layoutIssues.length === 0 ? 'clean' : observed.layoutIssues.join(' | ')
  );
  check(
    'CHK-07 Smooth Continuous Motion (No Snap-Back > 0.22m / 50ms)',
    observed.maxMotionStep <= 0.22 && observed.motionSamples >= 15,
    `maxStep=${observed.maxMotionStep}m, median=${observed.medianMotionStep}m, n=${observed.motionSamples}`
  );
  check(
    'CHK-08 More Popover Specialty Styles & Emotes + Help Modal Proven',
    observed.morePopoverVerified && observed.helpModalVerified,
    `morePopover=${observed.morePopoverVerified} (${observed.moreDetail}), helpModal=${observed.helpModalVerified}`
  );

  return results;
}

if (proveBroken) {
  const brokenFixture = {
    errors: ['console: 404'],
    hasSomaSkin: false,
    meshCount: 1,
    ps5KeyboardLocked: false,
    keyboardLeaksWhilePS5: 3,
    keyboardWorksWhenDisconnected: false,
    ps5ButtonsVerified: 4,
    xrayClearVerified: false,
    ps5ButtonLog: ['square:fail'],
    layoutIssues: ['1440x900: PS5 HUD overlaps bottom dock', 'badge truncated'],
    maxMotionStep: 0.65,
    medianMotionStep: 0.08,
    motionSamples: 20,
    morePopoverVerified: false,
    moreDetail: 'closed',
    helpModalVerified: false,
  };
  const brokenRes = evaluateChecks(brokenFixture);
  const allFailed = brokenRes.every((r) => !r.pass);
  console.log('[--prove-broken] Check self-test results:');
  for (const r of brokenRes) {
    console.log(`  ${r.pass ? 'UNEXPECTED PASS' : 'CAUGHT FAIL'} | ${r.id}: ${r.detail}`);
  }
  if (!allFailed) {
    console.error('[--prove-broken] FAILED: one or more checks did not catch the broken fixture!');
    process.exit(1);
  }
  console.log('[--prove-broken] PASSED: all 8 outcome checks catch real failures.');
  process.exit(0);
}

fs.mkdirSync(outDir, { recursive: true });

async function inspectLayout(page, label) {
  return page.evaluate((vpLabel) => {
    const issues = [];
    if (document.documentElement.scrollWidth > window.innerWidth + 1) {
      issues.push(`${vpLabel}: horizontal overflow (${document.documentElement.scrollWidth} > ${window.innerWidth})`);
    }
    const rectOf = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const st = getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) < 0.05) return null;
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0 ? r : null;
    };
    const overlaps = (a, b) =>
      a && b && a.left < b.right - 2 && a.right > b.left + 2 && a.top < b.bottom - 2 && a.bottom > b.top + 2;

    const topL = rectOf('.hud-top-left');
    const topR = rectOf('.hud-top-right');
    const ps5Card = rectOf('.ps5-hud');
    const dock = rectOf('.hud-dock');

    if (overlaps(topL, topR)) issues.push(`${vpLabel}: top-left overlaps top-right`);
    if (overlaps(ps5Card, dock)) issues.push(`${vpLabel}: PS5 HUD overlaps bottom dock`);
    if (overlaps(ps5Card, topL)) issues.push(`${vpLabel}: PS5 HUD overlaps top-left`);
    if (dock && dock.bottom > window.innerHeight - 10) {
      issues.push(`${vpLabel}: bottom dock too close to viewport bottom (${Math.round(window.innerHeight - dock.bottom)}px)`);
    }
    if (dock && (dock.left < 4 || dock.right > window.innerWidth - 4)) {
      issues.push(`${vpLabel}: bottom dock clipped horizontally (${Math.round(dock.left)}..${Math.round(dock.right)} / ${window.innerWidth})`);
    }

    for (const sel of ['#ps5-label', '#ps5-hud-status', '#ps5-haptic-pill']) {
      const el = document.querySelector(sel);
      if (el && el.offsetParent !== null && el.scrollWidth > el.clientWidth + 2) {
        issues.push(`${vpLabel}: truncated text in ${sel} ("${el.textContent}")`);
      }
    }
    return issues;
  }, label);
}

(async () => {
  let sha = 'unknown';
  try {
    const base = execSync('git rev-parse --short HEAD', {
      cwd: path.resolve(import.meta.dirname, '..'),
      encoding: 'utf-8',
    }).trim();
    const dirty = execSync('git status --porcelain', {
      cwd: path.resolve(import.meta.dirname, '..'),
      encoding: 'utf-8',
    }).trim();
    sha = dirty ? `${base}+wt` : base;
  } catch {}

  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome',
    args: ['--enable-webgl', '--ignore-gpu-blocklist'],
  });

  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push('console: ' + m.text());
  });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('response', (res) => {
    if (res.status() >= 400) errors.push(`http ${res.status()}: ${res.url()}`);
  });

  await page.goto(APP_URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(950);

  const webglInfo = await page.evaluate(() => {
    const c = document.getElementById('stage');
    const gl = c?.getContext('webgl2') || c?.getContext('webgl');
    const dbg = gl?.getExtension('WEBGL_debug_renderer_info');
    return {
      backend: gl instanceof WebGL2RenderingContext ? 'WebGL2' : 'WebGL',
      renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'unknown',
      dpr: window.devicePixelRatio,
    };
  });

  const baseState = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
  const layoutIssues = [];

  // 1. Hero cam-01 Dark Pavilion (default view)
  await page.evaluate(() => window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01'));
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, '01_hero_dark.png') });
  layoutIssues.push(...(await inspectLayout(page, '1440x900-dark')));

  // 2. Hero Close-Up of Dressed SOMA Character (cam-02)
  await page.evaluate(() => window.__MOTIONBRICKS__.setCameraBookmark?.('cam-02'));
  await page.waitForTimeout(480);
  await page.screenshot({ path: path.join(outDir, '06_dressed_closeup.png') });

  // 3. Test PS5 Exclusive Mode + Strict Keyboard Lockout + ALL 10 BAR.md C5 DualSense Buttons!
  await page.evaluate(() => window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01'));
  await page.waitForTimeout(300);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({
      connected: true,
      transport: 'Bluetooth',
      batteryLevel: 0.86,
      buttons: {},
    });
  });
  await page.waitForTimeout(280);
  layoutIssues.push(...(await inspectLayout(page, '1440x900-ps5')));

  // Verify ALL keyboard keys are strictly blocked while PS5 is connected!
  const beforeKb = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
  for (const key of ['KeyC', 'KeyJ', 'Space', 'Digit9', 'KeyX', 'KeyH', 'KeyL', 'KeyV', 'KeyB', 'KeyE']) {
    await page.keyboard.press(key);
    await page.waitForTimeout(35);
  }
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(200);
  await page.keyboard.up('KeyW');
  const afterKb = await page.evaluate(() => window.__MOTIONBRICKS__.getState());

  let keyboardLeaksWhilePS5 = 0;
  if (afterKb.activeStyle !== beforeKb.activeStyle) keyboardLeaksWhilePS5++;
  if (afterKb.mode !== beforeKb.mode) keyboardLeaksWhilePS5++;
  if (afterKb.xrayMode !== beforeKb.xrayMode) keyboardLeaksWhilePS5++;
  if (afterKb.showAllGuides !== beforeKb.showAllGuides) keyboardLeaksWhilePS5++;
  if (afterKb.lightTheme !== beforeKb.lightTheme) keyboardLeaksWhilePS5++;
  const kbDist = Math.hypot(afterKb.hips[0] - beforeKb.hips[0], afterKb.hips[2] - beforeKb.hips[2]);
  if (kbDist > 0.15) keyboardLeaksWhilePS5++;

  const ps5ButtonLog = [];
  let ps5ButtonsVerified = 0;

  // (1/10) Square -> Left Jab + Left Stick
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({
      connected: true,
      transport: 'Bluetooth',
      batteryLevel: 0.86,
      leftStick: { x: 0.78, y: 0.56 },
      buttons: { square: true },
    });
  });
  await page.waitForTimeout(380);
  const sqCheck = await page.evaluate(() => ({
    mode: window.__MOTIONBRICKS__.getState().mode,
    svgLit: document.querySelector('[data-btn="square"]')?.classList.contains('active'),
    pillLit: Array.from(document.querySelectorAll('.ps5-map-pill.active')).some((e) => e.textContent?.includes('Jab')),
  }));
  if (sqCheck.svgLit && sqCheck.pillLit && sqCheck.mode === 'punch_left') {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('square:OK');
  } else ps5ButtonLog.push(`square:FAIL(${JSON.stringify(sqCheck)})`);
  await page.screenshot({ path: path.join(outDir, '04_ps5_square_jab.png') });

  // (2/10) R2 -> Heavy Right Cross + L2 Slow-Mo Hold in Warm Daylight Theme
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.toggleTheme();
    window.__MOTIONBRICKS__.simulatePS5({
      connected: true,
      transport: 'Bluetooth',
      batteryLevel: 0.86,
      leftStick: { x: 0.25, y: 0.8 },
      rightStick: { x: -0.2, y: 0.0 },
      leftTrigger: 0.72,
      rightTrigger: 0.95,
      buttons: { r1: true },
    });
  });
  await page.waitForTimeout(380);
  const r2Check = await page.evaluate(() => ({
    mode: window.__MOTIONBRICKS__.getState().mode,
    r2W: parseFloat(document.querySelector('#svg-r2-fill')?.getAttribute('width') || '0'),
    pillLit: Array.from(document.querySelectorAll('.ps5-map-pill.active')).some((e) => e.textContent?.includes('Cross')),
  }));
  if (r2Check.r2W > 30 && r2Check.pillLit && r2Check.mode === 'punch_right') {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('r2:OK');
  } else ps5ButtonLog.push(`r2:FAIL(${JSON.stringify(r2Check)})`);
  await page.screenshot({ path: path.join(outDir, '05_ps5_r2_cross_light.png') });

  // Clean Warm Daylight Gallery hero (cam-01)
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5(null);
    window.__MOTIONBRICKS__.togglePS5Hud();
    window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01');
  });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, '02_hero_light.png') });
  layoutIssues.push(...(await inspectLayout(page, '1440x900-light')));

  // Switch back to Dark Pavilion & test remaining 8 BAR.md C5 DualSense buttons!
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.toggleTheme();
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(160);

  // (3/10) L1 -> Wide / Center Cam (strictly separate from L2 Slow-Mo!)
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { l1: true } });
  });
  await page.waitForTimeout(240);
  const l1Check = await page.evaluate(() => ({
    slowMo: window.__MOTIONBRICKS__.getState().slowMo,
    svgLit: document.querySelector('[data-btn="l1"]')?.classList.contains('active'),
    pillLit: document.querySelector('.ps5-map-pill[data-map-key="l1"]')?.classList.contains('active'),
    toast: document.querySelector('#ps5-action-toast')?.textContent || '',
  }));
  if (l1Check.svgLit && l1Check.pillLit && l1Check.slowMo === false && l1Check.toast.includes('Wide')) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('l1:OK');
  } else ps5ButtonLog.push(`l1:FAIL(${JSON.stringify(l1Check)})`);
  // Release L1 and toggle wideCam off cleanly
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(80);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { l1: true } });
  });
  await page.waitForTimeout(80);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(80);

  // (4/10) D-Pad Right -> Cycle Style
  const styleBefore = await page.evaluate(() => window.__MOTIONBRICKS__.getState().activeStyle);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { dpadRight: true } });
  });
  await page.waitForTimeout(240);
  const drCheck = await page.evaluate(() => ({
    style: window.__MOTIONBRICKS__.getState().activeStyle,
    svgLit: document.querySelector('[data-btn="dpadRight"]')?.classList.contains('active'),
    pillLit: document.querySelector('.ps5-map-pill[data-map-key="dpadLR"]')?.classList.contains('active'),
  }));
  if (drCheck.svgLit && drCheck.pillLit && drCheck.style !== styleBefore) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('dpadRight:OK');
  } else ps5ButtonLog.push(`dpadRight:FAIL(${JSON.stringify(drCheck)})`);

  // (5/10) Options -> AI Guides Toggle ON then OFF
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { options: true } });
  });
  await page.waitForTimeout(240);
  const optCheck = await page.evaluate(() => ({
    guides: window.__MOTIONBRICKS__.getState().showAllGuides,
    svgLit: document.querySelector('[data-btn="options"]')?.classList.contains('active'),
    pillLit: document.querySelector('.ps5-map-pill[data-map-key="options"]')?.classList.contains('active'),
  }));
  if (optCheck.svgLit && optCheck.pillLit && optCheck.guides === true) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('options:OK');
  } else ps5ButtonLog.push(`options:FAIL(${JSON.stringify(optCheck)})`);
  // Release & toggle Guides back OFF cleanly
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(70);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { options: true } });
  });
  await page.waitForTimeout(70);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(70);

  // (6/10) Circle -> Crouch
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { circle: true } });
  });
  await page.waitForTimeout(360);
  const cirCheck = await page.evaluate(() => ({
    style: window.__MOTIONBRICKS__.getState().activeStyle,
    svgLit: document.querySelector('[data-btn="circle"]')?.classList.contains('active'),
  }));
  if (cirCheck.svgLit && cirCheck.style === 'crouch') {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('circle:OK');
  } else ps5ButtonLog.push(`circle:FAIL(${JSON.stringify(cirCheck)})`);

  // (7/10) Touchpad -> Neural X-Ray ON (capture 10_pose_xray.png), then cleanly release & toggle OFF!
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { touchpad: true } });
  });
  await page.waitForTimeout(420);
  const padCheck = await page.evaluate(() => ({
    xray: window.__MOTIONBRICKS__.getState().xrayMode,
    svgLit: document.querySelector('[data-btn="touchpad"]')?.classList.contains('active'),
  }));
  if (padCheck.svgLit && padCheck.xray === true) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('touchpad:OK');
  } else ps5ButtonLog.push(`touchpad:FAIL(${JSON.stringify(padCheck)})`);
  await page.screenshot({ path: path.join(outDir, '10_pose_xray.png') });

  // Cleanly release touchpad -> re-press to turn X-Ray OFF -> release touchpad completely!
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(70);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { touchpad: true } });
  });
  await page.waitForTimeout(70);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: {} });
  });
  await page.waitForTimeout(500);
  const xrayClearCheck = await page.evaluate(() => ({
    xray: window.__MOTIONBRICKS__.getState().xrayMode,
    padSvgLit: document.querySelector('[data-btn="touchpad"]')?.classList.contains('active'),
    padPillLit: document.querySelector('.ps5-map-pill[data-map-key="touchpad"]')?.classList.contains('active'),
  }));
  const xrayClearVerified =
    xrayClearCheck.xray === false && !xrayClearCheck.padSvgLit && !xrayClearCheck.padPillLit;

  // (8/10) D-Pad Down ALONE -> Overhead Duck! (PAD X-Ray is OFF & unlit; ↓ Duck glows & toast shows Overhead Duck!)
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { dpadDown: true } });
  });
  await page.waitForTimeout(320);
  const duckState = await page.evaluate(() => ({
    mode: window.__MOTIONBRICKS__.getState().mode,
    svgLit: document.querySelector('[data-btn="dpadDown"]')?.classList.contains('active'),
    pillLit: document.querySelector('.ps5-map-pill[data-map-key="dpadDown"]')?.classList.contains('active'),
    toast: document.querySelector('#ps5-action-toast')?.textContent || '',
  }));
  if (duckState.svgLit && duckState.pillLit && duckState.mode.startsWith('duck') && duckState.toast.includes('Duck')) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('dpadDown:OK');
  } else ps5ButtonLog.push(`dpadDown:FAIL(${JSON.stringify(duckState)})`);

  await page.evaluate(() => window.__MOTIONBRICKS__.setCameraBookmark?.('cam-04'));
  for (let i = 0; i < 65; i++) {
    await page.waitForTimeout(100);
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (st.mode === 'duck_action' && st.playFrame > 6) {
      await page.screenshot({ path: path.join(outDir, '08_pose_duck.png') });
      break;
    }
  }
  if (!fs.existsSync(path.join(outDir, '08_pose_duck.png'))) {
    await page.screenshot({ path: path.join(outDir, '08_pose_duck.png') });
  }

  // (9/10) Triangle -> Sit!
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5({ connected: true, transport: 'USB', batteryLevel: 1.0, buttons: { triangle: true } });
  });
  await page.waitForTimeout(340);
  const triCheck = await page.evaluate(() => ({
    mode: window.__MOTIONBRICKS__.getState().mode,
    svgLit: document.querySelector('[data-btn="triangle"]')?.classList.contains('active'),
  }));
  if (triCheck.svgLit && triCheck.mode.startsWith('sit')) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('triangle:OK');
  } else ps5ButtonLog.push(`triangle:FAIL(${JSON.stringify(triCheck)})`);

  // Capture Seated pose (disconnect PS5 cleanly first so disconnected HUD has zero haptic glow!)
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5(null);
    window.__MOTIONBRICKS__.togglePS5Hud();
    window.__MOTIONBRICKS__.setCameraBookmark?.('cam-03');
  });
  for (let i = 0; i < 65; i++) {
    await page.waitForTimeout(120);
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (st.mode === 'sit_hold') {
      await page.screenshot({ path: path.join(outDir, '09_pose_sit.png') });
      break;
    }
  }
  if (!fs.existsSync(path.join(outDir, '09_pose_sit.png'))) {
    await page.screenshot({ path: path.join(outDir, '09_pose_sit.png') });
  }

  // Clean Parkour Vault Apex Capture (07_pose_vault.png): catch true airborne vault apex (hips.y >= 1.26 m right over x = -2.35!)
  await page.click('#btn-reset');
  await page.waitForTimeout(380);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.setCameraBookmark?.('cam-03');
    window.__MOTIONBRICKS__.triggerVault();
  });
  let vaultMaxY = 0;
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(45);
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (st.mode === 'vault_action' || st.mode === 'vault_exit') {
      vaultMaxY = Math.max(vaultMaxY, st.hips[1]);
      if (st.hips[1] >= 1.26 && Math.abs(st.hips[0] - (-2.35)) < 0.35) {
        await page.screenshot({ path: path.join(outDir, '07_pose_vault.png') });
        break;
      }
    }
  }
  if (!fs.existsSync(path.join(outDir, '07_pose_vault.png'))) {
    await page.screenshot({ path: path.join(outDir, '07_pose_vault.png') });
  }

  // (10/10) Cross -> Smart Jump / Vault (also test 1280x720 laptop viewport cleanly!)
  await page.click('#btn-reset');
  await page.waitForTimeout(380);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01');
    window.__MOTIONBRICKS__.simulatePS5({
      connected: true,
      transport: 'Bluetooth',
      batteryLevel: 0.86,
      leftStick: { x: 0.4, y: 0.6 },
      buttons: { cross: true },
    });
  });
  await page.waitForTimeout(420);
  const crossCheck = await page.evaluate(() => ({
    mode: window.__MOTIONBRICKS__.getState().mode,
    svgLit: document.querySelector('[data-btn="cross"]')?.classList.contains('active'),
    pillLit: document.querySelector('.ps5-map-pill[data-map-key="cross"]')?.classList.contains('active'),
  }));
  if (crossCheck.svgLit && crossCheck.pillLit && (crossCheck.mode.startsWith('jump') || crossCheck.mode.startsWith('vault'))) {
    ps5ButtonsVerified++;
    ps5ButtonLog.push('cross:OK');
  } else ps5ButtonLog.push(`cross:FAIL(${JSON.stringify(crossCheck)})`);
  layoutIssues.push(...(await inspectLayout(page, '1280x720-ps5')));
  await page.screenshot({ path: path.join(outDir, '03_laptop_1280.png') });

  // Restore 1440x900 & disconnect PS5 to test keyboard restoration, More popover, Help Modal, Stress Gauntlet, & Smoothness!
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.simulatePS5(null);
    window.__MOTIONBRICKS__.togglePS5Hud();
  });
  await page.waitForTimeout(220);

  // Test keyboard restoration after PS5 disconnect!
  await page.evaluate(() => window.__MOTIONBRICKS__.triggerStyle('jog'));
  await page.waitForTimeout(220);
  const preRestore = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
  await page.keyboard.press('KeyC');
  await page.waitForTimeout(240);
  const postRestore = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
  const keyboardWorksWhenDisconnected =
    preRestore.keyboardLocked === false && postRestore.activeStyle === 'crouch';

  // CHK-08 Part A: More Popover -> Specialty Style (Vogue) + Emote (Stretch) + Live Capture!
  await page.click('#btn-reset');
  await page.waitForTimeout(350);
  await page.evaluate(() => window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01'));
  await page.click('#btn-more-styles');
  await page.waitForTimeout(200);
  const popoverCheck = await page.evaluate(() => {
    const pop = document.querySelector('#more-styles-popover');
    const r = pop?.getBoundingClientRect();
    const open = Boolean(pop && !pop.classList.contains('hidden'));
    const fits = Boolean(r && r.left >= 0 && r.right <= window.innerWidth && r.top >= 0 && r.bottom <= window.innerHeight);
    return { open, fits, count: pop?.querySelectorAll('button').length ?? 0 };
  });
  // Select specialty style Vogue, then reopen popover and click Stretch emote + screenshot with popover visible!
  await page.locator('#more-styles-grid button', { hasText: 'Vogue' }).click();
  await page.waitForTimeout(260);
  const vogueStyle = await page.evaluate(() => window.__MOTIONBRICKS__.getState().activeStyle);
  await page.click('#btn-more-styles');
  await page.waitForTimeout(160);
  await page.locator('#more-emotes-grid button', { hasText: 'Stretch' }).click();
  await page.waitForTimeout(350);
  const emoteMode = await page.evaluate(() => window.__MOTIONBRICKS__.getState().mode);
  // Open popover cleanly so capture 12_more_popover_emote.png visibly proves the popover + active specialty/emote state!
  await page.click('#btn-more-styles');
  await page.waitForTimeout(180);
  await page.screenshot({ path: path.join(outDir, '12_more_popover_emote.png') });
  await page.click('#btn-more-styles');
  const morePopoverVerified =
    popoverCheck.open && popoverCheck.fits && popoverCheck.count >= 6 && vogueStyle === 'vogue' && emoteMode.startsWith('emote_warmup');
  const moreDetail = `open=${popoverCheck.open},fits=${popoverCheck.fits},items=${popoverCheck.count},style=${vogueStyle},emote=${emoteMode}`;

  // CHK-08 Part B: Fresh Help Modal capture (11_help_modal.png) from THIS exact build!
  await page.click('#btn-help');
  await page.waitForTimeout(220);
  const helpCheck = await page.evaluate(() => {
    const card = document.querySelector('.help-modal-card');
    const r = card?.getBoundingClientRect();
    const open = !document.querySelector('#help-modal')?.classList.contains('hidden');
    const fits = Boolean(r && r.left >= 0 && r.right <= window.innerWidth && r.top >= 0 && r.bottom <= window.innerHeight);
    return { open, fits };
  });
  await page.screenshot({ path: path.join(outDir, '11_help_modal.png') });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(180);
  const helpClosed = await page.evaluate(() => document.querySelector('#help-modal')?.classList.contains('hidden'));
  const helpModalVerified = helpCheck.open && helpCheck.fits && helpClosed;

  // Fresh Rear Stress-Test Gauntlet capture (13_stress_gauntlet.png) from THIS exact warm-lit 4-wall build!
  await page.click('#btn-reset');
  await page.waitForTimeout(350);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.setCameraBookmark?.('cam-01');
    window.__MOTIONBRICKS__.triggerStyle('run');
    window.__MOTIONBRICKS__.triggerGoto(0.0, -2.6);
  });
  for (let i = 0; i < 32; i++) {
    await page.waitForTimeout(95);
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (st.hips[2] < -2.1 && Math.abs(st.hips[0]) < 0.7) break;
  }
  await page.evaluate(() => window.__MOTIONBRICKS__.setSteer(0, -1));
  for (let i = 0; i < 55; i++) {
    await page.waitForTimeout(95);
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (st.hips[2] < -7.5) {
      await page.screenshot({ path: path.join(outDir, '13_stress_gauntlet.png') });
      break;
    }
  }
  if (!fs.existsSync(path.join(outDir, '13_stress_gauntlet.png'))) {
    await page.screenshot({ path: path.join(outDir, '13_stress_gauntlet.png') });
  }
  await page.evaluate(() => window.__MOTIONBRICKS__.setSteer(0, 0));

  // Smoothness check across 28 continuous 50ms steering frames
  await page.click('#btn-reset');
  await page.waitForTimeout(350);
  await page.evaluate(() => {
    window.__MOTIONBRICKS__.triggerStyle('jog');
    window.__MOTIONBRICKS__.setSteer(1, 0);
  });
  const steps = [];
  let prevH = null;
  for (let i = 0; i < 28; i++) {
    await page.waitForTimeout(50);
    if (i === 14) await page.evaluate(() => window.__MOTIONBRICKS__.setSteer(0, 1));
    const st = await page.evaluate(() => window.__MOTIONBRICKS__.getState());
    if (prevH) {
      steps.push(Math.hypot(st.hips[0] - prevH[0], st.hips[1] - prevH[1], st.hips[2] - prevH[2]));
    }
    prevH = st.hips;
  }
  await page.evaluate(() => window.__MOTIONBRICKS__.setSteer(0, 0));
  const sortedSteps = [...steps].sort((a, b) => a - b);
  const maxMotionStep = Number(Math.max(...steps, 0).toFixed(4));
  const medianMotionStep = Number((sortedSteps[Math.floor(sortedSteps.length / 2)] || 0).toFixed(4));

  const observed = {
    errors,
    hasSomaSkin: Boolean(baseState.hasSomaSkin),
    meshCount: Number(baseState.somaMeshCount ?? 0),
    ps5KeyboardLocked: Boolean(beforeKb.keyboardLocked),
    keyboardLeaksWhilePS5,
    keyboardWorksWhenDisconnected,
    ps5ButtonsVerified,
    xrayClearVerified,
    ps5ButtonLog,
    layoutIssues,
    maxMotionStep,
    medianMotionStep,
    motionSamples: steps.length,
    morePopoverVerified,
    moreDetail,
    helpModalVerified,
  };

  const results = evaluateChecks(observed);
  const allPassed = results.every((r) => r.pass);

  const manifest = `# Capture Manifest
- **Build:** \`${sha}\`
- **URL:** ${APP_URL}
- **Backend:** ${webglInfo.backend} (\`${webglInfo.renderer}\`, DPR=${webglInfo.dpr})
- **Primary Viewport:** 1440×900 + Laptop 1280×720
- **Generated:** ${new Date().toISOString()}
- **Outcome Checks:** ${allPassed ? 'ALL PASS (8/8)' : 'FAIL'}
- **Captures (Single Consistent Build):** 01_hero_dark.png, 02_hero_light.png, 03_laptop_1280.png, 04_ps5_square_jab.png, 05_ps5_r2_cross_light.png, 06_dressed_closeup.png, 07_pose_vault.png, 08_pose_duck.png, 09_pose_sit.png, 10_pose_xray.png, 11_help_modal.png, 12_more_popover_emote.png, 13_stress_gauntlet.png
`;
  fs.writeFileSync(path.join(outDir, 'MANIFEST.md'), manifest);

  const checkMd = [
    `# Automated Outcome Checks (Build \`${sha}\`)`,
    '',
    ...results.map((r) => `- **${r.pass ? 'PASS' : 'FAIL'}** | ${r.id} — ${r.detail}`),
    '',
    errors.length ? `## Errors\n${errors.map((e) => `- ${e}`).join('\n')}` : '## Errors\n- None (0 console, 0 page, 0 HTTP >= 400)',
  ].join('\n');
  fs.writeFileSync(path.join(outDir, 'checks.md'), checkMd);

  console.log(checkMd);
  await browser.close();
  process.exit(allPassed ? 0 : 1);
})();
