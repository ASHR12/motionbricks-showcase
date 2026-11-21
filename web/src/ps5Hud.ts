import type { PS5ControllerState } from 'ps5-controller-webhid';

export interface PS5HudActionHandlers {
  onConnectClick: () => void;
  onTriggerAction: (key: string) => void;
}

interface MappingItem {
  key: string;
  symbol: string;
  label: string;
  color: string;
  round?: boolean;
}

const MAPPING_ITEMS: MappingItem[] = [
  { key: 'cross', symbol: '✕', label: 'Jump / Vault', color: '#60a5fa', round: true },
  { key: 'circle', symbol: '○', label: 'Crouch', color: '#fb7185', round: true },
  { key: 'square', symbol: '□', label: 'Jab Combo', color: '#e879f9', round: true },
  { key: 'triangle', symbol: '△', label: 'Sit', color: '#34d399', round: true },
  { key: 'r2', symbol: 'R2', label: 'Cross', color: '#f43f5e' },
  { key: 'l2', symbol: 'L2', label: 'Slow-Mo', color: '#22d3ee' },
  { key: 'r1', symbol: 'R1', label: 'Sprint', color: '#f97316' },
  { key: 'l1', symbol: 'L1', label: 'Wide Cam', color: '#38bdf8' },
  { key: 'r3', symbol: 'R3', label: 'Roll', color: '#38bdf8' },
  { key: 'dpadDown', symbol: '↓', label: 'Duck', color: '#2dd4bf' },
  { key: 'dpadLR', symbol: '←/→', label: 'Style', color: '#a78bfa' },
  { key: 'touchpad', symbol: 'PAD', label: 'X-Ray', color: '#38bdf8' },
  { key: 'options', symbol: 'OPT', label: 'Guides', color: '#94a3b8' },
  { key: 'create', symbol: 'CRT', label: 'Reset', color: '#94a3b8' },
];

/**
 * Horizontal 2-pane interactive PS5 DualSense Controller HUD (matching targets/t01.png).
 *
 * Left Pane: Illuminated hairline DualSense schematic with live moving glowing L/R thumbstick orbs,
 *            live L2/R2 analog pressure bars, PlayStation-colored face buttons, 4-way D-pad,
 *            L1/R1 bumpers, Touchpad, Options/Create, and pulsing warm-amber grip haptic rings + live action banner.
 * Right Pane: Compact 2-column live Button -> Action mapping grid where every physical press (or click
 *             when disconnected) immediately illuminates both SVG button and mapping pill.
 * Footer Bar: Unmistakable bold `PS5 EXCLUSIVE · KEYS LOCKED` indicator + live haptic telemetry pill.
 */
export class PS5ControllerHUD {
  public readonly root: HTMLElement;
  private readonly statusBadge: HTMLElement;
  private readonly connectBtn: HTMLButtonElement;
  private readonly actionToast: HTMLElement;
  private readonly exclusiveStatus: HTMLElement;
  private readonly lStickDot: SVGCircleElement;
  private readonly lStickHalo: SVGCircleElement;
  private readonly rStickDot: SVGCircleElement;
  private readonly rStickHalo: SVGCircleElement;
  private readonly l2Bar: SVGRectElement;
  private readonly r2Bar: SVGRectElement;
  private readonly lHapticRing: SVGPathElement;
  private readonly rHapticRing: SVGPathElement;
  private readonly hapticLabel: HTMLElement;

  private readonly svgBtnEls = new Map<string, SVGElement>();
  private readonly mapPillEls = new Map<string, HTMLButtonElement>();
  private readonly flashUntil = new Map<string, number>();

  private toastClearAt = 0;
  private hapticLDecay = 0;
  private hapticRDecay = 0;
  private hapticReasonClearAt = 0;
  private visible = false;
  private connectedState = false;

  constructor(container: HTMLElement, handlers: PS5HudActionHandlers) {
    this.root = document.createElement('section');
    this.root.className = 'ps5-hud hidden';
    this.root.setAttribute('aria-label', 'PS5 DualSense Interactive Controller HUD');

    this.root.innerHTML = `
      <div class="ps5-hud-header">
        <div class="ps5-hud-title-group">
          <span class="ps5-conn-dot" id="ps5-hud-dot" aria-hidden="true"></span>
          <span class="ps5-hud-title">PS5 DualSense</span>
          <span class="ps5-hud-badge" id="ps5-hud-status">Disconnected</span>
        </div>
        <button type="button" class="ps5-hud-connect" id="ps5-hud-connect-btn">Connect</button>
      </div>

      <div class="ps5-hud-body">
        <div class="ps5-pane-left">
          <svg class="ps5-svg" viewBox="0 0 320 202" role="img" aria-label="Live PS5 DualSense schematic">
            <defs>
              <radialGradient id="ps5-lstick-glow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.92" />
                <stop offset="52%" stop-color="#38bdf8" stop-opacity="0.34" />
                <stop offset="100%" stop-color="#38bdf8" stop-opacity="0" />
              </radialGradient>
              <radialGradient id="ps5-rstick-glow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#818cf8" stop-opacity="0.88" />
                <stop offset="52%" stop-color="#818cf8" stop-opacity="0.3" />
                <stop offset="100%" stop-color="#818cf8" stop-opacity="0" />
              </radialGradient>
            </defs>

            <!-- Analog trigger pressure bars L2 / R2 -->
            <g class="svg-btn svg-trig-group" data-btn="l2">
              <rect x="42" y="5" width="48" height="10" rx="5" class="svg-well" />
              <rect x="42" y="5" width="0" height="10" rx="5" class="svg-trig-fill trig-l2" id="svg-l2-fill" />
              <text x="66" y="13" text-anchor="middle" class="svg-micro">L2</text>
            </g>

            <g class="svg-btn svg-trig-group" data-btn="r2">
              <rect x="230" y="5" width="48" height="10" rx="5" class="svg-well" />
              <rect x="230" y="5" width="0" height="10" rx="5" class="svg-trig-fill trig-r2" id="svg-r2-fill" />
              <text x="254" y="13" text-anchor="middle" class="svg-micro">R2</text>
            </g>

            <!-- L1 / R1 Shoulder Bumpers -->
            <rect x="38" y="20" width="56" height="12" rx="6" class="svg-btn" data-btn="l1" />
            <text x="66" y="29" text-anchor="middle" class="svg-micro-btn">L1</text>

            <rect x="226" y="20" width="56" height="12" rx="6" class="svg-btn" data-btn="r1" />
            <text x="254" y="29" text-anchor="middle" class="svg-micro-btn">R1</text>

            <!-- DualSense ergonomic silhouette + signature two-tone inner seam -->
            <path d="M56 36 C35 38 21 59 16 98 C11 137 14 179 34 187 C51 194 65 168 82 145 C96 127 115 127 160 127 C205 127 224 127 238 145 C255 168 269 194 286 187 C306 179 309 137 304 98 C299 59 285 38 264 36 L220 34 C196 32 124 32 100 34 Z"
                  class="svg-body" />
            <path d="M84 129 C96 116 114 115 160 115 C206 115 224 116 236 129"
                  class="svg-seam" />

            <!-- Left & Right Grip Haptic Rumble Arcs -->
            <path d="M35 121 C26 140 26 164 38 176 C48 182 56 169 65 151"
                  class="svg-haptic" id="svg-haptic-l" />
            <path d="M285 121 C294 140 294 164 282 176 C272 182 264 169 255 151"
                  class="svg-haptic" id="svg-haptic-r" />

            <!-- Touchpad + DualSense flanking lightbar lines -->
            <path d="M113 38 L108 82" class="svg-lightbar" />
            <path d="M207 38 L212 82" class="svg-lightbar" />
            <path d="M115 38 L205 38 L209 78 C209 82 206 84 201 84 L119 84 C114 84 111 82 111 78 Z"
                  class="svg-btn svg-touchpad" data-btn="touchpad" />

            <!-- Create / Options -->
            <rect x="96" y="45" width="8" height="16" rx="4" class="svg-btn" data-btn="create" />
            <rect x="216" y="45" width="8" height="16" rx="4" class="svg-btn" data-btn="options" />

            <!-- Center speaker dots + subtle PS mark -->
            <circle cx="152" cy="96" r="1.3" class="svg-speaker-dot" />
            <circle cx="160" cy="96" r="1.3" class="svg-speaker-dot" />
            <circle cx="168" cy="96" r="1.3" class="svg-speaker-dot" />
            <circle cx="160" cy="108" r="5" class="svg-ps-mark" />

            <!-- D-Pad (4 directional arms) -->
            <g class="svg-dpad">
              <rect x="59" y="53" width="14" height="17" rx="3.5" class="svg-btn" data-btn="dpadUp" />
              <rect x="59" y="83" width="14" height="17" rx="3.5" class="svg-btn" data-btn="dpadDown" />
              <rect x="42" y="70" width="17" height="14" rx="3.5" class="svg-btn" data-btn="dpadLeft" />
              <rect x="73" y="70" width="17" height="14" rx="3.5" class="svg-btn" data-btn="dpadRight" />
              <circle cx="66" cy="77" r="4" class="svg-dpad-center" />
            </g>

            <!-- PlayStation Face Buttons: Triangle, Circle, Cross, Square -->
            <g class="svg-face">
              <g class="svg-btn svg-face-tri" data-btn="triangle">
                <circle cx="254" cy="57" r="11.5" />
                <text x="254" y="61" text-anchor="middle" class="svg-sym svg-sym-tri">△</text>
              </g>
              <g class="svg-btn svg-face-cir" data-btn="circle">
                <circle cx="275" cy="77" r="11.5" />
                <text x="275" y="81.5" text-anchor="middle" class="svg-sym svg-sym-cir">○</text>
              </g>
              <g class="svg-btn svg-face-crs" data-btn="cross">
                <circle cx="254" cy="97" r="11.5" />
                <text x="254" y="101.5" text-anchor="middle" class="svg-sym svg-sym-crs">✕</text>
              </g>
              <g class="svg-btn svg-face-sqr" data-btn="square">
                <circle cx="233" cy="77" r="11.5" />
                <text x="233" y="81" text-anchor="middle" class="svg-sym svg-sym-sqr">□</text>
              </g>
            </g>

            <!-- Left & Right Analog Stick Wells + Glowing Live Orbs -->
            <g class="svg-sticks">
              <circle cx="115" cy="110" r="24" fill="url(#ps5-lstick-glow)" class="svg-stick-halo" id="svg-lstick-halo" />
              <circle cx="115" cy="110" r="19.5" class="svg-stick-well" data-btn="l3" />
              <circle cx="115" cy="110" r="8.5" class="svg-stick-dot svg-stick-l" id="svg-lstick-dot" />

              <circle cx="205" cy="110" r="24" fill="url(#ps5-rstick-glow)" class="svg-stick-halo" id="svg-rstick-halo" />
              <circle cx="205" cy="110" r="19.5" class="svg-stick-well" data-btn="r3" />
              <circle cx="205" cy="110" r="8.5" class="svg-stick-dot svg-stick-r" id="svg-rstick-dot" />
            </g>
          </svg>

          <div class="ps5-action-toast" id="ps5-action-toast">L-Stick Move · R-Stick Cam</div>
        </div>

        <div class="ps5-pane-right">
          <div class="ps5-map-grid" id="ps5-map-grid"></div>
        </div>
      </div>

      <div class="ps5-hud-footer">
        <div class="ps5-exclusive-status" id="ps5-exclusive-status">
          <span class="ps5-lock-dot" aria-hidden="true"></span>
          <span id="ps5-exclusive-text">Connect PS5 · WebHID</span>
        </div>
        <div class="ps5-haptic-pill" id="ps5-haptic-pill">Haptics Ready</div>
      </div>
    `;

    container.appendChild(this.root);

    this.statusBadge = this.root.querySelector('#ps5-hud-status')!;
    this.connectBtn = this.root.querySelector('#ps5-hud-connect-btn')!;
    this.actionToast = this.root.querySelector('#ps5-action-toast')!;
    this.exclusiveStatus = this.root.querySelector('#ps5-exclusive-text')!;
    this.hapticLabel = this.root.querySelector('#ps5-haptic-pill')!;

    this.lStickDot = this.root.querySelector('#svg-lstick-dot')!;
    this.lStickHalo = this.root.querySelector('#svg-lstick-halo')!;
    this.rStickDot = this.root.querySelector('#svg-rstick-dot')!;
    this.rStickHalo = this.root.querySelector('#svg-rstick-halo')!;
    this.l2Bar = this.root.querySelector('#svg-l2-fill')!;
    this.r2Bar = this.root.querySelector('#svg-r2-fill')!;
    this.lHapticRing = this.root.querySelector('#svg-haptic-l')!;
    this.rHapticRing = this.root.querySelector('#svg-haptic-r')!;

    this.root.querySelectorAll<SVGElement>('[data-btn]').forEach((el) => {
      this.svgBtnEls.set(el.dataset.btn!, el);
    });

    this.connectBtn.addEventListener('click', handlers.onConnectClick);

    const grid = this.root.querySelector('#ps5-map-grid')!;
    for (const item of MAPPING_ITEMS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'ps5-map-pill';
      btn.dataset.mapKey = item.key;
      btn.style.setProperty('--btn-accent', item.color);
      const symClass = item.round ? 'ps5-map-sym ps5-map-sym-round' : 'ps5-map-sym';
      btn.innerHTML = `<span class="${symClass}">${item.symbol}</span><span class="ps5-map-lbl">${item.label}</span>`;
      btn.addEventListener('click', () => {
        this.flashAction(item.key, item.label);
        handlers.onTriggerAction(item.key);
      });
      grid.appendChild(btn);
      this.mapPillEls.set(item.key, btn);
    }
  }

  public setVisible(show: boolean): void {
    this.visible = show;
    this.root.classList.toggle('hidden', !show);
  }

  public isVisible(): boolean {
    return this.visible;
  }

  public setConnectionStatus(
    connected: boolean,
    badgeLabel: string,
    _snapshot?: PS5ControllerState
  ): void {
    this.connectedState = connected;
    this.root.classList.toggle('connected', connected);
    this.statusBadge.textContent = badgeLabel;
    this.connectBtn.textContent = connected ? 'Reconnect' : 'Connect';
    this.exclusiveStatus.textContent = connected
      ? 'PS5 EXCLUSIVE · KEYS LOCKED'
      : 'Connect PS5 · WebHID';
    if (!connected) {
      this.hapticLDecay = 0;
      this.hapticRDecay = 0;
      this.lHapticRing.style.opacity = '0';
      this.rHapticRing.style.opacity = '0';
      this.hapticLabel.classList.remove('firing');
      this.hapticLabel.textContent = 'Haptics Ready';
      this.toastClearAt = 0;
      this.actionToast.classList.remove('active');
      this.actionToast.textContent = 'L-Stick Move · R-Stick Cam';
    } else if (!this.hapticLabel.classList.contains('firing')) {
      this.hapticLabel.textContent = 'Neural Haptics';
    }
  }

  public flashAction(buttonKey: string, actionLabel: string): void {
    const now = performance.now();
    this.flashUntil.set(buttonKey, now + 420);
    if (buttonKey === 'dpadLeft' || buttonKey === 'dpadRight') {
      this.flashUntil.set('dpadLR', now + 420);
    }

    const item = MAPPING_ITEMS.find(
      (m) =>
        m.key === buttonKey ||
        ((buttonKey === 'dpadLeft' || buttonKey === 'dpadRight') && m.key === 'dpadLR')
    );
    const sym = item ? item.symbol : buttonKey === 'dpadUp' ? '↑' : buttonKey.toUpperCase();
    this.actionToast.textContent = `${sym}  →  ${actionLabel}`;
    this.actionToast.classList.add('active');
    this.toastClearAt = now + 1350;
  }

  public pulseHapticVisual(left: number, right: number, reason: string): void {
    const now = performance.now();
    this.hapticLDecay = Math.max(this.hapticLDecay, Math.min(1, left * 1.45));
    this.hapticRDecay = Math.max(this.hapticRDecay, Math.min(1, right * 1.45));
    this.hapticLabel.textContent = `Haptic · ${reason}`;
    this.hapticLabel.classList.add('firing');
    this.hapticReasonClearAt = now + 480;
  }

  /**
   * Updates all live SVG buttons, analog sticks, trigger bars, haptic rings, and mapped pills at 60 fps.
   */
  public update(snapshot: PS5ControllerState, rawDt: number, nowMs: number): void {
    if (!this.visible) return;

    // 1. Live Analog Left & Right Thumbstick Orbs + Halo Glow
    const lx = snapshot.sticks.left.x;
    const ly = snapshot.sticks.left.y;
    const rx = snapshot.sticks.right.x;
    const ry = snapshot.sticks.right.y;

    const lCx = (115 + lx * 11).toFixed(2);
    const lCy = (110 - ly * 11).toFixed(2);
    const rCx = (205 + rx * 11).toFixed(2);
    const rCy = (110 - ry * 11).toFixed(2);

    const lMoving = Math.hypot(lx, ly) > 0.14;
    const rMoving = Math.hypot(rx, ry) > 0.14;
    const r3Active = Boolean(snapshot.buttons.r3) || (this.flashUntil.get('r3') ?? 0) > nowMs;

    this.lStickDot.setAttribute('cx', lCx);
    this.lStickDot.setAttribute('cy', lCy);
    this.lStickDot.classList.toggle('active', lMoving);
    this.lStickHalo.setAttribute('cx', lCx);
    this.lStickHalo.setAttribute('cy', lCy);
    this.lStickHalo.style.opacity = lMoving ? '1' : this.connectedState ? '0.68' : '0.22';

    this.rStickDot.setAttribute('cx', rCx);
    this.rStickDot.setAttribute('cy', rCy);
    this.rStickDot.classList.toggle('active', rMoving || r3Active);
    this.rStickHalo.setAttribute('cx', rCx);
    this.rStickHalo.setAttribute('cy', rCy);
    this.rStickHalo.style.opacity = rMoving || r3Active ? '1' : '0';

    // 2. Live Analog Trigger Bars (L2 & R2) + data-btn active state
    const l2Val = Math.max(snapshot.triggers.left, snapshot.buttons.l2Digital ? 1 : 0);
    const r2Val = Math.max(snapshot.triggers.right, snapshot.buttons.r2Digital ? 1 : 0);
    const l2Flash = (this.flashUntil.get('l2') ?? 0) > nowMs ? 1 : 0;
    const r2Flash = (this.flashUntil.get('r2') ?? 0) > nowMs ? 1 : 0;
    const l2Eff = Math.max(l2Val, l2Flash);
    const r2Eff = Math.max(r2Val, r2Flash);
    this.l2Bar.setAttribute('width', (48 * l2Eff).toFixed(1));
    this.r2Bar.setAttribute('width', (48 * r2Eff).toFixed(1));
    const l2Lit = l2Eff > 0.42;
    const r2Lit = r2Eff > 0.45;
    this.svgBtnEls.get('l2')?.classList.toggle('active', l2Lit);
    this.svgBtnEls.get('r2')?.classList.toggle('active', r2Lit);

    // 3. Every SVG button active glow
    const btns = snapshot.buttons;
    const checkSvg = (key: string, physical: boolean): boolean => {
      const lit = physical || (this.flashUntil.get(key) ?? 0) > nowMs;
      this.svgBtnEls.get(key)?.classList.toggle('active', lit);
      return lit;
    };

    const crossLit = checkSvg('cross', btns.cross);
    const circleLit = checkSvg('circle', btns.circle);
    const squareLit = checkSvg('square', btns.square);
    const triLit = checkSvg('triangle', btns.triangle);
    const l1Lit = checkSvg('l1', btns.l1);
    const r1Lit = checkSvg('r1', btns.r1);
    checkSvg('dpadUp', btns.dpadUp);
    const downLit = checkSvg('dpadDown', btns.dpadDown);
    const leftLit = checkSvg('dpadLeft', btns.dpadLeft);
    const rightLit = checkSvg('dpadRight', btns.dpadRight);
    const padLit = checkSvg('touchpad', btns.touchpad);
    const optLit = checkSvg('options', btns.options);
    const createLit = checkSvg('create', btns.create);
    const l3Lit = checkSvg('l3', btns.l3);
    const r3Lit = checkSvg('r3', btns.r3);

    // 4. Sync 2-column live Button -> Action mapping pills
    const setPill = (key: string, lit: boolean): void => {
      this.mapPillEls
        .get(key)
        ?.classList.toggle('active', lit || (this.flashUntil.get(key) ?? 0) > nowMs);
    };
    setPill('cross', crossLit);
    setPill('circle', circleLit);
    setPill('square', squareLit);
    setPill('triangle', triLit);
    setPill('r2', r2Lit);
    setPill('l2', l2Lit);
    setPill('r1', r1Lit || l3Lit);
    setPill('l1', l1Lit);
    setPill('r3', r3Lit);
    setPill('dpadLR', leftLit || rightLit);
    setPill('dpadDown', downLit);
    setPill('touchpad', padLit);
    setPill('options', optLit);
    setPill('create', createLit);

    // 5. Decay Live Haptic Grip Arcs
    const decay = Math.exp(-rawDt * 8.5);
    this.hapticLDecay = this.hapticLDecay < 0.02 ? 0 : this.hapticLDecay * decay;
    this.hapticRDecay = this.hapticRDecay < 0.02 ? 0 : this.hapticRDecay * decay;
    this.lHapticRing.style.opacity = this.hapticLDecay.toFixed(3);
    this.rHapticRing.style.opacity = this.hapticRDecay.toFixed(3);

    if (nowMs > this.hapticReasonClearAt && this.hapticLabel.classList.contains('firing')) {
      this.hapticLabel.classList.remove('firing');
      this.hapticLabel.textContent = this.connectedState ? 'Neural Haptics' : 'Haptics Ready';
    }

    // Keep action toast live whenever a physical DualSense button/trigger is actively held down
    const heldLabel =
      btns.dpadDown ? '↓  →  Overhead Duck' :
      btns.dpadUp ? '↑  →  Emote' :
      btns.dpadRight ? '→  →  Next Style' :
      btns.dpadLeft ? '←  →  Previous Style' :
      btns.square ? '□  →  Left Jab / Combo' :
      btns.cross ? '✕  →  Jump / Vault' :
      btns.circle ? '○  →  Crouch Toggle' :
      btns.triangle ? '△  →  Sit / Stand' :
      btns.r3 ? 'R3  →  360° Combat Roll' :
      btns.l1 ? 'L1  →  Wide / Center Cam' :
      btns.touchpad ? 'PAD  →  Neural X-Ray' :
      btns.options ? 'OPT  →  AI Guides' :
      btns.create ? 'CRT  →  Reset Rig' :
      r2Val > 0.55 ? 'R2  →  Heavy Right Cross' :
      l2Val > 0.42 ? 'L2  →  Slow-Mo (Hold)' :
      null;

    if (heldLabel) {
      this.actionToast.textContent = heldLabel;
      this.actionToast.classList.add('active');
    } else if (nowMs > this.toastClearAt && this.actionToast.classList.contains('active')) {
      this.actionToast.classList.remove('active');
      this.actionToast.textContent = 'L-Stick Move · R-Stick Cam';
    }
  }
}
