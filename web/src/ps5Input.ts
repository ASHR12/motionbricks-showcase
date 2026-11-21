import {
  ChargeStatus,
  PS5Controller,
  TriggerEffect,
  createNeutralSnapshot,
  getDualSenseSupport,
  type PS5ControllerButton,
  type PS5ControllerState,
} from 'ps5-controller-webhid';

export type PS5InputSource = 'webhid' | 'gamepad' | 'simulated' | 'none';

export interface PS5ActionCallbacks {
  onJump: () => void;
  onCrouch: () => void;
  onPunchLeft: () => void;
  onPunchRight: () => void;
  onSit: () => void;
  onDuck: () => void;
  onRoll?: () => void;
  onEmote: () => void;
  onCycleStyle: (delta: -1 | 1) => void;
  onToggleWideCam: () => void;
  onToggleSlowMo: () => void;
  onToggleXRay: () => void;
  onToggleGuides: () => void;
  onReset: () => void;
  onStatusChange?: (
    connected: boolean,
    label: string,
    snapshot: PS5ControllerState,
    source?: PS5InputSource
  ) => void;
  onActionTriggered?: (buttonKey: string, actionLabel: string) => void;
  onHapticPulse?: (left: number, right: number, reason: string) => void;
}

export interface PS5FrameInput {
  connected: boolean;
  source: PS5InputSource;
  transport: string;
  snapshot: PS5ControllerState;
  steerX: number;
  steerY: number; // +1 forward (stick up), -1 back
  camYawDelta: number;
  camPitchDelta: number;
  sprint: boolean;
  slowMoHold: boolean;
}

export type PS5SimOverride = Partial<{
  connected: boolean;
  transport: PS5ControllerState['transport'];
  batteryLevel: number;
  leftStick: { x: number; y: number };
  rightStick: { x: number; y: number };
  leftTrigger: number;
  rightTrigger: number;
  buttons: Partial<Record<PS5ControllerButton, boolean>>;
}> | null;

export function formatBatteryPercent(snapshot?: PS5ControllerState): number {
  if (!snapshot) return 100;
  const raw = snapshot.battery.level;
  const pct = raw <= 1 ? Math.round(raw * 100) : Math.round(raw);
  return Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 100));
}

export function formatPS5TopBarLabel(
  connected: boolean,
  source: PS5InputSource,
  snapshot?: PS5ControllerState
): string {
  if (!connected) return 'Connect PS5';
  const pct = formatBatteryPercent(snapshot);
  if (source === 'gamepad') {
    return `PS5 Gamepad · ${pct}%`;
  }
  const transport = snapshot?.transport ?? 'Bluetooth';
  if (transport === 'USB') {
    return `PS5 USB · ${pct}%`;
  }
  return `PS5 DualSense · ${pct}%`;
}

export function formatPS5HudBadge(
  connected: boolean,
  source: PS5InputSource,
  snapshot?: PS5ControllerState
): string {
  if (!connected) return 'Disconnected';
  const pct = formatBatteryPercent(snapshot);
  if (source === 'gamepad') {
    return `Gamepad · ${pct}%`;
  }
  const transport = snapshot?.transport ?? 'Bluetooth';
  return `${transport} · ${pct}%`;
}

/**
 * Full-featured PS5 DualSense controller bridge powered by `ps5-controller-webhid`
 * (WebHID USB & Bluetooth: adaptive triggers, dual-motor haptics, battery, named buttons),
 * with automatic fallback to browser Gamepad API when a controller is plugged in prior to
 * WebHID pairing, plus a clean synchronous test/simulation override for automated verification.
 */
export class DualSenseBridge {
  private readonly controller: PS5Controller;
  private readonly callbacks: PS5ActionCallbacks;
  private readonly prevButtons: Partial<Record<PS5ControllerButton, boolean>> = {};
  private prevL2Pressed = false;
  private prevR2Pressed = false;
  private prevConnected = false;
  private prevSource: PS5InputSource = 'none';
  private prevLeftContact = true;
  private prevRightContact = true;
  private lastFootHapticMs = 0;
  private triggersConfigured = false;
  private lastErrorMsg = '';
  private simOverride: PS5SimOverride = null;

  constructor(callbacks: PS5ActionCallbacks) {
    this.callbacks = callbacks;
    this.controller = new PS5Controller({
      pollRateHz: 60,
      stickDeadzone: 0.1,
    });

    this.controller.onError((err) => {
      this.lastErrorMsg = err.message;
      console.warn('[ps5-controller-webhid]', err.message);
      this.callbacks.onStatusChange?.(false, err.message, this.controller.current, 'none');
    });

    // Listen to async WebHID connection state transitions (handles USB & Bluetooth handshake cleanly)
    this.controller.subscribe((snap) => {
      if (this.simOverride) return;
      if (snap.connected && !this.prevConnected) {
        this.prevConnected = true;
        this.prevSource = 'webhid';
        this.resetEdgeState();
        this.configureAdaptiveTriggers();
        void this.pulseHaptic(0.35, 0.35, 130, 'Connected');
        this.callbacks.onStatusChange?.(
          true,
          formatPS5TopBarLabel(true, 'webhid', snap),
          snap,
          'webhid'
        );
      } else if (!snap.connected && this.prevConnected && this.prevSource === 'webhid') {
        this.prevConnected = false;
        this.prevSource = 'none';
        this.triggersConfigured = false;
        this.resetEdgeState();
        this.callbacks.onStatusChange?.(false, 'Connect PS5', snap, 'none');
      }
    }, false);

    window.addEventListener('beforeunload', () => {
      this.dispose();
    });
  }

  private resetEdgeState(): void {
    for (const k of Object.keys(this.prevButtons) as PS5ControllerButton[]) {
      delete this.prevButtons[k];
    }
    this.prevL2Pressed = false;
    this.prevR2Pressed = false;
  }

  public getSupport() {
    return getDualSenseSupport();
  }

  public isWebHIDSupported(): boolean {
    return getDualSenseSupport().supported;
  }

  public getLastError(): string {
    return this.lastErrorMsg;
  }

  public getSource(): PS5InputSource {
    if (this.simOverride) {
      return (this.simOverride.connected ?? true) ? 'simulated' : 'none';
    }
    return this.prevSource;
  }

  public isConnected(): boolean {
    if (this.simOverride) {
      return this.simOverride.connected ?? true;
    }
    return this.prevConnected || this.controller.current.connected;
  }

  public async connect(): Promise<{ ok: boolean; error?: string }> {
    const sup = getDualSenseSupport();
    if (!sup.supported) {
      const msg = !sup.webHID
        ? 'WebHID requires desktop Chrome or Edge'
        : 'WebHID requires localhost or HTTPS';
      this.lastErrorMsg = msg;
      this.callbacks.onStatusChange?.(false, msg, this.controller.current, 'none');
      return { ok: false, error: msg };
    }
    try {
      this.lastErrorMsg = '';
      await this.controller.connect();
      const snap = this.controller.read();
      if (snap.connected) {
        this.prevConnected = true;
        this.prevSource = 'webhid';
        this.resetEdgeState();
        this.configureAdaptiveTriggers();
        void this.pulseHaptic(0.35, 0.35, 130, 'Connected');
        this.callbacks.onStatusChange?.(
          true,
          formatPS5TopBarLabel(true, 'webhid', snap),
          snap,
          'webhid'
        );
      }
      return { ok: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'PS5 connection cancelled';
      this.lastErrorMsg = msg;
      return { ok: false, error: msg };
    }
  }

  public setSimulation(override: PS5SimOverride): void {
    const wasConnected = this.prevConnected;
    const prevSource = this.prevSource;
    this.simOverride = override;

    if (override) {
      const snap = this.buildSimulatedSnapshot();
      const nowConnected = snap.connected;
      const nowSource: PS5InputSource = nowConnected ? 'simulated' : 'none';
      if (nowConnected !== wasConnected || nowSource !== prevSource) {
        this.prevConnected = nowConnected;
        this.prevSource = nowSource;
        this.resetEdgeState();
        if (!nowConnected) {
          this.triggersConfigured = false;
        }
        this.callbacks.onStatusChange?.(
          nowConnected,
          formatPS5TopBarLabel(nowConnected, nowSource, snap),
          snap,
          nowSource
        );
      } else if (nowConnected && (override.transport !== undefined || override.batteryLevel !== undefined)) {
        this.callbacks.onStatusChange?.(
          true,
          formatPS5TopBarLabel(true, nowSource, snap),
          snap,
          nowSource
        );
      }
    } else {
      const hidSnap = this.controller.read();
      const gpSnap = hidSnap.connected ? null : this.readStandardGamepadFallback();
      const snap = hidSnap.connected ? hidSnap : (gpSnap ?? createNeutralSnapshot());
      const nowConnected = snap.connected;
      const nowSource: PS5InputSource = hidSnap.connected
        ? 'webhid'
        : gpSnap
          ? 'gamepad'
          : 'none';
      if (nowConnected !== wasConnected || nowSource !== prevSource) {
        this.prevConnected = nowConnected;
        this.prevSource = nowSource;
        this.resetEdgeState();
        if (!nowConnected) {
          this.triggersConfigured = false;
        }
        this.callbacks.onStatusChange?.(
          nowConnected,
          formatPS5TopBarLabel(nowConnected, nowSource, snap),
          snap,
          nowSource
        );
      }
    }
  }

  private configureAdaptiveTriggers(): void {
    if (this.triggersConfigured) return;
    try {
      // R2: weapon-style tactile resistance snap for Heavy Right Cross punch
      this.controller.setTriggerFeedback('right', {
        effect: TriggerEffect.Feedback,
        position: 0.24,
        strength: 0.68,
      });
      // L2: smooth progressive tension for Bullet-Time Slow-Mo
      this.controller.setTriggerFeedback('left', {
        effect: TriggerEffect.Feedback,
        position: 0.32,
        strength: 0.38,
      });
      this.triggersConfigured = true;
    } catch {
      // Transport may be limited
    }
  }

  private async pulseHaptic(
    left: number,
    right: number,
    durationMs: number,
    reason: string
  ): Promise<void> {
    if (!this.isConnected()) return;
    this.callbacks.onHapticPulse?.(left, right, reason);
    if (this.controller.current.connected) {
      try {
        await this.controller.pulseRumble(left, right, durationMs);
      } catch {
        // Ignore transient HID output errors
      }
    }
  }

  public triggerPunchHaptic(side: 'left' | 'right'): void {
    if (side === 'left') {
      void this.pulseHaptic(0.58, 0.22, 110, 'L Jab');
    } else {
      void this.pulseHaptic(0.32, 0.88, 150, 'R Cross');
    }
  }

  public triggerVaultHaptic(): void {
    void this.pulseHaptic(0.65, 0.65, 135, 'Vault');
  }

  /**
   * Fires subtle left/right haptic footsteps synced live to MotionBricks' ONNX `foot_contacts`!
   */
  public syncNeuralFootContacts(
    contacts: readonly [number, number, number, number],
    isMoving: boolean,
    nowMs: number
  ): void {
    const leftNow = Math.max(contacts[0], contacts[1]) > 0.78;
    const rightNow = Math.max(contacts[2], contacts[3]) > 0.78;

    if (isMoving && nowMs - this.lastFootHapticMs > 165) {
      if (leftNow && !this.prevLeftContact) {
        this.lastFootHapticMs = nowMs;
        void this.pulseHaptic(0.26, 0.03, 45, 'L Foot');
      } else if (rightNow && !this.prevRightContact) {
        this.lastFootHapticMs = nowMs;
        void this.pulseHaptic(0.03, 0.26, 45, 'R Foot');
      }
    }
    this.prevLeftContact = leftNow;
    this.prevRightContact = rightNow;
  }

  /**
   * Fallback reader converting standard browser Gamepad API into the exact PS5ControllerState shape
   * if user plugged in a DualSense before clicking Connect PS5 (WebHID).
   */
  private readStandardGamepadFallback(): PS5ControllerState | null {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    const gp = pads.find((p) => p && p.connected);
    if (!gp) return null;

    const snap = createNeutralSnapshot();
    const b = (i: number): boolean => Boolean(gp.buttons[i]?.pressed);
    const val = (i: number): number => gp.buttons[i]?.value ?? 0;
    const dz = (v: number): number => (Math.abs(v) > 0.12 ? v : 0);

    snap.connected = true;
    snap.transport = 'USB';
    snap.productName = gp.id || 'Gamepad';
    snap.battery.level = 1;
    snap.battery.status = ChargeStatus.Full;

    // Standard Gamepad Y is down (-1 up), DualSense WebHID Y is up (+1 up)
    snap.sticks.left.x = dz(gp.axes[0] ?? 0);
    snap.sticks.left.y = -dz(gp.axes[1] ?? 0);
    snap.sticks.right.x = dz(gp.axes[2] ?? 0);
    snap.sticks.right.y = -dz(gp.axes[3] ?? 0);

    snap.triggers.left = val(6);
    snap.triggers.right = val(7);

    snap.buttons.cross = b(0);
    snap.buttons.circle = b(1);
    snap.buttons.square = b(2);
    snap.buttons.triangle = b(3);
    snap.buttons.l1 = b(4);
    snap.buttons.r1 = b(5);
    snap.buttons.l2Digital = b(6);
    snap.buttons.r2Digital = b(7);
    snap.buttons.create = b(8);
    snap.buttons.options = b(9);
    snap.buttons.l3 = b(10);
    snap.buttons.r3 = b(11);
    snap.buttons.dpadUp = b(12);
    snap.buttons.dpadDown = b(13);
    snap.buttons.dpadLeft = b(14);
    snap.buttons.dpadRight = b(15);
    snap.buttons.ps = b(16);
    snap.buttons.touchpad = b(17);
    return snap;
  }

  private buildSimulatedSnapshot(): PS5ControllerState {
    const snap = createNeutralSnapshot();
    const sim = this.simOverride!;
    snap.connected = sim.connected ?? true;
    snap.transport = sim.transport ?? 'Bluetooth';
    snap.productName = 'DualSense Wireless Controller';
    snap.battery.level = sim.batteryLevel ?? 0.86;
    snap.battery.status = ChargeStatus.Discharging;
    if (sim.leftStick) snap.sticks.left = { ...sim.leftStick };
    if (sim.rightStick) snap.sticks.right = { ...sim.rightStick };
    snap.triggers.left = sim.leftTrigger ?? 0;
    snap.triggers.right = sim.rightTrigger ?? 0;
    if (sim.buttons) {
      for (const [k, v] of Object.entries(sim.buttons)) {
        snap.buttons[k as PS5ControllerButton] = Boolean(v);
      }
    }
    return snap;
  }

  public poll(rawDt: number): PS5FrameInput {
    let snap: PS5ControllerState;
    let source: PS5InputSource = 'none';

    if (this.simOverride) {
      snap = this.buildSimulatedSnapshot();
      source = snap.connected ? 'simulated' : 'none';
    } else {
      const hidSnap = this.controller.read();
      if (hidSnap.connected) {
        snap = hidSnap;
        source = 'webhid';
      } else {
        const gpSnap = this.readStandardGamepadFallback();
        if (gpSnap) {
          snap = gpSnap;
          source = 'gamepad';
        } else {
          snap = hidSnap;
          source = 'none';
        }
      }
    }

    const connected = snap.connected;
    if (connected !== this.prevConnected || source !== this.prevSource) {
      this.prevConnected = connected;
      this.prevSource = source;
      this.resetEdgeState();
      if (connected) {
        if (source === 'webhid') this.configureAdaptiveTriggers();
        this.callbacks.onStatusChange?.(
          true,
          formatPS5TopBarLabel(true, source, snap),
          snap,
          source
        );
      } else {
        this.triggersConfigured = false;
        this.callbacks.onStatusChange?.(false, 'Connect PS5', snap, 'none');
      }
    }

    if (!connected) {
      return {
        connected: false,
        source: 'none',
        transport: 'Disconnected',
        snapshot: snap,
        steerX: 0,
        steerY: 0,
        camYawDelta: 0,
        camPitchDelta: 0,
        sprint: false,
        slowMoHold: false,
      };
    }

    const justPressed = (key: PS5ControllerButton): boolean => {
      const cur = Boolean(snap.buttons[key]);
      const prev = Boolean(this.prevButtons[key]);
      this.prevButtons[key] = cur;
      return cur && !prev;
    };

    if (justPressed('cross')) {
      this.callbacks.onActionTriggered?.('cross', 'Jump / Vault');
      this.callbacks.onJump();
    }
    if (justPressed('circle')) {
      this.callbacks.onActionTriggered?.('circle', 'Crouch Toggle');
      this.callbacks.onCrouch();
    }
    if (justPressed('square')) {
      this.triggerPunchHaptic('left');
      this.callbacks.onActionTriggered?.('square', 'Left Jab / Combo');
      this.callbacks.onPunchLeft();
    }
    if (justPressed('triangle')) {
      this.callbacks.onActionTriggered?.('triangle', 'Sit / Stand');
      this.callbacks.onSit();
    }
    if (justPressed('dpadLeft')) {
      this.callbacks.onActionTriggered?.('dpadLeft', 'Previous Style');
      this.callbacks.onCycleStyle(-1);
    }
    if (justPressed('dpadRight')) {
      this.callbacks.onActionTriggered?.('dpadRight', 'Next Style');
      this.callbacks.onCycleStyle(1);
    }
    if (justPressed('dpadDown')) {
      this.callbacks.onActionTriggered?.('dpadDown', 'Overhead Duck');
      this.callbacks.onDuck();
    }
    if (justPressed('dpadUp')) {
      this.callbacks.onActionTriggered?.('dpadUp', 'Emote');
      this.callbacks.onEmote();
    }
    if (justPressed('l1')) {
      this.callbacks.onActionTriggered?.('l1', 'Wide / Center Cam');
      this.callbacks.onToggleWideCam();
    }
    if (justPressed('touchpad')) {
      this.callbacks.onActionTriggered?.('touchpad', 'Neural X-Ray');
      this.callbacks.onToggleXRay();
    }
    if (justPressed('options')) {
      this.callbacks.onActionTriggered?.('options', 'AI Guides');
      this.callbacks.onToggleGuides();
    }
    if (justPressed('create')) {
      this.callbacks.onActionTriggered?.('create', 'Reset Rig');
      this.callbacks.onReset();
    }
    if (justPressed('r3')) {
      this.callbacks.onActionTriggered?.('r3', '360° Combat Roll');
      this.callbacks.onRoll?.();
    }

    // L2 analog pull -> Hold 0.25x Bullet-Time Slow-Mo (strictly separate from L1!)
    const l2Now = snap.triggers.left > 0.42 || Boolean(snap.buttons.l2Digital);
    if (l2Now && !this.prevL2Pressed) {
      this.callbacks.onActionTriggered?.('l2', 'Slow-Mo (Hold)');
    }
    this.prevL2Pressed = l2Now;

    // R2 analog pull -> Heavy Right Cross Punch!
    const r2Now = snap.triggers.right > 0.55 || Boolean(snap.buttons.r2Digital);
    if (r2Now && !this.prevR2Pressed) {
      this.triggerPunchHaptic('right');
      this.callbacks.onActionTriggered?.('r2', 'Heavy Right Cross');
      this.callbacks.onPunchRight();
    }
    this.prevR2Pressed = r2Now;

    const lx = snap.sticks.left.x;
    const ly = snap.sticks.left.y;
    const rx = snap.sticks.right.x;
    const ry = snap.sticks.right.y;
    const lMag = Math.hypot(lx, ly);

    return {
      connected: true,
      source,
      transport: snap.transport,
      snapshot: snap,
      steerX: lMag > 0.13 ? lx : 0,
      steerY: lMag > 0.13 ? ly : 0,
      camYawDelta: -rx * rawDt * 2.55,
      camPitchDelta: -ry * rawDt * 1.65,
      sprint: Boolean(snap.buttons.r1 || snap.buttons.l3),
      slowMoHold: l2Now,
    };
  }

  public dispose(): void {
    try {
      this.controller.resetOutputs();
      this.controller.dispose();
    } catch {
      // ignore on unload
    }
  }
}
