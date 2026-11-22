import * as THREE from 'three';
import { cloneConcreteTextures } from './pavilion';

export interface StudioObstacleSpec {
  id: string;
  type: 'vault' | 'sit' | 'duck' | 'hurdle' | 'block';
  label: string;
  pos: [number, number, number];
  size: [number, number, number];
  clearance?: number;
  facing?: [number, number];
  color: string;
}

export interface StudioObstacleManager {
  getClickTargets(): THREE.Object3D[];
  getEditTargets(): THREE.Object3D[];
  syncObstacle(obs: StudioObstacleSpec): void;
  syncAll(specs: StudioObstacleSpec[]): void;
  setEditVisuals(editActive: boolean, hoveredId: string | null, selectedId: string | null): void;
}

/**
 * Builds a chamfered trapezoidal concrete vault monolith in the XY cross-section extruded along Z.
 * Final dimensions: bottom width = baseW, top width = topW, height = baseY..topY, length = lengthZ.
 */
function createVaultMonolithGeometry(
  baseW: number,
  topW: number,
  baseY: number,
  topY: number,
  lengthZ: number,
  bevel = 0.014
): THREE.BufferGeometry {
  const bHalf = baseW * 0.5 - bevel;
  const tHalf = topW * 0.5 - bevel;
  const y0 = baseY + bevel;
  const y1 = topY - bevel;

  const shape = new THREE.Shape();
  shape.moveTo(-bHalf, y0);
  shape.lineTo(bHalf, y0);
  shape.lineTo(tHalf, y1);
  shape.lineTo(-tHalf, y1);
  shape.closePath();

  const innerDepth = Math.max(0.05, lengthZ - bevel * 2);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: innerDepth,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 3,
    curveSegments: 2,
  });
  geo.translate(0, 0, -innerDepth * 0.5);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a chamfered architectural slab (used for the Gallery Bench seat) whose exact
 * top surface lies at `topY` and outer bounds equal `[widthX, thicknessY, lengthZ]`.
 */
function createChamferedSlabGeometry(
  widthX: number,
  thicknessY: number,
  lengthZ: number,
  topY: number,
  bevel = 0.012
): THREE.BufferGeometry {
  const hx = widthX * 0.5 - bevel;
  const yTop = topY - bevel;
  const yBot = topY - thicknessY + bevel;

  const shape = new THREE.Shape();
  shape.moveTo(-hx, yBot);
  shape.lineTo(hx, yBot);
  shape.lineTo(hx, yTop);
  shape.lineTo(-hx, yTop);
  shape.closePath();

  const innerDepth = Math.max(0.04, lengthZ - bevel * 2);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: innerDepth,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 3,
    curveSegments: 2,
  });
  geo.translate(0, 0, -innerDepth * 0.5);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds Nike/Apple-industrial brutalist studio parkour & interaction props matching targets/t01.png
 * while preserving exact collision/interaction datums (Vault top = 0.82 m, Sit top = 0.44 m,
 * Duck underside clearance = 1.35 / 1.32 m, Hurdle rail top = h, Plyo block top = h)
 * and click-target `userData.obstacle` bindings.
 */
export function buildStudioObstacles(
  specs: StudioObstacleSpec[],
  scene: THREE.Scene,
  sensorZoneGroup: THREE.Group
): THREE.Object3D[] {
  return createStudioObstacleManager(specs, scene, sensorZoneGroup).getClickTargets();
}

interface ManagedObstacleEntry {
  id: string;
  type: StudioObstacleSpec['type'];
  propGroup: THREE.Group;
  orientGroup: THREE.Group | null;
  zoneGroup: THREE.Group;
  selGroup: THREE.Group;
  selFillMat: THREE.MeshBasicMaterial;
  selEdgeMat: THREE.LineBasicMaterial;
  hitMeshes: THREE.Object3D[];
}

export function createStudioObstacleManager(
  specs: StudioObstacleSpec[],
  scene: THREE.Scene,
  sensorZoneGroup: THREE.Group
): StudioObstacleManager {
  const clickTargets: THREE.Object3D[] = [];
  const editTargets: THREE.Object3D[] = [];
  const entries = new Map<string, ManagedObstacleEntry>();
  const editOverlayGroup = new THREE.Group();
  editOverlayGroup.name = 'EditGymOverlays';
  editOverlayGroup.visible = false;
  scene.add(editOverlayGroup);

  const registerClickable = (meshes: THREE.Object3D[], obs: StudioObstacleSpec, bucket: THREE.Object3D[]) => {
    for (const m of meshes) {
      m.userData = { obstacle: obs };
      clickTargets.push(m);
      editTargets.push(m);
      bucket.push(m);
    }
  };

  const makeSelectionFootprint = (footW: number, footD: number, colorHex: string) => {
    const selGroup = new THREE.Group();
    const padW = footW + 0.24;
    const padD = footD + 0.24;
    const selFillMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(colorHex || '#38bdf8'),
      transparent: true,
      opacity: 0.08,
      depthWrite: false,
    });
    const fill = new THREE.Mesh(new THREE.PlaneGeometry(padW, padD).rotateX(-Math.PI / 2), selFillMat);
    fill.position.y = 0.011;

    const selEdgeMat = new THREE.LineBasicMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.55,
    });
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(padW, 0.014, padD)),
      selEdgeMat
    );
    outline.position.y = 0.014;
    selGroup.add(fill, outline);
    editOverlayGroup.add(selGroup);
    return { selGroup, selFill: fill, selFillMat, selEdgeMat };
  };

  // Architectural material palette shared across props
  const vaultConcreteTex = cloneConcreteTextures(0.45, 0.45);
  const vaultConcreteMat = new THREE.MeshStandardMaterial({
    color: 0x2f333c,
    map: vaultConcreteTex.map,
    roughnessMap: vaultConcreteTex.roughnessMap,
    bumpMap: vaultConcreteTex.bumpMap,
    bumpScale: 0.01,
    roughness: 0.84,
    metalness: 0.06,
  });

  const slateConcreteTex = cloneConcreteTextures(0.55, 0.55);
  const slateBenchMat = new THREE.MeshStandardMaterial({
    color: 0x2d333f,
    map: slateConcreteTex.map,
    roughnessMap: slateConcreteTex.roughnessMap,
    bumpMap: slateConcreteTex.bumpMap,
    bumpScale: 0.005,
    roughness: 0.58,
    metalness: 0.12,
  });

  const plyoConcreteTex = cloneConcreteTextures(0.5, 0.5);
  const plyoUpperConcreteMat = new THREE.MeshStandardMaterial({
    color: 0x3b404a,
    map: plyoConcreteTex.map,
    roughnessMap: plyoConcreteTex.roughnessMap,
    bumpMap: plyoConcreteTex.bumpMap,
    bumpScale: 0.008,
    roughness: 0.78,
    metalness: 0.08,
  });

  const brushedAluminumMat = new THREE.MeshStandardMaterial({
    color: 0x9aa4b2,
    roughness: 0.28,
    metalness: 0.84,
  });

  const darkAnodizedMat = new THREE.MeshStandardMaterial({
    color: 0x1a1f28,
    roughness: 0.44,
    metalness: 0.62,
  });

  const darkBrushedAluMat = new THREE.MeshStandardMaterial({
    color: 0x2a313d,
    roughness: 0.34,
    metalness: 0.78,
  });

  const copperGripMat = new THREE.MeshStandardMaterial({
    color: 0xe59656,
    emissive: 0x4a2008,
    emissiveIntensity: 0.24,
    roughness: 0.24,
    metalness: 0.86,
  });

  const hurdleAmberSightMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    emissive: 0x5c2b04,
    emissiveIntensity: 0.34,
    roughness: 0.24,
    metalness: 0.8,
  });

  const copperRoseInlayMat = new THREE.MeshStandardMaterial({
    color: 0xd98668,
    emissive: 0x461c10,
    emissiveIntensity: 0.22,
    roughness: 0.26,
    metalness: 0.82,
  });

  const matteTealGlowMat = new THREE.MeshStandardMaterial({
    color: 0x2dd4bf,
    emissive: 0x14b8a6,
    emissiveIntensity: 0.78,
    roughness: 0.32,
    metalness: 0.14,
  });

  const applyTransform = (entry: ManagedObstacleEntry, obs: StudioObstacleSpec) => {
    entry.propGroup.position.set(obs.pos[0], 0, obs.pos[2]);
    entry.zoneGroup.position.set(obs.pos[0], 0, obs.pos[2]);
    entry.selGroup.position.set(obs.pos[0], 0, obs.pos[2]);

    if (entry.type === 'vault') {
      const rot = obs.size[0] > obs.size[2] ? Math.PI / 2 : 0;
      if (entry.orientGroup) entry.orientGroup.rotation.y = rot;
      entry.zoneGroup.rotation.y = rot;
      entry.selGroup.rotation.y = rot;
    } else if (entry.type === 'duck' || entry.type === 'hurdle') {
      const rot = obs.size[2] > obs.size[0] ? Math.PI / 2 : 0;
      if (entry.orientGroup) entry.orientGroup.rotation.y = rot;
      entry.zoneGroup.rotation.y = rot;
      entry.selGroup.rotation.y = rot;
    } else {
      const yaw = obs.facing ? Math.atan2(-obs.facing[1], obs.facing[0]) : 0;
      entry.propGroup.rotation.y = yaw;
      entry.zoneGroup.rotation.y = yaw;
      entry.selGroup.rotation.y = yaw;
    }

    for (const m of entry.hitMeshes) {
      m.userData = { obstacle: obs };
    }
  };

  for (const obs of specs) {
    const [w, h, d] = obs.size;
    const propGroup = new THREE.Group();
    propGroup.name = `Prop_${obs.id}`;
    const zoneGroup = new THREE.Group();
    zoneGroup.name = `Zone_${obs.id}`;
    sensorZoneGroup.add(zoneGroup);
    const hitMeshes: THREE.Object3D[] = [];
    let orientGroup: THREE.Group | null = null;
    let selData: {
      selGroup: THREE.Group;
      selFill: THREE.Mesh;
      selFillMat: THREE.MeshBasicMaterial;
      selEdgeMat: THREE.LineBasicMaterial;
    };

    if (obs.type === 'vault') {
      // =====================================================================
      // SCULPTED TRAPEZOIDAL ARCHITECTURAL CONCRETE VAULT PLINTH (top = 0.82 m)
      // Built canonically along Z; orientGroup rotates 90° when size[0] > size[2]
      // =====================================================================
      const lengthSpan = Math.max(w, d);
      // Matches exact 0.34 m BVH vault obstacle envelope (clean shin & sneaker clearance on takeoff!)
      const baseW = 0.36;
      const topW = 0.24;
      const plinthH = 0.022;
      const monolithTopY = Math.max(0.4, h - 0.08); // 0.74 m when h = 0.82 m

      const vaultInner = new THREE.Group();
      orientGroup = vaultInner;

      // Recessed shadow-gap dark anodized base plinth anchoring monolith to floor
      const baseSkirt = new THREE.Mesh(
        new THREE.BoxGeometry(baseW - 0.04, plinthH, lengthSpan - 0.04),
        darkAnodizedMat
      );
      baseSkirt.position.y = plinthH * 0.5;
      baseSkirt.castShadow = true;
      baseSkirt.receiveShadow = true;

      // Tapered trapezoidal chamfered poured-concrete monolith (base 0.62 m -> top 0.32 m, height 0.74 m)
      const monolithGeo = createVaultMonolithGeometry(
        baseW,
        topW,
        plinthH,
        monolithTopY,
        lengthSpan,
        0.014
      );
      const monolith = new THREE.Mesh(monolithGeo, vaultConcreteMat);
      monolith.castShadow = true;
      monolith.receiveShadow = true;

      // Subtle machined dark-steel top inlay bed plate where the grip rail mounts
      const topBedPlate = new THREE.Mesh(
        new THREE.BoxGeometry(0.11, 0.008, Math.max(0.3, lengthSpan - 0.14)),
        darkAnodizedMat
      );
      topBedPlate.position.y = monolithTopY + 0.002;
      topBedPlate.receiveShadow = true;

      // Sleek brushed warm-copper/amber tubular vault grip rail reaching EXACT y = h (0.82 m)
      const tubeR = 0.021; // 42 mm grip diameter
      const railCenterY = h - tubeR; // 0.799 m => exact top tangent = 0.820 m
      const bendR = Math.max(0.025, railCenterY - monolithTopY); // 0.059 m exact quarter-turn to monolithTopY
      const straightSpan = Math.max(0.6, lengthSpan - 0.38); // continuous palm grip zone
      const halfSpan = straightSpan * 0.5;

      const railBar = new THREE.Mesh(
        new THREE.CylinderGeometry(tubeR, tubeR, straightSpan, 24),
        copperGripMat
      );
      railBar.rotation.x = Math.PI / 2;
      railBar.position.set(0, railCenterY, 0);
      railBar.castShadow = true;
      railBar.receiveShadow = true;

      // Smooth 90° curved end-turn stanchions joining horizontal bar seamlessly into monolith top
      const bendGeoPos = new THREE.TorusGeometry(bendR, tubeR, 18, 24, Math.PI / 2);
      bendGeoPos.rotateY(-Math.PI / 2);
      const bendPos = new THREE.Mesh(bendGeoPos, copperGripMat);
      bendPos.position.set(0, monolithTopY, halfSpan);
      bendPos.castShadow = true;

      const bendGeoNeg = new THREE.TorusGeometry(bendR, tubeR, 18, 24, Math.PI / 2);
      bendGeoNeg.rotateY(Math.PI / 2);
      const bendNeg = new THREE.Mesh(bendGeoNeg, copperGripMat);
      bendNeg.position.set(0, monolithTopY, -halfSpan);
      bendNeg.castShadow = true;

      // Machined dark-steel mounting flange escutcheons at both rail stanchion bases
      const flangeGeo = new THREE.CylinderGeometry(0.036, 0.044, 0.018, 20);
      const flangePos = new THREE.Mesh(flangeGeo, darkAnodizedMat);
      flangePos.position.set(0, monolithTopY + 0.009, halfSpan + bendR);
      flangePos.castShadow = true;

      const flangeNeg = new THREE.Mesh(flangeGeo, darkAnodizedMat);
      flangeNeg.position.set(0, monolithTopY + 0.009, -(halfSpan + bendR));
      flangeNeg.castShadow = true;

      vaultInner.add(
        baseSkirt,
        monolith,
        topBedPlate,
        railBar,
        bendPos,
        bendNeg,
        flangePos,
        flangeNeg
      );
      propGroup.add(vaultInner);
      registerClickable(
        [baseSkirt, monolith, topBedPlate, railBar, bendPos, bendNeg, flangePos, flangeNeg],
        obs,
        hitMeshes
      );

      // Floor proximity sensor pad aligned along canonical crossing axis (X crossing, Z length)
      const zoneW = 4.3;
      const zoneD = lengthSpan + 0.3;
      const zoneFill = new THREE.Mesh(
        new THREE.PlaneGeometry(zoneW, zoneD).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.055, depthWrite: false })
      );
      zoneFill.position.set(0, 0.005, 0);
      const zoneOutline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(zoneW, 0.01, zoneD)),
        new THREE.LineBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.42 })
      );
      zoneOutline.position.set(0, 0.007, 0);
      zoneGroup.add(zoneFill, zoneOutline);
      selData = makeSelectionFootprint(baseW + 0.16, lengthSpan, '#facc15');
    } else if (obs.type === 'sit') {
      // =====================================================================
      // SCULPTED GALLERY BENCH (exact seat top = 0.44 m, warm copper-rose inlay)
      // =====================================================================
      const seatThick = 0.074;
      const seatBottomY = h - seatThick; // 0.366 m when h = 0.44 m
      const glideH = 0.008;

      // Chamfered beveled architectural slate seat slab (exact top = h = 0.44 m)
      const seatGeo = createChamferedSlabGeometry(w, seatThick, d, h, 0.012);
      const seat = new THREE.Mesh(seatGeo, slateBenchMat);
      seat.castShadow = true;
      seat.receiveShadow = true;

      // Recessed dark brushed-aluminum plinth legs at left/right bench ends (open front for seated legs!)
      const legThickZ = 0.048;
      const legWidthX = w - 0.07;
      const legH = seatBottomY - glideH;
      const legOffsetZ = d * 0.5 - 0.062;
      const legGeo = new THREE.BoxGeometry(legWidthX, legH, legThickZ);

      const legL = new THREE.Mesh(legGeo, darkBrushedAluMat);
      legL.position.set(0, glideH + legH * 0.5, -legOffsetZ);
      legL.castShadow = true;
      legL.receiveShadow = true;

      const legR = new THREE.Mesh(legGeo, darkBrushedAluMat);
      legR.position.set(0, glideH + legH * 0.5, legOffsetZ);
      legR.castShadow = true;
      legR.receiveShadow = true;

      // Matte-black floor glides underneath each plinth leg
      const glideGeo = new THREE.BoxGeometry(legWidthX - 0.02, glideH, legThickZ - 0.008);
      const glideL = new THREE.Mesh(glideGeo, darkAnodizedMat);
      glideL.position.set(0, glideH * 0.5, -legOffsetZ);
      const glideR = new THREE.Mesh(glideGeo, darkAnodizedMat);
      glideR.position.set(0, glideH * 0.5, legOffsetZ);

      // Recessed horizontal architectural apron stretcher underneath the seat
      const stretcher = new THREE.Mesh(
        new THREE.BoxGeometry(w - 0.14, 0.032, legOffsetZ * 2 - legThickZ),
        darkAnodizedMat
      );
      stretcher.position.set(0, seatBottomY - 0.022, 0);
      stretcher.castShadow = true;

      // Crisp subtle warm copper-rose edge inlay wrapping the under-seat reveal & leg return
      const inlayPerimeter = new THREE.Mesh(
        new THREE.BoxGeometry(w - 0.016, 0.011, d - 0.016),
        copperRoseInlayMat
      );
      inlayPerimeter.position.set(0, seatBottomY - 0.005, 0);

      const legInlayGeo = new THREE.BoxGeometry(0.008, legH - 0.02, 0.01);
      const legInlayL = new THREE.Mesh(legInlayGeo, copperRoseInlayMat);
      legInlayL.position.set(legWidthX * 0.5 + 0.001, glideH + legH * 0.5, -legOffsetZ + legThickZ * 0.5 - 0.005);
      const legInlayR = new THREE.Mesh(legInlayGeo, copperRoseInlayMat);
      legInlayR.position.set(legWidthX * 0.5 + 0.001, glideH + legH * 0.5, legOffsetZ - legThickZ * 0.5 + 0.005);

      propGroup.add(seat, legL, legR, glideL, glideR, stretcher, inlayPerimeter, legInlayL, legInlayR);
      registerClickable([seat, legL, legR, stretcher, inlayPerimeter], obs, hitMeshes);
      selData = makeSelectionFootprint(w, d, '#f472b6');
    } else if (obs.type === 'duck') {
      // =====================================================================
      // OVERHEAD PARKOUR DUCK GANTRY (exact underside clearance = 1.35 m)
      // Built canonically spanning X; orientGroup rotates 90° when size[2] > size[0]
      // =====================================================================
      const spanW = Math.max(w, d);
      const clearH = obs.clearance ?? 1.35;
      const totalH = clearH + h; // 1.77 m
      const postW = 0.088;
      const postD = 0.098;
      const postX = spanW * 0.5 + postW * 0.5 + 0.008; // clear inner span > 2.2 m
      const baseH = 0.026;
      const topBeamH = 0.096;
      const topBeamD = 0.112;

      const duckInner = new THREE.Group();
      orientGroup = duckInner;

      // Twin brushed-aluminum square-tube portal legs
      const postGeo = new THREE.BoxGeometry(postW, totalH - baseH, postD);
      const postL = new THREE.Mesh(postGeo, brushedAluminumMat);
      postL.position.set(-postX, baseH + (totalH - baseH) * 0.5, 0);
      postL.castShadow = true;
      postL.receiveShadow = true;

      const postR = new THREE.Mesh(postGeo, brushedAluminumMat);
      postR.position.set(postX, baseH + (totalH - baseH) * 0.5, 0);
      postR.castShadow = true;
      postR.receiveShadow = true;

      // Bolted dark-steel architectural baseplates with 4 machined corner hex bolts each
      const footGeo = new THREE.BoxGeometry(0.22, baseH, 0.28);
      const boltGeo = new THREE.CylinderGeometry(0.009, 0.009, 0.012, 6);

      const footL = new THREE.Mesh(footGeo, darkAnodizedMat);
      footL.position.set(-postX, baseH * 0.5, 0);
      footL.receiveShadow = true;
      footL.castShadow = true;

      const footR = new THREE.Mesh(footGeo, darkAnodizedMat);
      footR.position.set(postX, baseH * 0.5, 0);
      footR.receiveShadow = true;
      footR.castShadow = true;

      for (const sideX of [-postX, postX]) {
        for (const bx of [-0.075, 0.075]) {
          for (const bz of [-0.105, 0.105]) {
            const bolt = new THREE.Mesh(boltGeo, brushedAluminumMat);
            bolt.position.set(sideX + bx, baseH + 0.006, bz);
            duckInner.add(bolt);
          }
        }
      }

      // Horizontal brushed-aluminum top portal truss beam spanning full portal width
      const topBeamW = postX * 2 + postW;
      const topBeam = new THREE.Mesh(
        new THREE.BoxGeometry(topBeamW, topBeamH, topBeamD),
        brushedAluminumMat
      );
      topBeam.position.set(0, totalH - topBeamH * 0.5, 0);
      topBeam.castShadow = true;
      topBeam.receiveShadow = true;

      // Recessed dark structural truss shadow channel running along the top beam
      const trussChannel = new THREE.Mesh(
        new THREE.BoxGeometry(spanW + 0.02, 0.036, topBeamD + 0.006),
        darkAnodizedMat
      );
      trussChannel.position.set(0, totalH - topBeamH * 0.5, 0);

      // Corner structural gusset sleeves joining legs cleanly to top portal beam
      const gussetGeo = new THREE.BoxGeometry(0.11, 0.11, topBeamD + 0.012);
      const gussetL = new THREE.Mesh(gussetGeo, darkBrushedAluMat);
      gussetL.position.set(-postX, totalH - topBeamH * 0.5, 0);
      const gussetR = new THREE.Mesh(gussetGeo, darkBrushedAluMat);
      gussetR.position.set(postX, totalH - topBeamH * 0.5, 0);

      // Suspended brushed-aluminum clearance housing + recessed matte-teal underside glow bar at EXACT y = 1.35 m
      const glowH = 0.012;
      const housingH = 0.032;
      const clearanceSpanW = spanW - 0.10;
      const housingBottomY = clearH + glowH; // 1.362 m
      const housingTopY = housingBottomY + housingH; // 1.394 m
      const topBeamBottomY = totalH - topBeamH; // 1.674 m

      const clearanceHousing = new THREE.Mesh(
        new THREE.BoxGeometry(clearanceSpanW + 0.04, housingH, 0.064),
        darkBrushedAluMat
      );
      clearanceHousing.position.set(0, housingBottomY + housingH * 0.5, 0);
      clearanceHousing.castShadow = true;
      clearanceHousing.receiveShadow = true;

      // Recessed glowing matte-teal underside clearance bar — bottom face sits at EXACT y = clearH (1.350 m)!
      const clearanceStrip = new THREE.Mesh(
        new THREE.BoxGeometry(clearanceSpanW, glowH, 0.032),
        matteTealGlowMat
      );
      clearanceStrip.position.set(0, clearH + glowH * 0.5, 0);

      // Sleek twin vertical brushed-aluminum suspension drop-rods connecting top beam to clearance bar
      const dropH = Math.max(0.02, topBeamBottomY - housingTopY);
      const dropGeo = new THREE.BoxGeometry(0.024, dropH, 0.024);
      const dropX = clearanceSpanW * 0.5 - 0.03;

      const dropL = new THREE.Mesh(dropGeo, brushedAluminumMat);
      dropL.position.set(-dropX, housingTopY + dropH * 0.5, 0);
      dropL.castShadow = true;

      const dropR = new THREE.Mesh(dropGeo, brushedAluminumMat);
      dropR.position.set(dropX, housingTopY + dropH * 0.5, 0);
      dropR.castShadow = true;

      // Mid-span subtle horizontal structural tie rod inside the truss bay
      const midTie = new THREE.Mesh(
        new THREE.BoxGeometry(dropX * 2, 0.018, 0.018),
        darkBrushedAluMat
      );
      midTie.position.set(0, housingTopY + dropH * 0.58, 0);

      duckInner.add(
        postL,
        postR,
        footL,
        footR,
        topBeam,
        trussChannel,
        gussetL,
        gussetR,
        clearanceHousing,
        clearanceStrip,
        dropL,
        dropR,
        midTie
      );
      propGroup.add(duckInner);
      registerClickable(
        [postL, postR, footL, footR, topBeam, trussChannel, clearanceHousing, clearanceStrip, dropL, dropR],
        obs,
        hitMeshes
      );

      // Floor proximity sensor pad (canonical X span, Z crossing)
      const zFill = new THREE.Mesh(
        new THREE.PlaneGeometry(spanW + 0.2, 3.8).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.055, depthWrite: false })
      );
      zFill.position.set(0, 0.005, 0);
      const zOut = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(spanW + 0.2, 0.01, 3.8)),
        new THREE.LineBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.42 })
      );
      zOut.position.set(0, 0.007, 0);
      zoneGroup.add(zFill, zOut);
      selData = makeSelectionFootprint(spanW + 0.26, 0.42, '#2dd4bf');
    } else if (obs.type === 'hurdle') {
      // =====================================================================
      // ARCHITECTURAL LOW STEEL PARKOUR HURDLE RAIL (exact top = h, 0.46–0.48 m)
      // Built canonically spanning X; orientGroup rotates 90° when size[2] > size[0]
      // =====================================================================
      const span = Math.max(w, d);
      const postX = Math.max(0.2, span * 0.5 - 0.055);
      const padH = 0.006;
      const footH = 0.018;
      const yBase = padH + footH; // 0.024 m top of L-baseplate
      const tubeR = 0.02; // 40 mm tubular horizontal rail
      const collarR = 0.024; // 48 mm CNC dark-anodized end collar -> top tangent = exact h
      const barCenterY = Math.max(yBase + 0.12, h - collarR);
      const sightH = 0.006;

      const hurdleInner = new THREE.Group();
      orientGroup = hurdleInner;

      // Twin brushed-aluminum L-shaped floor baseplates + dark anti-slip pads + machined hex bolts
      const padGeo = new THREE.BoxGeometry(0.118, padH, 0.376);
      const footGeo = new THREE.BoxGeometry(0.106, footH, 0.356);
      const boltGeo = new THREE.CylinderGeometry(0.007, 0.007, 0.008, 6);

      // Angled stanchion geometry connecting rear of baseplate (z = -0.055) up to crossbar collar (z = 0)
      const zBase = -0.055;
      const dy = Math.max(0.1, barCenterY - yBase);
      const dz = -zBase; // +0.055 m forward lean
      const stanchionLen = Math.hypot(dy, dz);
      const tiltX = Math.atan2(dz, dy);
      const stanchionGeo = new THREE.BoxGeometry(0.042, stanchionLen, 0.052);

      // Sculpted dark-anodized triangular gusset fin reinforcing the L-foot elbow
      const gussetShape = new THREE.Shape();
      gussetShape.moveTo(-0.045, yBase);
      gussetShape.lineTo(0.135, yBase);
      gussetShape.lineTo(-0.028, Math.min(barCenterY - 0.05, yBase + 0.132));
      gussetShape.closePath();
      const gussetGeo = new THREE.ExtrudeGeometry(gussetShape, {
        depth: 0.022,
        bevelEnabled: false,
      });
      gussetGeo.translate(0, 0, -0.011);
      gussetGeo.rotateY(-Math.PI / 2);
      gussetGeo.computeVertexNormals();

      // Twin machined dark-anodized end collars + subtle brushed-aluminum outer end plugs
      const collarGeo = new THREE.CylinderGeometry(collarR, collarR, 0.068, 24);
      collarGeo.rotateZ(Math.PI / 2);
      const endPlugGeo = new THREE.CylinderGeometry(tubeR * 0.72, tubeR * 0.72, 0.078, 18);
      endPlugGeo.rotateZ(Math.PI / 2);

      const hurdleClickables: THREE.Object3D[] = [];

      for (const sx of [-postX, postX]) {
        const pad = new THREE.Mesh(padGeo, darkAnodizedMat);
        pad.position.set(sx, padH * 0.5, 0.035);
        pad.receiveShadow = true;

        const foot = new THREE.Mesh(footGeo, brushedAluminumMat);
        foot.position.set(sx, padH + footH * 0.5, 0.035);
        foot.castShadow = true;
        foot.receiveShadow = true;

        const stanchion = new THREE.Mesh(stanchionGeo, brushedAluminumMat);
        stanchion.position.set(sx, (yBase + barCenterY) * 0.5, zBase * 0.5);
        stanchion.rotation.x = tiltX;
        stanchion.castShadow = true;
        stanchion.receiveShadow = true;

        const gusset = new THREE.Mesh(gussetGeo, darkAnodizedMat);
        gusset.position.set(sx, 0, 0);
        gusset.castShadow = true;
        gusset.receiveShadow = true;

        const collar = new THREE.Mesh(collarGeo, darkAnodizedMat);
        collar.position.set(sx, barCenterY, 0);
        collar.castShadow = true;
        collar.receiveShadow = true;

        const endPlug = new THREE.Mesh(endPlugGeo, brushedAluminumMat);
        endPlug.position.set(sx, barCenterY, 0);

        hurdleInner.add(pad, foot, stanchion, gusset, collar, endPlug);
        hurdleClickables.push(foot, stanchion, gusset, collar);

        for (const bz of [-0.095, 0.165]) {
          for (const bx of [-0.032, 0.032]) {
            const bolt = new THREE.Mesh(boltGeo, darkBrushedAluMat);
            bolt.position.set(sx + bx, yBase + 0.004, bz);
            hurdleInner.add(bolt);
          }
        }
      }

      // Crisp brushed-aluminum tubular horizontal crossbar spanning between collars
      const crossbarGeo = new THREE.CylinderGeometry(tubeR, tubeR, postX * 2, 24);
      crossbarGeo.rotateZ(Math.PI / 2);
      const crossbar = new THREE.Mesh(crossbarGeo, brushedAluminumMat);
      crossbar.position.set(0, barCenterY, 0);
      crossbar.castShadow = true;
      crossbar.receiveShadow = true;

      // Sleek warm brushed-amber/copper top sightline accent strip — exact top face sits at y = h!
      const sightlineStrip = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.3, span - 0.14), sightH, 0.015),
        hurdleAmberSightMat
      );
      sightlineStrip.position.set(0, h - sightH * 0.5, 0);
      sightlineStrip.castShadow = true;

      hurdleInner.add(crossbar, sightlineStrip);
      hurdleClickables.push(crossbar, sightlineStrip);
      propGroup.add(hurdleInner);
      registerClickable(hurdleClickables, obs, hitMeshes);

      // Subtle amber AI Guides floor sensor zone (canonical X span, Z crossing)
      const hZoneW = span + 0.24;
      const hZoneD = 2.2;
      const hFill = new THREE.Mesh(
        new THREE.PlaneGeometry(hZoneW, hZoneD).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.055, depthWrite: false })
      );
      hFill.position.set(0, 0.005, 0);
      const hOut = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(hZoneW, 0.01, hZoneD)),
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.42 })
      );
      hOut.position.set(0, 0.007, 0);
      zoneGroup.add(hFill, hOut);
      selData = makeSelectionFootprint(span + 0.06, 0.42, '#f59e0b');
    } else {
      // =====================================================================
      // STEPPED BRUTALIST CONCRETE PLYO BLOCK (exact top = h = 0.40 m)
      // =====================================================================
      const skirtH = 0.028;
      const revealH = 0.012;
      const lowerH = Math.max(0.08, (h - skirtH) * 0.46);
      const lowerTopY = skirtH + lowerH;
      const upperBottomY = lowerTopY + revealH;
      const upperH = Math.max(0.08, h - upperBottomY);
      const stepInset = Math.min(0.08, Math.min(w, d) * 0.12);
      const upperW = Math.max(0.24, w - stepInset * 2);
      const upperD = Math.max(0.24, d - stepInset * 2);

      // Recessed dark-anodized shadow-gap steel base skirt anchoring monolith to studio floor
      const skirt = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.2, w - 0.06), skirtH, Math.max(0.2, d - 0.06)),
        darkAnodizedMat
      );
      skirt.position.y = skirtH * 0.5;
      skirt.castShadow = true;
      skirt.receiveShadow = true;

      // Lower chamfered poured-concrete plinth tier
      const lowerTier = new THREE.Mesh(
        createChamferedSlabGeometry(w, lowerH, d, lowerTopY, 0.012),
        vaultConcreteMat
      );
      lowerTier.castShadow = true;
      lowerTier.receiveShadow = true;

      // Recessed architectural dark shadow-step groove between lower & upper concrete tiers
      const stepGroove = new THREE.Mesh(
        new THREE.BoxGeometry(
          Math.max(0.18, upperW - 0.025),
          revealH + 0.008,
          Math.max(0.18, upperD - 0.025)
        ),
        darkAnodizedMat
      );
      stepGroove.position.y = lowerTopY + revealH * 0.5;
      stepGroove.castShadow = true;

      // Upper stepped chamfered honed-concrete plyo deck — exact top surface lies at y = h!
      const upperTier = new THREE.Mesh(
        createChamferedSlabGeometry(upperW, upperH, upperD, h, 0.014),
        plyoUpperConcreteMat
      );
      upperTier.castShadow = true;
      upperTier.receiveShadow = true;

      // Subtle brushed-copper step-reveal fillet + 4 vertical corner accent beads (all strictly <= h)
      const copperReveal = new THREE.Mesh(
        new THREE.BoxGeometry(upperW - 0.008, 0.006, upperD - 0.008),
        copperGripMat
      );
      copperReveal.position.y = upperBottomY - 0.002;

      const beadH = Math.max(0.04, upperH - 0.036);
      const beadGeo = new THREE.BoxGeometry(0.012, beadH, 0.012);
      const beadX = upperW * 0.5 - 0.007;
      const beadZ = upperD * 0.5 - 0.007;
      const beadY = upperBottomY + upperH * 0.5;
      const cornerBeads: THREE.Mesh[] = [];
      for (const bx of [-beadX, beadX]) {
        for (const bz of [-beadZ, beadZ]) {
          const bead = new THREE.Mesh(beadGeo, copperGripMat);
          bead.position.set(bx, beadY, bz);
          cornerBeads.push(bead);
        }
      }

      propGroup.add(skirt, lowerTier, stepGroove, upperTier, copperReveal, ...cornerBeads);
      registerClickable([skirt, lowerTier, stepGroove, upperTier, copperReveal, ...cornerBeads], obs, hitMeshes);

      // Subtle AI Guides floor sensor zone framing the plyo block
      const bColor = new THREE.Color(obs.color || '#a8a29e');
      const bZoneW = w + 0.85;
      const bZoneD = d + 1.15;
      const bFill = new THREE.Mesh(
        new THREE.PlaneGeometry(bZoneW, bZoneD).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: bColor, transparent: true, opacity: 0.055, depthWrite: false })
      );
      bFill.position.set(0, 0.005, 0);
      const bOut = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(bZoneW, 0.01, bZoneD)),
        new THREE.LineBasicMaterial({ color: bColor, transparent: true, opacity: 0.42 })
      );
      bOut.position.set(0, 0.007, 0);
      zoneGroup.add(bFill, bOut);
      selData = makeSelectionFootprint(w, d, obs.color || '#a8a29e');
    }

    selData.selFill.userData = { obstacle: obs };
    editTargets.push(selData.selFill);
    hitMeshes.push(selData.selFill);

    const entry: ManagedObstacleEntry = {
      id: obs.id,
      type: obs.type,
      propGroup,
      orientGroup,
      zoneGroup,
      selGroup: selData.selGroup,
      selFillMat: selData.selFillMat,
      selEdgeMat: selData.selEdgeMat,
      hitMeshes,
    };
    applyTransform(entry, obs);
    entries.set(obs.id, entry);
    scene.add(propGroup);
  }

  return {
    getClickTargets: () => clickTargets,
    getEditTargets: () => editTargets,
    syncObstacle: (obs: StudioObstacleSpec) => {
      const entry = entries.get(obs.id);
      if (entry) applyTransform(entry, obs);
    },
    syncAll: (allSpecs: StudioObstacleSpec[]) => {
      for (const obs of allSpecs) {
        const entry = entries.get(obs.id);
        if (entry) applyTransform(entry, obs);
      }
    },
    setEditVisuals: (editActive: boolean, hoveredId: string | null, selectedId: string | null) => {
      editOverlayGroup.visible = editActive;
      if (!editActive) return;
      for (const [id, entry] of entries) {
        const isSel = id === selectedId;
        const isHov = id === hoveredId;
        if (isSel) {
          entry.selEdgeMat.color.setHex(0x38bdf8);
          entry.selEdgeMat.opacity = 0.98;
          entry.selFillMat.color.setHex(0x38bdf8);
          entry.selFillMat.opacity = 0.22;
        } else if (isHov) {
          entry.selEdgeMat.color.setHex(0xfbbf24);
          entry.selEdgeMat.opacity = 0.88;
          entry.selFillMat.color.setHex(0xfbbf24);
          entry.selFillMat.opacity = 0.14;
        } else {
          entry.selEdgeMat.color.setHex(0x94a3b8);
          entry.selEdgeMat.opacity = 0.45;
          entry.selFillMat.color.setHex(0x94a3b8);
          entry.selFillMat.opacity = 0.05;
        }
      }
    },
  };
}
