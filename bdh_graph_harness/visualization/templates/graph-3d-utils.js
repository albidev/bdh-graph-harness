(function attachBDH3DUtils(globalScope) {
  'use strict';

  function finiteOrUndefined(value) {
    return Number.isFinite(value) ? value : undefined;
  }

  function cloneNodeForStructuralUpdate(node, liveNode) {
    const incoming = node || {};
    const live = liveNode || {};
    const spatialKeys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'];
    const clone = {
      id: incoming.id,
      name: incoming.name,
      color: incoming.color,
      val: incoming.val,
      _opacity: incoming._opacity,
      _shape: incoming._shape,
      _dormant: incoming._dormant,
      _mass: incoming._mass,
      _synapticGlow: incoming._synapticGlow,
      _tags: incoming._tags,
      _title: incoming._title,
      _displayLabel: incoming._displayLabel,
      _path: incoming._path,
      _text: incoming._text,
      _hidden: incoming._hidden,
      _cluster: incoming._cluster,
    };

    spatialKeys.forEach(key => {
      clone[key] = finiteOrUndefined(incoming[key]) ?? finiteOrUndefined(live[key]);
    });
    return clone;
  }

  function cameraPositionForFocus(node, cameraPosition, focusDistance) {
    const target = node || { x: 0, y: 0, z: 0 };
    const camera = cameraPosition || { x: 0, y: 0, z: 1 };
    const distance = Number.isFinite(focusDistance) && focusDistance > 0 ? focusDistance : 120;
    const dx = (camera.x || 0) - (target.x || 0);
    const dy = (camera.y || 0) - (target.y || 0);
    const dz = (camera.z || 0) - (target.z || 0);
    const length = Math.hypot(dx, dy, dz);
    if (length === 0) {
      return { x: target.x || 0, y: target.y || 0, z: (target.z || 0) + distance };
    }
    const scale = distance / length;
    return {
      x: (target.x || 0) + dx * scale,
      y: (target.y || 0) + dy * scale,
      z: (target.z || 0) + dz * scale,
    };
  }

  function fogDensityForFitDistance(distance, constrained) {
    const baseDensity = constrained ? 0.00055 : 0.00038;
    const safeDistance = Number.isFinite(distance) && distance > 0 ? distance : 1000;
    // Keep the fitted graph above roughly 80% fog visibility regardless of
    // world-space extent. A fixed density erases large graphs at overview zoom.
    return Math.min(baseDensity, 0.45 / safeDistance);
  }

  function nodeScaleForFitDistance(distance, viewportHeight, fovDegrees) {
    const safeDistance = Number.isFinite(distance) && distance > 0 ? distance : 1000;
    const safeHeight = Number.isFinite(viewportHeight) && viewportHeight > 0 ? viewportHeight : 800;
    const safeFov = Number.isFinite(fovDegrees) && fovDegrees > 0 ? fovDegrees : 50;
    const worldUnitsPerPixel = (
      2 * safeDistance * Math.tan((safeFov * Math.PI) / 360)
    ) / safeHeight;
    // Fit is an overview, not a star chart. Keep nodes large enough to survive
    // dark-mode contrast and high-density/mobile displays in CSS pixels.
    const desiredRadiusPixels = safeHeight <= 700 ? 4.8 : 4.2;
    const representativeNodeRadius = 5.5;
    return Math.max(1, Math.min(20, (
      worldUnitsPerPixel * desiredRadiusPixels
    ) / representativeNodeRadius));
  }

  const api = {
    cloneNodeForStructuralUpdate,
    cameraPositionForFocus,
    fogDensityForFitDistance,
    nodeScaleForFitDistance,
  };
  globalScope.BDH3DUtils = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
