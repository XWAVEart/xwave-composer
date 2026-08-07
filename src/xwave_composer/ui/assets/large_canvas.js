/**
 * Infinite Canvas Mode — pan/zoom viewport + movable generation region.
 *
 * Scene arrives via #xwave-lc-root[data-scene] (base64 JSON).
 * Actions go to #xwave-lc-action (CSS-hidden textbox).
 */
(function () {
  "use strict";

  const GRID = 64;
  const SNAP_MS = 150;

  const state = {
    canvasW: 2048,
    canvasH: 2048,
    previewUrl: null,
    previewImg: null,
    stamp: { x: 0, y: 0, w: 1024, h: 1024 },
    panX: 0,
    panY: 0,
    zoom: 0.25,
    drag: null,
    snapFrame: null,
    lastSceneB64: "",
  };

  function $(id) {
    return document.getElementById(id);
  }

  function findActionInput() {
    const root = $("xwave-lc-action");
    if (!root) return null;
    if (root.tagName === "TEXTAREA" || root.tagName === "INPUT") return root;
    return root.querySelector("textarea, input");
  }

  function setNativeValue(input, value) {
    const proto =
      input.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && desc.set) desc.set.call(input, value);
    else input.value = value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function emitAction(action) {
    action._ts = Date.now();
    const input = findActionInput();
    if (!input) {
      console.warn("xwave-lc: action bridge not found");
      return;
    }
    setNativeValue(input, JSON.stringify(action));
  }

  function fitToView() {
    const wrap = $("xwave-lc-wrap");
    if (!wrap) return;
    const viewW = wrap.clientWidth || 640;
    const viewH = wrap.clientHeight || 520;
    state.zoom = fitZoom(viewW, viewH);
    state.panX = (viewW - state.canvasW * state.zoom) / 2;
    state.panY = (viewH - state.canvasH * state.zoom) / 2;
  }

  function parseScene(root) {
    const b64 = root.getAttribute("data-scene") || "";
    if (!b64 || b64 === state.lastSceneB64) return false;
    state.lastSceneB64 = b64;
    try {
      const json = atob(b64);
      const scene = JSON.parse(json);
      const prevW = state.canvasW;
      const prevH = state.canvasH;
      state.canvasW = scene.width || state.canvasW;
      state.canvasH = scene.height || state.canvasH;
      state.previewUrl = scene.preview_url || null;
      if (scene.stamp) {
        state.stamp = {
          x: scene.stamp.x | 0,
          y: scene.stamp.y | 0,
          w: scene.stamp.w | 0,
          h: scene.stamp.h | 0,
        };
        keepStampReachable();
      }
      const dimChanged =
        state.canvasW !== prevW || state.canvasH !== prevH || !!scene.fit;
      if (typeof scene.zoom === "number" && !dimChanged) state.zoom = scene.zoom;
      if (typeof scene.pan_x === "number" && !dimChanged) state.panX = scene.pan_x;
      if (typeof scene.pan_y === "number" && !dimChanged) state.panY = scene.pan_y;
      if (state.previewUrl) {
        const img = new Image();
        img.onload = () => {
          state.previewImg = img;
          if (dimChanged) fitToView();
          draw();
        };
        img.src = state.previewUrl;
      } else {
        state.previewImg = null;
        if (dimChanged) fitToView();
      }
      state._needsFit = dimChanged;
      return true;
    } catch (err) {
      console.warn("xwave-lc: bad scene", err);
      return false;
    }
  }

  function fitZoom(viewW, viewH) {
    if (state.canvasW <= 0 || state.canvasH <= 0) return 0.25;
    const zx = viewW / state.canvasW;
    const zy = viewH / state.canvasH;
    return Math.min(zx, zy) * 0.92;
  }

  function worldToScreen(wx, wy) {
    return {
      x: state.panX + wx * state.zoom,
      y: state.panY + wy * state.zoom,
    };
  }

  function screenToWorld(sx, sy) {
    return {
      x: (sx - state.panX) / state.zoom,
      y: (sy - state.panY) / state.zoom,
    };
  }

  function clamp(value, low, high) {
    return Math.min(Math.max(value, low), high);
  }

  function keepStampReachable() {
    const s = state.stamp;
    // Preserve edge-outpainting, but keep at least one grid cell on canvas so
    // the selection can always be grabbed again.
    s.x = clamp(s.x, GRID - s.w, state.canvasW - GRID);
    s.y = clamp(s.y, GRID - s.h, state.canvasH - GRID);
  }

  function snappedStampPosition() {
    const s = state.stamp;
    return {
      x: clamp(Math.round(s.x / GRID) * GRID, GRID - s.w, state.canvasW - GRID),
      y: clamp(Math.round(s.y / GRID) * GRID, GRID - s.h, state.canvasH - GRID),
    };
  }

  function cancelSnapAnimation() {
    if (state.snapFrame !== null) {
      cancelAnimationFrame(state.snapFrame);
      state.snapFrame = null;
    }
  }

  function animateStampTo(x, y, done) {
    cancelSnapAnimation();
    const x0 = state.stamp.x;
    const y0 = state.stamp.y;
    if (Math.abs(x - x0) < 0.01 && Math.abs(y - y0) < 0.01) {
      state.stamp.x = x;
      state.stamp.y = y;
      draw();
      done();
      return;
    }
    const started = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - started) / SNAP_MS);
      const eased = 1 - Math.pow(1 - t, 3);
      state.stamp.x = x0 + (x - x0) * eased;
      state.stamp.y = y0 + (y - y0) * eased;
      draw();
      if (t < 1) {
        state.snapFrame = requestAnimationFrame(tick);
      } else {
        state.snapFrame = null;
        state.stamp.x = x;
        state.stamp.y = y;
        draw();
        done();
      }
    };
    state.snapFrame = requestAnimationFrame(tick);
  }

  function emitStampPosition() {
    emitAction({
      type: "region",
      x: state.stamp.x,
      y: state.stamp.y,
      w: state.stamp.w,
      h: state.stamp.h,
    });
  }

  function finishStampDrag() {
    const target = snappedStampPosition();
    animateStampTo(target.x, target.y, emitStampPosition);
  }

  function draw() {
    const canvas = $("xwave-lc-canvas");
    if (!canvas) return;
    const wrap = $("xwave-lc-wrap");
    if (!wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(280, wrap.clientWidth || 640);
    // Fill the viewport column — no artificial 720px cap on Infinite Canvas.
    const cssH = Math.max(320, wrap.clientHeight || 520);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0a0c0f";
    ctx.fillRect(0, 0, cssW, cssH);

    // checker for empty
    const chk = 16;
    for (let y = 0; y < cssH; y += chk) {
      for (let x = 0; x < cssW; x += chk) {
        ctx.fillStyle = ((x / chk + y / chk) | 0) % 2 ? "#12151a" : "#171b21";
        ctx.fillRect(x, y, chk, chk);
      }
    }

    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);

    // Document plane — dark slate (not pure black) so empty size reads clearly
    // against the checkerboard “outside canvas” background.
    ctx.fillStyle = "#14181f";
    ctx.fillRect(0, 0, state.canvasW, state.canvasH);
    ctx.strokeStyle = "rgba(230, 233, 238, 0.35)";
    ctx.lineWidth = 2 / state.zoom;
    ctx.strokeRect(0.5, 0.5, state.canvasW - 1, state.canvasH - 1);
    if (state.previewImg) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(state.previewImg, 0, 0, state.canvasW, state.canvasH);
    }

    // region
    const s = state.stamp;
    ctx.strokeStyle = "rgba(94, 234, 212, 0.95)";
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([8 / state.zoom, 6 / state.zoom]);
    ctx.strokeRect(s.x + 0.5, s.y + 0.5, s.w - 1, s.h - 1);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(94, 234, 212, 0.08)";
    ctx.fillRect(s.x, s.y, s.w, s.h);

    ctx.restore();

    // HUD
    ctx.fillStyle = "rgba(230,233,238,0.75)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(
      state.canvasW +
        "×" +
        state.canvasH +
        "  region " +
        s.w +
        "×" +
        s.h +
        " @ " +
        s.x +
        "," +
        s.y +
        "  zoom " +
        Math.round(state.zoom * 100) +
        "%",
      10,
      18
    );
  }

  function hitStamp(sx, sy) {
    const w = screenToWorld(sx, sy);
    const s = state.stamp;
    return w.x >= s.x && w.x <= s.x + s.w && w.y >= s.y && w.y <= s.y + s.h;
  }

  function bind() {
    const canvas = $("xwave-lc-canvas");
    if (!canvas || canvas._xwaveLcBound) return;
    canvas._xwaveLcBound = true;

    canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const before = screenToWorld(sx, sy);
      const factor = ev.deltaY < 0 ? 1.1 : 0.9;
      state.zoom = Math.max(0.05, Math.min(4, state.zoom * factor));
      const after = {
        x: state.panX + before.x * state.zoom,
        y: state.panY + before.y * state.zoom,
      };
      state.panX += sx - after.x;
      state.panY += sy - after.y;
      draw();
    }, { passive: false });

    canvas.addEventListener("pointerdown", (ev) => {
      cancelSnapAnimation();
      canvas.setPointerCapture(ev.pointerId);
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      if (ev.button === 1 || ev.shiftKey || !hitStamp(sx, sy)) {
        state.drag = { kind: "pan", x: sx, y: sy, panX: state.panX, panY: state.panY };
      } else {
        const w = screenToWorld(sx, sy);
        state.drag = {
          kind: "stamp",
          ox: w.x - state.stamp.x,
          oy: w.y - state.stamp.y,
        };
      }
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (!state.drag) return;
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      if (state.drag.kind === "pan") {
        state.panX = state.drag.panX + (sx - state.drag.x);
        state.panY = state.drag.panY + (sy - state.drag.y);
        draw();
      } else if (state.drag.kind === "stamp") {
        const w = screenToWorld(sx, sy);
        state.stamp.x = w.x - state.drag.ox;
        state.stamp.y = w.y - state.drag.oy;
        keepStampReachable();
        draw();
      }
    });

    canvas.addEventListener("pointerup", (ev) => {
      if (!state.drag) return;
      const kind = state.drag.kind;
      state.drag = null;
      if (kind === "stamp") {
        finishStampDrag();
      } else {
        draw();
      }
    });

    canvas.addEventListener("pointercancel", () => {
      if (!state.drag) return;
      const kind = state.drag.kind;
      state.drag = null;
      if (kind === "stamp") finishStampDrag();
      else draw();
    });

    canvas.addEventListener("dblclick", () => {
      const wrap = $("xwave-lc-wrap");
      if (!wrap) return;
      state.zoom = fitZoom(wrap.clientWidth || 640, wrap.clientHeight || 520);
      state.panX = ((wrap.clientWidth || 640) - state.canvasW * state.zoom) / 2;
      state.panY = ((wrap.clientHeight || 520) - state.canvasH * state.zoom) / 2;
      draw();
    });
  }

  function syncFromDom() {
    const root = $("xwave-lc-root");
    if (!root) return;
    const changed = parseScene(root);
    bind();
    if (changed && state._needsFit) {
      fitToView();
      state._needsFit = false;
    }
    draw();
  }

  const obs = new MutationObserver(() => syncFromDom());
  let resizeObs = null;
  function boot() {
    const root = $("xwave-lc-root");
    if (root) {
      obs.observe(root, { attributes: true, attributeFilter: ["data-scene"] });
      syncFromDom();
    }
    const wrap = $("xwave-lc-wrap");
    if (wrap && typeof ResizeObserver !== "undefined") {
      if (resizeObs) resizeObs.disconnect();
      resizeObs = new ResizeObserver(() => draw());
      resizeObs.observe(wrap);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  setInterval(syncFromDom, 500);
  window.addEventListener("resize", () => draw());
})();
