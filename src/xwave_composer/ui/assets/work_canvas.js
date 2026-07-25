/**
 * xwave WORK canvas + layer stack interactions.
 *
 * Scene data arrives via #xwave-work-root[data-scene] (base64 JSON) rendered
 * by Python. Layer cards are server-rendered inside #xwave-layers-root; this
 * script only binds delegated events (click / delete / drag-reorder).
 *
 * Actions are sent to Python through the CSS-hidden #xwave-action-out textbox.
 */
(function () {
  "use strict";

  const HANDLE = 11;
  const ROT_OFFSET = 34;

  const state = {
    layers: [],
    selectedId: null,
    canvasW: 1024,
    canvasH: 1024,
    bgUrl: null,
    bgImg: null,
    bgScale: 1,
    bgRotation: 0,
    bgOffsetX: 0,
    bgOffsetY: 0,
    bgFlipX: false,
    bgFlipY: false,
    drag: null,
    viewScale: 1,
    lastSceneB64: "",
    liveTimer: null,
    nudging: false,
    nudgeTimer: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  // ── action bridge ────────────────────────────────────────────
  function findActionInput() {
    const root = $("xwave-action-out");
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
      console.warn("xwave: action bridge input not found");
      return;
    }
    setNativeValue(input, JSON.stringify(action));
  }

  function emitTransform(final) {
    const layer = state.layers.find((l) => l.id === state.selectedId);
    if (!layer) return;
    emitAction({
      type: "transform",
      final: final === true,
      id: layer.id,
      x: layer.x,
      y: layer.y,
      scale_x: layer.scale_x,
      scale_y: layer.scale_y,
      rotation: layer.rotation,
    });
  }

  function scheduleLiveTransform() {
    if (state.liveTimer) clearTimeout(state.liveTimer);
    state.liveTimer = setTimeout(function () {
      state.liveTimer = null;
      emitTransform(false);
    }, 120);
  }

  // ── scene ingest ─────────────────────────────────────────────
  function loadImage(url, prev) {
    if (!url) return null;
    if (prev && prev._url === url) return prev;
    const img = new Image();
    img._url = url;
    img.decoding = "async";
    img.onload = draw;
    img.src = url;
    return img;
  }

  function applyScene(data) {
    if (!data || typeof data !== "object") return;
    state.canvasW = data.width || 1024;
    state.canvasH = data.height || 1024;
    state.selectedId = data.selected_id || null;
    state.bgUrl = data.bg_data_url || null;
    state.bgImg = state.bgUrl ? loadImage(state.bgUrl, state.bgImg) : null;
    state.bgScale = data.bg_scale != null ? Number(data.bg_scale) : 1;
    state.bgRotation = data.bg_rotation != null ? Number(data.bg_rotation) : 0;
    state.bgOffsetX = data.bg_offset_x != null ? Number(data.bg_offset_x) : 0;
    state.bgOffsetY = data.bg_offset_y != null ? Number(data.bg_offset_y) : 0;
    state.bgFlipX = !!data.bg_flip_x;
    state.bgFlipY = !!data.bg_flip_y;

    const prev = {};
    state.layers.forEach(function (l) {
      if (l._img) prev[l.id] = l._img;
    });
    state.layers = (data.layers || []).map(function (l) {
      const layer = Object.assign({}, l);
      layer._img =
        prev[layer.id] && prev[layer.id]._url === layer.data_url
          ? prev[layer.id]
          : loadImage(layer.data_url, null);
      return layer;
    });

    ensureCanvasBound();
    draw();
  }

  function ingestFromDom(force) {
    if (state.drag && !force) return;
    const root = $("xwave-work-root");
    if (!root) return;
    const b64 = root.getAttribute("data-scene") || "";
    if (!b64 || (!force && b64 === state.lastSceneB64)) return;
    try {
      const data = JSON.parse(atob(b64));
      state.lastSceneB64 = b64;
      applyScene(data);
    } catch (e) {
      console.warn("xwave: scene parse failed", e);
    }
  }

  // ── canvas drawing ───────────────────────────────────────────
  function layerSize(layer) {
    return {
      w: (layer.w || 256) * (layer.scale_x || 1),
      h: (layer.h || 256) * (layer.scale_y || 1),
    };
  }

  function draw() {
    const canvas = $("xwave-work-canvas");
    if (!canvas) return;
    const wrap = canvas.parentElement;
    const maxW = Math.max(260, (wrap && wrap.clientWidth) || 520);
    const view = maxW;
    state.viewScale = view / state.canvasW;
    const viewH = state.canvasH * state.viewScale;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.max(1, Math.round(view * dpr));
    canvas.height = Math.max(1, Math.round(viewH * dpr));
    canvas.style.width = view + "px";
    canvas.style.height = viewH + "px";

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr * state.viewScale, 0, 0, dpr * state.viewScale, 0, 0);

    // checkerboard base
    ctx.fillStyle = "#101216";
    ctx.fillRect(0, 0, state.canvasW, state.canvasH);
    const step = 32;
    ctx.fillStyle = "#15181d";
    for (let y = 0; y < state.canvasH; y += step) {
      for (let x = 0; x < state.canvasW; x += step) {
        if (((x / step + y / step) | 0) % 2 === 0) ctx.fillRect(x, y, step, step);
      }
    }

    if (state.bgImg && state.bgImg.complete && state.bgImg.naturalWidth) {
      const scale = Math.max(0.05, state.bgScale || 1);
      const w = state.canvasW * scale;
      const h = state.canvasH * scale;
      ctx.save();
      ctx.translate(
        state.canvasW / 2 + (state.bgOffsetX || 0),
        state.canvasH / 2 + (state.bgOffsetY || 0)
      );
      ctx.scale(state.bgFlipX ? -1 : 1, state.bgFlipY ? -1 : 1);
      ctx.rotate(((state.bgRotation || 0) * Math.PI) / 180);
      ctx.drawImage(state.bgImg, -w / 2, -h / 2, w, h);
      ctx.restore();
    } else if (!state.bgUrl) {
      ctx.fillStyle = "rgba(229,231,235,0.28)";
      ctx.font = Math.round(30 / state.viewScale) + "px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Generate a background to begin", state.canvasW / 2, state.canvasH / 2);
    }

    for (let i = 0; i < state.layers.length; i++) {
      const layer = state.layers[i];
      if (layer.visible === false) continue;
      const sz = layerSize(layer);
      ctx.save();
      ctx.translate(layer.x, layer.y);
      ctx.rotate(((layer.rotation || 0) * Math.PI) / 180);
      ctx.globalAlpha = layer.opacity != null ? layer.opacity : 1;
      ctx.globalCompositeOperation = layer.blend_canvas || "source-over";
      if (layer._img && layer._img.complete && layer._img.naturalWidth) {
        ctx.drawImage(layer._img, -sz.w / 2, -sz.h / 2, sz.w, sz.h);
      }
      ctx.globalCompositeOperation = "source-over";
      ctx.restore();
    }

    const sel = state.layers.find(function (l) {
      return l.id === state.selectedId;
    });
    if (sel && sel.visible !== false) {
      const sz = layerSize(sel);
      const hs = HANDLE / state.viewScale;
      ctx.save();
      ctx.translate(sel.x, sel.y);
      ctx.rotate(((sel.rotation || 0) * Math.PI) / 180);
      ctx.strokeStyle = "#5eead4";
      ctx.lineWidth = 1.75 / state.viewScale;
      ctx.strokeRect(-sz.w / 2, -sz.h / 2, sz.w, sz.h);
      ctx.fillStyle = "#5eead4";
      [
        [-sz.w / 2, -sz.h / 2],
        [sz.w / 2, -sz.h / 2],
        [sz.w / 2, sz.h / 2],
        [-sz.w / 2, sz.h / 2],
      ].forEach(function (c) {
        ctx.fillRect(c[0] - hs / 2, c[1] - hs / 2, hs, hs);
      });
      ctx.beginPath();
      ctx.moveTo(0, -sz.h / 2);
      ctx.lineTo(0, -sz.h / 2 - ROT_OFFSET / state.viewScale);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, -sz.h / 2 - ROT_OFFSET / state.viewScale, hs * 0.65, 0, Math.PI * 2);
      ctx.fillStyle = "#f59e0b";
      ctx.fill();
      ctx.restore();
    }
  }

  // ── pointer interaction ──────────────────────────────────────
  function canvasPoint(evt) {
    const canvas = $("xwave-work-canvas");
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((evt.clientX - rect.left) / rect.width) * state.canvasW,
      y: ((evt.clientY - rect.top) / rect.height) * state.canvasH,
    };
  }

  function toLocal(layer, p) {
    const dx = p.x - layer.x;
    const dy = p.y - layer.y;
    const rad = (-(layer.rotation || 0) * Math.PI) / 180;
    return {
      x: dx * Math.cos(rad) - dy * Math.sin(rad),
      y: dx * Math.sin(rad) + dy * Math.cos(rad),
    };
  }

  function hitTest(p) {
    // Only the menu-selected layer is interactive — never pick another layer
    // by clicking through the stack (avoids jumpy handles with many objects).
    if (!state.selectedId || state.selectedId === "__bg__") return null;
    const layer = state.layers.find(function (l) {
      return l.id === state.selectedId;
    });
    if (!layer || layer.visible === false) return null;

    const sz = layerSize(layer);
    const loc = toLocal(layer, p);
    const hs = (HANDLE * 1.8) / state.viewScale;

    const ry = -sz.h / 2 - ROT_OFFSET / state.viewScale;
    if (Math.hypot(loc.x, loc.y - ry) < hs * 1.3) return { layer: layer, mode: "rotate" };
    const corners = [
      [-sz.w / 2, -sz.h / 2],
      [sz.w / 2, -sz.h / 2],
      [sz.w / 2, sz.h / 2],
      [-sz.w / 2, sz.h / 2],
    ];
    for (let c = 0; c < corners.length; c++) {
      if (Math.abs(loc.x - corners[c][0]) <= hs && Math.abs(loc.y - corners[c][1]) <= hs) {
        return { layer: layer, mode: "resize" };
      }
    }
    if (loc.x >= -sz.w / 2 && loc.x <= sz.w / 2 && loc.y >= -sz.h / 2 && loc.y <= sz.h / 2) {
      return { layer: layer, mode: "move" };
    }
    return null;
  }

  /**
   * Is this point on a visible pixel of the layer, rather than merely inside
   * its bounding box? Object layers are cut-outs, so their boxes are mostly
   * transparent and box-only hit testing would make a truck behave like the
   * rectangle it arrived in.
   *
   * The alpha map is cached on the image element, so it survives scene
   * updates and is rebuilt only when the image itself changes.
   */
  function opaqueAt(layer, loc, sz) {
    const img = layer._img;
    if (!img || !img.complete || !img.naturalWidth) return true;
    if (img._alpha === undefined) {
      try {
        // Hit testing does not need full resolution.
        const MAXD = 256;
        const k = Math.min(1, MAXD / Math.max(img.naturalWidth, img.naturalHeight));
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(img.naturalWidth * k));
        c.height = Math.max(1, Math.round(img.naturalHeight * k));
        const cx = c.getContext("2d", { willReadFrequently: true });
        cx.drawImage(img, 0, 0, c.width, c.height);
        img._alpha = {
          d: cx.getImageData(0, 0, c.width, c.height).data,
          w: c.width,
          h: c.height,
        };
      } catch (e) {
        img._alpha = null; // tainted canvas — fall back to the bounding box
      }
    }
    if (!img._alpha) return true;
    const u = (loc.x + sz.w / 2) / sz.w;
    const v = (loc.y + sz.h / 2) / sz.h;
    const px = Math.min(img._alpha.w - 1, Math.max(0, Math.floor(u * img._alpha.w)));
    const py = Math.min(img._alpha.h - 1, Math.max(0, Math.floor(v * img._alpha.h)));
    return img._alpha.d[(py * img._alpha.w + px) * 4 + 3] > 8;
  }

  /** Topmost layer under the point, ignoring which layer is selected. */
  function hitTestAny(p) {
    // state.layers is bottom-to-top draw order, so scan from the top down.
    for (let i = state.layers.length - 1; i >= 0; i--) {
      const layer = state.layers[i];
      if (layer.visible === false) continue;
      const sz = layerSize(layer);
      const loc = toLocal(layer, p);
      if (loc.x < -sz.w / 2 || loc.x > sz.w / 2) continue;
      if (loc.y < -sz.h / 2 || loc.y > sz.h / 2) continue;
      if (!opaqueAt(layer, loc, sz)) continue;
      return layer;
    }
    return null;
  }

  function selectedLayer() {
    return (
      state.layers.find(function (l) {
        return l.id === state.selectedId;
      }) || null
    );
  }

  function onDown(evt) {
    // Primary button only; ignore right-click and pen barrel buttons.
    if (evt.button !== undefined && evt.button !== 0) return;
    const p = canvasPoint(evt);

    // Check the selected layer first so its resize and rotate handles keep
    // priority when objects overlap. This is what made selection menu-only
    // originally: handles must not jump to whatever sits under the cursor.
    let hit = hitTest(p);
    let selectId = null;

    if (!hit) {
      // Nothing on the selected layer, so fall back to direct selection:
      // clicking an object picks it up, the way any layered editor behaves.
      // Transparent pixels fall through to the layer underneath.
      const under = hitTestAny(p);
      if (!under) return; // empty canvas keeps the current selection
      if (under.id !== state.selectedId) {
        // Apply locally straight away so the drag starts on this frame
        // instead of waiting for the server to echo the selection back.
        state.selectedId = under.id;
        selectId = under.id;
        draw();
      }
      hit = { layer: under, mode: "move" };
    }

    evt.preventDefault();
    if (evt.pointerId !== undefined && evt.currentTarget.setPointerCapture) {
      try {
        evt.currentTarget.setPointerCapture(evt.pointerId);
      } catch (e) {
        /* capture is an optimisation, not a requirement */
      }
    }
    // Record the pose before the drag changes it. The action bridge is a
    // single slot, so any selection change rides along instead of being sent
    // as a second message that would overwrite this one.
    emitAction({ type: "history_push", select: selectId });
    state.drag = {
      mode: hit.mode,
      start: p,
      ox: hit.layer.x,
      oy: hit.layer.y,
      ow: hit.layer.w || 256,
      oh: hit.layer.h || 256,
      orot: hit.layer.rotation || 0,
      // angle from layer center to grab point, for jump-free rotation
      grabAngle: (Math.atan2(p.y - hit.layer.y, p.x - hit.layer.x) * 180) / Math.PI,
    };
  }

  function onMove(evt) {
    if (!state.drag) return;
    const p = canvasPoint(evt);
    const layer = state.layers.find(function (l) {
      return l.id === state.selectedId;
    });
    if (!layer) return;
    const d = state.drag;
    if (d.mode === "move") {
      layer.x = d.ox + (p.x - d.start.x);
      layer.y = d.oy + (p.y - d.start.y);
    } else if (d.mode === "resize") {
      const loc = toLocal(layer, p);
      layer.scale_x = Math.max(0.05, (Math.abs(loc.x) * 2) / d.ow);
      layer.scale_y = Math.max(0.05, (Math.abs(loc.y) * 2) / d.oh);
    } else if (d.mode === "rotate") {
      const ang = (Math.atan2(p.y - layer.y, p.x - layer.x) * 180) / Math.PI;
      layer.rotation = d.orot + (ang - d.grabAngle);
    }
    draw();
    scheduleLiveTransform();
  }

  function onUp() {
    if (!state.drag) return;
    state.drag = null;
    if (state.liveTimer) {
      clearTimeout(state.liveTimer);
      state.liveTimer = null;
    }
    emitTransform(true);
  }

  // ── keyboard ─────────────────────────────────────────────────
  function isTextTarget(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function scheduleNudgeCommit() {
    if (state.nudgeTimer) clearTimeout(state.nudgeTimer);
    state.nudgeTimer = setTimeout(function () {
      state.nudgeTimer = null;
      state.nudging = false;
      emitTransform(true);
    }, 400);
  }

  function onKeyDown(evt) {
    // The page is full of prompt boxes and number fields. Never take a key
    // that the user is aiming at one of them.
    if (isTextTarget(evt.target)) return;

    const mod = evt.ctrlKey || evt.metaKey;
    const key = evt.key;

    if (mod && (key === "z" || key === "Z")) {
      evt.preventDefault();
      emitAction({ type: evt.shiftKey ? "redo" : "undo" });
      return;
    }
    if (mod && (key === "y" || key === "Y")) {
      evt.preventDefault();
      emitAction({ type: "redo" });
      return;
    }

    const layer = selectedLayer();
    if (!layer) return;

    if (key === "Delete" || key === "Backspace") {
      evt.preventDefault();
      emitAction({ type: "delete", id: layer.id });
      return;
    }

    const step = evt.shiftKey ? 10 : 1;
    let dx = 0;
    let dy = 0;
    if (key === "ArrowLeft") dx = -step;
    else if (key === "ArrowRight") dx = step;
    else if (key === "ArrowUp") dy = -step;
    else if (key === "ArrowDown") dy = step;
    else return;

    evt.preventDefault();
    if (!state.nudging) {
      // One undo entry for a burst of arrow presses, not one per pixel.
      state.nudging = true;
      emitAction({ type: "history_push" });
    }
    layer.x += dx;
    layer.y += dy;
    draw();
    scheduleNudgeCommit();
  }

  function ensureCanvasBound() {
    const canvas = $("xwave-work-canvas");
    if (!canvas || canvas._xwaveBound) return;
    canvas._xwaveBound = true;
    // Pointer events cover mouse, pen and touch through one path, which is
    // what the spec's "open it from any device" case needs. The matching
    // touch-action: none already lives in app.css.
    canvas.addEventListener("pointerdown", onDown);
    if (!window._xwaveWinBound) {
      window._xwaveWinBound = true;
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
      window.addEventListener("resize", draw);
      window.addEventListener("keydown", onKeyDown);
    }
  }

  // ── layer stack (server-rendered cards, delegated events) ────
  function bindStackDelegation() {
    const root = $("xwave-layers-root");
    if (!root) return;
    const host = root.closest(".xwave-layers-col") || root.parentElement || root;
    if (host._xwaveStackBound) return;
    host._xwaveStackBound = true;

    host.addEventListener("click", function (e) {
      const del = e.target.closest("[data-delete-id]");
      if (del) {
        e.preventDefault();
        e.stopPropagation();
        emitAction({ type: "delete", id: del.getAttribute("data-delete-id") });
        return;
      }
      const card = e.target.closest("[data-layer-id]");
      if (card) {
        emitAction({ type: "select", id: card.getAttribute("data-layer-id") });
      }
    });

    host.addEventListener("dragstart", function (e) {
      const card = e.target.closest("[data-layer-id]");
      if (!card || card.getAttribute("data-layer-id") === "__bg__") return;
      card.classList.add("is-dragging");
      e.dataTransfer.setData("text/plain", card.getAttribute("data-layer-id"));
    });
    host.addEventListener("dragend", function (e) {
      const card = e.target.closest("[data-layer-id]");
      if (card) card.classList.remove("is-dragging");
    });
    host.addEventListener("dragover", function (e) {
      const card = e.target.closest("[data-layer-id]");
      if (!card || card.getAttribute("data-layer-id") === "__bg__") return;
      e.preventDefault();
      card.classList.add("is-over");
    });
    host.addEventListener("dragleave", function (e) {
      const card = e.target.closest("[data-layer-id]");
      if (card) card.classList.remove("is-over");
    });
    host.addEventListener("drop", function (e) {
      const card = e.target.closest("[data-layer-id]");
      if (!card) return;
      e.preventDefault();
      card.classList.remove("is-over");
      const fromId = e.dataTransfer.getData("text/plain");
      const toId = card.getAttribute("data-layer-id");
      if (!fromId || fromId === toId || toId === "__bg__") return;
      // Card order in the DOM is top-most first; compute new bottom→top order.
      const stack = $("xwave-layer-stack");
      const topFirst = Array.from(stack.querySelectorAll("[data-layer-id]"))
        .map(function (c) {
          return c.getAttribute("data-layer-id");
        })
        .filter(function (id) {
          return id !== "__bg__";
        });
      const from = topFirst.indexOf(fromId);
      const to = topFirst.indexOf(toId);
      if (from < 0 || to < 0) return;
      topFirst.splice(from, 1);
      topFirst.splice(to, 0, fromId);
      emitAction({ type: "reorder", ids: topFirst.slice().reverse() });
    });
  }

  // ── simple / advanced mode ───────────────────────────────────
  // Purely client side: flipping an attribute on <body> lets CSS hide the
  // advanced controls instantly. Doing it through Gradio visibility would
  // re-render dozens of components and lose focus on every toggle.
  const MODE_KEY = "xwave-mode";

  function currentMode() {
    return document.body.getAttribute("data-xwave-mode") || "simple";
  }

  function applyMode(mode) {
    document.body.setAttribute("data-xwave-mode", mode);
    try {
      window.localStorage.setItem(MODE_KEY, mode);
    } catch (e) {
      /* private browsing — the mode just will not persist */
    }
    const btn = $("xwave-mode-toggle");
    if (btn) {
      const advanced = mode === "advanced";
      btn.textContent = advanced ? "Simple" : "Advanced";
      btn.setAttribute("aria-pressed", advanced ? "true" : "false");
      btn.title = advanced
        ? "Hide the technical controls"
        : "Show sampler, style and export controls";
    }
    // Layout changed underneath the canvas, so its scale is stale.
    draw();
  }

  function bindModeToggle() {
    const btn = $("xwave-mode-toggle");
    if (!btn || btn._xwaveBound) return;
    btn._xwaveBound = true;
    btn.addEventListener("click", function () {
      applyMode(currentMode() === "advanced" ? "simple" : "advanced");
    });
    applyMode(currentMode());
  }

  function initMode() {
    let saved = null;
    try {
      saved = window.localStorage.getItem(MODE_KEY);
    } catch (e) {
      /* ignore */
    }
    document.body.setAttribute("data-xwave-mode", saved === "advanced" ? "advanced" : "simple");
  }

  // ── boot / DOM watching ──────────────────────────────────────
  function tick() {
    ingestFromDom(false);
    ensureCanvasBound();
    bindStackDelegation();
    bindModeToggle();
  }

  function watchDom() {
    let scheduled = false;
    const obs = new MutationObserver(function () {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(function () {
        scheduled = false;
        tick();
      });
    });
    obs.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-scene"],
    });
    setInterval(tick, 900);
  }

  function boot() {
    // Set the mode before anything paints so the advanced controls never
    // flash on screen for a frame before being hidden.
    initMode();
    watchDom();
    ingestFromDom(true);
    ensureCanvasBound();
    bindStackDelegation();
    bindModeToggle();
    draw();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  setTimeout(tick, 400);
  setTimeout(tick, 1200);

  window.xwaveUI = { state: state, draw: draw, ingestFromDom: ingestFromDom };
})();
