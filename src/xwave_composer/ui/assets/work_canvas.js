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
    // After a drag, keep only that layer's local pose until the hold expires
    // (or the server catches up) so late scene HTML cannot snap it backward.
    holdTransformsUntil: 0,
    holdLayerId: null,
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
    // Data-URLs are local — sync decode so new layers paint with the
    // scene update instead of after OUTPUT has already refreshed.
    img.decoding = "sync";
    img.onload = draw;
    img.src = url;
    return img;
  }

  function heldLayerId() {
    if (state.drag) return state.selectedId || null;
    if (Date.now() < state.holdTransformsUntil) return state.holdLayerId;
    return null;
  }

  function posesMatch(a, b) {
    if (!a || !b) return false;
    return (
      Math.abs((a.x || 0) - (b.x || 0)) < 0.5 &&
      Math.abs((a.y || 0) - (b.y || 0)) < 0.5 &&
      Math.abs((a.scale_x || 1) - (b.scale_x || 1)) < 0.001 &&
      Math.abs((a.scale_y || 1) - (b.scale_y || 1)) < 0.001 &&
      Math.abs((a.rotation || 0) - (b.rotation || 0)) < 0.05
    );
  }

  function applyScene(data) {
    if (!data || typeof data !== "object") return;
    state.canvasW = data.width || 1024;
    state.canvasH = data.height || 1024;
    // Prefer the optimistic local selection while the pointer is down.
    if (data.selected_id != null && !state.drag) {
      state.selectedId = data.selected_id || null;
    } else if (data.selected_id != null && !state.selectedId) {
      state.selectedId = data.selected_id || null;
    }
    state.bgUrl = data.bg_data_url || null;
    state.bgImg = state.bgUrl ? loadImage(state.bgUrl, state.bgImg) : null;
    state.bgScale = data.bg_scale != null ? Number(data.bg_scale) : 1;
    state.bgRotation = data.bg_rotation != null ? Number(data.bg_rotation) : 0;
    state.bgOffsetX = data.bg_offset_x != null ? Number(data.bg_offset_x) : 0;
    state.bgOffsetY = data.bg_offset_y != null ? Number(data.bg_offset_y) : 0;
    state.bgFlipX = !!data.bg_flip_x;
    state.bgFlipY = !!data.bg_flip_y;

    const prev = {};
    const localPose = {};
    const keepId = heldLayerId();
    state.layers.forEach(function (l) {
      if (l._img) prev[l.id] = l._img;
      if (keepId && l.id === keepId) {
        localPose[l.id] = {
          x: l.x,
          y: l.y,
          scale_x: l.scale_x,
          scale_y: l.scale_y,
          rotation: l.rotation,
        };
      }
    });
    state.layers = (data.layers || []).map(function (l) {
      const layer = Object.assign({}, l);
      layer._img =
        prev[layer.id] && prev[layer.id]._url === layer.data_url
          ? prev[layer.id]
          : loadImage(layer.data_url, null);
      const pose = localPose[layer.id];
      if (pose) {
        // Server caught up — drop the hold so inspector edits apply immediately.
        if (posesMatch(pose, layer)) {
          state.holdTransformsUntil = 0;
          state.holdLayerId = null;
        } else {
          layer.x = pose.x;
          layer.y = pose.y;
          layer.scale_x = pose.scale_x;
          layer.scale_y = pose.scale_y;
          layer.rotation = pose.rotation;
        }
      }
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

  function selectLayerLocal(id) {
    state.selectedId = id || null;
    const stack = $("xwave-layer-stack");
    if (stack) {
      Array.prototype.forEach.call(
        stack.querySelectorAll("[data-layer-id]"),
        function (card) {
          card.classList.toggle(
            "is-selected",
            card.getAttribute("data-layer-id") === id
          );
        }
      );
    }
    draw();
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
      } else if (layer.data_url) {
        // Placeholder so a just-added layer is visible on WORK before
        // its bitmap finishes decoding (keeps OUTPUT from feeling first).
        ctx.fillStyle = "rgba(94, 234, 212, 0.18)";
        ctx.strokeStyle = "rgba(94, 234, 212, 0.55)";
        ctx.lineWidth = 1.5 / state.viewScale;
        ctx.fillRect(-sz.w / 2, -sz.h / 2, sz.w, sz.h);
        ctx.strokeRect(-sz.w / 2, -sz.h / 2, sz.w, sz.h);
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

  function onDown(evt) {
    evt.preventDefault();
    // Selection is menu-only. Empty canvas clicks do not deselect.
    if (!state.selectedId || state.selectedId === "__bg__") return;
    const p = canvasPoint(evt);
    const hit = hitTest(p);
    if (!hit) return;
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
    // Hold only the dragged layer briefly so late scene HTML cannot snap it.
    state.holdLayerId = state.selectedId;
    state.holdTransformsUntil = Date.now() + 600;
    emitTransform(true);
  }

  function ensureCanvasBound() {
    const canvas = $("xwave-work-canvas");
    if (!canvas || canvas._xwaveBound) return;
    canvas._xwaveBound = true;
    canvas.addEventListener("mousedown", onDown);
    if (!window._xwaveWinBound) {
      window._xwaveWinBound = true;
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      window.addEventListener("resize", draw);
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
        const id = card.getAttribute("data-layer-id");
        // Paint selection immediately; Python only syncs inspector state.
        selectLayerLocal(id);
        emitAction({ type: "select", id: id });
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

  // ── boot / DOM watching ──────────────────────────────────────
  function tick() {
    ingestFromDom(false);
    ensureCanvasBound();
    bindStackDelegation();
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
    watchDom();
    ingestFromDom(true);
    ensureCanvasBound();
    bindStackDelegation();
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
