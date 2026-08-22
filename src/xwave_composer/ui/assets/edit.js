/**
 * Edit tab: SAM2 magic-select canvas + layer stack.
 * Scene: #xwave-edit-root[data-scene] (base64 JSON).
 * Actions: #xwave-edit-action.
 */
(function () {
  "use strict";

  const HOVER_MS = 70;
  const state = {
    width: 1024,
    height: 1024,
    previewUrl: "",
    overlayUrl: "",
    mode: "include",
    empty: true,
    previewImg: null,
    overlayImg: null,
    lastScene: "",
    hoverTimer: null,
    lastHoverKey: "",
    lastLayout: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function findActionInput() {
    const root = $("xwave-edit-action");
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

  function emit(payload) {
    payload._ts = Date.now();
    const input = findActionInput();
    if (!input) return;
    setNativeValue(input, JSON.stringify(payload));
  }

  function selectLocal(id) {
    const stack = $("xwave-edit-layer-stack");
    if (!stack) return;
    stack.querySelectorAll("[data-layer-id]").forEach(function (card) {
      card.classList.toggle("is-selected", card.getAttribute("data-layer-id") === id);
    });
  }

  function layout() {
    const canvas = $("xwave-edit-canvas");
    if (!canvas) return null;
    const wrap = canvas.parentElement;
    const viewW = wrap.clientWidth || canvas.clientWidth || 640;
    const viewH = wrap.clientHeight || 560;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(viewW * dpr) || canvas.height !== Math.floor(viewH * dpr)) {
      canvas.width = Math.floor(viewW * dpr);
      canvas.height = Math.floor(viewH * dpr);
      canvas.style.width = viewW + "px";
      canvas.style.height = viewH + "px";
    }
    const iw = state.width || 1;
    const ih = state.height || 1;
    const scale = Math.min(viewW / iw, viewH / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    const ox = (viewW - dw) / 2;
    const oy = (viewH - dh) / 2;
    state.lastLayout = { viewW: viewW, viewH: viewH, scale: scale, ox: ox, oy: oy, dpr: dpr, dw: dw, dh: dh };
    return state.lastLayout;
  }

  function eventToImage(e) {
    const canvas = $("xwave-edit-canvas");
    const lay = layout();
    if (!canvas || !lay || state.empty) return null;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const x = (sx - lay.ox) / lay.scale;
    const y = (sy - lay.oy) / lay.scale;
    if (x < 0 || y < 0 || x >= state.width || y >= state.height) return null;
    return { x: x, y: y };
  }

  function draw() {
    const canvas = $("xwave-edit-canvas");
    if (!canvas) return;
    const lay = layout();
    if (!lay) return;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(lay.dpr, 0, 0, lay.dpr, 0, 0);
    ctx.clearRect(0, 0, lay.viewW, lay.viewH);
    ctx.fillStyle = "#0a0c0f";
    ctx.fillRect(0, 0, lay.viewW, lay.viewH);
    if (state.empty || !state.previewImg) {
      ctx.fillStyle = "#828a98";
      ctx.font = '14px "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = "center";
      ctx.fillText("Load an image, then hover an object", lay.viewW / 2, lay.viewH / 2);
      return;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(state.previewImg, lay.ox, lay.oy, lay.dw, lay.dh);
    if (state.overlayImg) {
      ctx.drawImage(state.overlayImg, lay.ox, lay.oy, lay.dw, lay.dh);
    }
  }

  function loadIfChanged(url, slot, onReady) {
    if (!url) {
      state[slot] = null;
      onReady();
      return;
    }
    if (state[slot] && state[slot].src && state[slot].src.indexOf(url) !== -1) {
      onReady();
      return;
    }
    const img = new Image();
    img.onload = function () {
      state[slot] = img;
      onReady();
    };
    img.src = url;
  }

  function parseScene() {
    const root = $("xwave-edit-root");
    if (!root) return;
    const b64 = root.getAttribute("data-scene") || "";
    if (!b64 || b64 === state.lastScene) return;
    state.lastScene = b64;
    let scene;
    try {
      scene = JSON.parse(atob(b64));
    } catch (err) {
      return;
    }
    state.width = scene.width || state.width;
    state.height = scene.height || state.height;
    state.mode = scene.mode || "include";
    state.empty = !!scene.empty;
    const canvas = $("xwave-edit-canvas");
    if (canvas) {
      canvas.style.cursor = state.mode === "off" || state.empty ? "default" : "crosshair";
    }
    let pending = 2;
    const done = function () {
      pending -= 1;
      if (pending <= 0) draw();
    };
    loadIfChanged(scene.preview_url || "", "previewImg", done);
    loadIfChanged(scene.overlay_url || "", "overlayImg", done);
  }

  function queueHover(pt) {
    if (!pt || state.mode === "off" || state.empty) return;
    const key = (pt.x | 0) + "," + (pt.y | 0);
    if (key === state.lastHoverKey) return;
    if (state.hoverTimer) clearTimeout(state.hoverTimer);
    state.hoverTimer = setTimeout(function () {
      state.lastHoverKey = key;
      emit({ type: "hover", x: pt.x, y: pt.y });
    }, HOVER_MS);
  }

  function bindCanvas() {
    const canvas = $("xwave-edit-canvas");
    if (!canvas || canvas._xwaveBound) return;
    canvas._xwaveBound = true;
    canvas.addEventListener("mousemove", function (e) {
      queueHover(eventToImage(e));
    });
    canvas.addEventListener("mouseleave", function () {
      if (state.hoverTimer) clearTimeout(state.hoverTimer);
      state.lastHoverKey = "";
      emit({ type: "hover_end" });
    });
    canvas.addEventListener("click", function (e) {
      e.preventDefault();
      const pt = eventToImage(e);
      if (!pt) return;
      if (state.hoverTimer) clearTimeout(state.hoverTimer);
      emit({ type: "cut", x: pt.x, y: pt.y });
    });
  }

  function sync() {
    parseScene();
    bindCanvas();
    bindFx();
    draw();
  }

  if (window._xwaveEditBound) return;
  window._xwaveEditBound = true;

  document.addEventListener("click", function (e) {
    const root = e.target.closest("#xwave-edit-layers-root");
    if (!root) return;
    const del = e.target.closest("[data-delete-id]");
    if (del) {
      e.preventDefault();
      e.stopPropagation();
      emit({ type: "delete", id: del.getAttribute("data-delete-id") });
      return;
    }
    const card = e.target.closest("[data-layer-id]");
    if (!card) return;
    const id = card.getAttribute("data-layer-id");
    selectLocal(id);
    emit({ type: "select", id: id });
  });

  document.addEventListener("dragstart", function (e) {
    const card = e.target.closest("#xwave-edit-layers-root [data-layer-id]");
    if (!card || card.getAttribute("data-layer-id") === "base") return;
    card.classList.add("is-dragging");
    e.dataTransfer.setData("text/plain", card.getAttribute("data-layer-id"));
  });
  document.addEventListener("dragend", function (e) {
    const card = e.target.closest("#xwave-edit-layers-root [data-layer-id]");
    if (card) card.classList.remove("is-dragging");
  });
  document.addEventListener("dragover", function (e) {
    const card = e.target.closest("#xwave-edit-layers-root [data-layer-id]");
    if (!card || card.getAttribute("data-layer-id") === "base") return;
    e.preventDefault();
    card.classList.add("is-over");
  });
  document.addEventListener("dragleave", function (e) {
    const card = e.target.closest("#xwave-edit-layers-root [data-layer-id]");
    if (card) card.classList.remove("is-over");
  });
  document.addEventListener("drop", function (e) {
    const card = e.target.closest("#xwave-edit-layers-root [data-layer-id]");
    if (!card) return;
    e.preventDefault();
    card.classList.remove("is-over");
    const fromId = e.dataTransfer.getData("text/plain");
    const toId = card.getAttribute("data-layer-id");
    if (!fromId || fromId === toId || toId === "base") return;
    const stack = $("xwave-edit-layer-stack");
    if (!stack) return;
    const topFirst = Array.from(stack.querySelectorAll("[data-layer-id]"))
      .map(function (c) {
        return c.getAttribute("data-layer-id");
      })
      .filter(function (id) {
        return id !== "base";
      });
    const from = topFirst.indexOf(fromId);
    const to = topFirst.indexOf(toId);
    if (from < 0 || to < 0) return;
    topFirst.splice(from, 1);
    topFirst.splice(to, 0, fromId);
    emit({ type: "reorder", ids: topFirst.slice().reverse() });
  });

  const fx = { schema: null, bound: false, lastLayers: "" };

  function parseFxSchema() {
    const root = $("xwave-fx-root");
    if (!root) return null;
    const b64 = root.getAttribute("data-schema") || "";
    if (!b64) return null;
    try {
      return JSON.parse(atob(b64));
    } catch (err) {
      return null;
    }
  }

  function currentEffect() {
    if (!fx.schema) return null;
    const sel = $("xwave-fx-effect");
    const id = sel ? sel.value : "";
    return fx.schema.effects.find(function (item) {
      return item.id === id;
    }) || null;
  }

  function fillSelect(select, items, valueKey, labelKey, current) {
    if (!select) return;
    const prev = current || select.value;
    select.innerHTML = "";
    items.forEach(function (item) {
      const opt = document.createElement("option");
      opt.value = item[valueKey];
      opt.textContent = item[labelKey];
      select.appendChild(opt);
    });
    if (prev) select.value = prev;
    if (!select.value && items[0]) select.value = items[0][valueKey];
  }

  function paramVisible(param, values) {
    const when = param.visible_when || {};
    const keys = Object.keys(when);
    if (!keys.length) return true;
    return keys.every(function (key) {
      const allowed = when[key] || [];
      const raw = values[key];
      const asText = raw === true ? "true" : raw === false ? "false" : String(raw);
      return allowed.indexOf(asText) !== -1;
    });
  }

  function collectParams() {
    const box = $("xwave-fx-params");
    const values = {};
    if (!box) return values;
    box.querySelectorAll("[data-param]").forEach(function (el) {
      const key = el.getAttribute("data-param");
      if (el.type === "checkbox") values[key] = el.checked;
      else values[key] = el.value;
    });
    return values;
  }

  function renderParams() {
    const spec = currentEffect();
    const box = $("xwave-fx-params");
    const secondaryWrap = $("xwave-fx-secondary-wrap");
    const warpWrap = $("xwave-fx-warp-wrap");
    if (!box || !spec) return;
    if (secondaryWrap) secondaryWrap.classList.toggle("is-hidden", !spec.two_image);
    if (warpWrap) warpWrap.classList.toggle("is-hidden", !spec.warp);
    const previous = collectParams();
    box.innerHTML = "";
    spec.params.forEach(function (param) {
      if (!paramVisible(param, previous)) return;
      const wrap = document.createElement("label");
      wrap.className = "xwave-fx-field";
      const title = document.createElement("span");
      title.textContent = param.label;
      wrap.appendChild(title);
      let input;
      const value = previous[param.key] !== undefined ? previous[param.key] : param.default;
      if (param.kind === "choice") {
        input = document.createElement("select");
        (param.choices || []).forEach(function (choice) {
          const opt = document.createElement("option");
          opt.value = choice.id;
          opt.textContent = choice.label;
          input.appendChild(opt);
        });
        input.value = value == null ? "" : String(value);
      } else if (param.kind === "bool") {
        wrap.classList.add("is-bool");
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = value === true || value === "true";
      } else if (param.kind === "color") {
        input = document.createElement("input");
        input.type = "color";
        input.value = value || "#ffffff";
      } else {
        input = document.createElement("input");
        input.type = "number";
        if (param.min != null) input.min = param.min;
        if (param.max != null) input.max = param.max;
        if (param.step != null) input.step = param.step;
        input.placeholder = param.kind === "seed" ? "random" : "";
        input.value = value == null ? "" : value;
      }
      input.setAttribute("data-param", param.key);
      input.addEventListener("change", renderParams);
      wrap.appendChild(input);
      box.appendChild(wrap);
    });
  }

  function fillEffects() {
    if (!fx.schema) return;
    const groupSel = $("xwave-fx-group");
    const effectSel = $("xwave-fx-effect");
    if (!groupSel || !effectSel) return;
    fillSelect(groupSel, fx.schema.groups, "id", "label");
    const group = groupSel.value;
    const items = fx.schema.effects.filter(function (item) {
      return item.group === group;
    });
    fillSelect(effectSel, items, "id", "label");
    renderParams();
  }

  function fillSecondary() {
    const select = $("xwave-fx-secondary");
    const root = $("xwave-edit-layers-root");
    if (!select || !root) return;
    const b64 = root.getAttribute("data-layers") || "";
    if (b64 === fx.lastLayers && select.options.length) return;
    fx.lastLayers = b64;
    let layers = [];
    try {
      layers = b64 ? JSON.parse(atob(b64)) : [];
    } catch (err) {
      layers = [];
    }
    const items = [{ id: "rest", name: "Rest of composite" }].concat(layers);
    fillSelect(select, items, "id", "name");
  }

  function bindFx() {
    const root = $("xwave-fx-root");
    if (!root) return;
    if (!fx.schema) fx.schema = parseFxSchema();
    fillSecondary();
    if (root._xwaveFxBound) return;
    root._xwaveFxBound = true;
    fillEffects();
    const groupSel = $("xwave-fx-group");
    const effectSel = $("xwave-fx-effect");
    if (groupSel) groupSel.addEventListener("change", fillEffects);
    if (effectSel) effectSel.addEventListener("change", renderParams);
    const apply = $("xwave-fx-apply");
    if (apply) {
      apply.addEventListener("click", function (e) {
        e.preventDefault();
        const spec = currentEffect();
        if (!spec) return;
        const warp = $("xwave-fx-warp");
        const secondary = $("xwave-fx-secondary");
        emit({
          type: "effect",
          id: spec.id,
          params: collectParams(),
          secondary: secondary ? secondary.value : "rest",
          warp: !!(warp && warp.checked),
        });
      });
    }
  }

  const obs = new MutationObserver(function () {
    sync();
  });
  function boot() {
    obs.observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["data-scene", "data-layers", "data-schema"],
    });
    sync();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  setInterval(sync, 400);
  window.addEventListener("resize", draw);
})();
