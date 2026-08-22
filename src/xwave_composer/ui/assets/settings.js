/* xwave-composer — interface preferences (Settings tab).
   Applies data-mode / data-accent / data-density / data-motion on <html>,
   persisted to localStorage. Runs pre-paint from the head bundle so the
   stored scheme is active before Gradio renders. */
(function () {
  "use strict";

  var KEY = "xwave.ui";
  var DEFAULTS = { mode: "dark", accent: "teal", density: "compact", motion: "full" };
  var ATTRS = ["mode", "accent", "density", "motion"];

  function read() {
    var prefs = {};
    try {
      prefs = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
    } catch (e) {
      prefs = {};
    }
    var out = {};
    ATTRS.forEach(function (k) {
      out[k] = typeof prefs[k] === "string" && prefs[k] ? prefs[k] : DEFAULTS[k];
    });
    return out;
  }

  function save(prefs) {
    try {
      localStorage.setItem(KEY, JSON.stringify(prefs));
    } catch (e) {
      /* private mode etc. — prefs just won't persist */
    }
  }

  function apply(prefs) {
    var el = document.documentElement;
    if (!el) return;
    ATTRS.forEach(function (k) {
      el.setAttribute("data-" + k, prefs[k]);
    });
  }

  /* Pre-paint: apply before the app renders (head script, html element exists). */
  apply(read());

  function syncPanel() {
    var root = document.getElementById("xwave-settings-root");
    if (!root) return;
    var prefs = read();
    ATTRS.forEach(function (k) {
      root.querySelectorAll("[data-set-" + k + "]").forEach(function (btn) {
        var on = btn.getAttribute("data-set-" + k) === prefs[k];
        btn.classList.toggle("is-selected", on);
        btn.setAttribute("aria-pressed", on ? "true" : "false");
      });
    });
  }

  if (!window._xwaveSettingsBound) {
    window._xwaveSettingsBound = true;
    document.addEventListener(
      "click",
      function (e) {
        var btn = e.target.closest(
          "#xwave-set-reset, [data-set-mode], [data-set-accent], [data-set-density], [data-set-motion]"
        );
        if (!btn) return;
        if (btn.id === "xwave-set-reset") {
          save(Object.assign({}, DEFAULTS));
        } else {
          var prefs = read();
          ATTRS.forEach(function (k) {
            var v = btn.getAttribute("data-set-" + k);
            if (v) prefs[k] = v;
          });
          save(prefs);
        }
        apply(read());
        syncPanel();
      },
      true
    );
  }

  function boot() {
    apply(read());
    syncPanel();
    /* Gradio can re-mount tab content; keep the panel's selection in sync. */
    var host = document.querySelector("gradio-app") || document.body;
    if (host && !window._xwaveSettingsObs) {
      window._xwaveSettingsObs = true;
      new MutationObserver(function () {
        syncPanel();
      }).observe(host, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
