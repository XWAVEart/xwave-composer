/**
 * Library thumbnail clicks → hidden Gradio textbox (#xwave-lib-action).
 * Payload: {context, id, _ts}
 */
(function () {
  "use strict";

  function findActionInput() {
    const root = document.getElementById("xwave-lib-action");
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

  if (window._xwaveLibBound) return;
  window._xwaveLibBound = true;
  document.addEventListener("click", function (e) {
    const item = e.target.closest(".xwave-lib-item[data-lib-id]");
    if (!item) return;
    const grid = item.closest("[data-lib-context]");
    if (!grid) return;
    e.preventDefault();
    emit({
      type: "select",
      context: grid.getAttribute("data-lib-context") || "browse",
      id: item.getAttribute("data-lib-id"),
    });
  });
})();
