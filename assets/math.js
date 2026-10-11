"use strict";
document.querySelectorAll(".prose code").forEach(function (node) {
  var display = node.classList.contains("language-math") || !!node.closest(".language-math");
  if (!display && (node.closest("pre") || !node.textContent.includes("\\"))) return;
  var target = document.createElement(display ? "div" : "span");
  target.className = display ? "math-display" : "math-inline";
  try {
    katex.render(node.textContent.trim(), target, {displayMode: display, throwOnError: true, trust: false});
    (display ? node.closest("div.highlighter-rouge") || node.closest("pre") : node).replaceWith(target);
  } catch (error) {
    console.error("Unable to render a notebook formula", error);
  }
});
