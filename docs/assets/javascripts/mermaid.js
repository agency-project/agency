// Keep the renderer version aligned with tools/docs/package-lock.json.
(async () => {
  const blocks = [...document.querySelectorAll('pre > code.language-mermaid')];
  if (!blocks.length) return;
  try {
    const { default: mermaid } = await import('https://cdn.jsdelivr.net/npm/mermaid@12.1.0/dist/mermaid.esm.min.mjs');
    mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral', flowchart: { htmlLabels: false } });
    const nodes = blocks.map(code => {
      const container = document.createElement('pre');
      container.className = 'mermaid';
      container.textContent = code.textContent;
      code.parentElement.replaceWith(container);
      return container;
    });
    await mermaid.run({ nodes });
  } catch (error) {
    console.error('Mermaid rendering failed; editable diagram source remains available.', error);
  }
})();
