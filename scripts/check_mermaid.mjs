// Validate canonical Markdown and .mmd diagrams with the site’s pinned Mermaid.
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createRequire } from 'node:module';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const requireDocs = createRequire(path.join(root, 'tools/docs/package.json'));
const version = requireDocs('mermaid/package.json').version;
const renderer = await readFile(path.join(root, 'docs/assets/javascripts/mermaid.js'), 'utf8');
if (!renderer.includes(`mermaid@${version}/dist/mermaid.esm.min.mjs`)) {
  throw new Error('Site renderer and diagram validator must use the same Mermaid version');
}
const { JSDOM } = requireDocs('jsdom');
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { default: mermaid } = await import(requireDocs.resolve('mermaid'));
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });

async function diagramFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (['archive', 'figures', 'node_modules'].includes(entry.name)) continue;
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await diagramFiles(filename));
    else if (entry.name.endsWith('.md') || entry.name.endsWith('.mmd')) files.push(filename);
  }
  return files.sort();
}

let count = 0;
for (const filename of await diagramFiles(path.join(root, 'docs'))) {
  const source = await readFile(filename, 'utf8');
  const diagrams = filename.endsWith('.mmd')
    ? [source]
    : [...source.matchAll(/^```mermaid\s*\n([\s\S]*?)^```/gm)].map(match => match[1]);
  for (const [index, diagram] of diagrams.entries()) {
    try {
      await mermaid.parse(diagram);
      count += 1;
    } catch (error) {
      throw new Error(`${path.relative(root, filename)} diagram ${index + 1}: ${error.message}`);
    }
  }
}
if (!count) throw new Error('No canonical Mermaid diagrams found');
console.log(`Validated ${count} Mermaid diagrams with version ${version}.`);
dom.window.close();
