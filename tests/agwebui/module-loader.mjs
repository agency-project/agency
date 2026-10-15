import {readFile} from 'node:fs/promises';
const staticRoot = new URL('../../agency/observability/agwebui/static/', import.meta.url);
const urls = new Map();
export async function moduleURL(name) {
  if (urls.has(name)) return urls.get(name);
  let source = await readFile(new URL(name, staticRoot), 'utf8');
  const dependencies = [...source.matchAll(/from ['"]\.\/([^'"]+)['"]/g)];
  for (const dependency of dependencies) source = source.replace(`./${dependency[1]}`, await moduleURL(dependency[1]));
  const url = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
  urls.set(name, url); return url;
}
export async function loadModule(name) {return import(await moduleURL(name));}
