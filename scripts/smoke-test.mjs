import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const dist = "dist";
const indexPath = join(dist, "index.html");
if (!existsSync(indexPath)) throw new Error("Build output is missing dist/index.html");

const html = readFileSync(indexPath, "utf8");
const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(match => match[1]);
if (assets.length === 0) throw new Error("No JavaScript or CSS assets are referenced by the built page");

for (const asset of assets) {
  const path = join(dist, asset.replace(/^\//, ""));
  if (!existsSync(path)) throw new Error(`Built asset is missing: ${asset}`);
  if (asset.endsWith(".js") && readFileSync(path, "utf8").length < 100) {
    throw new Error(`JavaScript bundle appears unexpectedly small: ${asset}`);
  }
}

const files = readdirSync(dist);
if (!files.includes("manifest.webmanifest")) throw new Error("PWA manifest is missing");
if (!files.includes("sw.js")) throw new Error("Service worker is missing");

console.log(`SIGNBRIDGE smoke test passed: index, ${assets.length} referenced assets, manifest and service worker exist.`);
