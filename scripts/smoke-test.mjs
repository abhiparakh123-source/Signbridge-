import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const dist = "dist";
const indexPath = join(dist, "index.html");
if (!existsSync(indexPath)) throw new Error("Build output is missing dist/index.html");

const html = readFileSync(indexPath, "utf8");
const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(match => match[1]);
if (assets.length < 2) throw new Error("Expected built JavaScript and CSS assets to be referenced by the page");

for (const asset of assets) {
  const path = join(dist, asset.replace(/^\//, ""));
  if (!existsSync(path)) throw new Error(`Built asset is missing: ${asset}`);
  if (readFileSync(path, "utf8").length < 100) throw new Error(`Built asset appears unexpectedly small: ${asset}`);
}

for (const file of ["manifest.webmanifest", "sw.js"]) {
  if (!existsSync(join(dist, file))) throw new Error(`PWA file is missing: ${file}`);
}

// Guard important user-facing safety and device-cleanup behaviours against accidental removal.
const app = readFileSync("src/App.jsx", "utf8");
const required = [
  ["React import", /import React/],
  ["camera permission flow", /getUserMedia/],
  ["call setup cancellation guard", /pcRef\.current!==pc/],
  ["call attempt invalidation", /callAttemptRef\.current\+=1/],
  ["stale call setup cannot overwrite a newer call", /callAttemptRef\.current!==attempt\|\|pcRef\.current!==pc/],
  ["stale call setup cannot clear a newer busy state", /if\(callAttemptRef\.current===attempt\)setBusy\(false\)/],
  // ICE diagnostics now register state handlers between peer creation and ontrack.
  // Ensure the peer is registered and its track handler is installed before requesting mic permission.
  ["peer connection registered before microphone permission", /pcRef\.current=pc;(?:(?!getUserMedia)[\s\S])*?pc\.ontrack=/],
  ["camera track cleanup", /getTracks\(\)\.forEach\(t=>t\.stop\(\)\)/],
  ["speech recognition error handling", /r\.onerror/],
  ["speech synthesis fallback", /speechSynthesis/],
  ["translation limitation disclosure", /MODEL NOT CONNECTED/],
  ["ASL-first default language", /const LANGS=\["American Sign Language \(ASL\)","Indian Sign Language \(ISL\)","British Sign Language \(BSL\)"\]/],
  ["iPhone-compatible media recorder formats", /video\/mp4/],
  ["family camera preview attachment lifecycle", /videoRef\.current&&streamRef\.current/],
  ["conversation confirmation warning", /Confirm, don't guess/],
  ["conversation board voice-language selector", /board-voice-language/],
  ["Hindi voice locale support", /utterance\.lang=language/],
  ["Hindi and English board voice options", /Hindi \(India\).*English/s]
];
for (const [label, pattern] of required) {
  if (!pattern.test(app)) throw new Error(`Safety smoke check failed: ${label}`);
}

const sw = readFileSync(join(dist, "sw.js"), "utf8");
if (!sw.includes("cache.put(request,response.clone())")) throw new Error("Service worker does not cache fetched app assets for offline use");
if (!sw.includes("signbridge-shell-v3")) throw new Error("Expected updated service worker cache version");

const files = readdirSync(dist);
console.log(`SIGNBRIDGE smoke test passed: index, ${assets.length} assets, PWA files and ${required.length} source safeguards and offline asset caching.`);
