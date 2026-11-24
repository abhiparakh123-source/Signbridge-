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
  ["call SDP validation is imported", /validateCallSdp/],
  ["invalid offers are rejected before microphone permission", /validateCallSdp\(remoteOffer,"offer"\);if\(!validation\.ok\)/],
  ["invalid answers are rejected before applying remote description", /validateCallSdp\(remoteAnswer,"answer"\);if\(!validation\.ok\)/],
  ["restart offers are validated before applying", /validateCallSdp\(remoteOffer,"offer"\);if\(!validation\.ok\)/],
  ["call setup cancellation guard", /pcRef\.current!==pc/],
  ["call attempt invalidation", /callAttemptRef\.current\+=1/],
  ["ICE restart refreshes candidate diagnostics", /setIceCandidateTypes\(\[\]\);pc\.restartIce\?\.\(\)/],
  ["caller can create an ICE restart offer", /createOffer\(\{iceRestart:true\}\)/],
  ["caller restarts ICE without replacing the peer connection", /pc\.restartIce\?\.\(\)/],
  ["receiver applies a renegotiation offer without closing the call", /setRemoteDescription\(\{type:"offer",sdp:remoteOffer\.trim\(\)\}\)/],
  ["network restart controls are exposed to both call roles", /Restart network path · create new offer[\s\S]*?Apply new network offer without ending call/],
  ["call reset clears stale signaling codes", /setOffer\(""\);setAnswer\(""\);setRemoteOffer\(""\);setRemoteAnswer\(""\)/],
  ["ICE candidate type diagnostics", /pc\.onicecandidate=e=>\{const match=e\.candidate\?\.candidate\?\.match\(\/ typ \(host\|srflx\|prflx\|relay\)/],
  ["ICE candidate types exposed in call diagnostics", /Candidate types: \{iceCandidateTypes\.length/],
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

// Guard the managed WebRTC path separately from the legacy manual-SDP fallback.
const metered = readFileSync("src/MeteredCall.jsx", "utf8");
const meteredRequired = [
  ["Metered SDK import", /import\\s*\\{\\s*MeteredPeer\\s*\\}\\s*from\\s*["']@metered-ca\\/realtime["']/],
  ["publishable key passed only from build environment", /new MeteredPeer\\(\\{\\s*apiKey\\s*\\}\\)/],
  ["room codes are generated with secure randomness", /crypto\\.getRandomValues/],
  ["private room join", /peer\\.join\\(["']signbridge-/],
  ["local audio is attached to the peer", /peer\\.addStream\\(stream,\\s*\\{\\s*role:\\s*["']audio["']\\s*\\}\\)/],
  ["remote peer stream is attached to audio playback", /remote\\.on\\(["']stream-added["']/],
  ["peer-left clears remote audio", /peer\\.on\\(["']peer-left["']/],
  ["local tracks are stopped during teardown", /getTracks\\(\\)\\.forEach\\(track\\s*=>\\s*track\\.stop\\(\\)\\)/],
  ["remote captions require an explicit user action", /onClick=\\{toggleRemoteCaptions\\}/],
  ["remote captions use the remote audio track", /recognition\\.start\\(track\\)/],
  ["mobile remote-track captions remain disabled", /Android\\|iPhone\\|iPad\\|iPod/],
  ["free-tier quota limitations are disclosed", /Free-tier quotas apply/]
];
for (const [label, pattern] of meteredRequired) {
  if (!pattern.test(metered)) throw new Error(`Metered call smoke check failed: ${label}`);
}
console.log(`Metered call smoke checks passed: ${meteredRequired.length} integration safeguards.`);

const sw = readFileSync(join(dist, "sw.js"), "utf8");
if (!sw.includes("cache.put(request,response.clone())")) throw new Error("Service worker does not cache fetched app assets for offline use");
if (!sw.includes("signbridge-shell-v3")) throw new Error("Expected updated service worker cache version");

const files = readdirSync(dist);
console.log(`SIGNBRIDGE smoke test passed: index, ${assets.length} assets, PWA files and ${required.length} source safeguards and offline asset caching.`);
