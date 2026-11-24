import test from "node:test";
import assert from "node:assert/strict";
import { validateCallSdp } from "../src/call-sdp.js";

const validSdp = [
  "v=0",
  "o=- 123 2 IN IP4 127.0.0.1",
  "s=-",
  "t=0 0",
  "a=group:BUNDLE 0",
  "a=ice-ufrag:abcd",
  "a=ice-pwd:abcdefghijklmnopqrstuvwxyz",
  "a=fingerprint:sha-256 AA:BB:CC",
  "m=audio 9 UDP/TLS/RTP/SAVPF 111",
  "a=mid:0",
  "a=rtpmap:111 opus/48000/2"
].join("\r\n");

test("accepts a complete audio offer", () => {
  assert.deepEqual(validateCallSdp(validSdp, "offer"), { ok: true, message: "" });
});

test("accepts a complete audio answer", () => {
  assert.equal(validateCallSdp(validSdp, "answer").ok, true);
});

test("rejects empty and non-text connection codes", () => {
  assert.equal(validateCallSdp("", "offer").ok, false);
  assert.equal(validateCallSdp(null, "offer").ok, false);
});

test("rejects non-audio or malformed SDP", () => {
  assert.equal(validateCallSdp("hello", "offer").ok, false);
  assert.equal(validateCallSdp(validSdp.replace("m=audio", "m=video"), "offer").ok, false);
});

test("rejects SDP missing ICE or DTLS security attributes", () => {
  assert.equal(validateCallSdp(validSdp.replace("a=ice-pwd:abcdefghijklmnopqrstuvwxyz\r\n", ""), "offer").ok, false);
  assert.equal(validateCallSdp(validSdp.replace("a=fingerprint:sha-256 AA:BB:CC\r\n", ""), "answer").ok, false);
});

test("rejects oversized connection codes", () => {
  assert.equal(validateCallSdp(validSdp + "x".repeat(256_000), "offer").ok, false);
});

test("rejects an unexpected SDP type", () => {
  assert.equal(validateCallSdp(validSdp, "pranswer").ok, false);
});
