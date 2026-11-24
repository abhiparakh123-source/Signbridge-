const MAX_SDP_LENGTH = 256_000;

export function validateCallSdp(value, expectedType) {
  if (expectedType !== "offer" && expectedType !== "answer") {
    return { ok: false, message: "Invalid connection-code type." };
  }
  if (typeof value !== "string" || !value.trim()) {
    return { ok: false, message: "Paste the other person's connection code first." };
  }

  const sdp = value.trim();
  if (sdp.length > MAX_SDP_LENGTH) {
    return { ok: false, message: "Connection code is too large. Copy it again from SIGNBRIDGE." };
  }

  const lines = sdp.split(/\r?\n/);
  const has = pattern => lines.some(line => pattern.test(line));
  if (lines[0] !== "v=0" || !has(/^m=audio\s/)) {
    return { ok: false, message: "This is not a valid SIGNBRIDGE audio connection code. Check that you copied the complete offer or answer." };
  }
  if (!has(/^a=ice-ufrag:.+/) || !has(/^a=ice-pwd:.+/) || !has(/^a=fingerprint:.+/)) {
    return { ok: false, message: "The connection code is incomplete or missing secure network details. Copy the full code again." };
  }

  return { ok: true, message: "" };
}
