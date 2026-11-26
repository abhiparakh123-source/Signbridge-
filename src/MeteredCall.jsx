import React, { useEffect, useRef, useState } from "react";
import { MeteredPeer } from "@metered-ca/realtime";

function makeRoomCode() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
}

function remoteTrackCaptionSupport() {
  const ua = navigator.userAgent || "";
  if (/Android|iPhone|iPad|iPod/i.test(ua)) return false;
  const chrome = ua.match(/(?:Chrome|CriOS)\/(\d+)/);
  const edge = ua.match(/Edg\/(\d+)/);
  const firefox = ua.match(/Firefox\/(\d+)/);
  const opera = ua.match(/(?:OPR)\/(\d+)/);
  return !!(
    (chrome && Number(chrome[1]) >= 135) ||
    (edge && Number(edge[1]) >= 135) ||
    (firefox && Number(firefox[1]) >= 160) ||
    (opera && Number(opera[1]) >= 120)
  );
}

export function MeteredCall({ apiKey }) {
  const [room, setRoom] = useState("");
  const [activeRoom, setActiveRoom] = useState("");
  const [status, setStatus] = useState("Create a private call room or enter a room code");
  const [busy, setBusy] = useState(false);
  const [muted, setMuted] = useState(false);
  const [remoteCaption, setRemoteCaption] = useState("");
  const [remoteCaptionInterim, setRemoteCaptionInterim] = useState("");
  const [remoteCaptionLanguage, setRemoteCaptionLanguage] = useState("en-IN");
  const [remoteCaptionOn, setRemoteCaptionOn] = useState(false);
  const [iceRoute, setIceRoute] = useState("Waiting for connection diagnostics");

  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const audioRef = useRef(null);
  const remoteTrackRef = useRef(null);
  const remoteRecRef = useRef(null);
  const iceStatsTimerRef = useRef(null);
  const attemptRef = useRef(0);

  const stopRemoteCaptions = () => {
    const recognition = remoteRecRef.current;
    remoteRecRef.current = null;
    if (iceStatsTimerRef.current) clearInterval(iceStatsTimerRef.current);
    iceStatsTimerRef.current = null;
    try {
      recognition?.stop();
    } catch {
      // The browser may already have stopped recognition.
    }
    setRemoteCaptionOn(false);
    setRemoteCaptionInterim("");
  };

  const stopIceDiagnostics = () => {
    if (iceStatsTimerRef.current) clearInterval(iceStatsTimerRef.current);
    iceStatsTimerRef.current = null;
    setIceRoute("Call ended");
  };

  const stop = async () => {
    attemptRef.current++;
    stopRemoteCaptions();
    stopIceDiagnostics();
    remoteTrackRef.current = null;
    const peer = peerRef.current;
    peerRef.current = null;
    try {
      await peer?.close("call ended");
    } catch {
      // Continue releasing local media even if signalling is already closed.
    }
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (audioRef.current) audioRef.current.srcObject = null;
    setActiveRoom("");
    setMuted(false);
    setRemoteCaption("");
    setRemoteCaptionInterim("");
    setBusy(false);
    setStatus("Call ended");
  };

  useEffect(() => () => {
    attemptRef.current++;
    const recognition = remoteRecRef.current;
    remoteRecRef.current = null;
    try {
      recognition?.stop();
    } catch {
      // Ignore teardown races.
    }
    try {
      peerRef.current?.close("page closed");
    } catch {
      // Ignore teardown races.
    }
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  const join = async code => {
    const clean = code.trim().toLowerCase();
    if (!/^[a-f0-9]{16}$/.test(clean)) {
      setStatus("Enter a valid 16-character room code.");
      return;
    }
    if (peerRef.current || busy) return;

    const attempt = ++attemptRef.current;
    setBusy(true);
    setStatus("Requesting microphone permission…");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Microphone calling is unavailable in this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      if (attemptRef.current !== attempt) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      const peer = new MeteredPeer({ apiKey });
      peerRef.current = peer;
      streamRef.current = stream;

      peer.on("peer-joined", ({ peer: remote }) => {
        if (attemptRef.current !== attempt) return;

        const updateIceRoute = async () => {
          try {
            const pc = remote.pc; // Re-read after reconnect; Metered replaces this object.
            if (!pc || typeof pc.getStats !== "function") { setIceRoute("ICE diagnostics unavailable"); return; }
            const stats = await pc.getStats();
            const reports = [...stats.values()];
            const pair = reports.find(item => item.type === "candidate-pair" && (item.selected === true || (item.nominated === true && item.state === "succeeded")));
            if (!pair) { setIceRoute("ICE route not selected yet"); return; }
            const local = stats.get(pair.localCandidateId);
            const remoteCandidate = stats.get(pair.remoteCandidateId);
            const types = [local?.candidateType, remoteCandidate?.candidateType].filter(Boolean);
            setIceRoute(types.includes("relay") ? "TURN relay active" : types.length ? "Direct connection · " + [...new Set(types)].join(" / ") : "Connected · candidate type unavailable");
          } catch { setIceRoute("ICE diagnostics temporarily unavailable"); }
        };
        const startIceDiagnostics = () => {
          if (iceStatsTimerRef.current) clearInterval(iceStatsTimerRef.current);
          setIceRoute("Checking selected ICE route…");
          updateIceRoute();
          iceStatsTimerRef.current = setInterval(updateIceRoute, 5000);
        };
        startIceDiagnostics();
        remote.on("connection-reset", startIceDiagnostics);

        remote.on("stream-added", ({ stream: remoteStream }) => {
          if (attemptRef.current !== attempt) return;

          const audioTracks = remoteStream?.getAudioTracks?.() || [];
          const remoteTrack = audioTracks.find(track => track.readyState === "live") || audioTracks[0] || null;

          if (remoteTrackRef.current && remoteTrackRef.current !== remoteTrack) {
            stopRemoteCaptions();
          }
          remoteTrackRef.current = remoteTrack;

          if (remoteTrack) {
            remoteTrack.onended = () => {
              if (remoteTrackRef.current !== remoteTrack) return;
              stopRemoteCaptions();
              remoteTrackRef.current = null;
              setStatus("Other caller audio ended");
            };
          }

          if (audioRef.current) {
            audioRef.current.srcObject = remoteStream;
            audioRef.current.play().catch(() => {
              setStatus("Connected · tap Play to hear the other caller");
            });
          }
          if (remoteTrack) setStatus("Connected · encrypted WebRTC audio");
          else setStatus("Connected · waiting for the other caller's audio track");
        });

        remote.on("state-change", ({ to }) => {
          if (attemptRef.current !== attempt) return;
          if (to === "connected") {
            setStatus("Connected · encrypted WebRTC audio");
          } else if (to === "disconnected") {
            setStatus("Other caller's network connection was interrupted · waiting for recovery…");
          } else if (to === "failed") {
            setStatus("Other caller's connection failed · try rejoining the room or changing networks");
          }
        });
      });

      peer.on("peer-left", () => {
        if (attemptRef.current !== attempt) return;
        stopRemoteCaptions();
        stopIceDiagnostics();
        remoteTrackRef.current = null;
        if (audioRef.current) audioRef.current.srcObject = null;
        setStatus("Other caller left the room");
      });

      peer.on("state-change", ({ to }) => {
        if (attemptRef.current !== attempt) return;
        if (to === "reconnecting") setStatus("Signalling interrupted · reconnecting…");
        else if (to === "joined") setStatus("Room ready · waiting for the other caller…");
        else if (to === "disconnected") setStatus("Call signalling disconnected · attempting recovery…");
        else if (to === "failed") setStatus("Call signalling failed · check the network and rejoin");
        else if (to === "closed" && peerRef.current === peer) setStatus("Call signalling closed");
      });

      peer.on("error", ({ err }) => {
        if (attemptRef.current === attempt) {
          setStatus("Call service error: " + (err?.message || err?.name || "connection failed"));
        }
      });

      peer.addStream(stream, { role: "audio" });
      setStatus("Joining call room…");
      await peer.join("signbridge-" + clean);
      if (attemptRef.current !== attempt) return;
      setRoom(clean);
      setActiveRoom(clean);
      setStatus("Room ready · waiting for the other caller…");
    } catch (error) {
      if (attemptRef.current === attempt) {
        const peer = peerRef.current;
        peerRef.current = null;
        stopRemoteCaptions();
        remoteTrackRef.current = null;
        try {
          await peer?.close("setup failed");
        } catch {
          // Continue cleanup after a signalling failure.
        }
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setStatus(error?.message || "Could not join the call room");
      }
    } finally {
      if (attemptRef.current === attempt) setBusy(false);
    }
  };

  const create = () => {
    const code = makeRoomCode();
    setRoom(code);
    join(code);
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(activeRoom);
      setStatus("Room code copied · share it only with the intended caller");
    } catch {
      setStatus("Copy unavailable · select the room code and copy it manually");
    }
  };

  const mute = () => {
    const nextMuted = !muted;
    streamRef.current?.getAudioTracks().forEach(track => {
      track.enabled = !nextMuted;
    });
    setMuted(nextMuted);
  };

  const toggleRemoteCaptions = () => {
    if (remoteCaptionOn) {
      stopRemoteCaptions();
      setStatus("Remote caller captions stopped");
      return;
    }

    if (!remoteTrackCaptionSupport()) {
      setStatus("Remote-audio captions are not supported on this device/browser yet. This control will not fall back to your microphone.");
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatus("Speech recognition is unavailable in this browser.");
      return;
    }

    const track = remoteTrackRef.current;
    if (!track || track.readyState !== "live") {
      setStatus("Remote caller audio is not connected yet.");
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      remoteRecRef.current = recognition;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = remoteCaptionLanguage;

      recognition.onresult = event => {
        if (remoteRecRef.current !== recognition) return;
        let interim = "";
        let finalText = "";
        for (let index = event.resultIndex; index < event.results.length; index++) {
          if (event.results[index].isFinal) finalText += event.results[index][0].transcript + " ";
          else interim += event.results[index][0].transcript;
        }
        if (finalText.trim()) {
          setRemoteCaption(value => (value + " " + finalText.trim()).trim().slice(-4000));
        }
        setRemoteCaptionInterim(interim.trim());
      };

      recognition.onerror = event => {
        if (remoteRecRef.current !== recognition) return;
        remoteRecRef.current = null;
        setRemoteCaptionOn(false);
        setRemoteCaptionInterim("");
        setStatus("Remote captions stopped (" + (event.error || "speech service error") + ").");
      };

      recognition.onend = () => {
        if (remoteRecRef.current !== recognition) return;
        remoteRecRef.current = null;
        setRemoteCaptionOn(false);
        setRemoteCaptionInterim("");
      };

      recognition.start(track);
      setRemoteCaptionOn(true);
      setStatus("Remote caller captions started");
    } catch (error) {
      if (remoteRecRef.current) remoteRecRef.current = null;
      setRemoteCaptionOn(false);
      setRemoteCaptionInterim("");
      setStatus("This browser cannot caption the remote audio track (" + (error?.name || "unsupported") + ").");
    }
  };

  const captionSupported = remoteTrackCaptionSupport();

  return (
    <div className="panel internet-call">
      <div className="paneltop">
        <b>FREE INTERNET AUDIO CALL</b>
        <small>METERED · BETA</small>
      </div>
      <p className="privacy">
        Create a room and share its code with one trusted SIGNBRIDGE user. Both callers join the same room.
        Calls use peer-to-peer WebRTC audio and Metered signalling. TURN relay is available only when the
        account has an active free TURN service, automatic TURN injection is enabled, and quota remains.
      </p>
      <div className="call-status" role="status">{status}</div>
      <p className="privacy" aria-live="polite">Network route: {iceRoute}. TURN is confirmed only when this reports “TURN relay active”.</p>

      {!activeRoom ? (
        <>
          <button className="primary full" disabled={busy} onClick={create}>
            {busy ? "Connecting…" : "Create a call room"}
          </button>
          <label className="board-label">Or enter the other person's room code</label>
          <input
            className="board-input"
            value={room}
            onChange={event => setRoom(event.target.value.replace(/[^a-fA-F0-9]/g, "").slice(0, 16))}
            placeholder="16-character room code"
            maxLength={16}
          />
          <button className="secondary full" disabled={busy || room.length !== 16} onClick={() => join(room)}>
            Join call room
          </button>
        </>
      ) : (
        <>
          <label className="board-label">Share this room code privately</label>
          <textarea className="board-input signal-code" rows="2" readOnly value={activeRoom} />
          <button className="secondary full" onClick={copy}>Copy room code</button>
          <div className="row wrap">
            <button className="secondary" onClick={mute}>
              {muted ? "Unmute microphone" : "Mute microphone"}
            </button>
            <button className="secondary" onClick={stop}>End call</button>
          </div>
        </>
      )}

      <div className="remote-audio">
        <label className="board-label" htmlFor="metered-remote-audio">Other person's audio</label>
        <audio id="metered-remote-audio" ref={audioRef} autoPlay playsInline controls />
        <div className="row wrap">
          <label className="board-label" htmlFor="metered-caption-language">Remote caller language</label>
          <select
            id="metered-caption-language"
            className="board-input"
            value={remoteCaptionLanguage}
            disabled={remoteCaptionOn}
            onChange={event => setRemoteCaptionLanguage(event.target.value)}
          >
            <option value="en-IN">English (India)</option>
            <option value="hi-IN">Hindi (India)</option>
          </select>
          <button
            className="secondary"
            disabled={!remoteTrackRef.current || remoteTrackRef.current.readyState !== "live" || !captionSupported}
            onClick={toggleRemoteCaptions}
          >
            {remoteCaptionOn ? "Stop remote captions" : "Caption other caller"}
          </button>
        </div>
        {(remoteCaption || remoteCaptionInterim) && (
          <div className="captionbox" aria-live="polite">
            {remoteCaption}
            {remoteCaptionInterim && <span className="caption-interim"> {remoteCaptionInterim}</span>}
          </div>
        )}
        <p className="privacy">
          Remote-audio captions require a supported desktop browser (Chrome/Edge 135+, Firefox 160+, or Opera 120+).
          Android and iOS browsers are not enabled because their remote-track input support is unavailable.
          On supported browsers, audio may be sent to the browser's speech service. Start captions only if both
          callers agree. Captions are not a certified transcript.
        </p>
      </div>

      <p className="privacy">
        Room codes are random but should still be shared privately. Calls require microphone permission and live
        internet. Free-tier quotas apply. Android WebView compatibility still requires real-device testing.
      </p>
    </div>
  );
}
