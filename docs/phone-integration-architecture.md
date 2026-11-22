# SIGNBRIDGE phone integration: design and release gates

**Status: NOT CONNECTED.** The current Call Assist screen opens the phone dialler with a `tel:` link, prepares a message, and can play a local text-to-speech preview. It does not receive either side of a cellular call, stream call audio, transcribe the other speaker, translate speech, or inject translated speech into a call.

## Why the normal SIM call cannot simply be captured

Android's public audio APIs do not grant ordinary third-party apps access to both cellular-call audio tracks. The `VOICE_CALL`, `VOICE_UPLINK`, and `VOICE_DOWNLINK` audio sources require `CAPTURE_AUDIO_OUTPUT`, a permission reserved for system components. Android's audio-input sharing rules also state that an active voice call receives the microphone input; an ordinary app cannot simply take the call's audio. See the Android references:
- https://developer.android.com/reference/android/media/MediaRecorder.AudioSource
- https://developer.android.com/media/platform/sharing-audio-input

We must not add a fake "record call" button or claim that a normal `tel:` call is being interpreted.

## Two implementation routes

### Route A — app-routed phone call (the route for a genuinely integrated experience)

Use a telephony provider that can originate/receive PSTN calls and stream the live call media to a secure backend. The backend must:
1. Verify provider webhook signatures and authenticate the SIGNBRIDGE user.
2. Place a call only after the user explicitly confirms the destination and expected charges.
3. Open a bidirectional media stream for both directions of the conversation.
4. Transcribe the provider's audio with a Hindi/English speech recogniser.
5. Display interim and corrected transcripts with language and uncertainty labels.
6. Let the Deaf caller type a reply; translate it only after user review, then synthesise it and return audio to the call.
7. Keep an auditable, user-controlled stop/end-call action and avoid retaining raw audio by default.

Twilio Media Streams is one possible technical route: its bidirectional stream can receive call audio and send audio back into the call over a secure WebSocket. It requires a Programmable Voice account, a public WSS backend, signature validation and a separate STT/translation/TTS pipeline. Provider availability, destination-country restrictions, phone-number eligibility and per-minute charges must be checked before choosing it. See https://www.twilio.com/docs/voice/media-streams and https://www.twilio.com/docs/voice/media-streams/websocket-messages.

This route requires a telephony provider account and credentials. Credentials must be stored as backend secrets, never in the React app, APK, repository or browser.

### Route B — keep the existing SIM call

Keep the phone's native dialler for ordinary calls. SIGNBRIDGE can still prepare text and speech before the call, and can offer a separate caption workflow where the device/browser actually supports it. This is not a guaranteed live two-way call interpreter: Android may give the active call exclusive microphone input, and an ordinary app cannot access the call's downlink. A second device for captions is a practical fallback, but is not integrated call audio.

## Audio, language and safety requirements

- Ask the user to confirm the service provider's number before every call.
- Announce that AI translation is being used and obtain consent from the other participant before streaming/transcribing the call.
- Default to spoken Hindi or English chosen by the user; do not infer language silently.
- Keep original transcript, translated text and speech playback visibly separate.
- Allow immediate correction, repeat, pause and end-call actions.
- Never auto-agree to prices, appointment dates, addresses, payment or work authorisation.
- Require explicit confirmation before speaking a translated response.
- Do not retain call audio or transcripts by default; disclose any temporary processing and retention.
- Keep manual text/SMS and a human interpreter fallback available.

## Dependencies before a live PSTN bridge can be enabled

- User-selected telephony provider and an account that is permitted to call the intended destination country.
- Provider phone number / caller ID and funded account (provider usage may cost money).
- Public HTTPS + WSS backend endpoint and production secret storage.
- Provider webhook signature validation and user authentication.
- Hindi and English streaming speech-to-text, reviewed translation, and text-to-speech services.
- End-to-end tests with both sides consenting, including interruptions, accents, numbers, names, poor network and call termination.
- Privacy notice and explicit in-call consent.

Until these dependencies are configured and tested, the phone integration status must remain `not_connected`.
