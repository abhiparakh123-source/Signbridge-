# SIGNBRIDGE India telephony provider evaluation

**Decision status: Exotel is the leading candidate for an India-first pilot, but account eligibility and commercial terms are not yet confirmed. No provider is connected.**

## Why Exotel is the first candidate

Exotel's current AgentStream Connect Voice AI API documents outbound calls through its India API endpoint and a bidirectional WebSocket stream. The API accepts the destination number, an ExoPhone caller ID, a WSS stream URL and `StreamType=bidirectional`. This is a close technical match for SIGNBRIDGE's app-routed interpreter.

Official documentation:
- Connect Voice AI API: https://developer.exotel.com/docs/agentstream/connect-voice-ai
- Outbound connect API: https://docs.exotel.com/voice-apis/outbound-call-to-connect
- Stream and Voicebot applet background: https://support.exotel.com/support/solutions/articles/3000108630-working-with-the-stream-and-voicebot-applet

The older Stream/Voicebot applet documentation says bidirectional streaming may not be enabled by default for every account. We should use the current AgentStream API documentation as the integration reference, and obtain written confirmation that the feature is enabled for the specific account.

## Alternatives and constraints

### Twilio

Twilio Media Streams supports bidirectional WebSocket audio. However, its India voice guidelines say outbound calls to Indian PSTN destinations can only be made from international (non-Indian) numbers, and its help article says outbound calls originating from +91 caller IDs are blocked. This is a poor fit for a family-first India domestic calling experience unless the user explicitly accepts an international caller ID and the route is confirmed for their use case.

References:
- https://www.twilio.com/docs/voice/media-streams
- https://static1.twilio.com/en-us/guidelines/in/voice
- https://help.twilio.com/articles/57566259695003

### Plivo

Plivo documents bidirectional audio streaming and outbound calls. Its India domestic voice routes and Indian numbers require business registration and KYC documentation; current requirements include business registration documents and GST documentation. This may not suit a personal family project without an eligible registered business.

References:
- https://www.plivo.com/docs/voice-agents/audio-streaming/overview
- https://www.plivo.com/docs/voice/concepts/india-calling
- https://www.plivo.com/docs/numbers/compliance

## Account checks required before implementation

Before selecting Exotel finally, the account holder must confirm directly with Exotel:
1. Can an individual/family account (not a registered business) use outbound transactional calls to ordinary Indian mobile and landline numbers?
2. Can the account obtain an ExoPhone/caller ID and enable AgentStream bidirectional streaming?
3. What KYC documents and approval steps apply?
4. What are the setup, number rental, per-minute, streaming and tax charges? Is there a trial or low-cost test balance?
5. Are there restrictions on personal accessibility calls, caller ID display, call hours, destination types or call volume?
6. What consent, privacy, recording and telecom compliance steps must SIGNBRIDGE implement?

Do not assume a free tier, personal-account eligibility, a local caller ID, or that all Indian destinations are reachable. Do not purchase or configure anything until these checks are answered.

## Cross-platform client design

The phone call must be independent of the handset OS:
- Android and iPhone clients both authenticate to the same SIGNBRIDGE backend.
- The backend alone stores provider credentials and starts/stops provider calls.
- The called plumber/service provider answers on an ordinary phone; no SIGNBRIDGE installation is required.
- The app sends reviewed text to the backend; the speech pipeline returns translated audio into the provider call.
- The backend streams the other party's speech to streaming Hindi/English recognition; corrected captions and translations return to the app.
- A mobile browser/PWA can be the first iPhone client. Native iOS packaging is a later option, not a prerequisite for provider-routed PSTN calls.

iPhone Safari and installed PWAs must be tested for camera/microphone permission, audio playback, foreground/background transitions, interruptions and reconnect behaviour. Do not promise continuous background operation until verified on real devices.

## Security and release gates

- Provider API credentials stay in backend secret storage; never in the app, APK, repository or logs.
- Require an authenticated user, rate limits, destination allow-list/confirmation, call-duration cap and explicit cost confirmation.
- Verify provider callback authenticity using the provider's supported signature/authentication mechanism.
- Require both-party consent before AI processing; recording remains off by default.
- Provide persistent mute/pause/end controls and a clear fallback to manual text.
- Do not auto-confirm prices, bookings, addresses, payments or work authorisations.
- Do not enable live calls until end-to-end tests pass on Android and iPhone with consenting participants.

**Current status remains NOT CONNECTED.** This evaluation is a provider-selection decision aid, not a working call integration.
