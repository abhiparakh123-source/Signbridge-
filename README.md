# SIGNBRIDGE

Every voice. Every sign. One connection.

An accessibility-first web app/PWA concept for communication between Deaf, hard-of-hearing and hearing people.

## Included
- Responsive dashboard and mobile navigation
- ISL, ASL and BSL language selector
- Camera preview for the sign-to-text workflow, with fast-signing mode requesting up to 60 fps where supported and a 30 fps/lower-resolution fallback
- Browser speech-recognition captions where supported
- Text-to-speech and expanded quick-help phrases
- Conversation Board with editable messages, large-text display, copy and speech controls
- Explicit confirmation prompts to reduce misunderstandings in important exchanges
- Starter learning path with locally saved progress
- PWA manifest and service-worker shell cache

## ASL-first research pilot

The immediate model-development priority is now ASL, using rights-cleared continuous-signing research data where permitted. See the [ASL-first pilot plan](docs/asl-first-pilot.md). ISL remains a separate target; ASL and ISL are different languages, and an ASL model cannot interpret someone signing ISL. No ASL model is currently trained or connected.

## Real-time ISL translation plan

The engineering research, candidate datasets, architecture, licensing checks and release gates for a real continuous ISL translator are documented in [docs/isl-realtime-translation-plan.md](docs/isl-realtime-translation-plan.md). This is a plan, not an implemented model: the current camera remains a preview until a real streaming model is integrated and evaluated with fluent Deaf signers.

## ISL recognition service status

A separate FastAPI scaffold is in `services/isl-stream/`. It intentionally reports `model_not_configured` and does not generate translations. It is not deployed or connected to the app. See its [service README](services/isl-stream/README.md) and the [real-time ISL engineering plan](docs/isl-realtime-translation-plan.md).

## Run locally
Run npm install, then npm run dev.

## Important capability note
The camera page is currently a preview, NOT a sign-language translator. Fast-signing mode requests up to 60 frames per second from supported cameras and falls back when unavailable; the displayed frame rate is the device-reported capture rate, not a translation or recognition speed guarantee. Reliable sign-language translation needs a validated model trained on appropriately licensed, representative video data, including hand shape, movement, body position, facial expression and sign-language grammar. It must be evaluated with fluent signers and Deaf community partners. Do not use this prototype for critical communication or emergency decisions. The Conversation Board shares only the message selected or typed by the user; it does not verify that another person understood it. Confirm important details directly, especially names, numbers, dates, medication and directions.

Live captions rely on browser speech recognition and may not be available in every browser or locale. Speech recognition may use a browser/vendor service; review that provider's privacy terms.

## Next milestones
1. Create a dedicated SIGNBRIDGE Supabase project, separate from other apps.
2. Add Supabase Auth, profiles and per-user learning progress with row-level security.
3. Add expert-reviewed ISL learning media and accessibility testing.
4. Integrate a properly licensed sign-language recognition model and avatar pipeline after evaluation.
5. Add automated tests, privacy review and Android/PWA device testing.

No secrets are required for this frontend starter. Never put a Supabase service-role key in browser code.

## Reliability and verification status

The hosted build is checked by Render's production build step (Vite). This confirms that the source compiles and the static bundle is generated; it does not prove camera, microphone, clipboard, speech output or every mobile browser behaves identically. Those require device/browser permission tests.

### Current feature boundaries

| Feature | What it does now | Important limitation |
|---|---|---|
| Sign language selector | Selects ISL, ASL or BSL in the interface | Does not change or enable a recognition model |
| Sign-to-text | Opens a local camera preview; fast mode requests up to 60 fps where supported and falls back to a lower-bandwidth mode | No sign recognition or translation model is connected; camera fps is not translation accuracy or latency |
| Live captions | Uses browser speech recognition for spoken English (India) | Browser, network, microphone permission and locale affect availability and accuracy |
| Text-to-speech | Uses the device/browser speech engine | Voice and pronunciation depend on installed voices and device settings |
| Quick Help | Copies or speaks fixed messages | User must verify the selected message before showing or sending |
| Conversation Board | Selects or edits text, shows large text, copies and speaks it | It cannot confirm that another person understood the message |
| Call Assist | Opens the native phone dialler, prepares a message and previews device speech | No cellular call audio access, live transcription, translation or two-way audio bridge; provider integration is not connected |
| Learning progress | Stores completed starter cards in local browser storage | Device-local only; clearing browser data or changing devices may erase progress |
| Animated design | CSS motion and hover/focus effects | Respects the operating system's reduced-motion preference |

### Communication safety

Do not rely on SIGNBRIDGE as the sole communication method for emergencies, medical decisions, consent, legal matters, navigation or other high-stakes exchanges. Repeat and confirm critical information, especially names, numbers, dates, medication, addresses and directions. Sign-language translation must not be presented as accurate until a validated model is tested with fluent signers and Deaf community reviewers.

### Manual device test checklist

- Open on Android Chrome and desktop Chrome.
- Navigate through every menu item and use browser back/refresh.
- Grant and deny camera permission; verify preview starts and camera indicator stops when leaving the page.
- Grant and deny microphone permission; verify captions start, stop, report errors and clear correctly.
- Test copy with clipboard permission available and unavailable.
- Test speech output with device volume on and off.
- Select every Quick Help message and Conversation Board option; test custom text and large-text mode.
- Complete and uncomplete lessons; refresh and verify local progress.
- Enable reduced-motion in the operating system and verify animations stop.
- Test offline/reload behaviour and PWA installation separately.

Automated production build success is not a substitute for this manual test matrix. Do not mark a feature fully verified until it has passed on the target device/browser.


## Call Assist (preview)

Call Assist helps a Deaf or hard-of-hearing user prepare a service call: enter a contact number, open the device phone app, draft a message, preview it using device text-to-speech, and prepare written replies and appointment-confirmation details.

**This is not a live call relay.** The current web app cannot route the phone call's audio into speech recognition, cannot automatically transcribe or translate the other party, and cannot stream AI speech into a PSTN call. The phone app handles the actual call. Voice preview is played locally by the browser/device and is not injected into the call audio. Users must not rely on it as a real-time interpreter.

### Requirements for a production two-way call bridge

A future release needs a telephony provider with an India-supported PSTN route and compliant caller ID, a secure backend that holds provider credentials, real-time streaming audio, speech recognition for Hindi and English, translation that preserves names/numbers/amounts, natural speech synthesis, interruption/turn-taking handling, explicit consent and call recording controls, and auditable confirmations. No provider keys may be placed in browser code. Every uncertain transcript must be visibly marked and repeated back for confirmation; high-impact details should be confirmed by SMS/text before any booking or payment. A human interpreter or text fallback must remain available.

Do not market Call Assist as fully automatic or perfectly accurate until it passes independent testing with Deaf users, Hindi/English speakers, service providers and noisy real-world calls.


### Fast-signing requirement

Fast signing is normal fluent signing, not an edge case. The camera preview includes a fast capture request (up to 60 fps where supported), a standard 30 fps option, a lower-bandwidth fallback, and a device-reported capture setting. These controls only affect camera capture. They do not translate signs.

Before any sign-recognition release, benchmark a continuous temporal model with real fluent signers at natural and fast speeds. Measure end-to-end latency (capture to stable text), dropped frames, missed short signs, false insertions, word/sign error rates and correction rate. Evaluate complete phrases and conversation turns, including facial grammar and body movement, not only isolated hand shapes. Never publish a speed or accuracy claim without test results from the target Android devices and Deaf community reviewers.

## Phone integration status

Call Assist currently opens the phone's native dialler and prepares a message. It does **not** listen to or translate a live phone call. Android restricts cellular call audio capture for ordinary apps. A true two-way interpreter needs an app-routed PSTN provider with bidirectional audio streaming, plus configured speech recognition, translation and speech synthesis services. No provider account or credentials are configured. See [the phone integration architecture and prerequisites](docs/phone-integration-architecture.md).
\n\n## Cross-platform phone integration research\n\nThe planned provider-routed call architecture is designed for both Android and iPhone. The person receiving a call can answer on an ordinary phone and does not need SIGNBRIDGE. Exotel AgentStream is the leading India-first provider candidate because its API documents bidirectional WebSocket voice streaming; individual/family account eligibility, KYC, enablement and pricing remain unconfirmed. No live call bridge is connected. See [India telephony provider evaluation](docs/india-telephony-provider-evaluation.md) and [phone integration architecture](docs/phone-integration-architecture.md).\n\n\n## ISL data rights and training plan\n\nBefore training, review [the ISL dataset access and licensing plan](docs/isl-dataset-access-and-licensing.md). FDMSE-ISL is a reviewed-access isolated-sign research dataset, not a continuous conversation corpus. ISLRTC dictionary use has explicit no-resale/profiteering and attribution conditions. Do not scrape copyrighted broadcast video or upload family recordings to GitHub.\n

## Fast continuous signing engineering plan

The current CTC Transformer is an offline research baseline, not a live translator. See [the low-latency continuous signing architecture](docs/low-latency-continuous-signing.md) for the intended bounded-queue, incremental inference, uncertainty and device-benchmark design. Live translation remains disabled until a real model is trained and independently evaluated.
