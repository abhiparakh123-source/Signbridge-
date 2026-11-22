# SIGNBRIDGE

Every voice. Every sign. One connection.

An accessibility-first web app/PWA concept for communication between Deaf, hard-of-hearing and hearing people.

## Included
- Responsive dashboard and mobile navigation
- ISL, ASL and BSL language selector
- Camera preview for the sign-to-text workflow
- Browser speech-recognition captions where supported
- Text-to-speech and quick emergency phrases
- Starter learning path with locally saved progress
- PWA manifest and service-worker shell cache

## Run locally
Run npm install, then npm run dev.

## Important capability note
The camera page is currently a preview, NOT a sign-language translator. Reliable sign-language translation needs a validated model trained on appropriately licensed, representative video data, including hand shape, movement, body position, facial expression and sign-language grammar. It must be evaluated with fluent signers and Deaf community partners. Do not use this prototype for critical communication or emergency decisions.

Live captions rely on browser speech recognition and may not be available in every browser or locale. Speech recognition may use a browser/vendor service; review that provider's privacy terms.

## Next milestones
1. Create a dedicated SIGNBRIDGE Supabase project, separate from other apps.
2. Add Supabase Auth, profiles and per-user learning progress with row-level security.
3. Add expert-reviewed ISL learning media and accessibility testing.
4. Integrate a properly licensed sign-language recognition model and avatar pipeline after evaluation.
5. Add automated tests, privacy review and Android/PWA device testing.

No secrets are required for this frontend starter. Never put a Supabase service-role key in browser code.