# SIGNBRIDGE ASL recording consent — discussion template

> **Not legal advice or a ready-to-sign legal instrument.** Have this reviewed
> for the participant's jurisdiction, the recording platform, and the exact
> dataset licence before use. A qualified Deaf ASL signer/reviewer should help
> make the explanation accessible in the participant's preferred language.

## Project and contact

- Project: SIGNBRIDGE, a private family accessibility project.
- Project contact: [name and secure contact route]
- Participant code (pseudonym): [code — do not put a legal name in filenames]
- Session date: [date]
- Consent version: [version]

Explain in plain language that the project is exploring ASL recognition for a
private family communication aid. The current system is experimental and is
not a substitute for a qualified interpreter or emergency service.

## What may be recorded

Describe the actual session before recording. Select only what is genuinely
needed:

- [ ] Video showing face, hands, upper body and signing space.
- [ ] Audio (only if separately explained and necessary).
- [ ] Session metadata such as timestamps, camera frame rate and device class.
- [ ] ASL gloss annotations and separate English meaning/translation labels.
- [ ] Derived pose/hand/face landmarks or other features extracted from video.

Do not collect unrelated personal information. Ask participants to avoid
sharing third-party names, addresses, medical details or confidential content
in example conversations.

## Separate, optional permissions

A participant must be able to decline optional uses without pressure. Record
each choice independently; silence is not consent.

- [ ] I agree to recording for the stated SIGNBRIDGE research session.
- [ ] I agree that the specified recordings may be used to train an ASL model.
- [ ] I agree that the specified recordings may be used for held-out evaluation.
- [ ] I agree that de-identified derived landmarks/features may be retained for
      the stated project period.
- [ ] I agree to a private demonstration to the named project team.
- [ ] I separately agree to any public presentation or publication described
      here: [exact scope]. (Default: not requested and not permitted.)

Do not bundle training, evaluation, public demonstration and publication into
one mandatory permission. Do not assume that permission to record grants
permission to train a model.

## Access, security, retention and withdrawal

Before asking for consent, fill in and explain:

- Storage location and country: [details]
- Who can access raw video/audio: [named roles]
- Encryption and access-control method: [details]
- Transfer method: [approved secure method]
- Retention/deletion date for raw media: [date or defined period]
- Retention/deletion date for derived features and labels: [date or period]
- Backup deletion schedule: [details]
- Withdrawal contact and process: [secure contact]
- What happens to already-trained model weights after withdrawal: [explain
  honestly; do not promise unlearning unless it is technically supported]

No raw participant media, consent forms, identifying information or private
features may be committed to GitHub, placed in public CI artifacts, or bundled
in the public website/APK. Use a private, access-controlled storage location.
Do not upload recordings to third-party AI services unless that specific
transfer is explained and separately authorised.

Explain that withdrawal can stop future use and trigger deletion of retained
source data where feasible and legally required. Explain any limits before
consent, including copies already shared under a valid licence or model weights
that cannot reliably be traced back to one participant.

## Participant rights and safety

Confirm that the participant:
- can ask questions in their preferred accessible language;
- can pause or stop the session at any time;
- can skip any prompt without giving a reason;
- can decline without penalty or loss of compensation already earned;
- knows whether compensation is offered and when it is paid;
- has had time to consider the request without pressure;
- understands that recognition may be inaccurate and is not a replacement for
  human interpretation.

If the participant is under 18, do not use this template; obtain an
appropriate child-assent/guardian-consent process and specialist review first.

## Session checklist for the project operator

Before recording:
1. Verify the participant understands the purpose, exact capture and each
   separate optional permission.
2. Confirm that the selected permissions match the planned dataset manifest.
3. Assign a pseudonymous signer ID; keep any identity-to-code key separately.
4. Confirm the secure destination, access list, retention and deletion process.
5. Explain the camera framing and allow a test recording that can be deleted.
6. Obtain the participant's affirmative choices before starting capture.

After recording:
1. Confirm the participant still agrees to the selected uses.
2. Record session ID, timestamps, language, consent version and permission
   evidence in a restricted audit record.
3. Have a fluent Deaf ASL reviewer check glosses, non-manual cues and meaning.
4. Keep English translation separate from ASL gloss.
5. Assign all clips from one signer to only one dataset split.
6. Verify no names or identifying media appear in repository commits, build
   artifacts, issue attachments or public logs.

## Acknowledgement

Participant pseudonym/code: [code]

Selected permissions: [list exact selections]

Participant acknowledgement (accessible format): [record how affirmative
consent was provided, without publishing identifying information]

Project representative: [name/role]

Date and consent version: [date/version]

A copy of the completed consent record must be provided to the participant in
an accessible format. Store the signed/affirmed record separately from model
training data.
