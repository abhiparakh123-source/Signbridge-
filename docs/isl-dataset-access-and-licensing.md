# ISL dataset access, licensing and training plan

This project is a personal family accessibility project. "Non-commercial" does not automatically mean every dataset may be downloaded, copied, trained on or redistributed. Follow each source's exact access terms and retain attribution.

## 1. FDMSE-ISL: candidate for an isolated-sign baseline

The FDMSE-ISL project at Ramakrishna Mission Vivekananda Educational and Research Institute describes 2,002 ISL words, 20 native Deaf signers, 40,033 videos per view, three synchronized viewpoints and 60 fps RGB recordings. Access is reviewed and requires a request. Its published agreement says academic and non-commercial research only; no commercial use, redistribution, public hosting, sublicensing or transfer; credentials and download links must not be shared.

- Project: https://cs.rkmvu.ac.in/~isl/
- Dataset details: https://cs.rkmvu.ac.in/~isl/dataset.html
- Request access: https://cs.rkmvu.ac.in/~isl/isl-dataset/

**Important limitation:** the published dataset describes isolated word recognition. It is useful for a vocabulary/isolated-sign baseline and possibly landmark pretraining, but it does not by itself solve continuous, natural-speed sentence translation, turn-taking, coarticulation or grammar. Obtain written approval that the intended family accessibility prototype fits the permitted research use before downloading or training.

## 2. ISLRTC Dictionary: useful reference with explicit conditions

ISLRTC's FAQ says its dictionary videos may be used for research, teaching and development of technology provided they are not resold or used for profiteering and proper acknowledgement is given to ISLRTC and the Department of Empowerment of Persons with Disabilities, Ministry of Social Justice and Empowerment, Government of India.

- Official FAQ: https://islrtc.nic.in/faq/
- Official dictionary: https://divyangjan.depwd.gov.in/islrtc/
- Government Open Data catalogue: https://www.data.gov.in/catalog/indian-sign-language-dictionary

The dictionary contains word-level demonstrations and useful synonyms/context distinctions. Treat it as a vocabulary/reference source, not a sentence-level continuous corpus. Keep attribution in app credits and model documentation. Check the current source terms and access method before bulk downloading or preprocessing.

## 3. Continuous signing and fingerspelling resources

The 2026 ISH-NEWS research paper reports 4,222 sentence videos and more than 6.5K words, but its article states the source broadcast videos remain the intellectual property of their respective owners and are not redistributed as raw video. Do not scrape or download broadcast footage for SIGNBRIDGE training based only on the paper.

- Paper and data-rights statement: https://www.nature.com/articles/s41598-026-60893-0

A 2025 paper describes 1,308 continuous ISL fingerspelling segments from 499 ISH News videos. Its benchmark is a specialised fingerspelling resource, not general conversation translation. Confirm data access and reuse rights with the authors before use.

- Paper: https://aclanthology.org/2025.wslp-main.6/

## 4. Family recordings: the most relevant adaptation data

The Family Recorder is designed for voluntary, informed recordings from family signers. Keep clips private and local. Before any model training:
- Obtain informed consent from each signer for recording, annotation, model training and the intended private family use as separate purposes.
- Let the signer withdraw clips; maintain a private clip-to-manifest mapping.
- Record the exact sign language and regional variation; never mix ISL, ASL and BSL labels.
- Have a fluent Deaf signer or qualified ISL expert review every gloss/meaning.
- Capture complete phrases and natural transitions, not only posed isolated signs.
- Record both hands, upper-body pose, face/non-manual cues and enough context before/after each phrase.
- Split train/validation/test by signer, never by randomly splitting clips from the same signer across partitions.
- Keep raw videos and identity keys outside GitHub and public hosting.

A tiny family dataset is suitable for a narrowly scoped experiment, not a claim of general-purpose accuracy. Test on signers and phrases not used in training.

## 5. Recommended staged training strategy

1. Obtain permission for FDMSE-ISL and/or permitted ISLRTC dictionary use. Use only approved files and preserve source attribution.
2. Establish an isolated-sign baseline with a signer-independent evaluation. Report per-class accuracy, confusion matrix and signer-level results.
3. Collect consented family phrases with expert-reviewed sentence/gloss annotations and time boundaries.
4. Build a separate continuous ISL dataset and temporal model. Evaluate phrase-level WER/CER, sign/gloss error rate, omissions, insertions, latency and abstention on held-out signers.
5. Add a Hindi/English translation layer only after the ISL recognition output is evaluated; translation quality is a separate metric.
6. Keep ASL and BSL datasets, gloss conventions, models and evaluations separate. A language selector does not make an ISL model understand ASL or BSL.

## Release gate

No trained model, accuracy claim or live translation is enabled merely because a dataset is accessible. Release requires documented rights, consent, expert label review, signer-independent evaluation, uncertainty handling, device latency testing and a manual text fallback.
