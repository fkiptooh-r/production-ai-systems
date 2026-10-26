# 05 · Real-Time Voice Agent

**Goal:** a voice agent people can talk to naturally (interrupt it, pause, change their mind) that completes a real task, such as booking a house viewing or checking an order, with end-to-end latency under ~800 ms.
**Time:** 3–4 weeks.

## What you'll learn
- Audio basics: sample rate, PCM, codecs, chunking
- The streaming pipeline: speech-to-text (STT) → LLM → text-to-speech (TTS)
- Voice activity detection (VAD), endpointing and barge-in (interruptions)
- WebRTC and real-time agent frameworks
- Tool calling inside a live conversation
- Latency budgeting and measuring per stage

## Milestones

- [ ] **M1 · Offline pipeline.** Record a WAV → transcribe (Whisper / faster-whisper locally) → LLM → TTS → play the audio. Time every stage. This shows where the latency comes from.
- [ ] **M2 · Streaming.** Stream STT partial transcripts, stream LLM tokens, and start TTS on the first complete sentence. Measure time-to-first-audio.
- [ ] **M3 · Framework.** Rebuild it with LiveKit Agents or Pipecat, talking from the browser over WebRTC.
- [ ] **M4 · Turn-taking.** VAD + endpointing so it doesn't cut people off or wait too long. Barge-in: stop speaking immediately when the user talks.
- [ ] **M5 · Tools.** Function calls such as `book_viewing(listing_id, datetime)` or `order_status(order_id)`. Always read critical details back ("Tuesday 3pm, correct?") before acting.
- [ ] **M6 · Plug in the system.** Answers come from 01 (RAG), model choice from 04 (router), every turn traced by 02 with per-stage latency.
- [ ] **M7 · Evaluate.** 20 scripted test conversations: task success rate, p50/p95 latency, transcripts scored by the 03 judge.
- [ ] **M8 · Phone (optional).** Connect a phone number through a SIP/telephony provider available in your country.

## Latency budget (fill in with your measurements)

| Stage | Target | Measured |
|-------|--------|----------|
| End-of-speech detection | ~200 ms | |
| STT final | ~150 ms | |
| LLM first token | ~250 ms | |
| TTS first audio | ~150 ms | |
| **Total to first audio** | **< 800 ms** | |

## Done when
- A friend can call/talk to it, interrupt it, and still complete a booking
- Latency and task-success numbers are in the README, with a demo video

## Safety and trust
- The agent says it is an AI at the start
- Get consent before recording; don't store audio longer than needed
- Confirm before any action that changes data (bookings, cancellations)

## Where this goes in your products
- **House hunting:** book viewings and answer listing questions by voice or phone
- **E-commerce:** order tracking and support by phone
- **EdTech:** speaking practice, reading tutors, oral quizzes

## Resources
See [docs/RESOURCES.md → Voice Agents](../docs/RESOURCES.md#05--voice-agents).
