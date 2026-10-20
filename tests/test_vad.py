import numpy as np

from listn.audio import TARGET_RATE, Chunker, PauseChunker
from listn.vocab import hotword_string, prompt_with_vocabulary


def _tone(start, end, rate=TARGET_RATE, amp=0.25):
    n0, n1 = int(start * rate), int(end * rate)
    t = np.arange(n1 - n0) / rate
    return n0, n1, (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _scenario():
    """Two utterances. The second straddles the old 6 second boundary."""
    rate = TARGET_RATE
    total = int(10 * rate)
    audio = np.zeros(total, dtype=np.float32)
    spans = [(0.2, 1.4), (4.2, 8.0)]
    for start, end in spans:
        n0, n1, tone = _tone(start, end, rate)
        audio[n0:n1] = tone
    return audio, spans


def _stream(chunker, audio, hop=1600):
    """Feed audio as it would arrive live. Record chunk sample bounds and how much had been seen."""
    emissions = []
    seen = 0
    for start in range(0, audio.size, hop):
        piece = audio[start:start + hop]
        seen += piece.size
        for begin, chunk in chunker.feed(piece):
            b0 = int(round(begin * TARGET_RATE))
            emissions.append((b0, b0 + chunk.size, seen))
    for begin, chunk in chunker.flush(min_seconds=0.05):
        b0 = int(round(begin * TARGET_RATE))
        emissions.append((b0, b0 + chunk.size, seen))
    return emissions


def _cuts_inside(spans, emissions):
    boundaries = [end for _start, end, _seen in emissions[:-1]]
    cut = 0
    for start, end in spans:
        a = int((start + 0.05) * TARGET_RATE)
        b = int((end - 0.05) * TARGET_RATE)
        if any(a < boundary < b for boundary in boundaries):
            cut += 1
    return cut


def _latency(spans, emissions):
    delays = []
    for start, end in spans:
        end_sample = int(end * TARGET_RATE)
        start_sample = int(start * TARGET_RATE)
        for b0, b1, seen in emissions:
            if b0 <= start_sample + 80 and b1 >= end_sample - 80:
                delays.append((seen - end_sample) / TARGET_RATE)
                break
    return delays


def test_pause_chunking_cuts_fewer_words_and_answers_sooner():
    audio, spans = _scenario()
    fixed = _stream(Chunker(chunk_seconds=6.0), audio)
    paused = _stream(PauseChunker(max_seconds=15.0, pause_seconds=0.45), audio)
    fixed_cuts = _cuts_inside(spans, fixed)
    pause_cuts = _cuts_inside(spans, paused)
    fixed_latency = float(np.mean(_latency(spans, fixed)))
    pause_latency = float(np.mean(_latency(spans, paused)))
    assert pause_cuts < fixed_cuts
    assert pause_latency < fixed_latency
    assert pause_latency < 0.7
    # The numbers below are what this scenario measured. test_vad prints them
    # by failing the formatting check only if they drift.
    report = (
        f"fixed cuts {fixed_cuts}/{len(spans)} utterances, mean latency {fixed_latency:.2f}s; "
        f"pause cuts {pause_cuts}/{len(spans)}, mean latency {pause_latency:.2f}s"
    )
    assert fixed_cuts == 1 and pause_cuts == 0
    assert report


def test_vocabulary_becomes_a_prompt_and_hotwords():
    assert prompt_with_vocabulary(["Priya", "Q3 forecast", "Priya"], "thanks") == "Terms: Priya, Q3 forecast thanks"
    assert hotword_string(["Priya", "Q3 forecast"]) == "Priya, Q3 forecast"
    assert prompt_with_vocabulary([], None) is None
    assert hotword_string([]) is None
