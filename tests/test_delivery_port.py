"""Parity tests for the Python port of apps/web/src/lib/delivery.ts (reference repo), mirroring its vitest cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interviewer.ai_text import observe, text_signals  # noqa: E402
from interviewer.delivery import CandTurn, answer_delivery, features  # noqa: E402


def speech(text: str, *, gap=0.1, dur=0.3, pause_at=(), pause_s=0.8):
    toks = text.split(" ")
    t = 0.0
    words = []
    for i, w in enumerate(toks):
        if i in pause_at:
            t += pause_s
        words.append([w.rstrip(".,?!"), round(t, 2), round(t + dur, 2)])
        t += dur + gap
    return {"text": text, "words": words}


READ = ("When a patient is anxious I explain each step before I do it. I keep my voice calm and I make sure they know "
        "they can ask me to pause. Then I choose the best vein, I anchor it, and I complete the draw quickly. Finally I "
        "check on them and thank them for their patience during the procedure today.")
SPONT = ("Um so there was this one guy, he was uh he was really scared of needles and I I kind of um just talked to him "
         "about his his dog and uh and that helped I think, he calmed down, and um I got it on the first try I mean the "
         "second try actually")


def ev(seq, type_, payload):
    return {"seq": seq, "type": type_, "payload": payload}


def test_features_count_fillers_restarts_and_mid_sentence_pauses():
    s = speech("I drew it. Then um I I labelled the the tube", pause_at=(3, 6))
    f = features([CandTurn("t", s["text"], s["words"])])
    assert f.words == 11
    assert abs(f.fillers_per100 - 9.1) < 0.1
    assert abs(f.restarts_per100 - 18.2) < 0.1
    assert abs(f.pauses_per100 - 9.1) < 0.1  # pause before "I" counts; the one after "it." does not


def test_flags_fluent_long_answer_whose_follow_up_stumbles_and_not_a_natural_one():
    fu = speech("um I think uh it was um maybe the uh the second week I uh do not remember exactly", pause_at=(2, 6, 10))
    main = speech(READ, gap=0.05, dur=0.3)
    nat = speech(SPONT, gap=0.15, pause_at=(5, 14, 30))
    events = [
        ev(1, "turn", {"role": "agent", "text": "Tell me about an anxious patient?", "ended_at": 100}),
        ev(2, "turn", {"role": "candidate", "turn_id": "c1", "started_at": 101.2, **main}),
        ev(3, "turn", {"role": "agent", "text": "When was that?", "ended_at": 130}),
        ev(4, "turn", {"role": "candidate", "turn_id": "c2", "started_at": 131, **fu}),
        ev(5, "tool_record_answer", {"question_id": "q_anxiety"}),
        ev(6, "turn", {"role": "agent", "text": "How do you handle a hard stick?", "ended_at": 200}),
        ev(7, "turn", {"role": "candidate", "turn_id": "c3", "started_at": 202.5, **nat}),
        ev(8, "tool_record_answer", {"question_id": "q_hard_stick"}),
    ]
    a, b = answer_delivery(events)
    assert a.question_id == "q_anxiety"
    assert a.onset_ms == 1200
    assert a.follow_up is not None and a.follow_up.words > 8
    assert a.sounds_read is True
    assert any("follow-up much less fluent" in s for s in a.signals)
    assert b.question_id == "q_hard_stick"
    assert b.sounds_read is False
    assert b.turn_ids == ["c3"]


def test_text_signals_separate_generated_from_spontaneous():
    gen = ("First, I review the medication in a reputable drug reference to understand its indications and contraindications. "
           "Second, I verify the order against the patient's allergies and renal function as appropriate. Finally, I "
           "communicate proactively with the pharmacist to ensure patient safety. It is important to maintain a "
           "comprehensive and structured approach in order to deliver optimal care.")
    g = text_signals(gen)
    s = text_signals(SPONT)
    assert g.reads_generated is True
    assert s.reads_generated is False
    assert s.fillers > 0 and g.fillers == 0


def test_observe_requires_both_sides_for_ai_assisted():
    gen = ("First, I review the medication in a reputable drug reference to understand its indications and contraindications. "
           "Second, I verify the order against the patient's allergies and renal function as appropriate. Finally, I "
           "communicate proactively with the pharmacist to ensure patient safety. It is important to maintain a "
           "comprehensive and structured approach in order to deliver optimal care.")
    o = observe("q", gen, None)
    assert o.verdict == "reads_generated"
    assert all(e.startswith("text:") for e in o.evidence)
