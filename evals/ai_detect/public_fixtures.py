"""Public-data fixtures: real spontaneous human speech from the AMI Meeting Corpus, plus matched AI-generated pairs.

    uv run python evals/ai_detect/public_fixtures.py --passages 20            # fetch AMI + generate pairs
    uv run python evals/ai_detect/public_fixtures.py --passages 20 --no-llm   # AMI only, no credits spent

Why AMI: verbatim transcripts of real spontaneous speech with the fillers kept, CC BY 4.0, no login
(https://huggingface.co/datasets/edinburghcstr/ami). It is meeting talk, not job interviews, and the text is uppercase
without punctuation, so this script joins one speaker's consecutive utterances into answer-length passages, lowercases
them and marks each utterance boundary as a sentence end. That is stated in every fixture's `source`.

Why matched pairs: for each human passage, Gemini (through LiveKit Inference) writes (a) a polished written answer
covering the same content at the same length (tier ai_verbatim) and (b) the same content re-spoken casually with a few
fillers (tier ai_paraphrased). Same topic, same length, so the detector is judged on style, not subject matter.

Output: fixtures/public/ami_human.json and fixtures/public/ami_ai_pairs.json, picked up by run.py automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import ssl
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "fixtures" / "public"
ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=edinburghcstr/ami&config=ihm&split={split}&offset={off}&length=100"
FILLER_RE = re.compile(r"\b(um|uh|hmm|mm|erm|uhm)\b", re.I)


def _ssl_context() -> ssl.SSLContext:
    """python.org builds ship without the system CA bundle; use certifi's (installed with livekit-agents)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_rows(split: str, pages: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(pages):
        url = ROWS_URL.format(split=split, off=i * 100)
        try:
            with urllib.request.urlopen(url, timeout=30, context=_ssl_context()) as r:
                data = json.load(r)
        except Exception as e:  # network or API hiccup: keep what we have
            print(f"fetch stopped at page {i}: {e}", file=sys.stderr)
            break
        rows.extend(x["row"] for x in data.get("rows", []))
    return rows


def passages(rows: list[dict], min_words: int, max_words: int, max_gap_s: float = 4.0) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["meeting_id"], r["begin_time"]))
    out: list[dict] = []
    cur: dict | None = None
    for r in rows:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        key = (r["meeting_id"], r["speaker_id"])
        if cur and cur["key"] == key and r["begin_time"] - cur["end"] < max_gap_s:
            cur["utts"].append(txt)
            cur["end"] = r["end_time"]
        else:
            if cur:
                out.append(cur)
            cur = {"key": key, "utts": [txt], "start": r["begin_time"], "end": r["end_time"]}
    if cur:
        out.append(cur)

    def n_words(p: dict) -> int:
        return sum(len(u.split()) for u in p["utts"])

    good = []
    for p in out:
        n = n_words(p)
        if min_words <= n <= max_words:
            text = ". ".join(u.lower().strip() for u in p["utts"]) + "."
            text = re.sub(r"\s+", " ", text)
            text = text[0].upper() + text[1:]
            good.append({"meeting_id": p["key"][0], "speaker_id": p["key"][1], "start_s": round(p["start"], 1),
                         "end_s": round(p["end"], 1), "words": n, "text": text,
                         "has_fillers": bool(FILLER_RE.search(text))})
    return good


def human_fixtures(ps: list[dict]) -> dict:
    answers = []
    for i, p in enumerate(ps, 1):
        answers.append({
            "id": f"ami_h_{i:02d}", "tier": "human_spontaneous", "question_id": "q_ami",
            "text": p["text"], "follow_up_text": "",
            "source": (f"AMI Meeting Corpus (edinburghcstr/ami, ihm, CC BY 4.0), meeting {p['meeting_id']} speaker "
                       f"{p['speaker_id']} {p['start_s']}-{p['end_s']}s; consecutive utterances joined, lowercased, "
                       "utterance boundaries marked as sentence ends"),
        })
    return {"_about": "Real spontaneous human speech from the AMI Meeting Corpus. Meeting talk, not interviews.", "answers": answers}


VERBATIM_PROMPT = (
    "Below is a verbatim transcript of a person speaking spontaneously in a meeting. Write what an AI assistant would "
    "produce if someone asked it to turn this into a polished spoken answer to an interviewer, covering the same "
    "content and points, about {n} words, as one paragraph of complete sentences. Output only the answer text.\n\n"
    "TRANSCRIPT:\n{text}"
)
PARAPHRASE_PROMPT = (
    "Rewrite the following polished answer as the same person would actually say it out loud from memory on a phone "
    "call: casual spoken English, contractions, two or three natural fillers such as um or uh, one self-correction, "
    "same content and about the same length ({n} words). Output only the spoken text.\n\nANSWER:\n{text}"
)


async def generate_pairs(ps: list[dict], model: str) -> dict:
    from dotenv import load_dotenv
    from livekit.agents import inference
    from livekit.agents.llm import ChatContext

    load_dotenv(".env.local")
    llm = inference.LLM(model=model)

    async def ask(prompt: str) -> str:
        ctx = ChatContext()
        ctx.add_message(role="user", content=prompt)
        res = await llm.chat(chat_ctx=ctx).collect()
        return (res.text or "").strip()

    answers = []
    for i, p in enumerate(ps, 1):
        n = p["words"]
        verbatim = await ask(VERBATIM_PROMPT.format(n=n, text=p["text"]))
        para = await ask(PARAPHRASE_PROMPT.format(n=n, text=verbatim)) if verbatim else ""
        src = f"generated by {model} via LiveKit Inference from AMI passage {p['meeting_id']}/{p['speaker_id']} {p['start_s']}s"
        if verbatim:
            answers.append({"id": f"ami_v_{i:02d}", "tier": "ai_verbatim", "question_id": "q_ami", "text": verbatim,
                            "follow_up_text": "", "pair_of": f"ami_h_{i:02d}", "source": src + " (polished written answer)"})
        if para:
            answers.append({"id": f"ami_p_{i:02d}", "tier": "ai_paraphrased", "question_id": "q_ami", "text": para,
                            "follow_up_text": "", "pair_of": f"ami_h_{i:02d}", "source": src + " (spoken-style rewrite)"})
        print(f"  pair {i}/{len(ps)}: {len(verbatim.split())} / {len(para.split())} words", file=sys.stderr)
    await llm.aclose()
    return {"_about": "AI-generated matched pairs for the AMI human passages: same content and length, generated style.",
            "answers": answers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=int, default=20)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--pages", type=int, default=15, help="rows pages of 100 to fetch")
    ap.add_argument("--min-words", type=int, default=45)
    ap.add_argument("--max-words", type=int, default=140)
    ap.add_argument("--model", default="google/gemini-3.5-flash-lite")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    rows = fetch_rows(args.split, args.pages)
    ps = passages(rows, args.min_words, args.max_words)
    # prefer passages with fillers (the point of real speech), then spread across meetings
    ps.sort(key=lambda p: (not p["has_fillers"], p["meeting_id"], p["start_s"]))
    seen: dict[str, int] = {}
    picked = []
    for p in ps:
        if seen.get(p["meeting_id"], 0) >= 4:
            continue
        seen[p["meeting_id"]] = seen.get(p["meeting_id"], 0) + 1
        picked.append(p)
        if len(picked) >= args.passages:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ami_human.json").write_text(json.dumps(human_fixtures(picked), indent=1))
    print(f"{len(rows)} rows → {len(ps)} passages of {args.min_words}-{args.max_words} words → {len(picked)} picked "
          f"({sum(p['has_fillers'] for p in picked)} with fillers) → {OUT / 'ami_human.json'}")
    if args.no_llm:
        return
    pairs = asyncio.run(generate_pairs(picked, args.model))
    (OUT / "ami_ai_pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"{len(pairs['answers'])} generated answers → {OUT / 'ami_ai_pairs.json'}")


if __name__ == "__main__":
    main()
