# Canon: a continuity checker for AI microdrama scripts

AI microdramas are generated from their scripts: a voice says every word as written, voices are cast per speaker label, and video is drawn from each heading and description. A slip in the script therefore becomes an audible or visible error.

Canon finds those slips before generation:

- one person, family, house or place spelled several ways
- one character speaking under several speaker labels (or one label for two people)
- a day/night or building change across a cut the script calls continuous
- a misspelled word inside a spoken line
- inconsistent header spelling

Code finds every candidate and every location. A model only judges what code found, from snippets code picked. Code vetoes impossible merges. A person approves each fix before anything is patched.

Live app: https://canon-krishnachaitkc.streamlit.app · Prototype: this repository, with the run steps below. The solution doc is a 2-page PDF in the submission.

## Run

Windows:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

macOS or Linux: use `.venv/bin/pip` and `cp` instead.

Add a key to `.env` only for the model you use. `GEMINI_API_KEY` is free; `DEEPSEEK_API_KEY` is paid but cheap. Code-only mode needs no key and sends nothing.

```
python -m canon doctor --input samples/ravenscar
python -m canon run --input samples/ravenscar --out out/s --no-model
python -m canon run --input samples/ravenscar --out out/s --model gemini/gemini-3.5-flash-lite
python -m canon run --input show.zip --out out/x --model openai/<model> --base-url <url> --api-key-env MY_KEY
python -m canon patch --run out/s --approvals approvals.json
streamlit run app.py
python -m pytest tests -q
```

- Run `doctor` first on any new show: it reports parse coverage and the format report. The parser expects the script format of the input pack (scene heading, scene start/end, `**SPEAKER (tone):**`, `Start: | End:`).
- A run writes `findings.csv`, `cast.csv`, `summary.md`, `report.html` and `manifest.json` (calls, tokens, seconds, cost).
- `patch` needs `approvals.json` from the app or written by hand, e.g. `{"F0003": true}`. Without it, it stops with an error.
- `--model` swaps the judge; `--base-url` and `--api-key-env` point at any OpenAI-compatible endpoint.

Kuku's script pack is not in this repository. `samples/ravenscar` is an original four-episode show written for testing.
