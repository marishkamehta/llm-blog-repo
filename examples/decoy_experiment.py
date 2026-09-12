import csv
import time
from collections import Counter
from pathlib import Path

import requests
from silico.registry import make_llm


llm = make_llm("gemini-flash")

COMMON = (
    "You are booking a hotel for one night. The hotels differ only in guest "
    "rating and price. Choose one hotel. Reply with only its letter.\n\n"
    "A: Guest rating 8.5 out of 10; price $220.\n"
    "B: Guest rating 7.5 out of 10; price $130."
)

PROMPTS = {
    "control": COMMON,
    "decoy": COMMON + "\nC: Guest rating 8.3 out of 10; price $235.",
}
SCHEDULE = ["control", "decoy"] * 5
RESULTS = Path("decoy_results.csv")

# Resume after the last complete row if an earlier run was interrupted.
completed = 0
if RESULTS.exists():
    with RESULTS.open(newline="", encoding="utf-8") as existing_file:
        completed = sum(1 for _ in csv.DictReader(existing_file))

mode = "a" if completed else "w"
with RESULTS.open(mode, newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(
        output,
        fieldnames=["observation", "condition", "prompt", "raw_response", "choice"],
    )
    if not completed:
        writer.writeheader()

    for observation, condition in list(enumerate(SCHEDULE, start=1))[completed:]:
        for attempt in range(1, 4):
            try:
                raw_response = llm.respond(PROMPTS[condition])
                break
            except (requests.ReadTimeout, requests.HTTPError) as error:
                retryable = isinstance(error, requests.ReadTimeout) or (
                    error.response.status_code == 503
                )
                if not retryable or attempt == 3:
                    raise
                wait_seconds = 15 * attempt
                print(f"Temporary error; retrying in {wait_seconds} seconds.")
                time.sleep(wait_seconds)

        normalized = raw_response.strip().upper()
        choice = normalized if normalized in {"A", "B", "C"} else "INVALID"
        writer.writerow(
            {
                "observation": observation,
                "condition": condition,
                "prompt": PROMPTS[condition],
                "raw_response": raw_response,
                "choice": choice,
            }
        )
        output.flush()
        print(condition, choice)

with RESULTS.open(newline="", encoding="utf-8") as results_file:
    rows = list(csv.DictReader(results_file))

print("\nSummary")
for condition in PROMPTS:
    counts = Counter(row["choice"] for row in rows if row["condition"] == condition)
    print(condition, dict(counts))
