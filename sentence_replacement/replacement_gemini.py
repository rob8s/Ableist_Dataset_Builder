import requests
import json
import pandas as pd
import time
import os
from google import genai
from google.genai import types


with open("flagged_sentences.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
sentences = df["sentence"].tolist()
phrases = df["detected_phrase"].tolist()


MAX_RETRIES = 3
RETRY_DELAY = 5

results = []
existing_ids = set()
output_path = "rewritten_sentences.json"

if os.path.exists(output_path):
    with open(output_path, "r") as out_file:
        results = json.load(out_file)
        existing_ids = set(item["original_sentence"] for item in results)


# For Gemini
failed_sentences = []
client = genai.Client(api_key="AIzaSyAJCXF94YdQDi5dLL9TZrW8YFy4jsNYkrA")

system_prompt = "You are a helpful assistant that rewrites sentences to be more inclusive. Replace ableist language with non-ableist alternatives while keeping the meaning and structure of the sentence. Only return the full rewritten sentence as a single line."

for i, sentence in enumerate(sentences):

    if sentence in existing_ids:
        print(f"Skipping {i}: already processed")
        continue

    for attempt in range(MAX_RETRIES):
        try:

            response = client.models.generate_content(
                model="gemini-1.5-pro",
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=f"Rewrite the following sentence by replacing the phrase '{phrases[i]}' with a non-ableist alternative. Try to retain as much of the original sentence as possible while reducing the ableism of the sentence. Return only the complete revised sentence as a single line without explanation, commentary, or line breaks. Sentence: '{sentence}'"
            )

            if response.text:

                replacement = response.text
                print(f"\n\n {i} \n\n Original Sentence: {sentence} \n New Sentence: {replacement}")
                results.append({
                    "original_sentence": sentence,
                    "detected_phrase": phrases[i],
                    "replacement_sentence": replacement
                })

                with open(output_path, "w") as out_file:
                    json.dump(results, out_file, indent=2, ensure_ascii=False)
                break
            else:
                print(f"Warning: No choices in response at index {i}, attempt {attempt+1}")
                time.sleep(RETRY_DELAY)

        except genai.errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                retry_delay = 60  # fallback delay
                if hasattr(e, "response") and "retryDelay" in str(e):
                    # Extract from response text manually if needed
                    import re
                    match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                    if match:
                        retry_delay = int(match.group(1))
                print(f"Gemini rate limit hit at index {i}, waiting {retry_delay} seconds...")
                time.sleep(retry_delay)

        except Exception as e:
            print(f"[Unhandled] Error at index {i}, attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
    else:
        print(f"Failed after {MAX_RETRIES} attempts at index {i}. Skipping...")
        failed_sentences.append({
            "original_sentence": sentence,
            "detected_phrase": phrases[i]
        })   

    time.sleep(3)

with open("rewritten_sentences.json", "w") as out_file:
    json.dump(results, out_file, indent=2, ensure_ascii=False)

if failed_sentences:
    with open("failed_sentences.json", "w") as fail_file:
        json.dump(failed_sentences, fail_file, indent=2, ensure_ascii=False)
    print(f"⚠️ Logged {len(failed_sentences)} failed rewrites.")