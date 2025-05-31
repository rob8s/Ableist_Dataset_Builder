import requests
import json
import pandas as pd

with open("flagged_sentences.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
sentences = df["sentence"].tolist()
phrases = df["detected_phrase"].tolist()


results = []

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer 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",
    "HTTP-Referer": "YOUR_WEBSITE_URL",  # Optional
    "X-Title": "YOUR_APP_NAME",          # Optional
}
for i, sentence in enumerate(sentences):
    data = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",  # Check exact model ID on OpenRouter
        "messages": [{"role": "user", "content": f"Rewrite the following sentence to be non-ableist by replacing the phrase '{phrases[i]}'. Only return a single rewritten sentence, without explanation, commentary, or line breaks. Sentence: '{sentence}'"}]
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    # Assuming `response` is the JSON response dictionary:
    replacement = response["choices"][0]["message"]["content"]
    print(f"\n\n {i} \n\n Original Sentence: {sentence} \n New Sentence: {replacement}")
    results.append({
        "original_sentence": sentence,
        "detected_phrase": phrases[i],
        "replacement_sentence": replacement
    })

with open("rewritten_sentences.json", "w") as out_file:
    json.dump(results, out_file, indent=2, ensure_ascii=False)