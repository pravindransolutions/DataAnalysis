import fitz  # PyMuPDF
import re
import pandas as pd

# Step 1: Extract text
doc = fitz.open("VotingMinutes9-26-24.pdf")
text = "\n".join(page.get_text() for page in doc)
text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\xa0', ' ')

# Step 2: Split entries
entries = re.split(r"\n_{10,}\n", text)

# Step 3: Parse entries into list of dicts
records = []
for entry in entries:
    entry_data = {}
    lines = entry.strip().splitlines()
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().replace(" ", "_").replace("#", "Number").replace("/", "_")
            entry_data[key] = val.strip()
        else:
            entry_data.setdefault("Text", []).append(line.strip())
    if "Text" in entry_data:
        entry_data["Text"] = " ".join(entry_data["Text"])
    records.append(entry_data)

# Step 4: Save to CSV
df = pd.DataFrame(records)
df.to_csv("Voting_Minutes_Parsed.csv", index=False)
print("CSV saved as Voting_Minutes_Parsed.csv")
