import pdfplumber
import re
import numpy as n

file_path = "VotingMinutes2-13-25.docx.pdf"  # replace with your file path
#file_path = "Voting AgendaJuly11.docx_2.pdf"  # replace with your file path

entries = []
text_start = False
start_pattern = "^Transactional Hearing"

with pdfplumber.open(file_path) as pdf:

    full_text = ""
    for page in pdf.pages:
        lines= page.extract_text() + "\n"
        
        if text_start:
            full_text+=page.extract_text()
        else:
            for line in lines.splitlines():
                if text_start == True:
                    full_text+=line+"\n"
                    continue
                match = re.match(start_pattern, line)
                if match == n.nan or match is None:
                    continue
                elif match!= n.nan or match is not None or match.pos == 0:
                    text_start = True

#pattern = re.compile(r'\d+\.\s+(.*?)\n*(?:Doing business as:\s+(.*?))?\n(.*?)\n*License #:\s+([\w‑-]+)\n*(.*?)\n*(.*?Has Applied for an\s (\S(\n)))? License ',  re.DOTALL | re.IGNORECASE | re.MULTILINE)
pattern = re.compile(r'^\d+\.\s+(.*?)\n*(?:Doing business as:\s+(.*?))?\n(.*?)\n*License #:\s+([\w‑-]+)\n*(?:Has applied for a\s+(.*?)(\s+License))?',  re.IGNORECASE | re.MULTILINE)
matches = pattern.findall(full_text)
data = []
for m in matches:
    data.append({
            'business_name': m[0].strip(),
            'DBA': m[1].strip(),
            'Address':m[2],
            'License':m[3],
            'LicenseType':m[4]
                })

for d in data:
    print(d)
    print('--------------')