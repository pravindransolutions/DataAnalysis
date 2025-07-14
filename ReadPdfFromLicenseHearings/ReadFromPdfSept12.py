import pdfplumber
import pymupdf
import re
import numpy as n


# fix couple of \n lines an and read the new derferred licenses and write to xcel file and read all files to see if granted at a future date
#count all files
#file_path = "VotingMinutes9-12-24.pdf"  # replace with your file path
file_path="VotingMinutes9-26-24.pdf"
common_writefile = "allhearings.csv"
entries = []
text_start = False
start_pattern = '^Transactional Hearing'

with pymupdf.open(file_path) as pdf:
    full_text = ""
    for page in pdf.pages():
        lines = page.get_text() +"\n"

        if text_start:
            full_text += page.get_text()
        else:
            for line in lines.splitlines():
                if text_start == True:
                    full_text += line+"\n"
                    continue
                match = re.match(start_pattern, line)
                if match == n.nan or match is None:
                    continue
                elif match != n.nan or match is not None or match.pos == 0:
                    text_start = True

# pattern = re.compile(r'\d+\.\s+(.*?)\n*(?:Doing business as:\s+(.*?))?\n(.*?)\n*License #:\s+([\w‑-]+)\n*(.*?)\n*(.*?Has Applied for an\s (\S(\n)))? License ',  re.DOTALL | re.IGNORECASE | re.MULTILINE)
pattern = re.compile(
    r'^\d+\.\s+(.*?)\n*Doing business as:\s+(.*?)?\n*(\d{1,5}[\w\s.,\-]+?,\s\w+,\sMA\s\d{5})\n*License #:\s([\w‑-]+)\n*(?:Has applied for a\s+(.*?)(\s+License).*?|\n)?\n(Granted|Deferred)(.*?)?\n', re.DOTALL | re.IGNORECASE | re.MULTILINE)
full_text=full_text.replace('u2011','-'). replace('u2013','-').replace('/xa0','')
matches = pattern.findall(full_text)
data = []
for m in matches:
    data.append({
        'business_name': m[0].strip(),
        'DBA': m[1].strip(),
        'Address': m[2],
        'License': m[3],
        'LicenseType': m[4],
        'Status' : m[6],
        'Notes':m[7]
    })

for d in data:
    print(d)
    print('--------------')