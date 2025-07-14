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
        full_text+= page.get_text()+"\n"


# pattern = re.compile(r'\d+\.\s+(.*?)\n*(?:Doing business as:\s+(.*?))?\n(.*?)\n*License #:\s+([\w‑-]+)\n*(.*?)\n*(.*?Has Applied for an\s (\S(\n)))? License ',  re.DOTALL | re.IGNORECASE | re.MULTILINE)
pattern = re.compile(
    r'^\d+\.\s+(.*?)\n*Doing business as:\s+(.*?)?\n*(\d{1,5}[\w\s.,\-]+?,\s\w+,\sMA\s\d{5})\n*License #:\s([\w‑-]+)\n*(?:Has applied for a\s+(.*?)(\s+License).*?|\n)?\n(Granted|Deferred)(.*?)?\n', re.DOTALL | re.IGNORECASE | re.MULTILINE)
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