import pymupdf
import pytesseract
from pdf2image import convert_from_path
import numpy as n
import re

entries = []
text_start = False
start_pattern = '^Transactional Hearing'
file_name = 'VotingMinutes9-26-24.pdf'

with pymupdf.open(file_name,'r') as pdf:
    full_text= convert_from_path(pdf.pages())


pattern = re.compile(
    r'^\d+\.\s+(.*?)\nDoing business as:\s+(.*?)?\n*(\d{1,5}[\w\s.,\-]+?,\s\w+,\sMA\s\d{5})\n*License #:\s([\w‑-]+)\n*(?:Has applied for a\s+(.*?)(\s+License).*?|\n)?\n(Granted|Deferred)(.*?)?\n', re.DOTALL | re.IGNORECASE | re.MULTILINE)

text = pytesseract.image_to_string(full_text)
matches = pattern.findall(text)

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