import re

p = re.compile(r'\\section')

data = "\section123"

print(p.search(data))