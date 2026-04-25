import re

data = """park 010-8489-0830
kim 010-1234-5678
lee 016-869-0830"""

def hexrepl(match):
    value = int(match.group())
    return hex(value)

p = re.compile(r"\d+")
m = p.sub(hexrepl, "i live in 52 street, 2726 avenue 231spot" )
print(m)