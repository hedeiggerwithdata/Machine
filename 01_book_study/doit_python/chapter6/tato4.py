import sys

src = sys.argv[1]
dst = sys.argv[2]

f = open(src)
tap = f.read()
f.close()

jump = tap.replace("\t", " "*4)



f = open(dst, 'w', encoding = 'utf-8')
f.write(jump)
f.close()
