# f = open("C:/python/연습용.txt", 'w', encoding = 'utf-8' )

# A = ['덴지', '마키마', '파워', '아키', '레제', '아사']

# for x in A:
#     f.write("체인소맨의 주인공은 %s이다.\n" %x)
    
# f.close()

# f = open("C:/python/연습용.txt", 'r', encoding = 'utf-8')
# a = f.read()
# print(a)
# f.close()

f = open("C:/python/연습용.txt", 'a', encoding = 'utf-8')
list = ['포치타', '악마', '마인']
for y in list:
    f.write("체인소맨의 악마는 %s이다.\n" %y)
f.close()

f = open("C:/python/연습용.txt", 'r', encoding = 'utf-8')
read = f.readlines()
for component in read:
    print(component, end = ' ')
f.close()
    