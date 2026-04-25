# 메모장 만들기 (메모장 추가, 읽기) 쓰는 대로 메모장 파일에 쓰이고, 읽을 수 있는 기능

import sys
option = sys.argv[1]
if option == '-a':
    f = open("C:/python/chapter6/memo.txt", 'a', encoding = 'utf-8')
    M = sys.argv[2:]
    f.write(" ".join(M))
    f.write('\n')
    f.close()
elif option == '-r':
    f = open("C:/python/chapter6/memo.txt", 'r', encoding = 'utf-8')
    R = f.read()
    f.close()
    print(R)
    
else:
    print("옵션이 정확하지 않습니다. 옵션을 확인하고 다시 입력해주세요")