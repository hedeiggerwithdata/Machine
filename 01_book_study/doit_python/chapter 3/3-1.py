coffee = 30
money = 1100

def coffemachine(coffee, money, user):
    while True:
        user = input("구매를 원하지 않으신다면 Q를 눌러주세요 : ")
        
        if user == 'Q':
            break
        else:
            pass   
        coin = int(input("""동전을 투입하세요: """))
                
        print("동전을 넣으셨습니다. 구매를 계속합니다.")
            
        if coin>=100:
            coffee -= 1
            print("커피를 제작합니다. 거스름돈 %d원을 지급합니다." %(coin - 100))
            money = money - 100
            print("남은 커피 수량: %d개" %coffee)
        else:
            print("돈이 부족합니다. 돈을 반환합니다. 추가 구매를 원하시면 동전을 넣어주세요.")
            continue
        
        if coffee == 0:
            print("커피 수량이 부족해 판매를 종료합니다.")
            break
        
        if money == 0:
            print("돈이 부족해 집으로 갑니다")
            break
        

