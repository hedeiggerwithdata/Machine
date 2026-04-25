import time


# 시간 측정 데코레이터로 새로운 함수 변환
# 데코레이터를 클로저(외부 함수 및 변수 상태 저장)로 만들기
# wrapper는 기존 함수를 새로운 기능을 탑재한 함수인 wrapper로 만들어서 함수를 저장한다.

def timelap(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        print(f"{function.__name__} 수행시간: %.6f초" %(end - start))
        return result
    return wrapper


@timelap
def machine(coin):
    if coin>=500:print('커피 나왔습니다')
    else: print('잔액부족')
    
@timelap
def g(a, b):
    return a + b

@timelap
def f(name, say='안녕'):
    return f"{say} {name}"

@timelap
def dic(**kwargs):
    return kwargs

print(machine(500), end='\n')
print(g(1, 3), end='\n')
print(f('철수', say = '반가워'), end = '\n')
print(dic(a=1, b=2))