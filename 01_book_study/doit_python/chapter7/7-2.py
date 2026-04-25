class MyIteration:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def __iter__(self): #어떤 것이 이터레이터가 되느냐? 이터레이터를 불러올 때 실행
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            print("이터레이터 끝")
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result
    
class Re_Iteration:
    def __init__(self, data):
        self.data = data
        self.index = len(self.data) - 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < 0:
            print("이터레이터 끝")
            raise StopIteration
        result = self.data[self.index]
        self.index -= 1
        return result
    
            
class MyIterable:
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return Re_Iteration(self.data)
    
    
    
if __name__ == "__main__":
    a = MyIterable([1, 2, 3, 4])
    for i in a:
        print(i)
        
    for x in a:
        print(x)
        
