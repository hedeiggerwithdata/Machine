class FourCal:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        
    def setdata(self, first, second):
        self.first = first
        self.second = second
    
    def add(self):
        result = self.first + self.second
        return result
    
    def div(self):
        result = self.first / self.second
        return result
    
    def mul(self):
        result = self.first * self.second
        return result
    
    def sub(self):
        result = self.first - self.second
        return result

class MoreCal(FourCal):
    def pow(self):
        result = self.first ** self.second
        return result
    
    def div(self):
        if self.second == 0:
            return 0
        else:
            return self.first / self.second
        

a = MoreCal(3, 1)
print(a.add())

f = open("C:/python/mod1.py", 'w', encoding = 'utf-8')
code = """class mode:
    def database(self, first, second):
            self.first = first
            self.second = second
        
    def add(self):
            result = self.first + self.second
            return result
        
    def mul(self):
            result = self.first * self.second
            return result"""

f.write(code)
f.close()
        
