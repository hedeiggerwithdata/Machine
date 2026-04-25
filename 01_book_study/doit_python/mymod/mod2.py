PI = 3.141592

class Circle:
    def __init__(self, r):
        self.r = r
    
    def extent(self):
        return PI * (self.r**2)
    
def add(a, b):
    return a + b

if __name__ == "__main__":
    a = Circle(4)
    print(a.extent())
    
f = open('C:/python/mymod/raise_error.py', 'w', encoding = 'utf-8')
f.close()