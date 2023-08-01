class MyZoo(object):
    def __init__(self, a):
        self.a = a
        self.val = 0
        for i in a.values():
            self.val += i
        print("My Zoo!")
    
    def __eq__(self, other):
        return self.a.keys() == other.a.keys()
    
    def __len__(self):
        return self.val
    

myzoooo1 = MyZoo({'pig':1})
myzoooo2 = MyZoo({'pig':5})
print(myzoooo1 == myzoooo2)
print(len(myzoooo1))
print(len(myzoooo2))
