import time
import datetime

def work(a):
    def w1():
        print("Name:", a.__name__ )
        s = time.time()
        sd = datetime.datetime.now()
        x = a()
        t = time.time()
        td = datetime.datetime.now()
        print("Start time:", sd, "\nEnd time:", td, "\nSpent:",t-s,"seconds")
        return x
    return w1

@work
def do():
    x = int(input()) + 1
    return x

print(do())
