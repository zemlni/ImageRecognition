import threading
import time

a = [] 

def t2():
    for i in range (0, 1000):
        print a
        time.sleep(1)

def t1():
    for i in range(0, 100):
        a.append(i)
        time.sleep(1)

thread1 = threading.Thread(target=t1)
thread2 = threading.Thread(target=t2)

thread2.start()
thread1.start()

print "end " + a
