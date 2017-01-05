'''import RPi.GPIO as GPIO
import time

Motor1Pin1   = 11    # pin11
Motor1Pin2   = 12    # pin12
Motor1Enable = 13    # pin13
Motor2Pin1   = 15
Motor2Pin2   = 16
Motor2Enable = 18
GPIO.setmode(GPIO.BOARD)          # Numbers GPIOs by physical locationGPIO.setup(MotorPin1, GPIO.OUT)   # mode --- output
GPIO.setup(Motor1Pin1, GPIO.OUT)   # mode --- output
GPIO.setup(Motor1Pin2, GPIO.OUT)
GPIO.setup(Motor1Enable, GPIO.OUT)

GPIO.setup(Motor2Pin1, GPIO.OUT)   # mode --- output
GPIO.setup(Motor2Pin2, GPIO.OUT)
GPIO.setup(Motor2Enable, GPIO.OUT)

GPIO.output(Motor1Pin1, 1)
GPIO.output(Motor1Pin2, 0)

GPIO.output(Motor2Pin1, 0)
GPIO.output(Motor2Pin2, 1)

GPIO.output(Motor1Enable, 1)
GPIO.output(Motor2Enable, 1)

time.sleep(4)
GPIO.output(Motor1Enable, 0)
GPIO.output(Motor2Enable, 0)
'''
import numpy as np
import math
from multiprocessing import Queue

v1 = (1, 1)
v2 = (1, 2)
dot = v1[0] * v2[0] + v1[1] * v2[1]
bottom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
angle = math.acos(dot / bottom)
print angle

print v2[0] * v1[1] - v2[1] * v1[0]# 2d-cross product tells you whether vector is to the right or left. negative = left, positive = right
q = Queue()
print q.qsize()

angle = math.pi
matrix = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
v1 = np.matrix(v1).transpose()
print v1
l = (matrix*v1).A.T
print tuple([i for x in l for i in x])
print matrix

