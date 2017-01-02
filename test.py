import RPi.GPIO as GPIO
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
