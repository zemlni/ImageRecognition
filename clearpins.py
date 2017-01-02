import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup([11, 12, 13, 15, 16, 18], GPIO.OUT)
GPIO.output([11, 12, 13, 15, 16, 18], 0)
