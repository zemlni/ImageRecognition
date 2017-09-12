import RPi.GPIO as GPIO
import time

def loop():
        GPIO.output(MotorEnable, 1)
        p1.start(0)
        #p2.start(0)
        for i in range(20, 101):
                print i
                p1.ChangeDutyCycle(i)
                #p2.ChangeDutyCycle(i)
                time.sleep(.3)
        GPIO.output(MotorEnable, 0) 
        p1.stop()
        #p2.stop()

def destroy():
	GPIO.output(MotorEnable, GPIO.LOW) # motor stop
	GPIO.cleanup()                     # Release resource

if __name__ == '__main__':     # Program start from here
	
        MotorPin1   = 11    # pin11
	MotorPin2   = 12    # pin12
	MotorEnable = 13    # pin13
        GPIO.setmode(GPIO.BOARD)          # Numbers GPIOs by physical location
	GPIO.setup(MotorPin1, GPIO.OUT)   # mode --- output
	GPIO.setup(MotorPin2, GPIO.OUT)
	GPIO.setup(MotorEnable, GPIO.OUT)

	p1 = GPIO.PWM(MotorEnable, 30)
	#p2 = GPIO.PWM(MotorPin2, 30)
	p1.start(100)
        GPIO.output(MotorPin1, 1)
        GPIO.output(MotorPin2, 0)
        GPIO.output(MotorEnable, 1)
	time.sleep(10)
        p1.stop()
        GPIO.output(MotorEnable, 0)
	try:
		loop()
	except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
		destroy()
