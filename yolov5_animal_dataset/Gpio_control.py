import time
import RPi.GPIO as GPIO

class GpioLaserController:
    def __init__(self):
        self.LASER1 = 26
        self.LASER2 = 26
        self.LASER3 = 26
        self.LASER4 = 26

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.LASER1, GPIO.OUT)
        GPIO.setup(self.LASER2, GPIO.OUT)
        GPIO.setup(self.LASER3, GPIO.OUT)
        GPIO.setup(self.LASER4, GPIO.OUT)
        # 激光初始化关闭
        GPIO.output(self.LASER1, GPIO.LOW)
        GPIO.output(self.LASER2, GPIO.LOW)
        GPIO.output(self.LASER3, GPIO.LOW)
        GPIO.output(self.LASER4, GPIO.LOW)

        print("[GPIO] 初始化完成（四个激光已就绪）")
    def set_laser(self,duration):
        self.laser_on()
        time.sleep(duration)
        self.laser_off()

    def laser_on(self):
        GPIO.output(self.LASER1, GPIO.HIGH)
        GPIO.output(self.LASER2, GPIO.HIGH)
        GPIO.output(self.LASER3, GPIO.HIGH)
        GPIO.output(self.LASER4, GPIO.HIGH)
        print("[激光] 开启")

    def laser_off(self):
        GPIO.output(self.LASER1, GPIO.LOW)
        GPIO.output(self.LASER2, GPIO.LOW)
        GPIO.output(self.LASER3, GPIO.LOW)
        GPIO.output(self.LASER4, GPIO.LOW)
        print("[激光] 关闭")

    def cleanup(self):
        self.laser_off()
        GPIO.cleanup()
        print("[GPIO] 已清理并关闭激光")

