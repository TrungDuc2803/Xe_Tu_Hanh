import RPi.GPIO as GPIO
import time
import threading
from flask import Flask, request, jsonify

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup GPIO pins
ENAST = 11  
DIRST = 9   
PULST = 10  

ENASP = 26 
DIRSP = 19  
PULSP = 13  

ENATT = 4 
DIRTT = 3 
PULTT = 2 

ENATP = 22  
DIRTP = 27 
PULTP = 17 

TRIGT = 5
ECHOT = 6

TRIGS = 16
ECHOS = 20

TRIGTrai = 8
ECHOTrai = 7

TRIGPhai = 23
ECHOPhai = 24

# Motor rear left
GPIO.setup(ENAST, GPIO.OUT)
GPIO.setup(DIRST, GPIO.OUT)
GPIO.setup(PULST, GPIO.OUT)
GPIO.output(ENAST, GPIO.HIGH)

# Motor rear right
GPIO.setup(ENASP, GPIO.OUT)
GPIO.setup(DIRSP, GPIO.OUT)
GPIO.setup(PULSP, GPIO.OUT)
GPIO.output(ENASP, GPIO.HIGH)

# Motor front left
GPIO.setup(ENATT, GPIO.OUT)
GPIO.setup(DIRTT, GPIO.OUT)
GPIO.setup(PULTT, GPIO.OUT)
GPIO.output(ENATT, GPIO.HIGH)

# Motor front right
GPIO.setup(ENATP, GPIO.OUT)
GPIO.setup(DIRTP, GPIO.OUT)
GPIO.setup(PULTP, GPIO.OUT)
GPIO.output(ENATP, GPIO.HIGH)

GPIO.setup(TRIGT, GPIO.OUT)
GPIO.setup(ECHOT, GPIO.IN)

GPIO.setup(TRIGS, GPIO.OUT)
GPIO.setup(ECHOS, GPIO.IN)

GPIO.setup(TRIGTrai, GPIO.OUT)
GPIO.setup(ECHOTrai, GPIO.IN)

GPIO.setup(TRIGPhai, GPIO.OUT)
GPIO.setup(ECHOPhai, GPIO.IN)

timeRearLeft = 0.1
timeRearRight = 0.1
timeFrontLeft = 0.1
timeFrontRight = 0.1

front_distance = 0
behind_distance = 0
left_distance = 0
right_distance = 0

def MotorRearLeft():
    while True:
        GPIO.output(PULST, GPIO.HIGH)
        time.sleep(timeRearLeft)
        GPIO.output(PULST, GPIO.LOW)
        time.sleep(timeRearLeft)

def MotorRearRight():
    while True:
        GPIO.output(PULSP, GPIO.HIGH)
        time.sleep(timeRearRight)
        GPIO.output(PULSP, GPIO.LOW)
        time.sleep(timeRearRight)

def MotorFrontLeft():
    while True:
        GPIO.output(PULTT, GPIO.HIGH)
        time.sleep(timeFrontLeft)
        GPIO.output(PULTT, GPIO.LOW)
        time.sleep(timeFrontLeft)

def MotorFrontRight():
    while True:
        GPIO.output(PULTP, GPIO.HIGH)
        time.sleep(timeFrontRight)
        GPIO.output(PULTP, GPIO.LOW)
        time.sleep(timeFrontRight)

def RunDistance():
    global front_distance, behind_distance, right_distance, left_distance
    while True:
        GPIO.output(TRIGT, GPIO.LOW)
        GPIO.output(TRIGS, GPIO.LOW)
        GPIO.output(TRIGTrai, GPIO.LOW)
        GPIO.output(TRIGPhai, GPIO.LOW)
        time.sleep(2)
        
        GPIO.output(TRIGT, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIGT, GPIO.LOW)
        
        pulse_start = None
        pulse_end = None
        
        while pulse_start is None:
            pulse_start = time.time() if GPIO.input(ECHOT) == GPIO.HIGH else None
        
        while pulse_end is None:
            pulse_end = time.time() if GPIO.input(ECHOT) == GPIO.LOW else None
        
        pulse_duration = pulse_end - pulse_start
        front_distance = int(pulse_duration * 17150)
        
        GPIO.output(TRIGS, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIGS, GPIO.LOW)
        pulse_start = None
        pulse_end = None
        while pulse_start is None:
            pulse_start = time.time() if GPIO.input(ECHOS) == GPIO.HIGH else None
        while pulse_end is None:
            pulse_end = time.time() if GPIO.input(ECHOS) == GPIO.LOW else None
        pulse_duration = pulse_end - pulse_start
        behind_distance = int(pulse_duration * 17150)
        GPIO.output(TRIGTrai, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIGTrai, GPIO.LOW)
        pulse_start = None
        pulse_end = None
        while pulse_start is None:
            pulse_start = time.time() if GPIO.input(ECHOTrai) == GPIO.HIGH else None
        while pulse_end is None:
            pulse_end = time.time() if GPIO.input(ECHOTrai) == GPIO.LOW else None
        pulse_duration = pulse_end - pulse_start
        left_distance = int(pulse_duration * 17150)
        GPIO.output(TRIGPhai, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIGPhai, GPIO.LOW)
        pulse_start = None
        pulse_end = None
        while pulse_start is None:
            pulse_start = time.time() if GPIO.input(ECHOPhai) == GPIO.HIGH else None
        while pulse_end is None:
            pulse_end = time.time() if GPIO.input(ECHOPhai) == GPIO.LOW else None
        pulse_duration = pulse_end - pulse_start
        right_distance = int(pulse_duration * 17150)

app = Flask(__name__)
@app.route('/Distance', methods=['GET'])
def distance():
    global front_distance, behind_distance, right_distance, left_distance
    response = {
        "success": True,
        "message": "Moving forward",
        "front_distance": front_distance,
        "behind_distance": behind_distance,
        "right_distance": right_distance,
        "left_distance": left_distance
    }
    
    return jsonify(response), 200
@app.route('/Forward', methods=['GET'])
def Forward():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if front_distance > 30:
        GPIO.output(DIRST, GPIO.LOW)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
    
        GPIO.output(DIRSP, GPIO.HIGH)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
    
        GPIO.output(DIRTT, GPIO.LOW)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.HIGH)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()
        
    return jsonify({"success": True, "message": "Moving forward"}), 200

@app.route('/Backward', methods=['GET'])
def Backward():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if behind_distance > 10:
        GPIO.output(DIRST, GPIO.HIGH)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
        
        GPIO.output(DIRSP, GPIO.LOW)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
        
        GPIO.output(DIRTT, GPIO.HIGH)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.LOW)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()
        
    return jsonify({"success": True, "message": "Moving backward"}), 200

@app.route('/Right', methods=['GET'])
def Right():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if right_distance > 10:
        GPIO.output(DIRST, GPIO.HIGH)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
        
        GPIO.output(DIRSP, GPIO.HIGH)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
        
        GPIO.output(DIRTT, GPIO.LOW)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.LOW)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()
    
    return jsonify({"success": True, "message": "Moving left"}), 200

@app.route('/Left', methods=['GET'])
def Left():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if left_distance > 10:
        GPIO.output(DIRST, GPIO.LOW)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
        
        GPIO.output(DIRSP, GPIO.LOW)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
        
        GPIO.output(DIRTT, GPIO.HIGH)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.HIGH)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()
    
    return jsonify({"success": True, "message": "Moving right"}), 200

@app.route('/TurnLeft', methods=['GET'])
def TurnLeft():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if front_distance > 10 and behind_distance > 10 and left_distance > 25 and right_distance > 25:
        GPIO.output(DIRST, GPIO.HIGH)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
    
        GPIO.output(DIRSP, GPIO.HIGH)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
    
        GPIO.output(DIRTT, GPIO.HIGH)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
    
        GPIO.output(DIRTP, GPIO.HIGH)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()
        
    return jsonify({"success": True, "message": "Moving turnleft"}), 200

@app.route('/TurnRight', methods=['GET'])
def TurnRight():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if front_distance > 10 and behind_distance > 10 and left_distance > 25 and right_distance > 25:
        GPIO.output(DIRST, GPIO.LOW)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
    
        GPIO.output(DIRSP, GPIO.LOW)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
    
        GPIO.output(DIRTT, GPIO.LOW)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
    
        GPIO.output(DIRTP, GPIO.LOW)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()

    return jsonify({"success": True, "message": "Moving turnright"}), 200

@app.route('/ForwardLeft', methods=['GET'])
def ForwardLeft():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if front_distance > 10 and behind_distance > 10 and left_distance > 25 and right_distance > 25:
        GPIO.output(DIRST, GPIO.LOW)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
        
        GPIO.output(DIRSP, GPIO.HIGH)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.01
        
        GPIO.output(DIRTT, GPIO.LOW)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.01
        
        GPIO.output(DIRTP, GPIO.HIGH)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()

    return jsonify({"success": True, "message": "Moving forwardleft"}), 200

@app.route('/ForwardRight', methods=['GET'])
def ForwardRight():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if front_distance > 10 and right_distance > 10:
        GPIO.output(DIRST, GPIO.LOW)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.01
        
        GPIO.output(DIRSP, GPIO.HIGH)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
        
        GPIO.output(DIRTT, GPIO.LOW)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.HIGH)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.01
    else:
        Stop()

    return jsonify({"success": True, "message": "Moving forwardright"}), 200

@app.route('/BackwardRight', methods=['GET'])
def BackwardRight():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if behind_distance > 10 and right_distance > 10:
        GPIO.output(DIRST, GPIO.HIGH)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.0001
        
        GPIO.output(DIRSP, GPIO.LOW)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.01
        
        GPIO.output(DIRTT, GPIO.HIGH)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.01
        
        GPIO.output(DIRTP, GPIO.LOW)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.0001
    else:
        Stop()

    return jsonify({"success": True, "message": "Moving backwardleft"}), 200

@app.route('/BackwardLeft', methods=['GET'])
def BackwardLeft():
    global front_distance, behind_distance, left_distance, right_distance, timeRearLeft, timeRearRight, timeFrontLeft, timeFrontRight
    if behind_distance > 10 and left_distance > 10:
        GPIO.output(DIRST, GPIO.HIGH)
        GPIO.output(ENAST, GPIO.LOW)
        timeRearLeft =  0.01
        
        GPIO.output(DIRSP, GPIO.LOW)
        GPIO.output(ENASP, GPIO.LOW)
        timeRearRight =  0.0001
        
        GPIO.output(DIRTT, GPIO.HIGH)
        GPIO.output(ENATT, GPIO.LOW)
        timeFrontLeft =  0.0001
        
        GPIO.output(DIRTP, GPIO.LOW)
        GPIO.output(ENATP, GPIO.LOW)
        timeFrontRight =  0.01
    else:
        Stop()
        
    return jsonify({"success": True, "message": "Moving backwardright"}), 200

@app.route('/Stop', methods=['GET'])
def Stop():
    GPIO.output(ENAST, GPIO.HIGH)
    GPIO.output(ENASP, GPIO.HIGH)
    GPIO.output(ENATT, GPIO.HIGH)
    GPIO.output(ENATP, GPIO.HIGH)

    return jsonify({"success": True, "message": "Moving stop"}), 200

if __name__ == '__main__':
    try:
        
        RearLeft = threading.Thread(target=MotorRearLeft)
        RearRight = threading.Thread(target=MotorRearRight)
        FrontLeft = threading.Thread(target=MotorFrontLeft)
        FrontRight = threading.Thread(target=MotorFrontRight)
        runDistance = threading.Thread(target=RunDistance)
        RearLeft.start()
        RearRight.start()
        FrontLeft.start()
        FrontRight.start()
        runDistance.start()
        app.run(host='0.0.0.0', port=5000)

    except KeyboardInterrupt:
        print("error:", str(e))

