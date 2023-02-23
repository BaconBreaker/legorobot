import rpyc

try:
    ROBOT_HOSTNAME = "10.42.0.1"
    conn = rpyc.classic.connect(ROBOT_HOSTNAME)
    conn.execute("print('Hello Slave. I am your , master!')")
except Exception as e:
    print(e)
    raise Exception('No conection to rpyc server on robot possible! Is the robot conneced? Is the rpyc server started?')
