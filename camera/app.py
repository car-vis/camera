import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

import signal, sys
from threading import Thread, Event


from flask import Flask, render_template, Response

frame = None # 글로벌 변수 (이미지 받아오기)

bridge = CvBridge()
event = Event()

# ROS2-> openCV 변환과정
def on_image(msg):
    global frame
    
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding = "passthrough")
    frame = cv2.imencode(".jpg",cv_image)[1].tobytes()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    event.set()

# 노드 초기화 과정
rclpy.init(args=None)
node = rclpy.create_node('Show_image_python')

Thread(target=lambda:node).start()

subscription = node.create_subscription(Image,"/usb_cam/image_raw", on_image, 10)

app = Flask(__name__)

def get_frame():
    rclpy.spin_once(node, timeout_sec=1.0)
    event.wait()
    event.clear()
    return frame

@app.route('/')
def Index():
    return render_template('index.html')

def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/stream')
def Stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main(args=Node):
    app.run(host="10.10.15.64", port="8080", debug=True)

if __name__ == '__main__':
    main()
