import os

import cv2
import rospy
from genpy import Message
from numpy import ndarray
from cv_bridge import CvBridge
from pyautogui import size as window

from std_srvs.srv import SetBool

import tracking_msgs.srv as srv

__author__ = "Yue Lin"    # Email: yuelin@mail.dlut.edu.cn
__version__ = "24.06.25"  # Last updated date

INIT, WAIT, RUN = range(3)     # Status
SOT, MOT, DET, SAM = range(4)  # Algorithms
CLICK = cv2.EVENT_LBUTTONDOWN  # Mouse left click event
MAX_NUM = {SOT: 3, MOT: 10}    # Maximum number of targets

# Magic
TASKS = "SOT MOT DET SAM".split()
exec('='.join(("function", "type(lambda *_: ...)")))

# Colors
RED, BLACK, GRAY = (0, 0, 0xFF), (0,) * 3, (0x80,) * 3

# Classes
CLASSES = open(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "names.txt"
)).read().rstrip().split('\n') + ["tracked"]


class Window:
    def __init__(self, name: str, file: str):
        self.fps = [0, 0]      # [Total time, total number of frames]
        self.draw = None       # Starting drawing position
        self.mode = INIT       # Running status: INIT, WAIT, or RUN
        self.name = name       # The name of the window
        self.lock = False      # Whether is being accessed
        self.image = None      # Current image
        self.mouse = None      # Current mouse position
        self.frames = 0        # Frame id
        self.running = True    # Whether the windows is alive
        self.algorithm = None  # Currently running algorithm
        
        # Properties of all objects in the screen
        self.ids = []          # IDs of all selected objects
        self.chosens = []      # Whether the object is selected
        self.classes = []      # Categories of objects
        self.display = []      # Specially marked box
        self.objects = []      # All boxes
        self.tracked = []      # IDs output by ByteTrack
        
        # Initialize ROS node
        rospy.init_node("client")
        Service = rospy.ServiceProxy
        self.img2msg = CvBridge().cv2_to_imgmsg
        self.clients = {
            SAM: {
                "encode": Service("/SOT/frame0", srv.Image),
                "decode": Service("/SOT/point2box", srv.Point2Box)
            },
            MOT: {
                "reset": Service(rospy.get_param("~res"), SetBool),
                "track": Service(rospy.get_param("~mot"), srv.Image),
                "detect": Service(rospy.get_param("~det"), srv.Image),
                "sot&mot": Service(rospy.get_param("~mot1"), srv.Image),
                "sot&det": Service(rospy.get_param("~det1"), srv.Image)
            },
            SOT: {
                "track": Service(rospy.get_param("~sot"), srv.Image),
                "init": Service(rospy.get_param("~box"), srv.ImageWithBoxes)
            }
        }
        
        # All algorithms
        self.algorithms = (
            self.track1, self.track, self.detect, self.sot8det, self.sot8mot
        )
        
        # Create window
        cv2.namedWindow(name, 0x10)                # cv2.WINDOW_GUI_NORMAL = 16
        cv2.resizeWindow(name, *window())
        cv2.setMouseCallback(name, self.callback)

        # Save results
        self.mp4, self.txt = None, None
        self.save = rospy.get_param("~save")
        if self.save:
            path = __file__
            for _ in range(4):
                path = os.path.dirname(path)
            if "logs" not in os.listdir(path):
                os.mkdir(os.path.join(path, "logs"))
            self.mp4 = os.path.join(path, "logs", "{}.mp4".format(file))
            self.txt = open("{}.txt".format(os.path.splitext(self.mp4)[0]), 'w')
    
    @property
    def num(self) -> int:
        """Number of objects"""
        return len(self.objects)

    @property
    def ready(self) -> bool:
        """Whether it is possible to start"""
        if self.algorithm == DET or self.mode == RUN:
            return True
        if self.mode == INIT:
            return False
        objects = self.chosens.count(True)
        if self.algorithm == MOT:
            return objects > 0
        return 0 < objects <= MAX_NUM[SOT]

    @property
    def indexes(self) -> list:
        """Index of all selected targets"""
        return [idx for idx in range(self.num) if None 
                is self.objects[idx] or self.chosens[idx]]

    def read(self) -> None:
        """Read an image"""
        raise NotImplementedError()
    
    def run(self) -> None:
        """Show image once"""
        self.wait4lock()

        # Initial state, no box
        if self.mode == INIT:
            frame = self.image
        
        # Select targets
        elif self.mode == WAIT:
            frame = self.image.copy()
            for b in range(self.num):  # Draw boxes
                if self.objects[b] is not None:
                    cv2.rectangle(
                        frame,
                        self.objects[b][:2],
                        self.objects[b][2:],
                        self.color(b + 1) if self.chosens[b] 
                        else GRAY, 1 + int(self.chosens[b])
                    )
                    self.text(
                        "id:{}".format(b + 1), frame,
                        *self.objects[b][:2], (0xFF,) * 3,
                        BLACK if self.chosens[b] else GRAY
                    )
            if self.display and self.algorithm:  # Draw specially box
                cv2.rectangle(
                    frame, self.display[1:3], self.display[3:], 
                    GRAY if self.display[0] else BLACK,
                    1 if self.display[0] else 3
                )
            if self.draw:  # Draw a box using mouse
                cv2.rectangle(
                    frame, self.draw, self.mouse, BLACK, 2
                )
        
        # Running algorithm
        else:
            self.read()
            if not self.running:
                return
            frame = self.image.copy()

            # Run algorithm
            self.algorithms[self.algorithm]()
            
            # Draw boxes
            for b in range(self.num):
                if self.objects[b] is not None:
                    if abs(self.algorithm) == DET:
                        color = self.color(self.classes[b] + 1)
                    elif self.chosens[b]:
                        color = self.color(b + 1)
                    else:
                        color = GRAY
                    cv2.rectangle(
                        frame,
                        self.objects[b][:2],
                        self.objects[b][2:], color,
                        2 if self.chosens[b] or self.algorithm == DET else 1
                    )
                    cv2.circle(frame, (
                        (self.objects[b][0] + self.objects[b][2]) >> 1,
                        (self.objects[b][1] + self.objects[b][3]) >> 1
                    ), 1, color, -1)
                    if self.algorithm < DET:
                        self.text(
                            "id:{}".format(b + 1), frame,
                            *self.objects[b][:2], (0xFF,) * 3,
                            BLACK if self.chosens[b] else GRAY
                        )
            
            # Draw specially box
            if self.display and self.algorithm:
                cv2.rectangle(
                    frame, self.display[1:3], self.display[3:],
                    GRAY if self.display[0] else BLACK,
                    1 if self.display[0] else 3
                )
    
        # Draw buttons
        h, w, _ = frame.shape
        frame = cv2.copyMakeBorder(
            frame, 0, h >> 3, 0, w >> 1, 
            cv2.BORDER_CONSTANT, value=(0xFF,) * 3
        )
        for button in ("SOT", "MOT", "DET"):  # Algorithms
            algorithm = eval(button)
            cv2.putText(
                frame, button, (int((
                    algorithm + 0.5
                ) * w / 3), h + (h >> 4)),
                cv2.FONT_HERSHEY_PLAIN, 2,
                RED if self.algorithm is not None and
                abs(self.algorithm) == algorithm else GRAY, 2
            )
        if self.num:                          # Targets
            for obj in range(self.num):
                cv2.putText(
                    frame, "id: {}, box: {}, {}".format(
                        obj + 1, self.objects[obj],
                        CLASSES[self.classes[obj]] if 
                        self.chosens[obj] else "lost"
                    ), (w, int((obj + 0.75) * h / max(MAX_NUM[MOT], 0xA))), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    self.color(obj + 1) if self.chosens[obj] else GRAY
                )
        else:                                 # No object
            cv2.putText(
                frame, f"No object detected",
                (int(1.1 * w), int(1.75 * h / max(MAX_NUM[MOT], 0xA))), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.7, RED
            )
        if self.mode == WAIT:                 # Start button
            cv2.putText(
                frame, "Resume" if 
                self.frames > 0 else "Start",
                (int(1.25 * w), h + (h >> 4)),
                cv2.FONT_HERSHEY_PLAIN, 0x2,
                BLACK if self.ready else GRAY, 2
            )
        if self.mode == RUN:                  # Pause button
            cv2.putText(
                frame, "Pause",
                (int(1.25 * w), h + (h >> 4)),
                cv2.FONT_HERSHEY_PLAIN, 2, BLACK, 2
            )

        # FPS and frame id
        if self.frames > 0:
            self.fps[1] = max(self.fps[1], 1e-6)
            cv2.putText(
                frame, "FPS: {:.2f}. Frame id: {}".format(
                    self.fps[0] / self.fps[1], self.frames
                ), (0, h // 10), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2
            )
        
        # Show image
        self.lock = False
        cv2.imshow(self.name, frame)
        self.running = (cv2.waitKey(WAIT) & 0xFF) != 0x1B
        
        # Save results
        if self.save and self.mode == RUN:
            if type(self.mp4) is str:
                self.mp4 = cv2.VideoWriter(
                    self.mp4, cv2.VideoWriter_fourcc(*"MP4V"), 0x3C, (w, h)
                )
            assert isinstance(self.mp4, cv2.VideoWriter)
            self.mp4.write(frame[:h, :w, :])
            if self.algorithm == SOT:
                for b in range(self.num):
                    if self.objects[b] is not None:
                        self.txt.write("{}\n".format(','.join(
                            map(str, self.objects[b]) if
                            self.chosens[b] else (["None"] * 4)
                        )))
                        break

    def show(self) -> None:
        """Main loop"""
        while self.running and not rospy.is_shutdown():
            self.run()
            try:
                cv2.getWindowProperty(self.name, INIT)
            except cv2.error:
                self.running = False
        if self.txt is not None:
            self.txt.close()
        if self.mp4 is not None:
            self.mp4.release()
        cv2.destroyAllWindows()
    
    def wait(self) -> None:
        """Wait for services"""
        for task in self.clients.keys():
            rospy.loginfo("Waiting for {} service...".format(TASKS[task]))
            for client in self.clients[task].values():
                client.wait_for_service()
            rospy.loginfo("{} service is ready.".format(TASKS[task]))
        self.read()
        rospy.loginfo("Client has been started.")
    
    def call(self, algorithm: int, task: str, func: function) -> Message:
        """Call service"""
        self.clear()
        boxes = self.clients[algorithm][task].call(self.request(False))
        for b in range(len(boxes.cls)):
            self.append(
                func(boxes, b),
                boxes.cls[b] if 
                algorithm == MOT else -1,
                boxes.x[b], boxes.y[b],
                boxes.x[b] + boxes.w[b],
                boxes.y[b] + boxes.h[b]
            )
        if self.mode == RUN and self.algorithm == SOT:
            self.reid()
        self.fps[1] += 1
        self.fps[0] += 1 / boxes.time
        return boxes
    
    def request(self, box: bool) -> Message:
        request = (srv.ImageWithBoxesRequest if box else srv.ImageRequest)()
        request.image = self.img2msg(self.image, "bgr8")
        return request
    
    def reid(self) -> None:
        """Reset track id"""
        numbers = max(self.ids) + 1
        classes = [-1] * numbers
        objects = [None] * numbers
        chosens = [False] * numbers
        for idx in range(self.num):
            classes[self.ids[idx]] = self.classes[idx]
            objects[self.ids[idx]] = self.objects[idx]
            chosens[self.ids[idx]] = self.chosens[idx]
        self.classes, self.objects, self.chosens = classes, objects, chosens
    
    def init(self, index: int = None, box: list = None) -> None:
        """SOT initialize"""
        boxes = self.request(True)
        
        # Initialize boxes[index]
        if index is not None:
            boxes.num = 1
            box = self.objects[index]
            boxes.x.append(box[0])
            boxes.y.append(box[1])
            boxes.w.append(box[2] - box[0])
            boxes.h.append(box[3] - box[1])
        
        # Initialize box
        elif box is not None:
            boxes.num = 1
            boxes.x.append(box[0])
            boxes.y.append(box[1])
            boxes.w.append(box[2] - box[0])
            boxes.h.append(box[3] - box[1])
        
        # Initialize all boxes
        else:
            boxes.num = 0
            for b in range(self.num):
                if self.chosens[b]:
                    boxes.num += 1
                    self.classes[b] = -1
                    box = self.objects[b]
                    boxes.x.append(box[0])
                    boxes.y.append(box[1])
                    boxes.w.append(box[2] - box[0])
                    boxes.h.append(box[3] - box[1])
        
        self.clients[SOT]["init"].call(boxes)
    
    def init1(self, index: int = None) -> None:
        """SOT initialize"""
        boxes = self.request(True)
        if index is None:
            boxes.num = 0
            for b in range(self.num):
                if self.chosens[b]:
                    boxes.num += 1
                    self.classes[b] = -1
                    box = self.objects[b]
                    boxes.x.append(box[0])
                    boxes.y.append(box[1])
                    boxes.w.append(box[2] - box[0])
                    boxes.h.append(box[3] - box[1])
        else:
            boxes.num = 1
            box = self.objects[index]
            boxes.x.append(box[0])
            boxes.y.append(box[1])
            boxes.w.append(box[2] - box[0])
            boxes.h.append(box[3] - box[1])
        self.clients[SOT]["init"].call(boxes)
    
    def reset(self) -> None:
        """Reset MOT"""
        self.clients[MOT]["reset"].call(SetBool._request_class(True))
    
    def track(self, first: bool = False, sot: bool = False) -> Message:
        """MOT"""
        classes = [c for c in self.classes]
        boxes = self.call(MOT, "sot&mot" if sot else "track", lambda *_: True)
        if not first:
            self.clear()
            for obj in self.tracked:
                if obj < 0:
                    continue
                try:
                    obj = boxes.id.index(obj)
                    self.append(
                        True, boxes.cls[obj], 
                        boxes.x[obj], boxes.y[obj],
                        boxes.x[obj] + boxes.w[obj],
                        boxes.y[obj] + boxes.h[obj]
                    )
                except ValueError:
                    self.append(False, -1, *((0,) * 4))
            self.reid()
            self.classes = classes[:self.num]
            tracked = set(map(abs, self.tracked))
            for obj in range(len(boxes.cls)):
                if boxes.id[obj] not in tracked:
                    add = self.append(
                        True, boxes.cls[obj],
                        boxes.x[obj], boxes.y[obj],
                        boxes.x[obj] + boxes.w[obj],
                        boxes.y[obj] + boxes.h[obj]
                    )
                    if add not in self.ids:
                        self.ids.append(add)
                    self.tracked.append(boxes.id[obj])
        return boxes
    
    def track1(self) -> Message:
        """SOT"""
        return self.call(SOT, "track", lambda boxes, b: not boxes.cls[b])
    
    def detect(self, sot: bool = False) -> Message:
        """DET"""
        return self.call(MOT, "sot&det" if sot else "detect", lambda *_: True)
    
    def sot8mot(self) -> Message:
        """SOT & MOT"""
        boxes = self.track(sot=True)
        self.display = (
            boxes.occ,
            boxes.x0, boxes.y0, 
            boxes.x0 + boxes.w0, 
            boxes.y0 + boxes.h0
        )
        return boxes

    def sot8det(self) -> Message:
        boxes = self.detect(True)
        self.display = (
            boxes.occ,
            boxes.x0, boxes.y0, 
            boxes.x0 + boxes.w0, 
            boxes.y0 + boxes.h0
        )
        return boxes

    def encode(self) -> None:
        """EdgeSAM encode"""
        image = srv.ImageRequest()
        image.image = self.img2msg(self.image, "bgr8")
        self.clients[SAM]["encode"].call(image)

    def clear(self) -> None:
        """Clear the objects"""
        self.classes.clear()
        self.chosens.clear()
        self.objects.clear()

    def choose(self, index: int, double: bool = False) -> bool:
        """Select or deselect a target"""
        if index >= len(self.objects):
            return False
        if self.objects[index] is None:
            return False
        if double:  # Double click -> SOT
            self.chosens = [False] * self.num
            self.chosens[index] = True
            self.init(index)
        else:
            self.chosens[index] = not self.chosens[index]
        return True
    
    def remove(self, index: int) -> None:
        """Remove a target"""
        if index < len(self.objects):
            del self.classes[index]
            del self.chosens[index]
            del self.objects[index]
            if self.algorithm == MOT:
                del self.tracked[index]
    
    def append(self, selected: bool, cls: int, *box: int) -> int:
        assert len(box) == 4
        if sum(box) == 0:
            box = None
        index = self.num
        if self.num >= MAX_NUM[MOT]:
            index -= 1
            for b in range(MAX_NUM[MOT]):
                if not self.chosens[b]:
                    index = b
                    break
            self.classes[index] = cls
            self.objects[index] = box
            self.chosens[index] = selected
        else:
            self.classes.append(cls)
            self.objects.append(box)
            self.chosens.append(selected)
        return index
    
    def wait4lock(self) -> None:
        """Wait for lock"""
        while self.lock:
            pass
        self.lock = True

    def point2box(self, x: int, y: int) -> None:
        """EdgeSAM decode"""
        box = self.clients[SAM]["decode"].call(srv.Point2BoxRequest(x, y))
        if self.algorithm == SOT:
            self.append(True, -1, box.x1, box.y1, box.x2, box.y2)
        else:
            self.display = (False, box.x1, box.y1, box.x2, box.y2)
    
    def callback(self, event: int, x: int, y: int, *_: tuple) -> None:
        """Mouse callback function"""
        height, width = self.image.shape[:2]
        x0, y0 = x < width, y < height
        self.mouse = x, y
        
        # Choose mode SOT, MOT or DET
        if self.mode < RUN and not y0 and event == CLICK:
            for p in range(1, 4):
                if x < p * width / 3:
                    self.encode()
                    self.wait4lock()
                    self.mode = WAIT
                    self.algorithm = p - 1
                    
                    # SOT: Do nothing
                    if self.algorithm == SOT:
                        pass
                    
                    # DET: Detection
                    elif self.algorithm == DET:
                        self.detect()

                    # MOT: Reset tracker and track id
                    else:
                        self.reset()
                        self.tracked.clear()
                        boxes = self.track(True)
                        for b in range(len(boxes.id)):
                            self.tracked.append(boxes.id[b])
                        self.tracked = self.tracked[-MAX_NUM[MOT]:]
                    
                    self.lock = False
                    return
        
        if self.mode == WAIT:
            # Click "Start" or "Resume"
            if not x0 and not y0 and self.ready and event == CLICK:
                self.wait4lock()
                if self.algorithm == MOT:
                    for index in range(self.num):
                        if not self.chosens[index] and self.tracked[index] > 0:
                            self.tracked[index] *= -1
                if self.algorithm == SOT:
                    self.init()
                if self.display and not self.display[0]:
                    self.algorithm *= -1
                if self.algorithm < 0:
                    self.init(box=self.display[1:])
                self.ids, self.fps = self.indexes, [0, 0]
                self.mode, self.lock = RUN, False
                return

            # Double click
            if not x0 and y0 and event == cv2.EVENT_LBUTTONDBLCLK:
                self.wait4lock()
                if self.choose(int(max(MAX_NUM[MOT], 0xA) * y / height), True):
                    self.init()
                    self.algorithm, self.mode = SOT, RUN
                    self.ids, self.fps = self.indexes, [0, 0]
                self.lock = False
                return
            
            # Choose or remove box
            if self.algorithm < DET and not x0 and y0:
                self.wait4lock()
                if event == CLICK:                  # Choose
                    self.choose(int(max(MAX_NUM[MOT], 0xA) * y / height))
                # if event == cv2.EVENT_RBUTTONDOWN:  # Remove
                #     self.remove(int(max(MAX_NUM[MOT], 0xA) * y / height))
                self.lock = False
                return
            
            # SOT initialize
            if x0 and y0:
                self.wait4lock()

                # Point to box
                if event == cv2.EVENT_RBUTTONDOWN:
                    self.point2box(x, y)
                
                # Draw box
                if event == CLICK:                              # Start
                    self.draw = x, y
                if event == cv2.EVENT_LBUTTONUP and self.draw:  # End
                    x0, y0 = self.draw
                    x1, x2 = sorted([x, x0])
                    y1, y2 = sorted([y, y0])
                    if (y2 - y1) * (x2 - x1) > 0x10:
                        if self.algorithm == SOT:
                            self.append(True, -1, x1, y1, x2, y2)
                        else:
                            self.display = (False, x1, y1, x2, y2)
                    else:
                        self.display = []
                    self.draw = None

                self.lock = False
                return
        
        # Click "Puese"
        if self.mode == RUN and not x0 and not y0 and event == CLICK:
            self.wait4lock()
            self.encode()  # EdgeSAM encode
            self.algorithm = abs(self.algorithm)
            self.mode, self.lock = WAIT, False
    
    @staticmethod
    def color(index: int) -> tuple:
        """Get box color"""
        return tuple(map(lambda p: p * (index << 2) % 0xFF, (0x25, 0x11, 0x1D)))
    
    @staticmethod
    def text(text: str,
             img: ndarray,
             x: int, y: int,
             background: tuple,
             foreground: tuple,
             offset: int = -1,
             size: float = 0.5,
             font: int = cv2.FONT_HERSHEY_TRIPLEX) -> None:
        """Put text in the image"""
        cv2.putText(
            img, text, (
                x + offset, y + offset
            ), font, size, background
        )
        cv2.putText(img, text, (x, y), font, size, foreground)


class VideoWindow(Window):
    def __init__(self, name: str, video: str):
        super().__init__(name, os.path.splitext(os.path.split(video)[1])[0])
        self.frames -= 1
        self.video = cv2.VideoCapture(video)
        frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames > 0:
            self.wait()
            rospy.loginfo("There are {} frames in total.".format(frames))
        else:
            rospy.logerr("Unable to load video at {}.".format(video))

    def read(self) -> None:
        """Read an image"""
        self.running, frame = self.video.read()
        if self.running:
            self.frames += 1
            self.image = frame


class ImageWindow(Window):
    def __init__(self, name: str, images: str, suffix: str = "jpg"):
        super().__init__(name, os.path.split(images)[1])
        frames = 0
        self.folder = images
        self.suffix = suffix
        self.images = set(os.listdir(images))
        while frames < 100000:
            frames += 1
            file = "{:05d}.{}".format(frames, suffix)
            if file not in self.images:
                frames -= 1
                break
        if frames:
            self.wait()
            rospy.loginfo("There are {} images in total.".format(frames))
        else:
            rospy.logerr("Unable to load images at {}.".format(images))
    
    def read(self) -> None:
        """Read an image"""
        self.frames += 1
        name = "{:05d}.{}".format(self.frames, self.suffix)
        if name in self.images:
            rospy.loginfo(f"Loading image: {name}")
            self.image = cv2.imread(os.path.join(self.folder, name))
        else:
            self.running = False


class CameraWindow(Window):
    def __init__(self, name: str, camera: str):
        super().__init__(name, camera)
        self.camera = cv2.VideoCapture(int(camera))
        if self.camera.read()[0]:
            self.wait()
            rospy.loginfo("Successfully opened the camera {}.".format(camera))
        else:
            rospy.logerr("Opening camera {} failed.".format(camera))

    def read(self) -> None:
        """Read an image"""
        read, frame = self.camera.read()
        if read:
            self.frames += 1
            self.image = cv2.flip(frame, 1)
    
    def run(self) -> None:
        """The camera needs to constantly read images"""
        if self.mode < RUN:
            self.read()
        super().run()

    def encode(self) -> None:
        """For camera, EdgeSAM dose not require pre-encoding"""
        pass

    def point2box(self, x: int, y: int) -> None:
        """EdgeSAM encoding and decoding"""
        super().encode()
        super().point2box(x, y)


if __name__ == "__main__":
    from sys import argv

    assert argv[1].lower() in ("image", "video", "camera"), \
        "Only supports 'video', 'image', or 'camera'"

    eval("{}Window".format(argv[1].capitalize()))(argv[3], argv[2]).show()
