import sys
import cv2
import time
import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
                             QGroupBox, QGridLayout, QLCDNumber)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO

# ==========================================
# 核心设置区域
# ==========================================
# 1. 这里填入你训练好的模型路径，如果还没训练好，用 'yolov8n.pt'
MODEL_PATH = 'yolov8s.pt' 

# 2. 模拟业务逻辑：定义哪些物体算NG（缺陷）
# 如果你用官方模型，这里我们假设检测到 "cell phone"(手机) 算 NG，"cup"(杯子) 算 OK
# 如果是你自己的模型，这里应该写 ['scratch', 'dent'] 等缺陷标签
NG_CLASSES = ['cell phone', 'scissors'] 

# ==========================================
# 1. 后台工作线程：负责跑 AI 模型 (不卡界面)
# ==========================================
class AIWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage) # 发送图像信号
    update_stats_signal = pyqtSignal(str, str) # 发送统计信号 (结果, 类别)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = YOLO(MODEL_PATH)

    def run(self):
        # 打开摄像头，0是默认摄像头
        cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 1. AI 推理
                results = self.model(frame, verbose=False)
                
                # 2. 结果判读
                detection_status = "WAITING" # 默认状态
                detected_cls = ""

                for r in results:
                    # 画框
                    annotated_frame = r.plot()
                    
                    # 检查检测到的类别
                    if len(r.boxes) > 0:
                        # 取置信度最高的一个
                        box = r.boxes[0]
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        
                        # 简单的业务逻辑判断
                        if class_name in NG_CLASSES:
                            detection_status = "NG"
                            detected_cls = class_name
                            # 在图上画个大大的 NG
                            cv2.putText(annotated_frame, "NG", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        else:
                            detection_status = "OK"
                            detected_cls = class_name
                            cv2.putText(annotated_frame, "OK", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    else:
                        # 没检测到东西，保持原图
                        annotated_frame = frame
                        detection_status = "WAITING"

                # 3. 发送结果给 UI (每隔一点时间发送，避免刷屏太快，实际可视情况调整)
                if detection_status != "WAITING":
                    self.update_stats_signal.emit(detection_status, detected_cls)

                # 4. 转换图像格式供 PyQt 显示 (BGR -> RGB)
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(800, 600, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)

            # 稍微休眠一下释放CPU
            time.sleep(0.03) 
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# 2. 主界面逻辑
# ==========================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("supOS 工业AI视觉质检系统 - 演示版")
        self.resize(1200, 750)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # 数据统计变量
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.last_process_time = 0 # 用于简单的去抖动

        self.init_ui()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout() # 水平布局：左边视频，右边数据

        # --- 左侧：视频显示区 ---
        video_group = QGroupBox("实时监控 (Camera Feed)")
        video_group.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        video_layout = QVBoxLayout()
        
        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)
        self.image_label.setStyleSheet("background-color: #000; border: 2px solid #444;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("摄像头未启动\nCamera OFF")
        
        video_layout.addWidget(self.image_label)
        video_group.setLayout(video_layout)

        # --- 右侧：控制与数据区 ---
        control_layout = QVBoxLayout()
        
        # 1. 状态指示灯 (Result Indicator)
        self.status_label = QLabel("READY")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 30, QFont.Bold))
        self.status_label.setStyleSheet("background-color: #555; color: white; border-radius: 10px; padding: 20px;")
        self.status_label.setFixedHeight(120)
        
        # 2. 数据统计面板 (Statistics)
        stats_group = QGroupBox("生产统计 (Statistics)")
        stats_group.setStyleSheet("QGroupBox { border: 1px solid #555; font-weight: bold; }")
        stats_grid = QGridLayout()
        
        # 定义 LCD 显示屏
        self.lcd_total = self.create_lcd()
        self.lcd_ok = self.create_lcd()
        self.lcd_ng = self.create_lcd()
        self.lcd_rate = self.create_lcd()
        
        stats_grid.addWidget(QLabel("检测总数 (Total):"), 0, 0)
        stats_grid.addWidget(self.lcd_total, 0, 1)
        stats_grid.addWidget(QLabel("良品数量 (OK):"), 1, 0)
        stats_grid.addWidget(self.lcd_ok, 1, 1)
        stats_grid.addWidget(QLabel("缺陷数量 (NG):"), 2, 0)
        stats_grid.addWidget(self.lcd_ng, 2, 1)
        stats_grid.addWidget(QLabel("良品率 (Yield %):"), 3, 0)
        stats_grid.addWidget(self.lcd_rate, 3, 1)
        
        stats_group.setLayout(stats_grid)

        # 3. 系统日志 (System Log)
        log_group = QGroupBox("系统日志 / supOS上传状态")
        log_group.setStyleSheet("QGroupBox { border: 1px solid #555; font-weight: bold; }")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 12px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # 4. 按钮区
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("启动系统 (Start)")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        
        self.btn_stop = QPushButton("停止系统 (Stop)")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        # 组装右侧
        control_layout.addWidget(self.status_label)
        control_layout.addSpacing(20)
        control_layout.addWidget(stats_group)
        control_layout.addWidget(log_group)
        control_layout.addLayout(btn_layout)

        # 组装整体布局 (左视频占比 70%，右控制占比 30%)
        main_layout.addWidget(video_group, 70)
        main_layout.addLayout(control_layout, 30)
        main_widget.setLayout(main_layout)

    def create_lcd(self):
        lcd = QLCDNumber()
        lcd.setDigitCount(5)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        lcd.setStyleSheet("border: none; color: #00e5ff;")
        return lcd

    def update_image(self, qt_img):
        """更新视频画面"""
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_logic(self, status, cls_name):
        """
        核心业务逻辑：处理计数、防抖动、上传supOS
        """
        current_time = time.time()
        # 简单防抖：同一目标1.5秒内不重复计数
        if current_time - self.last_process_time < 1.5:
            return

        self.last_process_time = current_time
        
        # 更新计数
        self.total_count += 1
        if status == "NG":
            self.ng_count += 1
            # UI 变红
            self.status_label.setText(f"NG\n{cls_name}")
            self.status_label.setStyleSheet("background-color: #d32f2f; color: white; border-radius: 10px; font-weight: bold;")
        else:
            self.ok_count += 1
            # UI 变绿
            self.status_label.setText(f"OK\n{cls_name}")
            self.status_label.setStyleSheet("background-color: #388e3c; color: white; border-radius: 10px; font-weight: bold;")

        # 更新 LCD 数据
        self.lcd_total.display(self.total_count)
        self.lcd_ok.display(self.ok_count)
        self.lcd_ng.display(self.ng_count)
        if self.total_count > 0:
            yield_rate = (self.ok_count / self.total_count) * 100
            self.lcd_rate.display(f"{yield_rate:.1f}")

        # 写日志并模拟上传 supOS
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] 检测: {status} ({cls_name}) | 正在上传 supOS..."
        self.log_text.append(log_msg)
        
        # TODO: 这里调用你之前写的 requests.post 到 supOS 的代码
        # threading.Thread(target=push_to_supos, args=(...)).start()

    def start_detection(self):
        self.thread = AIWorker()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_stats_signal.connect(self.update_logic)
        self.thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("RUNNING")
        self.status_label.setStyleSheet("background-color: #1976d2; color: white; border-radius: 10px;")
        self.log_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 系统启动成功，已连接 supOS 数据底座")

    def stop_detection(self):
        if hasattr(self, 'thread'):
            self.thread.stop()
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("STOPPED")
        self.status_label.setStyleSheet("background-color: #555; color: white; border-radius: 10px;")
        self.image_label.setText("检测已停止")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())