#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import cv2
import os
import shutil
import mediapipe as mp
import numpy as np
import time
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QMessageBox, QLineEdit, QDialog
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal, QLocale


russian_locale = QLocale(QLocale.Russian, QLocale.Russia)
QLocale.setDefault(russian_locale)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
THRESHOLD_SECONDS = 5



class UserInfoWindow(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Информация о пользователе")
        self.setFixedSize(500, 500)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        
        name_label = QLabel("Имя и фамилия:", self)
        self.name_edit = QLineEdit(self)
        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)

        date_label = QLabel("Дата сессии приложения: (дд.мм.гггг)", self)
        self.date_edit = QLineEdit(self)
        layout.addWidget(date_label)
        layout.addWidget(self.date_edit)
        times_label = QLabel("Введите текущее время: (чч.мм)", self)
        self.times_edit = QLineEdit(self)
        layout.addWidget(times_label)
        layout.addWidget(self.times_edit)
        info_label = QLabel("Дополнительная информация или напишите ---", self)
        self.info_edit = QLineEdit(self)
        layout.addWidget(info_label)
        layout.addWidget(self.info_edit)

        

        save_button = QPushButton("Сохранить", self)
        save_button.clicked.connect(self.save_user_info)
        layout.addWidget(save_button)

    def save_user_info(self):
        name = self.name_edit.text()
        date = self.date_edit.text()
        info = self.date_edit.text()
        times = self.times_edit.text()
        if name and date and info and times:
            file_name = f"{name}_{date}_{times}_{info}_inform.txt"
            with open(file_name, "w") as file:
                file.write(f"Name: {name}\n")
                file.write(f"Date of App Use: {date}\n")
                file.write(f"Start time of app use: {times}\n")
                file.write(f"Additional info or put ---: {info}\n")
            
            self.main_window.total_inattention_time = 0
            self.main_window.attention_detected = True
            self.close()
            self.main_window.start_session()
            destination_folder = "your_info"
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            destination_file_name = os.path.join(destination_folder, file_name)
            shutil.move(file_name, destination_file_name)
        else:
            QMessageBox.warning(self, "Ошибка!", "Пожалуйста, заполните все поля.")
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

        self.image_label = QLabel(self)
        self.total_inattention_label = QLabel(self)
        self.attention_status_label = QLabel(self)
        self.session_time_label = QLabel(self)  # Added session time label
        self.total_inattention_time = 0
        self.inattention_start_time = None
        self.attention_detected = True
        self.session_start_time = None  # Added session start time

        self.init_ui()
        self.start_video_stream()

        self.statistics_window = None
        

        self.show_user_info_window()
        
    def calculate_inattention_duration(self):
        current_time = time.time()
        if not self.attention_detected and self.inattention_start_time is not None:
            inattention_duration = current_time - self.inattention_start_time
            if inattention_duration >= THRESHOLD_SECONDS:
                self.inattention_start_time = current_time
                return inattention_duration
        return 0
    def start_session(self):
        self.total_inattention_time = 0
        self.inattention_start_time = None
        self.attention_detected = True
        self.session_start_time = time.time()  # Store the start time of the session
        self.start_video_stream()

    def show_user_info_window(self):
        self.user_info_window = UserInfoWindow(self)
        self.user_info_window.exec_()
    def init_ui(self):
        self.setWindowTitle("Attention Checker (создан tsmrtvrts)")
        self.setFixedSize(800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Attention Checker", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)

        layout.addWidget(self.image_label)

        self.total_inattention_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.total_inattention_label)

        self.attention_status_label.setAlignment(Qt.AlignCenter)
        self.attention_status_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.attention_status_label)
        
        # Added session time label
        self.session_time_label.setAlignment(Qt.AlignCenter)
        self.session_time_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.session_time_label)
        instructions_button = QPushButton("Инструкция")
        instructions_button.clicked.connect(self.show_instructions)
        layout.addWidget(instructions_button)
        
        end_session_button = QPushButton("Завершить сессию")
        end_session_button.clicked.connect(self.end_session)
        layout.addWidget(end_session_button)
        

        help_button = QPushButton("Поддержка")
        help_button.clicked.connect(self.show_help)
        layout.addWidget(help_button)

    def check_inattention(self):
        if not self.attention_detected:
            current_time = time.time()
            if self.inattention_start_time is None:
                self.inattention_start_time = current_time
            else:
                inattention_duration = current_time - self.inattention_start_time
                if inattention_duration >= THRESHOLD_SECONDS:
                # Increment the total inattention time
                    self.total_inattention_time += inattention_duration
                    self.inattention_start_time = current_time
    

    def show_help(self):
        help_message = "Если у вас возникли проблемы при использовании приложения, свяжитесь с нами по адресу help_tsmrtvrts_creators@mail.ru"
        QMessageBox.information(self, "Поддержка", help_message)
    def show_instructions(self):
        instructions_message= "Это приложение создано tsmrtvrts.\n Сядьте в удобную позу и посмотрите на экран. Подвигайте головой и посмотрите, как работает приложение.\n Как только вы покинете поле зрения камеры, приложение не перестанет считать вас невнимательным.\n После нажатия кнопок (End Session) и (Exit) у вас сохранится ваша информация , а также данные сессии. \n Приятного использования!"
        QMessageBox.information(self, "Инструкция", instructions_message)
    def start_session(self):
        self.session_start_time = time.time()  # Store the start time of the session

    def start_video_stream(self):
        self.start_session()  # Start the session by setting the start time
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds (33 FPS)

    def update_frame(self):
        success, image = self.cap.read()
        start = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, Im in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (Im.x * img_w, Im.y * img_h)
                            nose_3d = (Im.x * img_w, Im.y * img_h, Im.z * 3000)
                        x, y = int(Im.x * img_w), int(Im.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, Im.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -8:
                    text = "Looking Left"
                elif y > 8:
                    text = "Looking Right"
                elif x < -4:
                    text = "Looking Down"
                elif x > 8:
                    text = "Looking Up"
                else:
                    text = "Forward"
                    self.attention_detected = True

                if text != "Forward":
                # Start the inattention timer if not in forward position
                    if self.attention_detected:
                        self.inattention_start_time = time.time()
                        self.attention_detected = False
                        self.attention_status_label.setText("Невнимательно")
                        self.attention_status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
                else:
                # End the inattention timer and accumulate the total inattention time
                    if self.inattention_start_time is not None:
                        inattention_end_time = time.time()
                        inattention_duration = inattention_end_time - self.inattention_start_time
                        self.total_inattention_time += inattention_duration
                        self.inattention_start_time = None
                        self.attention_status_label.setText("Внмание определено")
                        self.attention_status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                cv2.line(image, p1, p2, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=drawing_spec,
                                          connection_drawing_spec=drawing_spec)

    # In case no face is detected
        else:
            self.attention_detected = False
            self.check_inattention()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.display_image(image)
        self.update_labels()

        end = time.time()
    #print("FPS:", int(1 / (end - start)))


    def display_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_labels(self):
        # Update the total inattention label
        total_inattention_minutes = int(self.total_inattention_time / 60)
        total_inattention_seconds = int(self.total_inattention_time % 60)
        total_inattention_text = f"Общее время невнимательности: {total_inattention_minutes} минут {total_inattention_seconds} секунд"
        self.total_inattention_label.setText(total_inattention_text)

        # Update the session time label
        session_duration = time.time() - self.session_start_time
        session_minutes = int(session_duration / 60)
        session_seconds = int(session_duration % 60)
        session_time_text = f"Время сессии: {session_minutes} минут {session_seconds} секунд"
        self.session_time_label.setText(session_time_text)

    def end_session(self):
        self.timer.stop()

        self.show_statistics_window()
        #self.close()

    def show_statistics_window(self):
        if self.statistics_window is None:
            self.statistics_window = StatisticsWindow(self.total_inattention_time, self.session_start_time)
            self.statistics_window.closed.connect(self.on_statistics_window_closed)
        self.statistics_window.show()
        #self.hide() #

    def on_statistics_window_closed(self):
        self.statistics_window = None
        self.statistics_window.close()
    
    def closeEvent(self, event):
        cv2.destroyAllWindows()
        for window in QApplication.topLevelWidgets():
            window.close()
        self.close()
        #sys.exit(0)

class StatisticsWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, total_inattention_time, session_start_time):
        super().__init__()

        self.setWindowTitle("Attention Checker - Статистика")
        self.setFixedSize(800, 750)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Статистика Сессии", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Calculate the attention ratio and determine the color
        session_duration = time.time() - session_start_time
        attention_ratio = (total_inattention_time / session_duration)

        total_inattention_label = QLabel(f"Общее время невнимательности: {total_inattention_time} секунд", self)
        total_inattention_label.setAlignment(Qt.AlignCenter)
        total_inattention_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(total_inattention_label)
        
        total_inattention_ratio_label = QLabel(f"Общий процент времени невнимательности: {attention_ratio * 100}%", self)
        total_inattention_ratio_label.setAlignment(Qt.AlignCenter)
        total_inattention_ratio_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(total_inattention_ratio_label)
        
        total_info_label = QLabel(f"В папке session_data ваши диаграмма и статистика имеют значения \n _{total_inattention_time}_{attention_ratio}_\n", self)
        total_info_label.setAlignment(Qt.AlignCenter)
        total_info_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(total_info_label)
        
        if attention_ratio < 0.1:
            color = "green"
            attention_text = "Прекрасно! Вы были очень внимательны!"
        elif attention_ratio < 0.29:
            color = "yellow"
            attention_text = "Хорошо! Вы были внимательны, все в норме!"
        else:
            color = "red"
            attention_text = "Ну и ну! Вы были совсем невнимательны!"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([attention_ratio, 1 - attention_ratio], colors=[color, "lightgray"], startangle=90)
        ax.axis("equal")
        plt.legend(labels=["Невнимателен", "Внимателен"], loc="center")
        plt.tight_layout()

        diagram_path = f"diagram_{total_inattention_time}_{attention_ratio}.png"
        plt.savefig(diagram_path)

        diagram_label = QLabel(self)
        diagram_label.setPixmap(QPixmap(diagram_path))

        layout.addWidget(diagram_label)

        if color == "red":
            attention_label = QLabel(attention_text, self)
            attention_label.setAlignment(Qt.AlignCenter)
            attention_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            layout.addWidget(attention_label)
        if color == "yellow":
            attention_label = QLabel(attention_text, self)
            attention_label.setAlignment(Qt.AlignCenter)
            attention_label.setStyleSheet("font-size: 18px; font-weight: bold; color: yellow;")
            layout.addWidget(attention_label)
        if color == "green":
            attention_label = QLabel(attention_text, self)
            attention_label.setAlignment(Qt.AlignCenter)
            attention_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
            layout.addWidget(attention_label)
        destination_folder = "session_data"
        if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

        destination_diagram_file = os.path.join(destination_folder, diagram_path)
        shutil.move(diagram_path, destination_diagram_file)

        # Save the statistics to a file
        statistics_file = f"statistics_{total_inattention_time}_{attention_ratio}.txt"
        
        with open(statistics_file, "w") as file:
            file.write(f"Total inattention time: {total_inattention_time} seconds\n")
            file.write(f"Total inattention ratio: {attention_ratio * 100}%\n")
            
        destination_statistics_file = os.path.join(destination_folder, statistics_file)
        shutil.move(statistics_file, destination_statistics_file)
        
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close_application)
        layout.addWidget(exit_button)

    def close_application(self):
        cv2.destroyAllWindows()
        #QApplication.quit()
        
        for window in QApplication.topLevelWidgets():
            window.close()
        self.close()
    
    def closeEvent(self, event):
        cv2.destroyAllWindows()
        for window in QApplication.topLevelWidgets():
            window.close()
        self.close()
        #sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    #window.closeEvent = closeEvent

    sys.exit(app.exec_())


# In[1]:


### 


# In[ ]:




