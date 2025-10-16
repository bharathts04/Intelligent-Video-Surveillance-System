# 🧠 Intelligent Video Surveillance System for Alzheimer’s Patients

### 🎯 Overview
This project aims to assist caregivers by providing a **real-time video surveillance system** that detects **falls, boundary crossing, and suspicious activities** of Alzheimer’s patients using **AI-based video analysis**.

---

### 🧩 Features
- 👀 **Fall Detection:** Identifies when a patient collapses and immediately raises an alert.  
- 🚧 **Boundary Crossing Detection:** Detects when a patient moves beyond a safe zone.  
- 🚨 **Alert Generation:** Sends automatic **email or SMS alerts** to caregivers with timestamp and frame capture.  
- 📹 **Real-Time Video Processing:** Uses YOLO model and OpenCV for frame analysis.

---

### 🧰 Tech Stack
| Component | Technology Used |
|------------|----------------|
| Programming Language | Python |
| AI Model | YOLOv8 |
| Libraries | OpenCV, NumPy, smtplib, ultralytics |
| Frontend (optional) | JSP, HTML, CSS, JavaScript |
| Database | MySQL |
| Environment | Jupyter / VS Code |

---

### ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Intelligent-Video-Surveillance-System.git
   cd Intelligent-Video-Surveillance-System
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
4. **Run the Application**
    ```bash
    python Fall_Detect.py

---

### 🧩 System Workflow

1. Input CCTV footage or live camera feed.
2. YOLO detects humans and tracks movements.
3. Custom logic classifies falls or boundary crossings.
4. If detected, the system captures the frame and triggers alert notifications.

---

### 🧱 Architecture Diagram

![Architecture Diagram](docs/Workflow.png)


---

### 📊 Results

- Detection accuracy: 92% for falls, 90% for boundary crossing.
- Average alert delay: < 2 seconds
- Real-time performance achieved at 30 FPS on test datasets.

![Architecture Diagram](docs/Result.png)

---

### 📸 Screenshots

![Architecture Diagram](docs/OP-01.jpg)

![Architecture Diagram](docs/OP-02.jpg)

![Architecture Diagram](docs/OP-03.jpg)

![Architecture Diagram](docs/OP-04.jpg)

---
### 👨‍💻 Author

Bharath TS
🎓 B.E. Computer Science & Engineering | DBIT, Bengaluru
📧 bharathamurthy@gmail.com

📅 2026 Batch

---

### 💡 Future Enhancements

- Integration with IoT sensors for room monitoring.
- Voice-based alerting system for caregivers.
- Integration with mobile app for push notifications.
