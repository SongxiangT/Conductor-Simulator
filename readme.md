# Conductor Simulator

![Conductor Simulator Icon](./path-to-your-icon.png) 

A Python-based conductor simulator that uses hand gestures to control music playback tempo and volume. Leveraging computer vision and machine learning, this tool offers an interactive experience for users to conduct music in real-time.

## 🎶 Features

- **GUI Setup**: User-friendly interface to select music files and input BPM.
- **Hand Gesture Recognition**: Utilizes MediaPipe for real-time hand tracking.
- **Music Control**:
  - **Pause/Play**: Toggle music playback with a "Pause" gesture.
  - **Volume Control**: Adjust volume based on hand extensivity.
  - **Tempo Control**: Modify playback speed using tempo control gestures.
- **Visualization**: Real-time graphs displaying volume and tempo changes.

## 🎶Instructions for Playing
### Terminology in Music**<br>
**BPM (Beats Per Minute):**
BPM is a measurement of the speed of a musical piece. It tells how many beats occur in one minute. For example, a BPM of 120 means there are 120 beats in one minute, which would be a moderate tempo.<br>

**Volume:** 
Volume is the loudness or softness of the music. In this app, you can control the music's volume by adjusting the position of your hand.<br>

**Cue:**
A cue in music is a signal for a specific action to happen. In this app, cues are gestures that control the Pause and Continue functions. The Pause Cue stops the music, while the Continue Cue resumes playback.<br>

**Tempo:**
Tempo refers to the speed or pace of the music. It is controlled by adjusting the BPM. Higher BPM means faster music, and lower BPM means slower music.
<br>

### Controls
**Left Hand: Volume Control:**

Use your left hand to adjust the volume. The further your fingertips are from your palm (i.e., greater extensivity), the higher the volume.
Closing your hand reduces the volume, and opening it increases the volume.<br>
**Left Hand: Tempo Control:** 

Moving your left hand up and down controls the tempo. Quick, sharp movements are detected as BPM, adjusting the speed of the music accordingly.
Slow movements will lower the BPM, while rapid movements will increase it.<br>
**Right Hand**:
*Cue Control (Pause and Continue):*<br>
The right hand is used to trigger cue gestures.<br>
For a Pause Cue, extend all fingers and hold them steady.<br>
For a Continue Cue, move your right hand in a waving motion to resume the music.<br>
Note: The Pause Cue is sensitive and may result in false detections, so it’s best to keep movements deliberate and clear.<br>



## 🎶 Pipline Overview
![8ce2380aa3429793821979631b7d51c](https://github.com/user-attachments/assets/105b2e90-f5f7-42a8-8e13-cc0b362a2154)


## 🛠️ Installation

### Prerequisites

- **Python 3.7 or higher** installed on your system.
- **VLC Media Player:** Ensure VLC is installed as it's used for audio playback.


1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/orchestra-conductor.git
   cd orchestra-conductor
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```
4. **Ensure VLC is Installed**

Download and install VLC Media Player appropriate for your operating system.



## 🎶Usage
1. **Run the GUI**
```bash
python gui.py
```

2. **Setup**

Select Music File: Click the "Browse" button to choose your desired music file (.mp3 or .wav).
Enter BPM: BPM is a very basic concept in music, means how fast the music is. Input the BPM (Beats Per Minute) of your selected music. 
Start: Click the "Start" button to launch the application.<br>
<img width="387" alt="a8edfddfaf16287b10a77dad8a1760e" src="https://github.com/user-attachments/assets/09070ced-d046-4e14-86a7-5bbfb0fa3373"> <br>

3. **Control Music with Gestures**
When the camera is initialize, the play should present both hand in the camera like this: <br>
 <img width="296" alt="43a1374db124b6cdc72fb8f5cb1b68d" src="https://github.com/user-attachments/assets/f8ac9a0c-0b21-45f2-94ec-470487a22045"><br>

Pause/Play: Show a "Pause" posture (all five fingers extended and close together) to toggle music playback. <br>
Volume Control: Adjust the volume by changing the extensivity of your left hand's fingers.<br>
Tempo Control: Perform tempo control gestures with your left hand to adjust the playback rate.<br>

4. **Exit**

Press the ESC key in the video window to stop the music and close all application windows.




## 📷 Demo
<img width="591" alt="6fe304d963ad3c3a8b339d342864411" src="https://github.com/user-attachments/assets/731f9448-5c23-4bcb-8616-8f28192f1839">

https://youtu.be/tgY-Cf44HLk?si=veikNqNHiWVGvR_0


## 🤝 Contributing
Contributions are welcome! Please follow these steps:


📞 Contact
Songxiang TANG
Email: songxiangtang@gmail.com
GitHub: SongxiangT



## 📂Error Warnings
 1. **Initialization Error**:

At the initialization of the app, please make sure both hands are inside the camera’s field of view. If not, you might encounter the following error:
```bash

Error: local variable 'cue_detected' referenced before assignment
```
This error can occur because the system is unable to detect the hands during startup.
2. **Cue Detection Issue**:

The Cue Detection currently only supports two types of cues: the Pause Cue and the Continue Cue. The Pause Cue can sometimes be falsely detected due to the sensitivity of the gesture recognition. It’s recommended to use large, distinct gestures to avoid this.

