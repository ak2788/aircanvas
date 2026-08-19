# AirCanvas: Gesture-Controlled Creative Capture System :D
## Project track
Computer Vision

## What the project does
AirCanvas is a real-time creative camera application controlled through hand gestures. The system uses a webcam to track a user's hand, recognize gestures, and trigger creative camera actions without using keyboard or mouse controls during normal interaction.

Immersive Gestures🔥:
- 👈👉  Swipe left/right → cycle through 5 camera modes (Photo, Portrait, Artistic, B&W, Cool)
- 👍 Thumbs up → 3-second countdown & then captures a photo
- ✋🔺 Open palm + triangle → captures a photo & randomly applies one of 7 artistic filters
- ✋◽ Open palm + square → captures a photo & splits the scene into a draggable 4×4 grid
- ✌️ Peace sign → opens a sticker menu (heart, smile, star, shark)
- 🤏 Pinch + drag → move stickers or photo pieces anywhere you like
- 🤏🗑️ Drag to trash can → delete stickers or photo pieces
- 🤏🧹 Pinch + hold Clear button → reset the entire app
- 🤏➕➖ Pinch + hold +/- buttons → zoom in and out

Captured photos are saved automatically into project folders created by the program.

## Project Demos

<br>

<img width="60%" height="auto" alt="AirCanva Demo 1" src="https://github.com/user-attachments/assets/4d720759-2803-4dea-bc44-6e8951813dd2" />
<br>Demo Video 1: Demonstrating swipe gesture (change camera mode)

<br><br>

<img width="60%" height="auto" alt="AirCanva Demo 2" src="https://github.com/user-attachments/assets/0bfffb2d-4beb-4cee-af22-857d32374f5d" />
<br>Demo Video 2: Demonstrating thumbs up gestures (capture photo)

<br><br>

<img width="60%" height="auto" alt="AirCanvas Demo 3" src="https://github.com/user-attachments/assets/d6bce0e2-b3e0-412e-888f-99982de11772" />
<br>Demo Video 3: Demonstrating zoom in, zoom out, and clear buttons

<br><br>

<img width="60%" height="auto" alt="AirCanvas Demo 4" src="https://github.com/user-attachments/assets/890a1858-783d-4daa-8cea-86ecd14afb6b" />
<br>Demo Video 4: Demonstrating triangle gesture (random artistic filter)

<br><br>

<img width="60%" height="auto" alt="AirCanvas Demo 5 (2)" src="https://github.com/user-attachments/assets/0be5e358-bab5-406f-987a-6ef826c02f25" />
<br>Demo Video 5: Demonstrating peace sign gesture (sticker menu) and delete button

<br><br>

<img width="60%" height="auto" alt="AirCanvas Demo 6" src="https://github.com/user-attachments/assets/495be895-d3d2-46c8-b5c3-e03515f75134" />
<br>Demo Video 6: Demonstrating the square gesture (image segmentation), delete button, and thumbs up gesture

<br>
GIFs created with Canva


## How to run the project
1. Make sure Python is installed.
2. Open a terminal or command prompt in the project folder.
3. Create and activate a virtual environment.

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the integrated project file:
   ```bash
   python aircanvas_integrated.py
   ```

6. When the program opens, review the guide window for the gesture instructions.
7. Press `Q` in the camera window to quit the application.

## Required tools, libraries, software, and materials
Software and libraries:
- Python 3
- OpenCV
- MediaPipe
- NumPy

Hardware/materials:
- Laptop or desktop computer
- Built-in webcam or USB webcam
- Good lighting for reliable hand detection

The required Python packages are listed in `requirements.txt`:
```txt
opencv-python
numpy
mediapipe==0.10.14
```

## Files included
- `aircanvas_integrated.py` — Final integrated Python program containing the full AirCanvas system, including camera modes, gesture recognition, photo capture, random filters, stickers, zoom controls, trash/delete behavior, clear button, and movable photo pieces.
- `requirements.txt` — List of Python libraries required to run the project.
- `README.md` — Short project guide explaining what the project does, how to run it, required tools, included files, and limitations.

The program also creates these output folders automatically when run:
- `captured_photos/` — Stores photos captured using the thumbs-up gesture.
- `triangle_random_filtered_photos/` — Stores photos captured with random filters after the triangle gesture.

## Limitations and special instructions
- The project requires access to a working webcam.
- Gesture detection works best in bright, even lighting with the hand clearly visible.
- Only one hand is tracked at a time.
- Busy backgrounds, fast hand movement, or partially hidden fingers may reduce detection accuracy.
- The system depends on MediaPipe hand landmark detection, so accuracy may vary depending on camera quality and lighting.
- If the webcam does not open, make sure no other application is using the camera.
- Press `Q` in the main camera window to close the application properly.
- If gestures feel too sensitive or not sensitive enough, the timing and distance thresholds in `aircanvas_integrated.py` can be adjusted.
