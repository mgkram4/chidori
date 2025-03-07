# Cool MediaPipe Effects

A real-time computer vision application that combines MediaPipe's hand tracking and face mesh detection with cool visual effects. The application creates Sasuke Chidori-style lightning effects when making a fist gesture and applies an interesting mesh overlay on detected faces.

## Features

- Real-time hand tracking with MediaPipe
- Face mesh detection and overlay
- Dynamic lightning effects triggered by hand gestures
- Particle animation system
- FPS display and performance optimization

## Requirements

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy
- SciPy

## Installation

1. Clone the repository:
```bash
gh repo clone mgkram4/chidori
cd coolmediapipe
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy scipy
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Controls:
- Make a fist to trigger the Chidori-style lightning effect
- Press 'q' to quit the application

## Performance Notes

- The application includes performance optimization settings in the code:
  - Adjustable processing frame width
  - FPS limiting
  - Buffer size optimization
- If experiencing performance issues, you can adjust these settings in `main.py`

## License

This project is open source and available under the MIT License. # chidori
