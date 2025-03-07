# import the necessary libraries
import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d  # For smooth animations

# initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh  # Changed from face_detection to face_mesh

# Setup webcam with increased buffer size and resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Smaller buffer for less latency

# Performance settings
PROCESS_FRAME_WIDTH = 640  # Process at lower resolution for speed
DISPLAY_ORIGINAL_SIZE = True  # But display at original size
LIMIT_FRAMERATE = True
TARGET_FPS = 30

# Animation library and parameters - DEFINE CLASSES FIRST
class AnimationSystem:
    def __init__(self):
        self.effects = []
        self.max_effects = 300  # Reduced for performance
        
    def update(self, frame):
        # Update all effects and remove expired ones
        for effect in self.effects[:]:
            effect.update()
            if effect.is_expired():
                self.effects.remove(effect)
            else:
                effect.draw(frame)
    
    def add_effect(self, effect):
        if len(self.effects) < self.max_effects:
            self.effects.append(effect)
    
    def add_burst(self, x, y, effect_type="particle", count=20, **kwargs):
        for _ in range(count):
            if effect_type == "particle":
                self.add_effect(Particle(x, y, **kwargs))
            elif effect_type == "trail":
                self.add_effect(TrailEffect(x, y, **kwargs))
            elif effect_type == "energy":
                self.add_effect(EnergyBall(x, y, **kwargs))
            elif effect_type == "lightning":
                self.add_effect(LightningEffect(x, y, **kwargs))

# Base class for all visual effects
class Effect:
    def __init__(self, x, y, lifetime=30):
        self.x = x
        self.y = y
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.alpha = 1.0
    
    def update(self):
        self.lifetime -= 1
        # Update alpha for fade-out effect
        self.alpha = self.lifetime / self.max_lifetime
    
    def is_expired(self):
        return self.lifetime <= 0
    
    def draw(self, frame):
        pass  # Implemented by subclasses

# Enhanced particle effect with more customization
class Particle(Effect):
    def __init__(self, x, y, lifetime=30, size_range=(3, 8), 
                 velocity_range=(-3, 3), color=None, gravity=0.05):
        super().__init__(x, y, lifetime)
        self.vx = random.uniform(velocity_range[0], velocity_range[1])
        self.vy = random.uniform(velocity_range[0], velocity_range[1])
        self.size = random.randint(size_range[0], size_range[1])
        self.gravity = gravity
        
        if color is None:
            # Generate vibrant colors by using HSV
            hue = random.randint(0, 360)
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self.color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        else:
            self.color = color
            
        # Create easing function for size
        self.size_start = self.size
        self.size_end = max(1, self.size // 2)
        
    def update(self):
        super().update()
        self.x += self.vx
        self.y += self.vy
        # Add gravity effect
        self.vy += self.gravity
        # Calculate current size with easing
        progress = 1 - (self.lifetime / self.max_lifetime)
        self.current_size = int(self.size_start + (self.size_end - self.size_start) * progress)
        
    def draw(self, frame):
        # Apply alpha to color
        alpha_color = (
            int(self.color[0] * self.alpha),
            int(self.color[1] * self.alpha),
            int(self.color[2] * self.alpha)
        )
        cv2.circle(frame, (int(self.x), int(self.y)), self.current_size, alpha_color, -1)

# Trail effect that creates smooth paths
class TrailEffect(Effect):
    def __init__(self, x, y, lifetime=20, length=10, color=None, width=3):
        super().__init__(x, y, lifetime)
        self.points = [(x, y)]
        self.max_length = length
        self.width = width
        
        if color is None:
            # Blue-purple glow
            self.color = (255, 0, 200)
        else:
            self.color = color
    
    def update(self):
        super().update()
        # Add some random movement
        new_x = self.points[-1][0] + random.randint(-5, 5)
        new_y = self.points[-1][1] + random.randint(-5, 5)
        self.points.append((new_x, new_y))
        
        # Keep only the most recent points
        if len(self.points) > self.max_length:
            self.points.pop(0)
    
    def draw(self, frame):
        if len(self.points) < 2:
            return
            
        # Draw smooth trail with varying transparency
        for i in range(len(self.points) - 1):
            progress = i / (len(self.points) - 1)
            alpha = progress * self.alpha
            color = (
                int(self.color[0] * alpha),
                int(self.color[1] * alpha),
                int(self.color[2] * alpha)
            )
            width = int(self.width * alpha) + 1
            cv2.line(frame, self.points[i], self.points[i+1], color, width)

# Energy ball effect for powerful gestures
class EnergyBall(Effect):
    def __init__(self, x, y, lifetime=40, radius_max=60, color=None):
        super().__init__(x, y, lifetime)
        self.radius_max = radius_max
        
        if color is None:
            # Cyan with glow
            self.color = (255, 255, 0)  # Yellow
            self.inner_color = (0, 255, 255)  # Cyan
        else:
            self.color = color
            self.inner_color = color
    
    def update(self):
        super().update()
    
    def draw(self, frame):
        # Calculate current radius with pulse effect
        progress = 1 - (self.lifetime / self.max_lifetime)
        # Add pulsing with sine wave
        pulse = 0.5 + 0.5 * math.sin(progress * 6 * math.pi)
        radius = int(self.radius_max * (0.5 + 0.5 * pulse) * self.alpha)
        
        # Draw outer glow
        for r in range(radius, max(0, radius - 20), -2):
            alpha = (r / radius) * self.alpha
            color = (
                int(self.color[0] * alpha),
                int(self.color[1] * alpha),
                int(self.color[2] * alpha)
            )
            cv2.circle(frame, (int(self.x), int(self.y)), r, color, 1)
        
        # Draw inner circle
        inner_radius = max(1, int(radius * 0.3))
        cv2.circle(frame, (int(self.x), int(self.y)), inner_radius, self.inner_color, -1)

# Lightning effect for dramatic impact
class LightningEffect(Effect):
    def __init__(self, x, y, lifetime=15, length=100, color=None, branches=3):
        super().__init__(x, y, lifetime)
        self.length = length
        self.branches = branches
        
        if color is None:
            self.color = (255, 255, 255)  # White lightning
        else:
            self.color = color
            
        # Generate lightning path
        self.generate_path()
    
    def generate_path(self):
        self.paths = []
        # Main path
        main_path = [(self.x, self.y)]
        end_x = self.x + random.randint(-self.length, self.length)
        end_y = self.y + random.randint(-self.length, self.length)
        
        # Add some zigzag points
        segments = random.randint(3, 6)
        for i in range(segments):
            progress = (i + 1) / segments
            mid_x = self.x + (end_x - self.x) * progress
            mid_y = self.y + (end_y - self.y) * progress
            # Add randomness
            mid_x += random.randint(-30, 30)
            mid_y += random.randint(-30, 30)
            main_path.append((mid_x, mid_y))
        
        self.paths.append(main_path)
        
        # Add branches
        for _ in range(self.branches):
            if len(main_path) < 2:
                continue
                
            # Start from a random point on the main path
            start_idx = random.randint(0, len(main_path) - 2)
            branch = [main_path[start_idx]]
            
            # Create branch
            branch_length = random.randint(2, 4)
            start_x, start_y = main_path[start_idx]
            end_x = start_x + random.randint(-50, 50)
            end_y = start_y + random.randint(-50, 50)
            
            for i in range(branch_length):
                progress = (i + 1) / branch_length
                mid_x = start_x + (end_x - start_x) * progress
                mid_y = start_y + (end_y - start_y) * progress
                # Add randomness
                mid_x += random.randint(-15, 15)
                mid_y += random.randint(-15, 15)
                branch.append((mid_x, mid_y))
                
            self.paths.append(branch)
    
    def update(self):
        super().update()
    
    def draw(self, frame):
        # Draw all paths
        for path in self.paths:
            for i in range(len(path) - 1):
                # Add flicker effect
                flicker = random.uniform(0.7, 1.0)
                alpha = self.alpha * flicker
                color = (
                    int(self.color[0] * alpha),
                    int(self.color[1] * alpha),
                    int(self.color[2] * alpha)
                )
                cv2.line(frame, 
                         (int(path[i][0]), int(path[i][1])), 
                         (int(path[i+1][0]), int(path[i+1][1])), 
                         color, 
                         random.randint(1, 3))

# Function to detect if hand is making a fist
def is_fist(hand_landmarks):
    # Check if fingers are bent (simplified)
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    finger_pips = [6, 10, 14, 18]  # Corresponding middle joints
    
    # Check if fingertips are below their middle joints (bent fingers)
    bent_fingers = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            bent_fingers += 1
    
    # If most fingers are bent, it's likely a fist
    return bent_fingers >= 3

# NOW create the animation system instance AFTER classes are defined
animation_system = AnimationSystem()

# Main processing loop
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, 
                   min_tracking_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, 
                          refine_landmarks=False) as face_mesh:
    
    prev_frame_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # FPS control
        if LIMIT_FRAMERATE:
            curr_time = time.time()
            time_elapsed = curr_time - prev_frame_time
            if time_elapsed < 1.0/TARGET_FPS:
                continue
            prev_frame_time = curr_time
        
        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Resize for faster processing
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        if PROCESS_FRAME_WIDTH < w:
            scale = PROCESS_FRAME_WIDTH / w
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        # To improve performance, optionally mark the image as not writeable
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        hand_results = hands.process(frame)
        
        # Detect face mesh landmarks
        face_results = face_mesh.process(frame)
        
        # Draw the annotations on the image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Update and draw particles using animation system
        animation_system.update(frame)
        
        # Process hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get wrist position for animation center
                wrist = hand_landmarks.landmark[0]
                wrist_x = int(wrist.x * frame.shape[1])
                wrist_y = int(wrist.y * frame.shape[0])
                
                # Get fingertip positions for better animations
                fingertips = []
                for tip_id in [4, 8, 12, 16, 20]:  # Thumb and fingers
                    tip_x = int(hand_landmarks.landmark[tip_id].x * frame.shape[1])
                    tip_y = int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
                    fingertips.append((tip_x, tip_y))
                
                # Get palm center
                palm_center_x = sum(int(hand_landmarks.landmark[i].x * frame.shape[1]) for i in [0, 5, 9, 13, 17]) // 5
                palm_center_y = sum(int(hand_landmarks.landmark[i].y * frame.shape[0]) for i in [0, 5, 9, 13, 17]) // 5
                
                # Check if making a fist for Sasuke Chidori animation
                is_making_fist = is_fist(hand_landmarks)
                
                # CONSISTENT COLORS - Sasuke Chidori theme (more transparent)
                # Electric blue for lightning
                lightning_color = (180, 120, 0)  # Less intense orange-yellow in BGR
                # Light blue for the glow
                glow_color = (60, 0, 0)  # Lighter blue in BGR
                # Electric effects
                energy_color = (200, 200, 255)  # Light blue-white
                
                # Set flame parameters based on hand gesture
                if is_making_fist:
                    # SASUKE CHIDORI ACTIVATION! (toned down)
                    flame_lifetime = 30
                    flame_length = 20
                    flame_width = (4, 10)  # Thinner flames
                    flame_count = 2   # Fewer trails per finger
                    glow_alpha = 0.3  # Less intense glow
                    
                    # Add concentrated lightning in the palm (Chidori core) - fewer effects
                    for _ in range(4):  # Reduced from 10
                        # Lightning effects from palm center
                        start_x = palm_center_x + random.randint(-10, 10)
                        start_y = palm_center_y + random.randint(-10, 10)
                        
                        # Create lightning extending outward from palm
                        animation_system.add_effect(LightningEffect(
                            start_x, start_y,
                            lifetime=8,
                            length=random.randint(30, 60),  # Shorter lightning
                            color=lightning_color,
                            branches=3  # Fewer branches
                        ))
                    
                    # Create smaller energy ball in palm (Chidori orb)
                    animation_system.add_effect(EnergyBall(
                        palm_center_x, palm_center_y,
                        lifetime=15,
                        radius_max=20,  # Smaller energy ball
                        color=lightning_color
                    ))
                    
                    # SASUKE STYLE: Add lightning up the arm (fewer)
                    for _ in range(2):  # Reduced from 5
                        offset_x = random.randint(-10, 10)
                        offset_y = random.randint(-10, 10)
                        
                        # Calculate arm direction
                        arm_direction_x = wrist_x - palm_center_x
                        arm_direction_y = wrist_y - palm_center_y
                        
                        # Extend the lightning up the arm (shorter)
                        end_x = wrist_x + arm_direction_x * 2 + offset_x
                        end_y = wrist_y + arm_direction_y * 2 + offset_y
                        
                        animation_system.add_effect(LightningEffect(
                            wrist_x, wrist_y,
                            lifetime=10,
                            length=int(math.hypot(end_x-wrist_x, end_y-wrist_y)),
                            color=lightning_color,
                            branches=2
                        ))
                    
                    # Add crackling lightning between fingertips (reduced)
                    for i in range(len(fingertips)):
                        for j in range(i+1, len(fingertips)):
                            if random.random() > 0.8:  # Less frequent connections (0.6 -> 0.8)
                                animation_system.add_effect(LightningEffect(
                                    fingertips[i][0], fingertips[i][1],
                                    lifetime=8,
                                    length=int(math.hypot(fingertips[i][0]-fingertips[j][0], 
                                                     fingertips[i][1]-fingertips[j][1])),
                                    color=lightning_color,
                                    branches=1  # Fewer branches
                                ))
                    
                    # Add fewer particles from palm
                    animation_system.add_burst(
                        palm_center_x, palm_center_y,
                        effect_type="particle",
                        count=20,  # Reduced from 60
                        lifetime=15,
                        size_range=(2, 5),  # Smaller particles
                        velocity_range=(-5, 5),  # Less movement
                        color=energy_color,
                        gravity=0.05
                    )
                    
                    # Just one subtle pulsing circle
                    t = time.time() * 8
                    radius = 30 + int(5 * math.sin(t))
                    cv2.circle(frame, (palm_center_x, palm_center_y), radius, 
                              lightning_color, 1)
                else:
                    # Normal mode - very subtle blue glow
                    flame_lifetime = 15
                    flame_length = 10
                    flame_width = (2, 4)
                    flame_count = 1
                    glow_alpha = 0.1
                
                # Consistent flame color for fingers (blue flame trails) - fewer and more transparent
                for i, fingertip in enumerate(fingertips):
                    # Create trails per finger (reduced)
                    for _ in range(flame_count):
                        animation_system.add_effect(TrailEffect(
                            fingertip[0] + random.randint(-3, 3),  # Less random movement
                            fingertip[1] + random.randint(-3, 3),
                            lifetime=flame_lifetime, 
                            length=flame_length, 
                            width=random.randint(flame_width[0], flame_width[1]),
                            color=lightning_color
                        ))
                    
                    # Add occasional sparkling particles at fingertips
                    if random.random() > 0.6:  # Less frequent
                        animation_system.add_effect(Particle(
                            fingertip[0], 
                            fingertip[1],
                            lifetime=flame_lifetime - 5,
                            size_range=(2, 4),  # Smaller particles
                            velocity_range=(-3, 3),  # Less movement
                            color=energy_color,
                            gravity=0
                        ))
        
        # Process face with completely filled mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Get face bounding box from landmarks
                face_points = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    face_points.append((x, y))
                
                # Convert to numpy array for convex hull
                face_points_np = np.array(face_points, dtype=np.int32)
                hull = cv2.convexHull(face_points_np)
                
                # Create a solid black mask over the face
                mask = np.zeros_like(frame)
                cv2.fillConvexPoly(mask, hull, (0, 0, 0))  # Simple black mask
                
                # Apply the mask with complete opacity
                alpha = 1.0  # 100% opacity for completely black mask
                mask_area = (mask > 0)
                frame[mask_area] = cv2.addWeighted(frame, 1-alpha, mask, alpha, 0)[mask_area]
                
                # Draw all MediaPipe face landmarks and connections with brighter lines
                # Use custom drawing spec for more visible landmarks and connections
                landmark_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                
                # Draw all possible connections for a more filled-in look
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )
                
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_spec
                )
                
                # Draw lips
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 160, 255), thickness=2)
                )
                
                # Add contour to the face mask
                cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        
        # If needed, scale back to original size for display
        if DISPLAY_ORIGINAL_SIZE and PROCESS_FRAME_WIDTH < w:
            frame = cv2.resize(frame, (w, h))
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('MediaPipe Hands and Face Mesh', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

