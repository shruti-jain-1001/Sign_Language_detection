import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
from difflib import get_close_matches
from collections import deque
from flask import Flask, render_template, Response, request, jsonify
import pyttsx3
import threading

app = Flask(__name__)





def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    
    
# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Load dictionary of English words
with open("words_alpha.txt", "r") as f:
    english_words = set(word.strip().lower() for word in f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

labels_dict = {i: chr(65 + i) for i in range(23)}  # A to W

# Global variables
text_buffer = ""
suggested_words = []
prediction_history = deque(maxlen=5)
last_letter_time = time.time()
last_hand_time = time.time()
spoken_words = set()


# Route for rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global text_buffer, suggested_words, prediction_history, last_letter_time, last_hand_time

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        x_, y_, data_aux = [], [], []

        if results.multi_hand_landmarks:
            last_hand_time = time.time()  # Update last hand detection time

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                if len(x_) == 21:
                    for i in range(21):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))

                    if len(data_aux) == 42:
                        input_data = np.array(data_aux).reshape(1, 21, 2, 1)
                        prediction = model.predict(input_data)
                        predicted_index = int(np.argmax(prediction))
                        predicted_letter = labels_dict.get(predicted_index, "Unknown")

                        prediction_history.append(predicted_letter)

                        # Draw bounding box
                        h, w, _ = frame.shape
                        x1, y1 = int(min(x_) * w), int(min(y_) * h)
                        x2, y2 = int(max(x_) * w), int(max(y_) * h)
                        cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 165, 0), 2)
                        cv2.putText(frame, predicted_letter, (x1 - 10, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 165, 0), 3)
        else:
            prediction_history.clear()

        # Insert space after 3 seconds of no hand detected
        if time.time() - last_hand_time > 3:
            if not text_buffer.endswith(" "):
                text_buffer += " "
            words = text_buffer.strip().split()
            if words:
                last_word = words[-1]
                if last_word not in spoken_words:
                    threading.Thread(target=speak_text, args=(last_word,), daemon=True).start()
                    spoken_words.add(last_word)
                suggested_words = get_close_matches(last_word, english_words, n=3)
            else:
                suggested_words = []
            

        # Append letter every 1.5s
        if len(prediction_history) == prediction_history.maxlen:
            majority_letter = max(set(prediction_history), key=prediction_history.count)
            if majority_letter != "Unknown" and time.time() - last_letter_time > 1.5:
                if not text_buffer.endswith(majority_letter.lower()):
                    text_buffer += majority_letter.lower()
                    last_letter_time = time.time()

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame for video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text')
def get_text():
    return jsonify({'text': text_buffer, 'suggestions': suggested_words})

@app.route('/update_text', methods=['POST'])
def update_text():
    global text_buffer, suggested_words

    # Get the word that was clicked
    new_word = request.form.get('word')

    if new_word:
        # Split the text_buffer into words and replace the last word
        words = text_buffer.strip().split()
        if words:
            words[-1] = new_word  # Replace the last word with the new word
        else:
            words.append(new_word)  # If there are no words, just append the new word


        text_buffer = " ".join(words) + " "  # Join the words back into a string

        # Clear suggestions after update
        suggested_words = []

    return jsonify({'text': text_buffer, 'suggestions': suggested_words})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
