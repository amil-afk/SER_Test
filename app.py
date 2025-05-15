from flask import Flask, request, render_template
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import io
import base64
#from replit import web

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan label encoder
model = load_model('best_model1.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocessing constants
SAMPLE_RATE = 16000
MAX_DURATION = 8
MAX_LEN = SAMPLE_RATE * MAX_DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    else:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel_spec).T
    return log_mel[np.newaxis, ..., np.newaxis]  # shape: (1, time, mel, 1)

def plot_prediction(probs, labels):
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs, color='skyblue')
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])
    ax.set_title('Prediction Confidence per Emotion')
    plt.xticks(rotation=45)

    # Tambah label angka di atas bar
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    plot_url = None
    audio_file_name = None
    if request.method == 'POST':
        audio_file = request.files['file']
        if audio_file and audio_file.filename.endswith('.wav'):
            audio_file_name = audio_file.filename
            save_path = os.path.join('static', 'audio', audio_file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            audio_file.save(save_path)

            try:
                features = preprocess_audio(save_path)
                probs = model.predict(features)[0]
                pred_idx = np.argmax(probs)
                prediction = label_encoder.inverse_transform([pred_idx])[0]
                confidence = round(float(probs[pred_idx]) * 100, 2)
                plot_url = plot_prediction(probs, label_encoder.classes_)
            except Exception as e:
                prediction = f"Error: {e}"
                confidence = None

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        plot_url=plot_url,
        audio_file=audio_file_name
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
    #web.run(app) #to run in replit

