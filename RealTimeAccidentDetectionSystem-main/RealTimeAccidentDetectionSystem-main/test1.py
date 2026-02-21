import os
import cv2
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras import backend as K
from keras.models import load_model
from tensorflow.keras.applications import EfficientNetB1

# ---------------------------
# FORCE channels_last (must be before model creation)
# ---------------------------
K.set_image_data_format('channels_last')

# ---------------------------
# USER VARIABLES
# ---------------------------
VIDEO_PATH = "data/raw/AuQz_-J2kzc.mp4"
MODEL_WEIGHTS = "transformer_temporal_head.keras"   # transformer-only head
SEQ_LEN = 48
TARGET_FPS = 16   # None = use original fps
THRESH = 0.5        # overlay threshold
IMG_SIZE = 224      # must match training

PRED_EVERY = 3        # run transformer every 3rd sampled frame
DISPLAY_STRIDE = 2    # show only every 2nd raw frame


# ---------------------------
# Build backbone (RGB, 3 channels, ImageNet weights)
# ---------------------------
try:
    backbone = EfficientNetB1(
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),  # 3 channels
        weights="imagenet"
    )
    # sanity check
    print("Backbone built. input_shape:", backbone.input_shape)
    print("Backbone data_format:", K.image_data_format())
except Exception as e:
    print("ERROR building EfficientNetB1. If this repeats, delete the local efficientnetb1_notop.h5 and retry.")
    print("Full error:")
    raise

feat_dim = backbone.output_shape[-1]
print("Backbone feat_dim =", feat_dim)

# ---------------------------
# Load transformer model (expects (None, SEQ_LEN, feat_dim))
# ---------------------------
model = load_model(MODEL_WEIGHTS, compile=False)
try:
    print("Loaded transformer model. input_shape:", model.input_shape)
except Exception:
    print("Loaded transformer model (could not read input_shape)")

# ---------------------------
# Preprocessing helper
# (MATCH TRAINING: scale to [-1, 1])
# ---------------------------
def preprocess_frame_for_backbone(bgr_frame):
    """
    Resize, BGR->RGB, float32, then normalize to [-1, 1]
    to match the training pipeline.
    """
    img = cv2.resize(bgr_frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype("float32")
    arr = (arr / 127.5) - 1.0    # SAME AS TRAINING
    return arr

# ---------------------------
# Run inference on video
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
target_fps = float(orig_fps if TARGET_FPS is None else TARGET_FPS)
step = max(1, int(round(orig_fps / target_fps)))
print(f"Video FPS: {orig_fps:.2f}  -> sampling step={step} (target_fps={target_fps})")

feat_buffer = []          # will hold feat vectors (SEQ_LEN, feat_dim)
time_buffer = []          # corresponding timestamps
frame_batch_for_backbone = []
batch_size_backbone = 1   # keep 1 for simplicity

last_prob = 0.0
frame_idx = 0
sampled_count = 0
t0_global = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # sample for target fps
        if frame_idx % step == 0:
            sampled_count += 1

            # preprocess for backbone
            arr = preprocess_frame_for_backbone(frame)
            frame_batch_for_backbone.append(arr)

            # run backbone once we have a micro-batch
            if len(frame_batch_for_backbone) >= batch_size_backbone:
                batch_arr = np.stack(frame_batch_for_backbone, axis=0)  # (B, H, W, 3)
                feats = backbone.predict(batch_arr, verbose=0)          # (B, feat_dim)

                for i, f in enumerate(feats):
                    feat_buffer.append(f.astype(np.float32))
                    ts = cur_time - (len(feats) - 1 - i) * (1.0 / target_fps)
                    time_buffer.append(ts)
                    if len(feat_buffer) > SEQ_LEN:
                        feat_buffer.pop(0)
                        time_buffer.pop(0)

                frame_batch_for_backbone = []

            # when buffer is full, run transformer
            '''if len(feat_buffer) == SEQ_LEN:
                X = np.array(feat_buffer, dtype=np.float32)[None, ...]  # (1, SEQ_LEN, feat_dim)
                try:
                    prob = float(model.predict(X, verbose=0)[0, 0])
                except Exception as e:
                    print(f"Error during model.predict. Expected (None, {SEQ_LEN}, {feat_dim}).")
                    raise
                last_prob = prob
                center_time = time_buffer[len(time_buffer)//2] if time_buffer else cur_time
                if sampled_count % 100 == 0:
                    print(f"[{sampled_count}] vt={center_time:.2f}s prob={prob:.3f}")'''
            
            # when buffer is full, run transformer but not on every sampled frame
            if len(feat_buffer) == SEQ_LEN and (sampled_count % PRED_EVERY == 0):
                X = np.array(feat_buffer, dtype=np.float32)[None, ...]  # shape (1, SEQ_LEN, feat_dim)
                try:
                    prob = float(model.predict(X, verbose=0)[0,0])
                except Exception as e:
                    print("Error during model.predict. Check that transformer expects (None, SEQ_LEN, %d)." % feat_dim)
                    raise
                last_prob = prob
                center_time = time_buffer[len(time_buffer)//2] if time_buffer else cur_time
                if sampled_count % 100 == 0:
                    print(f"[{sampled_count}] vt={center_time:.2f}s prob={prob:.3f}")



        # draw overlay
        disp = frame.copy()
        color = (0, 0, 255) if last_prob >= THRESH else (0, 255, 0)
        txt = f"AccidentProb: {last_prob:.3f}"
        cv2.rectangle(disp, (10, 10), (360, 90), color, 2)
        cv2.putText(disp, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        '''cv2.imshow("Accident Detection", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1'''

        if frame_idx % DISPLAY_STRIDE == 0:
            cv2.imshow("Accident Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1


finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Done. total sampled frames:", sampled_count, "elapsed:", time.time() - t0_global)
