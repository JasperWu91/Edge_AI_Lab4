import os
import sys
import argparse
import glob
import time
import json
from datetime import datetime

import cv2
import numpy as np
import ollama
from ultralytics import YOLO

def main():
    # -------------------------------
    # Parse command-line arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Path to YOLO model file (e.g., runs/detect/train/weights/best.pt)')
    parser.add_argument('--source', required=True,
                        help='Image source: file, folder, video, usb camera (e.g., usb0)')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--resolution', default=None,
                        help='Display/record resolution WxH (e.g., 640x480)')
    parser.add_argument('--record', action='store_true',
                        help='Record detection video to demo1.avi (requires --resolution)')
    args = parser.parse_args()

    model_path = args.model
    source_arg = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record_video = args.record

    # -------------------------------
    # Validate model path
    # -------------------------------
    if not os.path.exists(model_path):
        print(f"ERROR: model file '{model_path}' not found")
        sys.exit(1)

    # -------------------------------
    # Load YOLO detection model
    # -------------------------------
    model = YOLO(model_path, task='detect')
    labels = model.names

    # -------------------------------
    # Determine source type
    # -------------------------------
    img_exts = {'.jpg','.jpeg','.png','.bmp'}
    vid_exts = {'.avi','.mov','.mp4','.mkv','.wmv'}

    if os.path.isdir(source_arg):
        source_type = 'folder'
    elif os.path.isfile(source_arg):
        ext = os.path.splitext(source_arg)[1].lower()
        if ext in img_exts:
            source_type = 'image'
        elif ext in vid_exts:
            source_type = 'video'
        else:
            print(f"ERROR: unsupported file extension '{ext}'")
            sys.exit(1)
    elif source_arg.startswith('usb'):
        source_type = 'usb'
        usb_idx = int(source_arg[3:])
    elif source_arg.startswith('picamera'):
        source_type = 'picamera'
        picam_idx = int(source_arg[8:])
    else:
        print(f"ERROR: invalid source '{source_arg}'")
        sys.exit(1)

    # -------------------------------
    # Parse resolution if given
    # -------------------------------
    resize = False
    if user_res:
        try:
            resW, resH = map(int, user_res.split('x'))
            resize = True
        except:
            print("ERROR: resolution must be in WxH format, e.g., 640x480")
            sys.exit(1)

    # -------------------------------
    # Setup video recording if requested
    # -------------------------------
    if record_video:
        if source_type not in ['video','usb']:
            print("ERROR: recording only works for video or camera sources")
            sys.exit(1)
        if not resize:
            print("ERROR: please specify --resolution when using --record")
            sys.exit(1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        recorder = cv2.VideoWriter('demo1.avi', fourcc, 30, (resW, resH))

    # -------------------------------
    # Initialize image/video source
    # -------------------------------
    if source_type == 'image':
        img_list = [source_arg]
    elif source_type == 'folder':
        img_list = [f for f in glob.glob(os.path.join(source_arg, '*'))
                    if os.path.splitext(f)[1].lower() in img_exts]
    elif source_type in ['video','usb']:
        cap = cv2.VideoCapture(source_arg if source_type=='video' else usb_idx)
        if resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    else:  # picamera
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(
            main={'format':'XRGB8888','size':(resW,resH)}))
        cap.start()

    # -------------------------------
    # Variables for FPS & recording
    # -------------------------------
    fps_buffer = []
    fps_avg_len = 200
    img_idx = 0

    # Variables for 30s JSON recording on 'c'
    recording = False
    record_start = None
    next_offsets = []
    recorded_data = []
    out_json = None

    # Predefined colors for bounding boxes
    colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
              (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

    # -------------------------------
    # Main loop: read frame, detect, display, record
    # -------------------------------
    while True:
        t0 = time.perf_counter()

        # --- grab a frame or image ---
        if source_type in ['image','folder']:
            if img_idx >= len(img_list):
                break
            frame = cv2.imread(img_list[img_idx])
            img_idx += 1
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        # --- resize if needed ---
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # --- run YOLO detection ---
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # --- check for 'c' key to start 30s JSON recording ---
        key = cv2.waitKey(1)
        if key in (ord('c'), ord('C')) and not recording:
            recording = True
            record_start = time.time()
            next_offsets = [0, 10, 20]
            recorded_data = []
            print(">>> Recording events for 30s (every 10s)...")

        # --- if in recording mode, capture at appropriate times ---
        if recording:
            elapsed = time.time() - record_start
            if next_offsets and elapsed >= next_offsets[0]:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                objs = []
                for det in detections:
                    conf = det.conf.item()
                    if conf < min_thresh:
                        continue
                    bbox = det.xyxy.cpu().numpy().squeeze().astype(int).tolist()
                    cls = int(det.cls.item())
                    objs.append({
                        "label": labels[cls],
                        "confidence": round(conf,2),
                        "bbox": bbox
                    })
                recorded_data.append({
                    "timestamp": ts,
                    "objects": objs,
                    "source": source_arg
                })
                next_offsets.pop(0)
                print(f"  captured +{int(elapsed)}s: {len(objs)} objects")

            # finish after 30s or all captures done
            if elapsed >= 30 or not next_offsets:
                out_json = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(recorded_data, f, ensure_ascii=False, indent=2)
                print(f">>> Saved JSON to {out_json}")
                break  # exit main loop

        # --- draw detections ---
        obj_count = 0
        for det in detections:
            conf = det.conf.item()
            if conf < min_thresh:
                continue
            x1,y1,x2,y2 = det.xyxy.cpu().numpy().squeeze().astype(int)
            cls = int(det.cls.item())
            color = colors[cls % len(colors)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{labels[cls]}:{int(conf*100)}%"
            ls, bs = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1,y1-ls[1]-4), (x1+ls[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            obj_count += 1

        # --- overlay FPS & object count ---
        fps = 1.0 / (time.perf_counter() - t0)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        if source_type in ['video','usb','picamera']:
            cv2.putText(frame, f"FPS:{avg_fps:.2f}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Objs:{obj_count}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("YOLO Detection", frame)
        if record_video:
            recorder.write(frame)

        # exit on 'q'
        if key in (ord('q'), ord('Q')):
            break

    # -------------------------------
    # Clean up resources
    # -------------------------------
    print("Cleaning up...")
    if source_type in ['video','usb','picamera']:
        cap.release()
    if record_video:
        recorder.release()
    cv2.destroyAllWindows()

    # -------------------------------
    # If we recorded JSON, send to Ollama for summary
    # -------------------------------
    if out_json:
        # load the recorded events
        with open(out_json, 'r', encoding='utf-8') as f:
            events = json.load(f)

        # build the prompt
        json_block = json.dumps(events, ensure_ascii=False, indent=2)
        prompt = f"""You are a helpful assistant.
                Here is a JSON list of object detection results over a 30-second window (every 10s):{json_block}
                Please write a brief paragraph (2–3 sentences) summarizing what happened during these 30 seconds."""

        # call Ollama synchronous API
        resp = ollama.generate(
            model="llama3.2:1b",
            prompt=prompt,
        )
        # print the generated report
        #summary = resp['response'] if isinstance(resp, dict) else resp
        print("\n=== 30s SUMMARY REPORT ===")
        print(resp['response'])

if __name__ == "__main__":
    main()


