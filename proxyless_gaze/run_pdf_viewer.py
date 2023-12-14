from deployment.onnx.gaze_ctrl_pdf_viewer import *
from deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter

from training.gaze_estimation.quantize import LinearQuantizer, quantize_linear

USE_QUANTIZED = True

def get_quantized_model(config_path="./training/gaze_estimation/configs/config.yaml"):
    model = quantize_linear(config_path)
    print(model)
    return model


if __name__ == "__main__":
    if USE_QUANTIZED:
        quantized_linear = get_quantized_model()
        quantized_linear.eval()
    face_detection_session = onnxruntime.InferenceSession("./models/face_detection.onnx")
    landmark_detection_session = onnxruntime.InferenceSession("./models/landmark_detection.onnx")
    
    if USE_QUANTIZED:
        gaze_estimation_session = quantized_linear
    else:
        gaze_estimation_session = onnxruntime.InferenceSession("./models/gaze.onnx")
    
    cap = cv2.VideoCapture(0)
    timer = Timer()

    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)

    print('look at top right corner in 3..')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    ret, frame = cap.read()
    faces = detect_face(frame, face_detection_session, timer)
    face = faces[0]
    x1, y1, x2, y2 = face[:4]
    face = np.array([x1,y1,x2,y2,face[-1]])
    landmark, landmark_on_cropped, cropped = detect_landmark(frame, face, landmark_detection_session, timer)
    TR_gaze_pitchyaw, _, _ = estimate_gaze(frame, landmark, gaze_estimation_session, timer, use_quantized=USE_QUANTIZED)

    print('look at top left corner in 3..')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    ret, frame = cap.read()
    faces = detect_face(frame, face_detection_session, timer)
    face = faces[0]
    x1, y1, x2, y2 = face[:4]
    face = np.array([x1,y1,x2,y2,face[-1]])
    landmark, landmark_on_cropped, cropped = detect_landmark(frame, face, landmark_detection_session, timer)
    TL_gaze_pitchyaw, _, _ = estimate_gaze(frame, landmark, gaze_estimation_session, timer, use_quantized=USE_QUANTIZED)
    
    print(TR_gaze_pitchyaw, TL_gaze_pitchyaw)
    app = QApplication(sys.argv)
    window = PDFViewer()
    window.show()
    window.open_pdf()

    cnt = 0
    mode = None
    gesture_timestamp = timer.get_current_timestamp()
    while True:
        if not window.viewer_widget.doc:
            print('no PDF open')
            time.sleep(1)
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        timer.start_record("whole_pipeline")
        show_frame = frame.copy()
        CURRENT_TIMESTAMP = timer.get_current_timestamp()
        cnt += 1
        if cnt % 2 == 1:
            faces = detect_face(frame, face_detection_session, timer)
        if faces is not None:
            face = faces[0]
            x1, y1, x2, y2 = face[:4]
            [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
            face = np.array([x1,y1,x2,y2,face[-1]])
            landmark, landmark_on_cropped, cropped = detect_landmark(frame, face, landmark_detection_session, timer)
            landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
            gaze_pitchyaw, rvec, tvec = estimate_gaze(frame, landmark, gaze_estimation_session, timer, use_quantized=USE_QUANTIZED)
            gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
            timer.start_record("visualize")
            show_frame = visualize(show_frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
            timer.end_record("visualize")
        timer.end_record("whole_pipeline")

        loading_bar = window.viewer_widget.loading_bar
        GESTURE_THRESH = 0.5
        topright = all(abs(gaze_pitchyaw[i] - TR_gaze_pitchyaw[i]) < 0.15 for i in range(2))
        topleft = all(abs(gaze_pitchyaw[i] - TL_gaze_pitchyaw[i]) < 0.15 for i in range(2))
        if topright and topleft:
            print('TOPLEFT AND TOPRIGHT DETECTED. EXITING...')
            break

        if not (topleft or topright) or (topleft and mode != 'TOPLEFT') and (topright and mode != 'TOPRIGHT'):
            # no gesture
            loading_bar.setValue(0)
            gesture_timestamp = CURRENT_TIMESTAMP
        else:
            # gesture logic
            if topright:
                if mode == 'TOPRIGHT':
                    gesture_time = CURRENT_TIMESTAMP - gesture_timestamp
                    if gesture_time < GESTURE_THRESH:
                        loading_bar.setValue(int(gesture_time * 100 // GESTURE_THRESH))
                    else:
                        loading_bar.setValue(0)
                        print('GESTURE DETECTED: NEXT PAGE')
                        gesture_timestamp = CURRENT_TIMESTAMP
                        window.viewer_widget.next_page()
                    # display
                    cv2.putText(show_frame, f"TOPRIGHT", 
                                (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                else:
                    mode = 'TOPRIGHT'
            if topleft:
                if mode == 'TOPLEFT':
                    gesture_time = CURRENT_TIMESTAMP - gesture_timestamp
                    if gesture_time < GESTURE_THRESH:
                        loading_bar.setValue(int(gesture_time * 100 // GESTURE_THRESH))
                    else:
                        loading_bar.setValue(0)
                        print('GESTURE DETECTED: PREVIOUS PAGE')
                        gesture_timestamp = CURRENT_TIMESTAMP
                        window.viewer_widget.previous_page()
                    # display
                    cv2.putText(show_frame, f"TOPLEFT", 
                                (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                else:
                    mode = 'TOPLEFT'
        show_frame = timer.print_on_image(show_frame)
        cv2.imshow("onnx_demo", show_frame)

        code = cv2.waitKey(1)
        if code == 27:
            break

    sys.exit(app.exec_())
