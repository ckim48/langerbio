from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from pose_detection import PoseDetector

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/pose_task/<pose_name>')
def pose_task(pose_name):
	return render_template('pose_task.html', pose_name=pose_name)


@app.route('/start_pose', methods=['POST'])
def start_pose():
    pose_name = request.json.get('pose_name')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({'error': 'Failed to access the camera'}), 500

    detector = PoseDetector(pose_name)

    try:
        with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    return jsonify({'error': 'Failed to read frame from the camera'}), 500

                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                # Process the frame
                frame = detector.detect_pose(frame, pose)

                # Optionally encode the frame (if needed for UI feedback)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                if detector.num_tries > 3:
                    break

    except Exception as e:
        app.logger.error(f"Error during pose detection: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        cap.release()

    return jsonify({
        'num_tries': detector.num_tries,
        'avg_score': detector.avg_score,
        'scores': detector.score_list
    })


if __name__ == '__main__':
	app.run(debug=True)
