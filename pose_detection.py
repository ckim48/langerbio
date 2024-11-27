import cv2
import mediapipe as mp
import numpy as np
import sys

import utils

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PoseDetector:
	def __init__(self, pose_name):
		self.pose_name = pose_name
		self.phase = 'start'
		self.num_tries = 0
		self.score = 0
		self.score_list = []
		self.avg_score = 0

	def calculate_angle_score(self, landmarks):
		if self.pose_name == 'squat':
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
			left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
			left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
			left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

			angle1 = utils.calculate_angle(left_shoulder, left_hip, left_knee)
			angle2 = utils.calculate_angle(left_hip, left_knee, left_ankle)

			# Phase detection and scoring
			if angle1 and angle2:
				if angle1 > 160 and angle2 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 < 100 and angle2 < 100 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.squat_calcualte_score(angle1, angle2, self.phase)
					self.score_list.append(self.score)

		elif self.pose_name == 'superman':
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

			left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			middle_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

			left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
			left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
			right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
			left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
			right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

			angle1 = utils.calculate_angle(left_wrist, neck, right_wrist)
			angle2 = utils.calculate_angle(middle_hip, left_knee, left_ankle)
			angle3 = utils.calculate_angle(neck, middle_hip, right_ankle)

			if angle1 and angle2 and angle3:
				if angle1 > 160 and angle2 > 160 and angle3 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 < 100 and angle2 < 100 and angle3 > 160 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.superman_calculate_score(angle1, angle2, angle3, self.phase)
					self.score_list.append(self.score)

		elif self.pose_name == 'plank':
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

			left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			middle_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

			left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
			left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
			left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

			angle1 = utils.calculate_angle(left_elbow, neck, middle_hip)
			angle2 = utils.calculate_angle(neck, middle_hip, left_knee)
			angle3 = utils.calculate_angle(middle_hip, left_knee, left_ankle)

			if angle1 and angle2 and angle3:
				if angle1 > 70 and angle2 > 160 and angle3 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 < 80 and angle2 > 160 and angle3 > 160 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.plank_calculate_score(angle1, angle2, angle3, self.phase)
					self.score_list.append(self.score)

		elif self.pose_name == 'sideplankA':
			right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

			angle1 = utils.calculate_angle(left_elbow, left_shoulder, left_hip)
			angle2 = utils.calculate_angle(left_shoulder, left_hip, left_knee)

			if angle1 and angle2:
				if angle1 > 70 and angle2 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 < 80 and angle2 > 160 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.sideplank_calculate_score(angle1, angle2, self.phase)
					self.score_list.append(self.score)

		elif self.pose_name == 'sideplankB':
			right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

			angle1 = utils.calculate_angle(right_elbow, right_shoulder, right_hip)
			angle2 = utils.calculate_angle(right_shoulder, right_hip, right_knee)

			if angle1 and angle2:
				if angle1 > 70 and angle2 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 < 80 and angle2 > 160 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.sideplank_calculate_score(angle1, angle2, self.phase)
					self.score_list.append(self.score)

		elif self.pose_name == 'bridge':
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

			left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			middle_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

			right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
			right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

			angle1 = utils.calculate_angle(neck, middle_hip, right_knee)
			angle2 = utils.calculate_angle(middle_hip, right_knee, right_ankle)

			if angle1 and angle2:
				if angle1 > 160 and angle2 > 160 and self.phase == 'action':
					self.phase = 'start'
					self.num_tries += 1
				elif angle1 > 160 and angle2 < 100 and self.phase == 'start':
					self.phase = 'action'
					self.score = utils.bridge_calculate_score(angle1, angle2, self.phase)
					self.score_list.append(self.score)

	def detect_pose(self, image, pose):
		output_image = image.copy()
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		height, width, _ = image.shape
		landmarks = []

		if results.pose_landmarks:
			# Draw pose landmarks on the image
			mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
			                          connections=mp_pose.POSE_CONNECTIONS)

			# Extract landmark positions
			for landmark in results.pose_landmarks.landmark:
				landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

			########################################################
			## calculate and display neck and middlehip positions ##
			########################################################
			# Calculate neck position as the midpoint between left and right shoulders
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			neck = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

			# Calculate the midpoint for left and right hips
			left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
			right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
			middle_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

			# Add points you need
			left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][:2]
			right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][:2]

			# Display the neck and middle hip as a point on the image
			cv2.circle(output_image, neck, 5, (0, 0, 255), -1)
			cv2.circle(output_image, middle_hip, 5, (0, 0, 255), -1)

			# Draw lines
			cv2.line(output_image, neck, middle_hip, (255, 255, 255), 2)
			cv2.line(output_image, middle_hip, left_knee, (255, 255, 255), 2)
			cv2.line(output_image, middle_hip, right_knee, (255, 255, 255), 2)
			########################################################

			# run function 'calculate_angle_score' to update angles and scores
			self.calculate_angle_score(landmarks)

			# Display phase and score on the screen
			# need to calculate how many 'starts'. List all the scores. Average them.
			if self.score_list:
				self.avg_score = sum(self.score_list) / len(self.score_list)
			else:
				self.avg_score = 0
			cv2.putText(output_image, f'Phase: {self.phase}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.putText(output_image, f'Score: {self.score}', (10, 69), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.putText(output_image, f'Num Tries: {self.num_tries}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
			            (255, 255, 255), 2)
			cv2.putText(output_image, f'Avg Scores: {self.avg_score}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
			            (255, 255, 255), 2)

		return output_image


# Main function to run pose detection
def main(pose_name):
	cap = cv2.VideoCapture(1)
	detector = PoseDetector(pose_name)

	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break

			# Detect pose and get the processed frame
			frame = detector.detect_pose(frame, pose)

			# Display the frame
			cv2.imshow(f'Detecting {pose_name}', frame)

			# Exit trigger
			if cv2.waitKey(10) & 0xFF == ord('q'):
				print(
					f"Exiting... {detector.num_tries} attempts recorded. Avg score: {sum(detector.score_list) / len(detector.score_list) if detector.score_list else 0}")
				break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python pose_detection.py <pose_name>")
		sys.exit(1)

	pose_name = sys.argv[1]
	main(pose_name)
