import numpy as np


def calculate_angle(a, b, c):
	a = np.array(a[:2])  # First point (2D)
	b = np.array(b[:2])  # Middle point (2D)
	c = np.array(c[:2])  # Last point (2D)

	ab = a - b
	cb = c - b

	# Calculate the dot product and magnitude
	dot_product = np.dot(ab, cb)
	magnitude_ab = np.linalg.norm(ab)
	magnitude_cb = np.linalg.norm(cb)

	# Avoid division by zero
	if magnitude_ab < 1e-5 or magnitude_cb < 1e-5:
		return None

	# Calculate the angle in radians and convert to degrees
	angle = np.arccos(dot_product / (magnitude_ab * magnitude_cb))
	return np.degrees(angle)


def squat_calcualte_score(angle1, angle2, phase):
	score = 0
	if phase == 'action':
		if 70 <= angle1 <= 110 and 70 <= angle2 <= 110:
			score += 10
		else:
			score = 0
	return score


def superman_calculate_score(angle1, angle2, angle3, phase):
	score = 0
	if phase == 'action':
		if 70 <= angle1 <= 110 and 70 <= angle2 <= 110 and angle3 >= 160:
			score += 10
		else:
			score = 0
	return score


def plank_calculate_score(angle1, angle2, angle3, phase):
	score = 0
	if phase == 'action':
		if 70 <= angle1 <= 110 and angle2 >= 160 and angle3 >= 160:
			score += 10
		else:
			score = 0
	return score


def sideplank_calculate_score(angle1, angle2, phase):
	score = 0
	if phase == 'action':
		if 40 <= angle1 <= 80 and angle2 <= 160:
			score += 10
		else:
			score = 0
	return score


def bridge_calculate_score(angle1, angle2, phase):
	score = 0
	if phase == 'action':
		if angle1 >= 160 and 70 <= angle2 <= 110:
			score += 10
		else:
			score = 0
	return score