import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    h, w = image.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, (h, w), (new_h, new_w)

# Function to manually set points on the image
def set_points(image, num_points, max_size):
    points = []
    resized_image, original_size, resized_size = resize_image(image, max_size)
    scaling_factor = (original_size[0] / resized_size[0], original_size[1] / resized_size[1])

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x * scaling_factor[1]), int(y * scaling_factor[0])))
            cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('image', resized_image)
    cv2.imshow('image', resized_image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points[:num_points]

# Function to draw lines and calculated axes on the image
def draw_lines_and_axes(image, points, finger_line_indices, knuckle_indices, extension_length=50):
    for i, j in finger_line_indices:
        p1, p2 = np.array(points[i]), np.array(points[j])
        direction = p2 - p1
        length = np.linalg.norm(direction)
        unit_vector = direction / length
        start_point = (p1 - unit_vector * extension_length).astype(int)
        end_point = (p2 + unit_vector * extension_length).astype(int)
        cv2.line(image, tuple(start_point), tuple(end_point), (255, 0, 0), 2)
    
    knuckle_start, knuckle_end = np.array(points[knuckle_indices[0]]), np.array(points[knuckle_indices[1]])
    knuckle_direction = knuckle_end - knuckle_start
    knuckle_unit_vector = knuckle_direction / np.linalg.norm(knuckle_direction)
    knuckle_line_start = (knuckle_start - knuckle_unit_vector * extension_length).astype(int)
    knuckle_line_end = (knuckle_end + knuckle_unit_vector * extension_length).astype(int)
    cv2.line(image, tuple(knuckle_line_start), tuple(knuckle_line_end), (0, 0, 255), 2)

    for i, j in finger_line_indices:
        p1, p2 = np.array(points[i]), np.array(points[j])
        direction = p2 - p1
        unit_vector = direction / np.linalg.norm(direction)
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])
        
        for p in [p1, p2]:
            axis_start_point = (p - perpendicular_vector * extension_length).astype(int)
            axis_end_point = (p + perpendicular_vector * extension_length).astype(int)
            cv2.line(image, tuple(axis_start_point), tuple(axis_end_point), (0, 255, 0), 1)
    return image

# Function to calculate perpendicular distances
def calculate_distances(points, line_indices):
    distances = []
    for i, j in line_indices:
        p1, p2 = np.array(points[i]), np.array(points[j])
        for k in range(len(points)):
            if k != i and k != j:
                p3 = np.array(points[k])
                dist = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
                distances.append(dist)
    return distances

# Load hand images
hand_images = [cv2.imread(f'hand{i}.jpg') for i in range(1, 6)]

# Set points and calculate feature vectors for each image
feature_vectors = []
for image in hand_images:
    points = set_points(image.copy(), 12, 800)  # Keep the number of points to 12
    finger_line_indices = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    knuckle_indices = (10, 11)  # Use points 6 and 7 as knuckle line endpoints
    image_with_lines_and_axes = draw_lines_and_axes(image.copy(), points, finger_line_indices, knuckle_indices, extension_length=200)
    distances = calculate_distances(points, finger_line_indices)
    feature_vectors.append(distances)
    plt.imshow(cv2.cvtColor(image_with_lines_and_axes, cv2.COLOR_BGR2RGB))
    plt.show()

# Calculate pairwise Euclidean distances
distance_matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(i+1, 5):
        dist = distance.euclidean(feature_vectors[i], feature_vectors[j])
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

print("Pairwise Euclidean Distance Matrix:")
print(distance_matrix)
