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
def draw_lines_and_axes(image, points, finger_line_indices, knuckle_indices, extension_length=100):
    knuckle_start, knuckle_end = np.array(points[knuckle_indices[0]]), np.array(points[knuckle_indices[1]])
    knuckle_direction = knuckle_end - knuckle_start
    knuckle_unit_vector = knuckle_direction / np.linalg.norm(knuckle_direction)
    knuckle_line_start = (knuckle_start - knuckle_unit_vector * extension_length * 2).astype(int)
    knuckle_line_end = (knuckle_end + knuckle_unit_vector * extension_length * 2).astype(int)
    cv2.line(image, tuple(knuckle_line_start), tuple(knuckle_line_end), (0, 0, 255), 2)
    label_position = ((knuckle_line_start + knuckle_line_end) / 2).astype(int)
    cv2.putText(image, "F1", tuple(label_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, (i, j) in enumerate(finger_line_indices):
        p1, p2 = np.array(points[i]), np.array(points[j])
        direction = p2 - p1
        length = np.linalg.norm(direction)
        unit_vector = direction / length
        start_point = (p1 - unit_vector * extension_length).astype(int)
        end_point = (p2 + unit_vector * extension_length).astype(int)
        cv2.line(image, tuple(start_point), tuple(end_point), (255, 0, 0), 2)

    for idx, (i, j) in enumerate(finger_line_indices):
        p1, p2 = np.array(points[i]), np.array(points[j])
        direction = p2 - p1
        unit_vector = direction / np.linalg.norm(direction)
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

        for k, p in enumerate([p1, p2]):
            axis_start_point = (p - perpendicular_vector * extension_length).astype(int)
            axis_end_point = (p + perpendicular_vector * extension_length).astype(int)
            cv2.line(image, tuple(axis_start_point), tuple(axis_end_point), (0, 255, 0), 2)

            if k == 0:
                label_position = ((axis_start_point + axis_end_point) / 2).astype(int)
                cv2.putText(image, f"F{idx + 2}", tuple(label_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

# Function to calculate perpendicular distances
def calculate_distances(points, line_indices):
    distances = []

    for i, j in line_indices:
        p1 = np.append(np.array(points[i]), 0)  
        p2 = np.append(np.array(points[j]), 0)  
        
        for k in range(len(points)):
            if k != i and k != j:
                p3 = np.append(np.array(points[k]), 0) 
                dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                distances.append(dist)

    return distances

# Function to calculate intensity profiles
def calculate_intensity_profiles(image, points, finger_line_indices, knuckle_indices, extension_length=200):
    profiles = []

    for i, j in finger_line_indices:
        p1, p2 = np.array(points[i]), np.array(points[j])
        direction = p2 - p1
        unit_vector = direction / np.linalg.norm(direction)
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

        axis_start_point = (p1 - perpendicular_vector * extension_length).astype(int)
        axis_end_point = (p1 + perpendicular_vector * extension_length).astype(int)
        profile = []

        for t in np.linspace(0, 1, int(np.linalg.norm(axis_end_point - axis_start_point))):
            pt = (1 - t) * axis_start_point + t * axis_end_point
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                profile.append(image[y, x])

        profiles.append(np.array(profile))

    # Calculate intensity profile for knuckle line
    if len(points) > max(knuckle_indices):
        knuckle_start, knuckle_end = np.array(points[knuckle_indices[0]]), np.array(points[knuckle_indices[1]])
        knuckle_direction = knuckle_end - knuckle_start
        knuckle_unit_vector = knuckle_direction / np.linalg.norm(knuckle_direction)
        knuckle_line_start = (knuckle_start - knuckle_unit_vector * extension_length * 2).astype(int)
        knuckle_line_end = (knuckle_end + knuckle_unit_vector * extension_length * 2).astype(int)

        knuckle_profile = []
        for t in np.linspace(0, 1, int(np.linalg.norm(knuckle_line_end - knuckle_line_start))):
            pt = (1 - t) * knuckle_line_start + t * knuckle_line_end
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                knuckle_profile.append(image[y, x])
        profiles.insert(0, np.array(knuckle_profile))  

    return profiles

# Load hand images
hand_images = [cv2.imread(f'hand{i}.jpg') for i in range(1, 6)]

# Set points and calculate feature vectors for each image
feature_vectors = []
for image_index in range(len(hand_images)):
    image = hand_images[image_index]

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    points = set_points(grayscale_image.copy(), 12, 800) 
    finger_line_indices = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)] 
    knuckle_indices = (0, 1)

    image_with_lines_and_axes = draw_lines_and_axes(grayscale_image.copy(), points, finger_line_indices, knuckle_indices, extension_length=175)
    profiles = calculate_intensity_profiles(grayscale_image, points, finger_line_indices, knuckle_indices)
    distances = calculate_distances(points, finger_line_indices)
    feature_vectors.append(np.array(distances).flatten())
    
    scaled_image, _, _ = resize_image(image_with_lines_and_axes, 800)

    cv2.imshow(f"Hand Image {image_index + 1} with Lines and Axes", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.figure(figsize=(15, 10))
    for idx, profile in enumerate(profiles):
        plt.subplot(3, 2, idx + 1)
        plt.plot(profile)
        if idx == 0:
            plt.title("Intensity Profile along Knuckle Line (F1)")
        else:
            plt.title(f"Intensity Profile along F{idx + 1}")
    
    plt.tight_layout()
    plt.show()
    

# Calculate pairwise Euclidean distances
distance_matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(i + 1, 5):
        dist = distance.euclidean(feature_vectors[i], feature_vectors[j])
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

print("Pairwise Euclidean Distance Matrix:")
print(distance_matrix)
