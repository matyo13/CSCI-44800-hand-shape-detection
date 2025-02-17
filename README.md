# CSCI-44800-hand-shape-detection

# The program input is 5 images, located at the same folder main is saved.
# The file name of images is "image{i}.jpg" where i is a number from 1 to 5

# The program first converts the image to grayscale for easier calculations of intensity profiles
# Then the program prompts the user to locate the vector points of joints of the hand
# The order of which the points aree placed is IMPORTANT
# The points are to be placed in pairs, two on each finger starting from the top joint, and a second point on the lower joint
# The knuckle line requires two points where it is CRUCIAL for these points to be placed last, as this is how the program detects the fingers vs. the knuckle vector
# There should be a total of 12 points placed in the end

# The program then takes these points, draws a line between each pair of points
# A perpendicular axis line is then plotted across each point

# The program processes each perpendicular axis line plotted, and creates an intensity profile
# The intensity profiles are calculated based on the contrast between the white background and the darker color of the fingers
# The intensity profiles are then plotted into a graph, which reveal the finger width of each finger

# After 5 iterations of the steps mentioned above have been completed, the programs calculates the euclidean distances between each axis, between each hand
# The output is a pairwise comparison of distances in a form of a 5x5 triangular matrix 