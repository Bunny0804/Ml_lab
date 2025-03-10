#KNN- Algorithm
import math
from collections import Counter
def euclidean_distance(point1, point2):
  distance = 0.0
  for i in range(len(point1)):
    distance += (point1[i] - point2[i]) ** 2
  return math.sqrt(distance)
def knn(train_data, test_point, k):
  """Performs K-Nearest Neighbors classification."""
  distances = []
  # Compute distances from test point to all training points
  for data_point in train_data:
    features = data_point[:-1] # All columns except last (features)
    label = data_point[-1] # Last column (label)
    distance = euclidean_distance(features, test_point)
    distances.append((distance, label))
  # Sort by distance and select k nearest neighbors
  distances.sort()
  neighbors = distances[:k]
  # Extract labels of k nearest neighbors
  labels = [label for _, label in neighbors]
  # Return the most common class label
  return Counter(labels).most_common(1)[0][0]
# Example usage
train_data = [
 [2.0, 3.0, 'A'],
 [1.0, 1.0, 'B'],
 [4.0, 5.0, 'A'],
 [7.0, 8.0, 'B'],
 [5.0, 6.0, 'A'],
 [2.0, 6.0, 'A'],
 [5.0, 4.0, 'B'],
 [6.0, 2.0, 'A']
]
test_point = [3.0, 3.5]
k = 3
prediction = knn(train_data, test_point, k)
print(f'Predicted class: {prediction}')
