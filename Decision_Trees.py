#Decision trees
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def gini_index(groups, classes):
    """Calculate the Gini index for a split."""
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / total_samples)
    return gini


def split_dataset(index, threshold, dataset):
    """Split a dataset based on an attribute and a threshold."""
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right


def best_split(dataset):
    """Find the best split for a dataset."""
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_threshold, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_threshold, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'threshold': best_threshold, 'groups': best_groups}


def build_tree(dataset, max_depth, min_size, depth=0):
    """Recursively build a decision tree."""
    split = best_split(dataset)
    left, right = split['groups']
    if not left or not right:
        combined = left + right
        return Node(value=max(set(row[-1] for row in combined), key=[row[-1] for row in combined].count))
    if depth >= max_depth or len(left) < min_size or len(right) < min_size:
        combined = left + right
        return Node(value=max(set(row[-1] for row in combined), key=[row[-1] for row in combined].count))
    return Node(
        feature=split['index'],
        threshold=split['threshold'],
        left=build_tree(left, max_depth, min_size, depth + 1),
        right=build_tree(right, max_depth, min_size, depth + 1)
    )


def predict(node, row):
    """Make a prediction using the decision tree."""
    if node.value is not None:
        return node.value
    if row[node.feature] < node.threshold:
        return predict(node.left, row)
    else:
        return predict(node.right, row)


# Example dataset (last column is the label)
dataset = [
    [2.7, 2.5, 'A'],
    [1.4, 1.8, 'B'],
    [3.3, 3.1, 'A'],
    [6.5, 7.2, 'B'],
    [4.9, 5.1, 'A']
]

# Build the tree
tree = build_tree(dataset, max_depth=3, min_size=1)

# Test prediction
test_sample = [3.0, 3.0]
print(f'Predicted class: {predict(tree, test_sample)}')
