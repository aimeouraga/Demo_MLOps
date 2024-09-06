import seaborn as sns
import matplotlib.pyplot as plt
from dataset import X, y 
from LogReg import prediction, y_test 
from sklearn.metrics import confusion_matrix


# Visualize the first feature (sepal length)
plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=20, color='blue', alpha=0.7)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('histogram_sepal_length.png')
plt.show()

# Scatter plot of two features (sepal length vs sepal width)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.grid(True)
plt.savefig('scatter_sepal_length_vs_width.png')
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='plasma', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('confusion_matrix.png')
plt.show()