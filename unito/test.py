from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X, y)
pred = clf.predict(X)

print(y)
print(pred)

print(log_loss(y_true=y, y_pred=pred))
