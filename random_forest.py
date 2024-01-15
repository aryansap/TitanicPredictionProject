from d_tree import DecisionTree
import numpy as np
from collections import Counter
class RandomForest:
    import numpy as np
    from collections import Counter
    def __init__(self, n_trees = 10, max_depth = 10, min_samples_split = 2, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    def fit(self,X,y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth,min_samples_split=self.min_samples_split,n_fe = self.n_features)
            xsample,ysample = self.bootstrapsamples(X,y)
            tree.fit(xsample,ysample)
            self.trees.append(tree)

    def bootstrapsamples(self,X,y):
        n_samples = X.shape[0]
        indxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[indxs],y[indxs]
    
    def _most_common_label(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value 

    def predict(self,X):
       predictions = np.array([tree.predict(X) for tree in self.trees])
       tree_preds = np.swapaxes(predictions,0,1)
       return np.array([self._most_common_label(pred) for pred in tree_preds])