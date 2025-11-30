import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
    """
    if len(np.unique(feature_vector)) == 1:
        return np.array([]), np.array([]), None, np.inf
   
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]
    
    
    unique_features = np.unique(feature_sorted)
    

    thresholds = (unique_features[:-1] + unique_features[1:]) / 2.0
    

    ginis = []
    valid_thresholds = []
    
 
    n_total = len(target_vector)
    
    for threshold in thresholds:
      
        left_mask = feature_sorted < threshold
        right_mask = ~left_mask
        
        n_left = np.sum(left_mask)
        n_right = n_total - n_left
        

        if n_left == 0 or n_right == 0:
            continue
        
     
        left_targets = target_sorted[left_mask]
        if len(left_targets) == 0:
            continue
            
        left_counts = np.bincount(left_targets, minlength=2)
        p_left_0 = left_counts[0] / n_left
        p_left_1 = left_counts[1] / n_left
        
        H_left = 1 - p_left_0**2 - p_left_1**2
        
        right_targets = target_sorted[right_mask]
        if len(right_targets) == 0:
            continue
            
        right_counts = np.bincount(right_targets, minlength=2)
        p_right_0 = right_counts[0] / n_right
        p_right_1 = right_counts[1] / n_right

        H_right = 1 - p_right_0**2 - p_right_1**2
        

        gini = -(n_left / n_total) * H_left - (n_right / n_total) * H_right
        
        ginis.append(gini)
        valid_thresholds.append(threshold)
    
    if not ginis:
        return np.array([]), np.array([]), None, np.inf
    
    ginis = np.array(ginis)
    valid_thresholds = np.array(valid_thresholds)
    
    best_idx = np.argmin(ginis) 
    min_gini = ginis[best_idx]
    best_indices = np.where(ginis == min_gini)[0]
    best_idx = best_indices[0]  
    
    threshold_best = valid_thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return valid_thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = 0

    def _fit_node(self, sub_X, sub_y, node, depth=0):
     
        if (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or \
           (len(np.unique(sub_y)) == 1):  
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        
        for feature in range(sub_X.shape[1]):  
            feature_type = self._feature_types[feature]
            
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                
                feature_vector_cat = sub_X[:, feature]
                categories = np.unique(feature_vector_cat)
                
               
                category_ratios = {}
                for category in categories:
                    mask = feature_vector_cat == category
                    if np.sum(mask) > 0:
                        ratio = np.sum(sub_y[mask]) / np.sum(mask)
                        category_ratios[category] = ratio
                
                sorted_categories = sorted(category_ratios.keys(), key=lambda x: category_ratios[x])
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                
                feature_vector = np.array([categories_map[x] for x in feature_vector_cat])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if gini is not None and (gini_best is None or gini < gini_best):  
                feature_best = feature
                gini_best = gini
                
                if feature_type == "real":
                    threshold_best = threshold
                    split = feature_vector < threshold
                elif feature_type == "categorical":
                    
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                    split = np.array([x in threshold_best for x in sub_X[:, feature]])
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return


        left_samples = np.sum(split)
        right_samples = len(sub_y) - left_samples
        
        if (self._min_samples_leaf is not None and 
            (left_samples < self._min_samples_leaf or right_samples < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        
        node["left_child"], node["right_child"] = {}, {}
        

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1) 
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)  

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание для одного объекта
        """
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        
        if feature_type == "real":

            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
