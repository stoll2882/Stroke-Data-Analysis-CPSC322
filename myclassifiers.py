import copy
import numpy as np
import math
import myutils as myutils

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        self.slope, self.intercept = myutils.compute_slope_intercept(X_train, y_train)
        return self

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i in range(len(X_test)):
            new_val = X_test[i][0] * self.slope + self.intercept
            y_predicted.append(new_val)
            
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for i, row in enumerate(X_test):
            result = self.get_k_neighbors(row)
            d = []
            n = []
            for j in result:
                d.append(j[0])
                n.append(j[1])
            distances.append(d)
            neighbor_indices.append(n)
            
        return distances, neighbor_indices # TODO: fix this
    
    def get_k_neighbors(self, X_test):
        output = []
        for i, instance in enumerate(self.X_train):
            dist = myutils.compute_euclidean_distance(X_test, instance)
            output_line = []
            output_line.append(dist)
            output_line.append(i)
            output_line.append(instance)
            output.append(output_line)
            
        # sort train by distance
        output_sorted = sorted(output)
        # grab the top k
        top_k = output_sorted[:self.n_neighbors]
            
        return top_k

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        dists, indexs = self.kneighbors(X_test)
        train_copy = copy.deepcopy(self.y_train)
        prediction = []
        
        unique = []
        for val in self.y_train:
            if val not in unique:
                unique.append(val)
            
        for i, val in enumerate(train_copy):
            index = unique.index(val)
            train_copy[i] = index
            
        for k in indexs:
            total = 0
            for j in k:
                total += train_copy[j]
            avg = round(total / len(k))
            final = unique[avg]
            prediction.append(final)
            
        return prediction

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None
        self.attribute_labels = None
        self.posterior_labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # establish new arrays
        newPriors = []
        parallel_labels = []
        parallel_counts = []
        newPosteriors = []
        posterior_parallel_labels = []
        # establish priors...
        # start with empty arrays and parallel label array
        for val in y_train:
            if val not in parallel_labels:
                newPriors.append(0)
                parallel_labels.append(val)
                parallel_counts.append(0)
        # count how many of each there are
        for label in y_train:
            index = parallel_labels.index(label)
            newPriors[index] += 1
            parallel_counts[index] += 1
        # establish final priors by dividing by length to get fraction in float form
        for i, prior in enumerate(newPriors):
            newPriors[i] = float(prior / len(y_train))

        # establish posteriors...
        for _ in range(len(X_train[0])):
            posterior_parallel_labels.append([])
            newPosteriors.append([])
        # first, establish how many different options are in each attribute and itialize newPosterios accordingly
        for _, _ in enumerate(parallel_labels):
            for i, attribute in enumerate(X_train):
                for j, val in enumerate(attribute):
                    if val not in posterior_parallel_labels[j]:
                        posterior_parallel_labels[j].append(val)
                        newPosteriors[j].append([])
        for i, attribute in enumerate(newPosteriors):
            for j, _ in enumerate(attribute):
                for _ in range(len(parallel_labels)):
                    newPosteriors[i][j].append(0)

        for i, row in enumerate(X_train):
            result_index = parallel_labels.index(y_train[i])
            for j, att in enumerate(row):
                att_index = posterior_parallel_labels[j].index(att)
                newPosteriors[j][att_index][result_index] += 1

        for i, att in enumerate(newPosteriors):
            for j, attribute in enumerate(att):
                for k, val in enumerate(attribute):
                    newPosteriors[i][j][k] = float(val / parallel_counts[k])
        
        self.attribute_labels = parallel_labels
        self.posterior_labels = posterior_parallel_labels
        self.priors = newPriors
        self.posteriors = newPosteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            results = []
            for i, label in enumerate(self.attribute_labels):
                total = []
                for j, characteristic in enumerate(instance):
                    att_index = self.posterior_labels[j].index(characteristic)
                    total.append(self.posteriors[j][att_index][i])
                result = total[0]
                for i in range(1, len(total)):
                    result = float(result * total[i])
                results.append(result)
            # multiply by posterior
            for i, r in enumerate(results):
                results[i] = r * self.priors[i]
            max_index = 0
            max_val = 0
            for i, r in enumerate(results):
                if r > max_val:
                    max_val = r
                    max_index = i
            y_predicted.append(self.attribute_labels[max_index])
        return y_predicted  # TODO: fix this

    
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def get_attribute_domains_and_header(self):
        attribute_domains = {}
        att_names = []
        att_values = []
        for i, col in enumerate(self.X_train[0]):
            att_name = str('att' + str(i))
            att_names.append(att_name)
            att_values.append([])

        for i, row in enumerate(self.X_train):
            for j, col in enumerate(row):
                if col not in att_values[j]:
                    att_values[j].append(col)

        for i, name in enumerate(att_names):
            attribute_domains[name] = att_values[i]
        
        return attribute_domains, att_names

    def select_attribute(self, instances, available_attributes):
        # for now, we are going to select an attribute randomly
        attribute_domains, header = self.get_attribute_domains_and_header()
        last_index = len(instances[0])
        y_att_name = str("att" + str(last_index))
        temp_header = copy.deepcopy(header)
        temp_header.append(y_att_name)
        y_col = myutils.get_column(instances, temp_header, y_att_name)
        unique_y_vals = []
        for instance in instances:
            if instance[len(instance) - 1] not in unique_y_vals:
                unique_y_vals.append(instance[len(instance) - 1])

        entropys = []

        for i, attribute in enumerate(available_attributes):
            possible_vals = attribute_domains[attribute]
            entropy = 0
            curr_column = myutils.get_column(instances, temp_header, attribute)
            for possible_val in possible_vals:
                indexs = []
                for i, col in enumerate(curr_column):
                    if col == possible_val:
                        indexs.append(i)
                # total_count = 0'
            
                for y_val in unique_y_vals:
                    count_unique = 0
                    for index in indexs:
                        if y_col[index] == y_val:
                            count_unique += 1
                    if count_unique == 0 or len(indexs) == 0:
                        total_entropy = 0.0
                    count_y_val = 0
                    for y in y_col:
                        if y == y_val:
                            count_y_val += 1
                    else:
                        # try:
                        fraction = 0.0
                        val_entropy = 0.0
                        if len(indexs) != 0:
                            try:
                                fraction = float(count_unique) / float(len(indexs))
                                val_entropy = -(fraction * math.log(fraction, 2))
                            except ValueError:
                                pass                  

                        total_entropy = (float(len(indexs)) / float(len(instances))) * val_entropy
                    entropy += total_entropy
            entropys.append(entropy)


        min_entropy_index = 0
        for i, entropy in enumerate(entropys):
            if entropy < entropys[min_entropy_index]:
                min_entropy_index = i

        return available_attributes[min_entropy_index]

    def partition_instances(self, instances, split_attribute):
        # this is a group by split_attribute's domain, not by the values of this attribute in instances
        # example: if split_attribute is "level"
        attribute_domains, header = self.get_attribute_domains_and_header()
        attribute_domain = attribute_domains[split_attribute]
        attribute_index = header.index(split_attribute) # 0
        partitions = {} # key (attribute value) : value (list of instances with this attribute value)
        for attribute_value in attribute_domain:
            partitions[attribute_value] = []
            for instance in instances:
                if instance[attribute_index] == attribute_value:
                    partitions[attribute_value].append(instance)
        return partitions

    def all_same_class(self, partition):
        label_index = len(partition[0]) - 1
        count = 0
        class_label = partition[0][label_index]
        for item in partition:
            if item[label_index] == class_label:
                count += 1
        if count == len(partition):
            return True
        else:
            return False

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):

        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        # print("splitting on:", split_attribute)
        available_attributes.remove(split_attribute)
        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # print("partitions:", partitions)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            # print("working with partition for:", attribute_value)
            value_subtree = ["Value", attribute_value]
            # TODO: appending lead nodes and subtrees appropriately to value_subtree
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and self.all_same_class(partition): # write function all_same_class() that returns true if all parititons in instance have the same class name
                end_index = len(partition[0]) - 1
                value_subtree.append(["Leaf", partition[0][end_index], len(partition), len(current_instances)])
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                value_subtree.append(["Leaf", self.get_majority_class(current_instances), len(partition), len(current_instances)])
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                return (["Leaf", self.get_majority_class(current_instances), len(partition), len(current_instances)])
            else: # all base cases are false... we recurse!!!
                subtree = self.tdidt(partition, copy.deepcopy(available_attributes))
                value_subtree.append(subtree)
            tree.append(value_subtree)
                # need to append subtree to value_subtree and appropriately append value_subtree to tree
        return tree
    
    def get_majority_class(self, current_instances):
        class_index = len(current_instances[0]) - 1
        checked_classes = []
        most_common_class = None
        max_count = 0

        for instance in current_instances:
            curr_class = instance[class_index]
            if curr_class not in checked_classes:
                count = 0
                for i in current_instances:
                    if i[class_index] == curr_class:
                        count += 1
                if count > max_count:
                    most_common_class = curr_class
                    max_count = count
        return most_common_class

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        new_tree = []

        attribute_domains, header = self.get_attribute_domains_and_header()


        combined_train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # # initial call to tdidt, current instances will be the whole table
        available_attributes = copy.deepcopy(header) # python is pass by object reference
        new_tree = self.tdidt(combined_train, available_attributes)
        
        self.tree = new_tree

    def tdidt_predict(self, header, tree, instance):
        info_type = tree[0]
        if info_type == "Attribute":
            attribute_index = header.index(tree[1])
            instance_value = instance[attribute_index]
            # now I need to find which "edge" to follow recursively
            for i in range(2, len(tree)):
                value_list = tree[i]
                if value_list[1] == instance_value:
                    # we have a match!! (decided which edge to recurse and go down)
                    return self.tdidt_predict(header, value_list[2], instance)
        else:  # "Leaf"
            return tree[1]
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        attribute_domains, header = self.get_attribute_domains_and_header()
        for instance in X_test:
            predictions.append(self.tdidt_predict(header, self.tree, instance))
        return predictions # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if self.tree == None:
            return
        attribute_domains, header = self.get_attribute_domains_and_header()
        header_to_use = header
        if attribute_names != None:
            assert len(attribute_names) == len(header)
            header_to_use = attribute_names
        
        domains_list = []
        parallel_names_list = []
        for attribute_value, domain in attribute_domains.items():
            domains_list.append(domain)
            parallel_names_list.append(attribute_value)

        num_domains = len(domains_list)
        indices = [0 for i in range(num_domains)]

        combinations = []

        while(True):
            new_combination = []
            for i in range(num_domains):
                new_combination.append(domains_list[i][indices[i]])
            combinations.append(new_combination)
            next_val = num_domains - 1
            while(next_val >= 0 and (indices[next_val] + 1 >= len(domains_list[next_val]))):
                next_val -= 1
            
            if (next_val < 0):
                break
            
            indices[next_val] += 1

            for i in range(next_val + 1, num_domains):
                indices[i] = 0
        
        for combination in combinations:
            sorted_combination = []
            for i, att in enumerate(header):
                att_index = parallel_names_list.index(att)
                sorted_combination.append(combination[att_index])
            print("If ", end='')
            for i, attribute in enumerate(header):
                if (i != len(header) - 1):
                    pass
                    print(str(header_to_use[i]) + ' == ' + str(sorted_combination[i]) + ' AND ', end='')
            print(str(header_to_use[len(header) - 1]) + ' == ' + str(str(sorted_combination[len(header) - 1])) + ' THEN ' + class_name + ' = ', end='')
            print(self.predict([sorted_combination])[0])

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
