import re
import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Load Data set
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()
    return stopwords

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            yield (row[1], int(row[0]))

# Preprocess the text and remove stopwords
class Preprocessor:
    def __init__(self, stopwords):
        self.stopwords = stopwords

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', '', text)  # Remove special characters
        words = text.split()
        return words

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def select_the_top_words(self, reviews):
        word_counts = Counter()
        for review in reviews:
            words = self.preprocess_text(review)
            words = self.remove_stopwords(words)
            word_counts.update(words)
        top_words = [word for word, count in word_counts.most_common(1000) if count > 1]
        return top_words

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self, preprocessor, alpha=1.0):  # Add alpha parameter for Laplace smoothing
        self.word_probs = {}
        self.class_probs = defaultdict(int)
        self.top_words_size = 0
        self.preprocessor = preprocessor
        self.alpha = alpha

    # Train the model
    def train(self, reviews, labels, top_words):
        labels = [1 if label == 5 else 0 for label in labels]
        self.top_words_size = len(top_words)
        word_counts = {word: [self.alpha, self.alpha] for word in top_words}

        for review, label in zip(reviews, labels):
            words = set(self.preprocessor.preprocess_text(review))
            words = self.preprocessor.remove_stopwords(words)
            for word in words:
                if word in top_words:
                    word_counts[word][label] += 1
            self.class_probs[label] += 1

        total_neg = self.class_probs[0] + self.top_words_size * self.alpha
        total_pos = self.class_probs[1] + self.top_words_size * self.alpha
        for word, counts in word_counts.items():
            self.word_probs[word] = [(counts[0] + 1) / (total_neg + self.top_words_size),
                                     (counts[1] + 1) / (total_pos + self.top_words_size)]

    # Predict the class of a review
    def predict(self, review, top_words):
        words = set(self.preprocessor.preprocess_text(review))
        words = self.preprocessor.remove_stopwords(words)
        pos_prob = self.class_probs[1] / sum(self.class_probs.values())
        neg_prob = self.class_probs[0] / sum(self.class_probs.values())

        for word in top_words:
            if word in words:
                pos_prob *= self.word_probs.get(word, [1 / (self.class_probs[0] + self.top_words_size * self.alpha),
                                                       1 / (self.class_probs[1] + self.top_words_size * self.alpha)])[1]
                neg_prob *= self.word_probs.get(word, [1 / (self.class_probs[0] + self.top_words_size * self.alpha),
                                                       1 / (self.class_probs[1] + self.top_words_size * self.alpha)])[0]
            else:
                pos_prob *= 1 - self.word_probs.get(word, [1 / (self.class_probs[0] + self.top_words_size * self.alpha),
                                                           1 / (self.class_probs[1] + self.top_words_size * self.alpha)])[1]
                neg_prob *= 1 - self.word_probs.get(word, [1 / (self.class_probs[0] + self.top_words_size * self.alpha),
                                                           1 / (self.class_probs[1] + self.top_words_size * self.alpha)])[0]

        return 1 if pos_prob > neg_prob else 0

    # Calculate accuracy of the model
    def get_accuracy(self, test_reviews, test_labels, top_words):
        test_labels = [1 if label == 5 else 0 for label in test_labels]
        predicted_labels = [self.predict(review, top_words) for review in test_reviews]

        correct_predictions = sum(1 for true_label, predicted_label in zip(test_labels, predicted_labels) if true_label == predicted_label)
        accuracy = correct_predictions / len(test_labels)
        return accuracy

# Cross-validation with k-fold
def cross_validate(reviews, labels, preprocessor, k=5, alpha_values=[0.1, 0.5, 1.0, 1.5, 2.0]):
    fold_size = len(reviews) // k
    best_alpha = 0
    best_score = 0

    for alpha in alpha_values:
        fold_accuracies = []

        for i in range(k):
            test_reviews = reviews[i * fold_size:(i + 1) * fold_size]
            test_labels = labels[i * fold_size:(i + 1) * fold_size]
            train_reviews = reviews[:i * fold_size] + reviews[(i + 1) * fold_size:]
            train_labels = labels[:i * fold_size] + labels[(i + 1) * fold_size:]

            top_words = preprocessor.select_the_top_words(train_reviews)
            classifier = NaiveBayesClassifier(preprocessor, alpha)
            classifier.train(train_reviews, train_labels, top_words)
            accuracy = classifier.get_accuracy(test_reviews, test_labels, top_words)

            fold_accuracies.append(accuracy)

        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f'Alpha: {alpha}, Avg Accuracy: {avg_accuracy}')

        if avg_accuracy > best_score:
            best_score = avg_accuracy
            best_alpha = alpha

    return best_alpha, best_score

def plot_learning_curve(train_reviews, train_labels, test_reviews, test_labels, top_words, alpha):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
    accuracies = []

    for train_size in train_sizes:
        size = int(len(train_reviews) * train_size)
        train_reviews_subset = train_reviews[:size]
        train_labels_subset = train_labels[:size]

        classifier = NaiveBayesClassifier(preprocessor, alpha)
        classifier.train(train_reviews_subset, train_labels_subset, top_words)
        accuracy = classifier.get_accuracy(test_reviews, test_labels, top_words)
        accuracies.append(accuracy)

    plt.plot(train_sizes, accuracies)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve Analysis')
    plt.show()

if __name__ == "__main__":
    print("Task 1: Feature Selection")
    # Load the dataset
    train_data = list(load_dataset('data/train.csv'))
    test_data = list(load_dataset('data/test.csv'))
    stopwords = load_stopwords('data/stopwords.txt')

    # Preprocess the data
    preprocessor = Preprocessor(stopwords)

    # Split the data into reviews and labels
    train_reviews, train_labels = zip(*train_data)
    test_reviews, test_labels = zip(*test_data)

    top_words = preprocessor.select_the_top_words(train_reviews)
    print("Top 20-50 words:", top_words[20:50])

    print("Task 2: Model Training and Evaluation")

    # Cross-validation to find the best alpha parameter
    best_alpha, best_score = cross_validate(list(train_reviews), list(train_labels), preprocessor, k=5)
    print(f'Best Alpha: {best_alpha}, Best Avg Accuracy: {best_score}')

    # Train the best model
    best_classifier = NaiveBayesClassifier(preprocessor, best_alpha)
    best_classifier.train(train_reviews, train_labels, top_words)

    # Evaluate the best model
    accuracy = best_classifier.get_accuracy(test_reviews, test_labels, top_words)
    print("Test Accuracy:", accuracy)

    print("Task 3: Learning Curve Analysis")
    plot_learning_curve(train_reviews, train_labels, test_reviews, test_labels, top_words, best_alpha)
