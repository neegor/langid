# langid

This incredibly fast and accurate library is designed for identifying the language of a given text. The class uses a model based on a Naive Bayes classifier combined with a finite state automaton to convert text into a feature vector. Let's examine its functionality in detail.

### Main components of the library:

### Model Initialization:
   - The model can be loaded from a string (`from_modelstring`) or from a file (`from_modelpath`). Model data is decoded from base64, decompressed using bz2, and deserialized using `pickle.loads`.
   - The model consists of several components:
     - `nb_ptc`: A matrix of feature probabilities for each class.
     - `nb_pc`: A vector of prior class probabilities.
     - `nb_classes`: A list of supported languages.
     - `tk_nextmove`: A dictionary of transitions in the finite state automaton.
     - `tk_output`: A dictionary mapping automaton states to feature indices.

### Class Methods:
   - `set_languages`: Restricts the set of languages considered by the classifier. If `None` is passed, all available languages are used.
   - `instance2fv`: Converts text into a feature vector. The text is encoded in UTF-8 and then processed using the finite automaton, which counts state frequencies and generates the feature vector.
   - `nb_classprobs`: Computes the probability of the text belonging to each class (language) based on the feature vector and the model.
   - `classify`: Classifies the text and returns the most likely language along with the confidence score.
   - `rank`: Returns a ranked list of languages with their corresponding probabilities.

### Probability Normalization:
   - If `norm_probs` is set to `True`, probabilities are normalized using `norm_func`, which converts log-probabilities into standard probabilities that sum up to 1.

### Example Usage:
```python
# Initialize the model
identifier = LanguageIdentifier.from_modelstring(model_string)

# Restrict the list of languages
identifier.set_languages(["en", "ru", "fr"])

# Classify text
language, confidence = identifier.classify("Hello, world!")
print(f"Language: {language}, Confidence: {confidence}")

# Get a ranked list of languages
ranked_languages = identifier.rank("Bonjour tout le monde")
for lang, prob in ranked_languages:
    print(f"{lang}: {prob:.4f}")
```

### Implementation Details:
- **Finite State Automaton**: Used for efficient feature counting (e.g., n-grams) in the input text. States correspond to character sequences, and outputs map to indices in the feature vector.
- **Naive Bayes Classifier**: Probabilities are computed as a linear combination of features and model weights, enabling fast classification.
- **Normalization**: Transforms log-probabilities into normalized values, making result interpretation more intuitive.

This code serves as a powerful tool for language identification and can be easily integrated into other applications such as text processing or data analysis pipelines.
