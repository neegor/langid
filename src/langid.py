"""Language Identification Library.

A high-performance language detection library using Naive Bayes classifiers and trie-based feature extraction.
Supports loading models from strings or files, restricting languages, and classifying text with confidence scores.

The model uses compressed base64-encoded pickled data containing pre-trained parameters for classification.
"""

import base64
import bz2
import logging
from collections import defaultdict
from pickle import loads
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NORM_PROBS: bool = True


class LanguageIdentifier:
    """A modern language identifier implementing a Naive Bayes classifier with Trie-based feature extraction.

    Attributes:
        nb_ptc (np.ndarray): Matrix of log probabilities of features given class.
        nb_pc (np.ndarray): Vector of prior log probabilities of classes.
        nb_numfeats (int): Number of total features in the model.
        nb_classes (List[str]): List of supported language codes.
        tk_nextmove (Dict[int, int]): Transition table for trie automaton.
        tk_output (Dict[int, List[int]]): Output mapping from trie states to feature indices.
        norm_probs (Callable): Function to normalize probability distributions.
    """

    __slots__ = (
        "__full_model",
        "nb_classes",
        "nb_numfeats",
        "nb_pc",
        "nb_ptc",
        "norm_probs",
        "tk_nextmove",
        "tk_output",
    )

    @classmethod
    def from_modelstring(cls, string: bytes, *args, **kwargs) -> "LanguageIdentifier":
        """Instantiate from a base64-encoded, bz2-compressed model string.

        Args:
            string (bytes): Compressed and encoded model data.
            *args: Additional positional arguments passed to constructor.
            **kwargs: Additional keyword arguments passed to constructor.

        Returns:
            LanguageIdentifier: Constructed instance from model data.
        """
        model_data = loads(bz2.decompress(base64.b64decode(string)))
        nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model_data
        nb_numfeats = len(nb_ptc) // len(nb_pc)

        return cls(
            np.array(nb_ptc).reshape(len(nb_ptc) // len(nb_pc), len(nb_pc)),
            np.array(nb_pc),
            nb_numfeats,
            nb_classes,
            tk_nextmove,
            tk_output,
            *args,
            **kwargs,
        )

    @classmethod
    def from_modelpath(cls, path: str, *args, **kwargs) -> "LanguageIdentifier":
        """Load model from file path.

        Args:
            path (str): Path to binary model file.
            *args: Additional positional arguments passed to constructor.
            **kwargs: Additional keyword arguments passed to constructor.

        Returns:
            LanguageIdentifier: Instantiated model loaded from file.
        """
        with open(path, "rb") as f:
            return cls.from_modelstring(f.read(), *args, **kwargs)

    def __init__(
        self,
        nb_ptc: np.ndarray,
        nb_pc: np.ndarray,
        nb_numfeats: int,
        nb_classes: list[str],
        tk_nextmove: dict[int, int],
        tk_output: dict[int, list[int]],
        norm_probs: bool = NORM_PROBS,
    ):
        """Initialize a new LanguageIdentifier instance.

        Args:
            nb_ptc (np.ndarray): Log probabilities of features given class.
            nb_pc (np.ndarray): Prior log probabilities of classes.
            nb_numfeats (int): Total number of features.
            nb_classes (List[str]): Supported language codes.
            tk_nextmove (Dict[int, int]): Trie transitions.
            tk_output (Dict[int, List[int]]): Mapping from trie states to features.
            norm_probs (bool): Whether to normalize output probabilities.
        """
        self.nb_ptc = nb_ptc
        self.nb_pc = nb_pc
        self.nb_numfeats = nb_numfeats
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        self.tk_output = tk_output
        self.norm_probs = self._create_norm_probs_func(norm_probs)
        self.__full_model = (nb_ptc, nb_pc, nb_classes)

    def _create_norm_probs_func(self, norm_probs: bool):
        """Create a normalization function for probability distributions.

        Args:
            norm_probs (bool): If True, returns softmax-like normalization function.

        Returns:
            Callable: Function that normalizes input probabilities.
        """
        if norm_probs:

            def norm_func(pd):
                return 1 / np.exp(pd[None, :] - pd[:, None]).sum(1)

            return norm_func
        return lambda pd: pd

    def set_languages(self, langs: Optional[list[str]] = None) -> None:
        """Restrict the set of identifiable languages.

        Args:
            langs (Optional[List[str]]): List of language codes to allow. If None, resets to full set.

        Raises:
            ValueError: If any provided language code is not recognized by the model.
        """
        logger.debug("Restricting languages to: %s", langs)
        nb_ptc, nb_pc, nb_classes = self.__full_model

        if langs is None:
            self.nb_classes = nb_classes
            self.nb_ptc = nb_ptc
            self.nb_pc = nb_pc
            return

        unknown_langs = [lang for lang in langs if lang not in nb_classes]
        if unknown_langs:
            raise ValueError(f"Unknown language codes: {unknown_langs}")

        subset_mask = np.array([lang in langs for lang in nb_classes], dtype=bool)
        self.nb_classes = [lang for lang in nb_classes if lang in langs]
        self.nb_ptc = nb_ptc[:, subset_mask]
        self.nb_pc = nb_pc[subset_mask]

    def instance2fv(self, text: str) -> np.ndarray:
        """Convert input text into a feature vector using trie-based tokenization.

        Args:
            text (str): Input text to convert.

        Returns:
            np.ndarray: Feature vector representing the input text.
        """
        text_bytes = text.encode("utf8") if isinstance(text, str) else text
        arr = np.zeros(self.nb_numfeats, dtype=np.uint32)
        state = 0
        statecount = defaultdict(int)

        for letter in text_bytes:
            state = self.tk_nextmove[(state << 8) + letter]
            statecount[state] += 1

        for state, count in statecount.items():
            for index in self.tk_output.get(state, ()):
                arr[index] += count

        return arr

    def nb_classprobs(self, fv: np.ndarray) -> np.ndarray:
        """Compute raw log-probabilities for each language class based on the feature vector.

        Args:
            fv (np.ndarray): Feature vector of shape (n_features,).

        Returns:
            np.ndarray: Array of log-probabilities for each class.
        """
        return np.dot(fv, self.nb_ptc) + self.nb_pc

    def classify(self, text: str) -> tuple[str, float]:
        """Classify the input text and return detected language and confidence score.

        Args:
            text (str): Input text to classify.

        Returns:
            Tuple[str, float]: Detected language code and confidence score.
        """
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        idx = np.argmax(probs)
        return str(self.nb_classes[idx]), float(probs[idx])

    def rank(self, text: str) -> list[tuple[str, float]]:
        """Return a ranked list of possible languages with confidence scores.

        Args:
            text (str): Input text to classify.

        Returns:
            List[Tuple[str, float]]: List of (language, confidence) pairs sorted descendingly by confidence.
        """
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        results = [(str(lang), float(prob)) for prob, lang in zip(probs, self.nb_classes)]
        return sorted(results, key=lambda x: x[1], reverse=True)
