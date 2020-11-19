"""
File for Generic helper functions
"""
import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """ Euclidean distance between n dimensional points"""
    return np.linalg.norm(a - b)


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ midpoint distance between n dimensional points"""
    return a + ((b - a) * 0.5)
