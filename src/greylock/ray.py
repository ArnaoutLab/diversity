from typing import List, Any, Callable, Union
from numpy import ndarray, empty, concatenate, float64, vstack, zeros
from pandas import DataFrame
from greylock.exceptions import InvalidArgumentError

# avoid mypy error: see https://github.com/jorenham/scipy-stubs/issues/100
from scipy.sparse import spmatrix  # type: ignore
import ray  # type: ignore
from greylock.similarity import (
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
    weighted_similarity_chunk_nonsymmetric,
    weighted_similarity_chunk_symmetric,
)


class SimilarityFromRayFunction(SimilarityFromFunction):
    """Implements Similarity by calculating similarities with a callable
    function."""

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks: int = 64,
        similarities_out: Union[ndarray, None] = None,
    ) -> None:
        """
        similarity:
            A Callable that calculates similarity between a pair of
            species. Must take two rows from X and return a numeric
            similarity value.
            If X is a 2D array, the rows will be 1D arrays.
            If X is a DataFrame, the rows will be named tuples.
        X:
            An array or DataFrame where each row contains the feature values
            for a given species.
        chunk_size:
            Determines how many rows of the similarity matrix each will
            be processes at a time. In general, choosing a larger
            chunk_size will make the calculation faster, but will also
            require more memory.
        """
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks

    def get_Y(self):
        return None

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, spmatrix],
    ) -> ndarray:
        weighted_similarity_chunk = ray.remote(weighted_similarity_chunk_nonsymmetric)
        X_ref = ray.put(self.X)
        Y_ref = ray.put(self.get_Y())
        abundance_ref = ray.put(relative_abundance)
        futures: List[Any] = []
        results = []

        def process_refs(refs):
            nonlocal results
            for chunk_index, abundance_chunk, similarity_chunk in ray.get(refs):
                results.append((chunk_index, abundance_chunk))
                if self.similarities_out is not None:
                    self.similarities_out[
                        chunk_index : chunk_index + similarity_chunk.shape[0], :
                    ] = similarity_chunk

        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            if len(futures) >= self.max_inflight_tasks:
                ready_refs, futures = ray.wait(futures)
                process_refs(ready_refs)
            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.func,
                X=X_ref,
                Y=Y_ref,
                relative_abundance=abundance_ref,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                return_Z=(self.similarities_out is not None),
            )
            futures.append(chunk_future)
        process_refs(futures)
        results.sort()  # This sorts by chunk index (1st in tuple)
        weighted_similarity_chunks = [r[1] for r in results]
        return concatenate(weighted_similarity_chunks)


class IntersetSimilarityFromRayFunction(SimilarityFromRayFunction):
    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        Y: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks=64,
        similarities_out: Union[ndarray, None] = None,
    ):
        super().__init__(func, X, chunk_size, max_inflight_tasks, similarities_out)
        self.Y = Y

    def get_Y(self):
        return self.Y

    def self_similar_weighted_abundances(
        self, relative_abundance: Union[ndarray, spmatrix]
    ):
        raise InvalidArgumentError(
            "Inappropriate similarity class for diversity measures"
        )


class SimilarityFromSymmetricRayFunction(SimilarityFromSymmetricFunction):
    """Implements Similarity by calculating similarities with a callable
    function for one triangle of the similarity matrix, and re-using those
    values for the other triangle.
    """

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks: int = 64,
        similarities_out: Union[ndarray, None] = None,
    ) -> None:
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, spmatrix],
    ) -> ndarray:
        weighted_similarity_chunk = ray.remote(weighted_similarity_chunk_symmetric)
        X_ref = ray.put(self.X)
        abundance_ref = ray.put(relative_abundance)
        futures: List[Any] = []
        result = relative_abundance
        if self.similarities_out is not None:
            self.similarities_out.fill(0.0)

        def process_refs(refs):
            nonlocal result
            for chunk_index, addend, similarities in ray.get(refs):
                result = result + addend
                if self.similarities_out is not None:
                    self.similarities_out[
                        chunk_index : chunk_index + similarities.shape[0], :
                    ] = similarities

        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            if len(futures) >= self.max_inflight_tasks:
                (ready_refs, futures) = ray.wait(futures)
                process_refs(ready_refs)

            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.func,
                X=X_ref,
                relative_abundance=abundance_ref,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                return_Z=(self.similarities_out is not None),
            )
            futures.append(chunk_future)
        process_refs(futures)
        if self.similarities_out is not None:
            self.similarities_out += self.similarities_out.T
            for i in range(self.X.shape[0]):
                self.similarities_out[i, i] = 1.0
        return result
