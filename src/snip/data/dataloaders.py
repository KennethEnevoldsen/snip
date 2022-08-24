"""Dataloaders for SNP formats."""
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from pandas_plink import read_plink1_bin, write_plink1_bin
from torch.utils.data import IterableDataset
from wasabi import msg
from xarray import DataArray


class PLINKIterableDataset(IterableDataset):
    """An iterable dataset for PLINK datasets.

    Attributes:
        buffer_size
        genotype
        seed
        shuffle
        path
        snp_imputation_method
        convert_to_tensor
    """

    def __init__(
        self,
        path: Union[str, Path],
        buffer_size: int = 1024,
        shuffle: bool = True,
        limit: Optional[int] = None,
        chromosome: Optional[int] = None,
        seed: int = 42,
        to_tensor: bool = True,
        impute_missing: Optional[str] = None,
        snp_replace_value: Union[int, float] = -1,
        verbose: bool = False,
    ) -> None:
        """Load a PLINK file as an iterable dataset.

        A PLINK iterable dataset loads .bed or .zarr files along with
        its metadata using XArray, which allow for loading the SNPs along with
        their metadata.

        Args:
            path (Union[str, Path]): Path to the .bed or .zarr file. If it is a
                .zarr file, it will load the "genotype" DataArray from the loaded Xarray
                dataset.
            buffer_size (int): Defaults to 1024.
            shuffle (bool): Should it shuffle the dataset using a shuffle buffer.
                Defaults to True.
            limit (Optional[int]): Limit the dataset to use to n samples. Defaults to
                None. In which case it does not limit the dataset.
            chromosome (Optional[int]): Defaults to None,
                indicating all chromosomes.
            seed (int): Random seed to use. Default to 42.
            to_tensor (bool): Should the iterable dataset use torch tensors? Defaults to
                True.
            impute_missing (Optional[str]): Impute missing snps. Valid strategies include,
                "mean", "replace with value" (replace missing with snp_replace_value).
                "mode" (most common snp). Default to None in which case it does not impute
                missing SNPs.
            snp_replace_value (Union[int, float]): If impute_missing =
                "replace with value" is true, what should it replace it with? Default
                to -1.
            verbose (bool): toggles the verbosity of the class.
        """
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.convert_to_tensor = to_tensor
        self.impute_missing_method = impute_missing
        self.snp_replace_value = snp_replace_value
        self.path = path
        self.limit = limit
        self.verbose = verbose

        self.__from_disk(path, limit=limit)
        self.set_chromosome(chromosome)

    def set_chromosome(self, chromosome: Optional[int]) -> None:
        """Filter the dataset according to chromosome.

        Args:
            chromosome (Optional[int]): The number of chromosome til filter.
        """
        if chromosome:
            self.genotype = self._genotype.where(
                self._genotype.chrom == str(chromosome),
                drop=True,
            )
        else:
            self.genotype = self._genotype

    def __create_iter(self) -> Iterator:
        for X in self.batch_iter(self.buffer_size):
            for x in X:
                yield x

    def create_data_array_iter(self, batch_size: Optional[int] = None) -> Iterator:
        """Create iterable of DataArrays.

        If shuffle is True, the data is self.shuffle using a shufflebuffer.

        Args:
            batch_size (Optional[int]): Defaults to None. If not None,
                the data is returned in batches of size batch_size.

        Returns:
            Iterator: An iterator of DataArray object containing the genotype data.
        """
        if self.impute_missing_method:
            self.impute_missing(self.impute_missing_method)

        if batch_size:
            dataset_iter = self.batch_iter(batch_size)
        else:
            dataset_iter = self.__create_iter()

        if self.shuffle:
            dataset_iter = self.shuffle_buffer(dataset_iter)
        return dataset_iter

    def batch_iter(self, batch_size: int) -> Iterator:
        """Create an iterator that returns batches of size batch_size.

        Args:
            batch_size (int): The batch size.

        Yields:
            DataArray: A DataArray object containing the genotype data.
        """
        n, _ = self.genotype.shape

        starts = range(0, n, batch_size)
        ends = range(batch_size, n, batch_size)

        if self.impute_missing_method:
            replacement_value = self.genotype.coords[
                f"{self.impute_missing_method}_snp"
            ]

        end = 0  # if batch_size > array.shape[1]
        for start, end in zip(starts, ends):
            X = self.genotype[start:end].compute()  # shape: batch size, snps
            if self.impute_missing_method:
                X.data = np.where(np.isnan(X.data), replacement_value, X.data)

            if self.convert_to_tensor:
                X = self.to_tensor(X)
            yield X
        if n > end:
            X = self.genotype[end:n].compute()
            if self.convert_to_tensor:
                X = self.to_tensor(X)
            yield X

    def __iter__(self) -> Iterator:
        """Create a iterator of the dataset."""
        dataset_iter = self.create_data_array_iter()

        for x in dataset_iter:
            yield x

    def to_disk(
        self,
        path: Union[str, Path],
        chunks: int = 2**13,
        overwrite: bool = True,
    ) -> None:
        """Save the dataset to disk.

        Args:
            path (Union[str, Path]): Path to save the dataset. Save format is determined
                by the file extension. Options include ".bed" or ".zarr". Defaults to
                ".zarr".
            chunks (int): Defaults to 2**13. The chunk size to be passed to
                Xarray.chunk, Defaults to 2**13.
            overwrite (bool): Should it overwrite? Default to True.
        """
        ext = os.path.splitext(path)[-1]
        if self.verbose:
            msg.info("Saving to disk at location: {path}")
        if ext == ".bed":
            write_plink1_bin(self.genotype, path)
        elif ext == ".zarr":
            genotype = self.genotype.chunk(chunks)
            ds = xr.Dataset(dict(genotype=genotype))
            if overwrite:
                ds.to_zarr(path, mode="w", consolidated=True, compute=True)
            else:
                ds.to_zarr(path, consolidated=True, compute=True)
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")

    def __from_disk(
        self,
        path: Union[str, Path],
        limit: Optional[int] = None,
        rechunk: Optional[bool] = None,
    ) -> None:
        """Load the dataset from disk.

        Args:
            path (Union[str, Path]): Path to the dataset. Read format is determined by the
                file extension. Options include ".bed" or ".zarr".
            limit (Optional[int]): Defaults to None. If not None,
                only the first limit number of rows will be loaded.
            rechunk (bool): Defaults to False. If True, the dataset will
                be rechunked into chunks of size 2**13.
        """
        ext = os.path.splitext(path)[-1]
        if ext == ".bed":
            self._genotype = read_plink1_bin(str(path))
        elif ext == ".zarr":
            zarr_ds = xr.open_zarr(path)
            self._genotype = zarr_ds.genotype
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")

        if limit:
            self._genotype = self._genotype[:limit]
        if rechunk is None and ext == ".zarr":
            self._genotype = self._genotype.chunk(2**13)
        elif rechunk:
            self._genotype = self._genotype.chunk(2**13)

    def to_tensor(self, x: DataArray) -> torch.Tensor:
        """Convert DataArray to tensor.

        Args:
            x (DataArray): A DataArray object containing the genotype data.

        Returns:
            torch.Tensor: The converted array as a torch tensor.
        """
        return torch.from_numpy(x.compute().data)

    def shuffle_buffer(self, dataset_iter: Iterator) -> Iterator:
        """Create a shuffle buffer of an iterator.

        Args:
            dataset_iter (Iterator): An iterator to shuffle.

        Yields:
            Union[torch.Tensor, DataArray]: An shuffled arrays
        """
        random.seed(self.seed)

        shufbuf = []
        try:
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

    def train_test_split(
        self,
        test_size: Union[float, int, None] = 0.20,
        train_size: Union[float, int, None] = None,
    ) -> Tuple[IterableDataset, IterableDataset]:
        """Create a train test split of the iterable dataset.

        Args:
            test_size (Union[float, int, None]): The test size. Either
                supplied as a percentage (float) or as a count (int). Defaults to 0.20.
            train_size (Union[float, int, None]): The train size. Should not
                be supplied if test_size is given. Either supplied as a percentage
                (float) or as a count (int). Defaults to None.

        Raises:
            ValueError: If both test_size and train_size is given.

        Returns:
            Tuple[IterableDataset, IterableDataset]: The train and test set,
                respectively.
        """
        samples = self.genotype.shape[0]

        if test_size and train_size:
            raise ValueError(
                "You can only supply either test_size or train_size. "
                + "The other is determined",
            )

        def __get_split_size(test_size, samples):
            if isinstance(test_size, int):
                n_test = test_size
            if isinstance(test_size, float):
                assert (
                    test_size > 0 and test_size < 1
                ), "if test size is a float it must be between zero and one."
                n_test = samples // (100 / (100 * test_size))
            return n_test

        if test_size:
            n_test = __get_split_size(test_size, samples)
        if train_size:
            n_train = __get_split_size(train_size, samples)
            n_test = samples - n_train

        # create random mask
        mask_array = np.zeros(samples)
        mask_array[0 : int(n_test)] = 1
        np.random.shuffle(mask_array)

        train = deepcopy(self)
        test = deepcopy(self)
        train.genotype = train.genotype[mask_array == 0]
        test.genotype = test.genotype[mask_array == 1]
        return train, test

    def impute_missing(self, method: str = "mean", save_after_imputation: bool = False):
        """Impute missing snps.

        Args:
            method (str): Method for imputing missing snps. Options include
                "mean" and "mode" (most common). Defaults to "mean".
            save_after_imputation (bool): Save after imputation to self.path
                (the path it was loaded from). Defaults to False.
        """
        imputated_snps = False
        if method == "mean" and "mean_snp" not in self.genotype.coords:
            imputated_snps = True
            if self.verbose:
                msg.info("Computing mean SNP")
            mean_snp = self.genotype.mean(axis=0, skipna=True).compute()
            mean_snp = mean_snp.fillna(0)  # if all of one snp is NA
            self.genotype = self.genotype.assign_coords(mean_snp=mean_snp)
        if method == "mode" and "mode_snp" not in self.genotype.coords:
            imputated_snps = True
            if self.verbose:
                msg.info("Computing most common SNP")
            most_common = np.array(
                [
                    Counter(snps.compute().data).most_common(1)[0][0]
                    for snps in self.genotype.data.T
                ],
            )
            most_common = np.nan_to_num(most_common, nan=0)  # if all of one snp is NA
            self.genotype = self.genotype.assign_coords(
                {"mode_snp": ("variant", most_common)},
            )
        if save_after_imputation and imputated_snps:
            self.to_disk(self.path, overwrite=True)
