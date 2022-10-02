"""Dataloaders for SNP formats."""
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

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
        genotype: Optional[DataArray] = None,
    ) -> None:
        """Load a PLINK file as an iterable dataset.

        A PLINK iterable dataset loads .bed or .zarr files along with
        its metadata using XArray, which allow for loading the SNPs along with
        their metadata.

        Args:
            path (Union[str, Path]): Path to the .bed or .zarr
                file. If it is a .zarr file, it will load the "genotype" DataArray from
                the loaded Xarray dataset.
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
            genotype (Optional[DataArray]): If supplied the path is ignored and this
                genotype array is used.
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
        self.chromosome = chromosome

        if genotype is None:
            self._genotype = self.__from_disk(path, limit=limit)
        else:
            self._genotype = genotype
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
            ].compute()

        end = 0  # if batch_size > array.shape[1]
        for start, end in zip(starts, ends):
            X = self.genotype[start:end].compute()  # shape: batch size, snps
            if self.impute_missing_method:
                X.data = np.where(np.isnan(X.data), replacement_value.data, X.data)

            if self.convert_to_tensor:
                X = self.to_tensor(X)
            yield X
        if n > end:
            X = self.genotype[end:n].compute()
            if self.impute_missing_method:
                X.data = np.where(np.isnan(X.data), replacement_value.data, X.data)

            if self.convert_to_tensor:
                X = self.to_tensor(X)
            yield X

    def get_X(self) -> DataArray:
        """Extract the dataset as a array, useful for e.g. sklearn
        pipelines."""

        X = self.genotype
        # extract the data
        if self.impute_missing_method:
            replacement_value = self.genotype.coords[
                f"{self.impute_missing_method}_snp"
            ].compute()
            X_ = np.where(np.isnan(X.data), replacement_value.data, X.data)
        return X_

    def __iter__(self) -> Iterator:
        """Create a iterator of the dataset."""
        dataset_iter = self.create_data_array_iter()

        for x in dataset_iter:
            yield x

    def update_on_disk(self) -> None:
        """Update the file on disk."""
        if not self.path:
            raise ValueError("Can't derive save path, as self.path is None or empty")
        if Path(self.path).suffix == ".zarr":
            self.to_disk(self.path, mode="a")
        else:
            self.to_disk(self.path, mode="w")

    def to_disk(
        self,
        path: Union[str, Path],
        chunks: int = 2**13,
        mode: Optional[str] = None,
    ) -> None:
        """Save the dataset to disk.

        Args:
            path (Union[str, Path, None]): Path to save the dataset. Save format is determined
                by the file extension. Options include ".bed" or ".zarr". Defaults to
                ".zarr".
            chunks (int): Defaults to 2**13. The chunk size to be passed to
                Xarray.chunk, Defaults to 2**13.
            mode (Optional[str]): Defaults to None. The mode to use when saving the
                dataset. Use "w" to overwrite the dataset. "a" to modify exisiting
                dataset.
        """
        ext = os.path.splitext(path)[-1]
        if self.verbose:
            msg.info("Saving to disk at location: {path}")
        if ext == ".bed":
            write_plink1_bin(self.genotype, path)
        elif ext == ".zarr":
            genotype = self.genotype.chunk(chunks)
            self.__to_zarr(path, genotype, mode=mode, verbose=self.verbose)
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")

    @staticmethod
    def __to_zarr(
        path: Union[str, Path],
        genotype: DataArray,
        mode: Optional[str],
        verbose: bool,
    ) -> None:
        if verbose:
            msg.warn(
                "Normalizing object dtypes to string dtypes, for more see "
                + "https://github.com/pydata/xarray/issues/3476",
            )
        for v in list(genotype.coords.keys()):
            if genotype.coords[v].dtype == object:
                genotype.coords[v] = genotype.coords[v].astype("unicode")
        ds = xr.Dataset(dict(genotype=genotype))
        ds.to_zarr(str(path), mode=mode, compute=True)

    def __from_disk(
        self,
        path: Union[str, Path],
        limit: Optional[int] = None,
        rechunk: Optional[bool] = None,
    ) -> DataArray:
        """Load the dataset from disk.

        Args:
            path (Union[str, Path]): Path to the dataset. Read format is determined by
                the file extension. Options include ".bed" or ".zarr".
            limit (Optional[int]): Defaults to None. If not None,
                only the first limit number of rows will be loaded.
            rechunk (bool): Defaults to False. If True, the dataset will
                be rechunked into chunks of size 2**13.

        Returns:
            DataArray: A DataArray object containing the genotype data.
        """
        ext = os.path.splitext(path)[-1]
        if ext == ".bed":
            genotype = read_plink1_bin(str(path))
        elif ext == ".zarr":
            zarr_ds = xr.open_zarr(path)
            genotype = zarr_ds.genotype
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")
        if rechunk is None and ext == ".zarr":
            genotype = genotype.chunk(2**13)

        if limit:
            genotype = genotype[:limit]
        elif rechunk:
            genotype = genotype.chunk(2**13)
        return genotype

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
                + "The other is determined.",
            )
        if test_size is None and train_size is None:
            raise ValueError("You must supply either test_size or train_size.")

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

    def is_missing_imputed(self, method: Optional[str] = None) -> bool:
        """Check if missing snps in the dataset is imputed.

        Args:
            method (Optional[str]): The imputation method. Defaults to None. In which
                case, the method is determined from self.impute_missing_method.

        Returns:
            bool: True if missing snps in the dataset is imputed.
        """
        if method is None:
            method = self.impute_missing_method

        if method == "mean" and "mean_snp" in self.genotype.coords:
            # check if mean_snp is not nan
            return not np.isnan(self.genotype.coords["mean_snp"]).any()
        if method == "mode" and "mode_snp" in self.genotype.coords:
            return not np.isnan(self.genotype.coords["mean_snp"]).any()
        if method == "replace with value":
            return True
        return False

    def impute_missing_from(
        self,
        dataset: "PLINKIterableDataset",
        method: Optional[str] = None,
    ):
        """Impute missing snps based on another dataset. Important for
        computing missing values on the test and validation set.

        Args:
            dataset (PLINKIterableDataset): Another plink dataset
            method (Optional[str], optional): Method for imputing snp. Defaults to None.
                Where it is used the method defined in the __init__.
        """
        if method is None:
            method = self.impute_missing_method
        if not dataset.is_missing_imputed(method):
            raise ValueError(
                "The dataset to impute from must be imputed. "
                + "Use dataset.impute_missing() to impute.",
            )

        if method == "mean":
            self.genotype = self.genotype.assign_coords(
                mean_snp=dataset.genotype.coords["mean_snp"],
            )
        elif method == "mode":
            self.genotype = self.genotype.assign_coords(
                mode_snp=dataset.genotype.coords["mode_snp"],
            )
        elif method == "replace with value":
            self.snp_replace_value = dataset.snp_replace_value
        else:
            raise ValueError("Unknown imputation method.")

    def impute_missing(
        self,
        method: Optional[str] = None,
    ):
        """Impute missing snps.

        Args:
            method (Optional[str]): Method for imputing missing snps. Options include
                "mean" and "mode" (most common). Defaults to "mean".
        """
        if method is None:
            method = self.impute_missing_method

        if self.is_missing_imputed(method):
            return

        if method == "mean":
            if self.verbose:
                msg.info("Computing mean SNP")
            mean_snp = self.genotype.mean(axis=0, skipna=True).compute()
            mean_snp = mean_snp.fillna(0)  # if all of one snp is NA
            self.genotype = self.genotype.assign_coords(mean_snp=mean_snp)
        if method == "mode":
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

    def split_into_strides(
        self,
        stride: int,
        drop_last: bool = True,
    ) -> List["PLINKIterableDataset"]:
        """Splits the dataset into multiple datasets.

        Each dataset contains {stides} SNPs

        Args:
            stride (int): The number of SNPs to split the dataset into.
            drop_last (bool): If True, the last dataset will be smaller than the others.
                Defaults to True.

        Returns:
            List[PLINKIterableDataset]: List of datasets
        """
        if self.verbose:
            msg.info(f"Splitting dataset into {stride} strides")
        datasets = []
        for i in range(0, self.genotype.shape[1], stride):
            datasets.append(
                PLINKIterableDataset(
                    self.path,
                    chromosome=self.chromosome,
                    genotype=self.genotype.isel(variant=slice(i, i + stride)),
                    verbose=self.verbose,
                    impute_missing=self.impute_missing_method,
                ),
            )
        if drop_last:
            datasets = datasets[:-1]
        return datasets

    @staticmethod
    def from_array(
        arr: Union[torch.Tensor, DataArray, np.ndarray],
        metadata_from: Union[DataArray, "PLINKIterableDataset"],
    ) -> "PLINKIterableDataset":
        """Create a PLINKIterableDataset from an array.

        Args:
            arr (Union[torch.Tensor, DataArray, np.ndarray]): The array to create the dataset from.
            metadata_from (Optional[DataArray, PLINKIterableDataset]): The metadata to
                use for the dataset.

        raises:
            ValueError: If the metadata is not supplied and the array is not a
                DataArray.

        Returns:
            PLINKIterableDataset: The dataset.
        """
        if not isinstance(arr, DataArray):
            if metadata_from is None:
                raise ValueError(
                    "You must supply metadata if you are not supplying a DataArray.",
                )
            # convert to np.array
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            elif not isinstance(arr, np.ndarray):
                raise ValueError(
                    "The array must be a torch.Tensor, np.ndarray, or "
                    + f"xarray.DataArray. Got {type(arr)}.",
                )
            if isinstance(metadata_from, PLINKIterableDataset):
                metadata_from = metadata_from.genotype
            if arr.shape == metadata_from.shape:
                arr = xr.DataArray(arr, coords=metadata_from.coords)
            else:
                coords = {
                    "variant": np.arange(0, arr.shape[1]),  # create variant id
                    "chrom": ("variant", np.repeat(1, arr.shape[1])),  # add chromosome
                    "a0": ("variant", np.repeat("A", arr.shape[1])),  # add allele 1
                    "a1": ("variant", np.repeat("B", arr.shape[1])),  # add allele 2
                    "snp": (
                        "variant",
                        np.array([f"c{t}" for t in range(arr.shape[1])]),
                    ),  # add SNP id
                    "pos": ("variant", np.arange(0, arr.shape[1])),  # add position
                }
                # transfer coords frome the first dimension (sample)
                for k in metadata_from.sample._coords:
                    coords[k] = ("sample", metadata_from.sample[k].data)
                arr = xr.DataArray(arr, dims=["sample", "variant"], coords=coords)
        return PLINKIterableDataset(path="", genotype=arr)


def combine_plinkdatasets(
    datasets: List[PLINKIterableDataset],
    along_dim: str = "sample",
    rewrite_variants: Optional[bool] = None,
) -> PLINKIterableDataset:
    """Merge multiple PLINKIterableDatasets into one.

    Args:
        datasets (List[PLINKIterableDataset]): The datasets to merge.
        along_dim (str): The dimension to merge along. Either variant or sample.
        rewrite_variants (Optional[bool]): If True, the variant IDs will be rewritten.
            Defaults to None. If None, the variant IDs will be rewritten if the
            datasets have overlapping variant IDs.

    Returns:
        PLINKIterableDataset: The merged dataset.
    """
    if len(datasets) == 1:
        return datasets[0]
    dataset = datasets[0]
    genotypes = [ds.genotype for ds in datasets]
    if along_dim == "variant":
        genotype = xr.concat(genotypes, dim="variant")

    elif along_dim == "sample":
        genotype = xr.concat(genotype, dim="sample")
    else:
        raise ValueError(f"Dimension {along_dim} not recognised.")
    if rewrite_variants is None:
        variants = genotype.variant.data
        rewrite_variants = np.unique(variants).shape == variants.shape

    if rewrite_variants:
        coords = {
            "variant": np.arange(0, genotype.shape[1]),  # create variant id
            "chrom": ("variant", np.repeat(1, genotype.shape[1])),  # add chromosome
            "a0": ("variant", np.repeat("A", genotype.shape[1])),  # add allele 1
            "a1": ("variant", np.repeat("B", genotype.shape[1])),  # add allele 2
            "snp": (
                "variant",
                np.array([f"c{t}" for t in range(genotype.shape[1])]),
            ),  # add SNP id
            "pos": ("variant", np.arange(0, genotype.shape[1])),  # add position
        }
        genotype.assign_coords(coords)

    dataset.genotype = genotype
    dataset.path = ""
    return dataset
