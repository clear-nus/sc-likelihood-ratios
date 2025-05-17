"""
Modified from 
https://github.com/lucaslie/torchprune/tree/b753745b773c3ed259bf819d193ce8573d89efbb/src/torchprune/torchprune/util/datasets
All credits go to the authors of the original code
"""

import os
from abc import ABC, abstractmethod
import json
import subprocess

import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader


def objectnet_to_imagenet113_mapping():

    #hardcode 113 unique indices from ObjectNet that overlap with ImageNet
    unique_elements = [409, 412, 414, 418, 419, 423, 434, 440, 446, 455, 457, 462, 463, 470,
        473, 479, 487, 499, 504, 507, 508, 531, 533, 539, 543, 545, 549, 550,
        560, 567, 578, 587, 588, 589, 601, 606, 608, 610, 618, 619, 620, 623,
        626, 629, 630, 632, 644, 647, 651, 658, 659, 664, 671, 673, 677, 679,
        695, 696, 700, 703, 720, 721, 725, 728, 729, 731, 732, 737, 740, 742,
        752, 761, 769, 770, 772, 773, 774, 778, 783, 790, 792, 797, 804, 806,
        809, 811, 813, 828, 834, 837, 841, 842, 846, 849, 850, 851, 859, 868,
        879, 882, 883, 893, 898, 902, 907, 909, 923, 930, 950, 951, 954, 968,
        999]
    mapping_dict= {old: new for new, old in enumerate(unique_elements)}
    return mapping_dict


class DownloadDataset(data.Dataset, ABC):
    """Custom abstract class for that can download and maintain dataset."""

    @property
    @abstractmethod
    def _train_tar_file_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _test_tar_file_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _train_dir(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _test_dir(self):
        raise NotImplementedError

    # @property
    # def _file_url(self):
    #     return f"file://{self._file_dir}/"

    @abstractmethod
    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        raise NotImplementedError

    @abstractmethod
    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        raise NotImplementedError

    @abstractmethod
    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        raise NotImplementedError

    @abstractmethod
    def _convert_target(self, target):
        """Convert target to correct format."""
        raise NotImplementedError

    def __init__(
        self,
        root,
        # file_dir,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """Initialize ImageNet dataset.

        Args:
            root (str): where to store the data set to be downloaded.
            file_dir (str): where to look before downloading it from S3.
            train (bool, optional): train or test data set. Defaults to True.
            transform (torchvision.transforms, optional): set of transforms to
            apply to input data. Defaults to None.
            target_transform (torchvision.transforms, optional): set of
            transforms to apply to target data. Defaults to None.
            download (bool, optional): try downloading. Defaults to False.

        """
        # expand the root to the full path and ensure it exists
        self._root = os.path.realpath(root)
        os.makedirs(self._root, exist_ok=True)

        # expand data_dir to the full path and ensure it exists
        # self._file_dir = os.path.realpath(file_dir)
        # os.makedirs(self._file_dir, exist_ok=True)

        # check whether it's training
        self._train = train

        # store transforms
        self.transform = transform
        self.target_transform = target_transform

        # check path and tar file to either train or test data
        if self._train:
            self._data_path = os.path.join(self._root, self._train_dir)
        else:
            self._data_path = os.path.join(self._root, self._test_dir)

        if download:
            # download now if needed
            self._download()
        elif not os.path.exists(self._data_path):
            # check if path exists
            raise FileNotFoundError("Data not found, please set download=True")

        # get the data
        if self._train:
            self._data = self._get_train_data(download)
        else:
            self._data = self._get_test_data(download)

    # def _download(self):
    #     """Download the data set from the cloud and return full file path."""
    #     # tar file name
    #     if self._train:
    #         tar_file = self._train_tar_file_name
    #     else:
    #         tar_file = self._test_tar_file_name

    #     def download_and_extract(url):
    #         """Use the torchvision fucntion to download and extract."""
    #         download_and_extract_archive(
    #             url=url + tar_file,
    #             download_root=self._root,
    #             filename=tar_file,
    #             extract_root=self._root,
    #             remove_finished=True,
    #         )

    #     # try downloading it from files URL.
    #     if tar_file is not None and not os.path.exists(self._data_path):
    #         download_and_extract(self._file_url)

    def __getitem__(self, index):
        """Return appropriate item."""
        img, target = self._data[index]

        # it might not be a PIL image yet ...
        img = self._convert_to_pil(img)

        # target might be weird, so convert it forst
        target = self._convert_target(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Get total number of data points."""
        return len(self._data)
    
    
class ImageNet(DownloadDataset):
    """Custom class for ImageNet that can download and maintain dataset."""

    @property
    def _train_tar_file_name(self):
        return "imagenet_object_localization.tar.gz"

    @property
    def _test_tar_file_name(self):
        return self._train_tar_file_name

    @property
    def _train_dir(self):
        return "ILSVRC/Data/CLS-LOC/train"

    @property
    def _test_dir(self):
        return "ILSVRC/Data/CLS-LOC/val"

    @property
    def _valprep_file(self):
        """Return file that gives the class for each validation image.

        File is taken from:
        https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
        """
        valprep = os.path.join(__file__, "../imagenetval/valprep.sh")
        return os.path.realpath(valprep)

    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        return ImageFolder(root=self._data_path)

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        # retrieve val image name to val target class map from valprep
        val_lookup = {}
        classes_lookup = {}
        with open(self._valprep_file, "r") as file:
            valprep = file.read().split("\t\n")
        for line in valprep:
            if "mv" not in line and ".JPEG" not in line:
                continue

            _, val_img_name, target_class = line[:-1].split(" ")

            # store hash map from img to target_class
            val_lookup[val_img_name] = target_class

            # store hash map to look up class keys
            classes_lookup[target_class] = None

        classes = list(classes_lookup.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        assert len(classes) == 1000

        files = list(val_lookup.keys())
        targets = [class_to_idx[val_lookup[file]] for file in files]

        return list(zip(files, targets))

    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        if self._train:
            return img
        else:
            return pil_loader(os.path.join(self._data_path, img))

    def _convert_target(self, target):
        return int(target)


class ObjectNet(ImageNet):
    """ObjectNet with ImageNet only classes."""

    @property
    def _test_tar_file_name(self):
        return "objectnet-1.0.zip"

    @property
    def _test_dir(self):
        return "objectnet-1.0/images"

    def _convert_to_pil(self, img):
        img = super()._convert_to_pil(img)

        if self._train:
            return img
        else:
            border = 3
            return img.crop(
                (border, border, img.size[0] - border, img.size[1] - border)
            )

    def _get_test_data(self, download):
        # have pytorch's ImageFolder class analyze the directories
        o_dataset = ImageFolder(self._data_path)

        # get mappings folder
        mappings_folder = os.path.abspath(
            os.path.join(self._data_path, "../mappings")
        )

        # get ObjectNet label to ImageNet label mapping
        with open(
            os.path.join(mappings_folder, "objectnet_to_imagenet_1k.json")
        ) as file_handle:
            o_label_to_all_i_labels = json.load(file_handle)

        # now remove double i labels to avoid confusion
        o_label_to_i_labels = {
            o_label: all_i_label.split("; ")
            for o_label, all_i_label in o_label_to_all_i_labels.items()
        }

        # some in-between mappings ...
        o_folder_to_o_idx = o_dataset.class_to_idx
        with open(
            os.path.join(mappings_folder, "folder_to_objectnet_label.json")
        ) as file_handle:
            o_folder_o_label = json.load(file_handle)

        # now get mapping from o_label to o_idx
        o_label_to_o_idx = {
            o_label: o_folder_to_o_idx[o_folder]
            for o_folder, o_label in o_folder_o_label.items()
        }

        # some in-between mappings ...
        with open(
            os.path.join(mappings_folder, "pytorch_to_imagenet_2012_id.json")
        ) as file_handle:
            i_idx_to_i_line = json.load(file_handle)
        with open(
            os.path.join(mappings_folder, "imagenet_to_label_2012_v2")
        ) as file_handle:
            i_line_to_i_label = file_handle.readlines()

        i_line_to_i_label = {
            i_line: i_label[:-1]
            for i_line, i_label in enumerate(i_line_to_i_label)
        }

        # now get mapping from i_label to i_idx
        i_label_to_i_idx = {
            i_line_to_i_label[i_line]: int(i_idx)
            for i_idx, i_line in i_idx_to_i_line.items()
        }

        # now get the final mapping of interest!!!
        o_idx_to_i_idxs = {
            o_label_to_o_idx[o_label]: [
                i_label_to_i_idx[i_label] for i_label in i_labels
            ]
            for o_label, i_labels in o_label_to_i_labels.items()
        }

        # now get a list of files of interest
        # map indices to [0,112]
        remapping_indices_dict = objectnet_to_imagenet113_mapping()
        overlapping_samples = []
        for filepath, o_idx in o_dataset.samples:
            if o_idx not in o_idx_to_i_idxs:
                continue
            
            rel_file = os.path.relpath(filepath, self._data_path)
            # overlapping_samples.append((rel_file, o_idx_to_i_idxs[o_idx][0]))
            overlapping_samples.append((rel_file, remapping_indices_dict[o_idx_to_i_idxs[o_idx][0]]))

        return overlapping_samples

    # def _download(self):
    #     """Download the data set from the cloud and return full file path."""
    #     # same download for training, otherwise we have pw-protected zip.
    #     if self._train:
    #         return super()._download()

    #     # download first.
    #     download_url(
    #         url=self._file_url + self._test_tar_file_name,
    #         root=self._root,
    #         filename=self._test_tar_file_name,
    #     )

    #     # now extract and consider password protection.
    #     from_path = os.path.join(self._root, self._test_tar_file_name)

    #     # with zipfile.ZipFile(from_path, "r") as f_zip:
    #     #     f_zip.extractall(
    #     #         self._root, pwd=bytes("objectnetisatestset", "utf-8")
    #     #     )

    #     # unzip with bash's unzip: so much faster due to password protection
    #     subprocess.run(
    #         ["unzip", "-P", "objectnetisatestset", from_path, "-d", self._root]
    #     )

    #     os.remove(from_path)