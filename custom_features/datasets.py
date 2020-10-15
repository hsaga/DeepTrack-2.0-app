import deeptrack as dt
import os
import glob
import itertools


def _split_to_folder_structure(path):
    return os.path.normpath(path).split(os.path.sep)


def _load(path):
    if isinstance(path, str):
        return dt.LoadImage(path=path).resolve()
    else:
        return path


class Dataset:
    def __init__(self, *args, **kwargs):
        self.iterators = {}
        super().__init__(*args, **kwargs)

    def _as_iterator(self, selector, sort_results=False):
        if selector in self.iterators:
            iterator = self.iterators[data]
        else:
            iterator_list = list(glob.glob(data))
            assert len(iterator_list) > 0, "Data selector {0} matches no files".format(
                iterator_list
            )
            if sort_results:
                iterator_list = sorted(iterator_list)
            iterator = itertools.cycle(iterator_list)
            self.iterators[selector] = iterator

        return iterator


class ImageToImageDataset(dt.Feature, Dataset):
    """Loads a dataset file by file.

    Dataset loader for image-to-image networks, where each sample is in its own file.

    Expects a file structure like

    .. code-block::

       root_path/
       . . path_to_data/
       . . . . file_1...
       . . . . file_2...
       . . path_to_label/
       . . . . label_of_file_1...
       . . . . label_of_file_2...

    root_path can be any path and is set by the property `root_path`
    The files are expected to be ordered such that they will be loaded in
    order by the glob module. A good way to guarantee this is to name the
    data file and the label file is the same, just in different folder, or
    just a different prefix, with the same suffix (such as a numbering 0,
    1...).

    Note that the data and the label do not be in different folders. The
    following is also a valid structure:

    .. code-block::

       root_path/
       . . file_1...
       . . file_2...
       ...
       . . label_of_file_1...
       . . label_of_file_2...

    Parameters
    ----------
    root_path : str
        The base path of the dataset
    data_selector : str
        A selector identifying the data files. Does not need to specify root_path.
        Follows the syntax of the glob module, where * is a wildcard. For
        example `'data/*'` matches all files in the folder data. Regex is
        in large parts supported.
    label_selector : str
        A selector identifying the label files. Does not need to specify root_path.
        Follows the syntax of the glob module, where * is a wildcard. For
        example `'label/*'` matches all files in the folder label. Regex is
        in large parts supported.
    sort_results : bool
        Sorts the lists of files. This is only useful if the glob module does not
        orders the files on some other metric than alphabetical ordering.

    """

    __distributed__ = False

    __extra_properties__ = ["data", "label"]

    def __init__(
        self,
        root_path="./",
        data_selector="data/*",
        label_selector="label/*",
        multiple_samples_per_file=True,
        sort_results=False,
        **kwargs
    ):
        def get_next_of_data_iterator(root_path, data_selector, sort_results):
            data = os.path.join(root_path, data_selector)
            iterator = self._as_iterator(data, sort_results)
            return next(iterator)

        def get_next_of_label_iterator(root_path, label_selector, sort_results):
            label = os.path.join(root_path, label_selector)
            iterator = self._as_iterator(label, sort_results)
            return next(iterator)

        super().__init__(
            root_path=root_path,
            data_selector=data_selector,
            label_selector=label_selector,
            data=get_next_of_data_iterator,
            label=get_next_of_label_iterator,
            **kwargs
        )

    def get(self, *args, data, label, is_label=False, **kwargs):
        if is_label:
            return dt.LoadImage(path=label).resolve()
        else:
            return dt.LoadImage(path=data).resolve()


class ImageToClassDataset(dt.Feature, Dataset):
    """Loads a classification dataset.

    Dataset loader for image-to-class networks, where each image is in its
    own file, and each class is a folder.

    Expects a file structure like

    .. code-block::

       root_path/
       . . class_1/
       . . . . file_1...
       . . . . file_2...
       . . class_2/
       . . . . file_1...
       . . . . file_2...
       . . class_3/
       . . . . file_1...
       . . . . file_2...
       ...

    root_path can be any path and is set by the property `root_path`

    Parameters
    ----------
    root_path : str
        The base path of the dataset
    class_selector : str
        Selector for classes. Default selects all classes in `root_path`.
        Uses the glob module syntax.
    balance_classes : bool
        If True, each class is equally likely to be sampled. If False,
        the probably of drawing a sample from a certain class is based
        on its relative population. That is, if a set has 100 samples,
        out of which 10 belongs to class 1 and 90 class 90, class 1
        will be chosen 10% of the time.

    """

    __distributed__ = False

    __extra_properties__ = [
        "data_class_tuple",
        "data",
        "class_name",
        "class_names",
    ]

    def __init__(
        self, root_path="./", class_selector="*", balance_classes=True, **kwargs
    ):
        setattr(self, "attr", None)

        def get_next_of_data_iterator(root_path, class_selector, balance_classes):
            if balance_classes:
                selector = os.path.join(root_path, class_selector)
                class_iterator = self._as_iterator(selector)
                next_class = next(class_iterator)

                structure = _split_to_folder_structure(next_class)
                class_name = structure[-1] or structure[-2]

                selector = os.path.join(next_class, "*")
                data_iterator = self._as_iterator(selector)
                data = next(data_iterator)

            else:
                selector = os.path.join(root_path, class_selector, "*")
                data_iterator = self._as_iterator(selector)
                data = next(data_itertor)
                class_name = _split_to_folder_structure(next_class)[-2]

            return data, class_name

        def get_class_names(root_path, class_selector):
            selector = os.path.join(root_path, class_selector)
            return [
                l[-1] or l[-2]
                for l in [
                    _split_to_folder_structure(path) for path in glob.glob(selector)
                ]
            ]

        super().__init__(
            root_path=root_path,
            class_selector=class_selector,
            balance_classes=balance_classes,
            data_class_tuple=get_next_of_data_iterator,
            class_names=get_class_names,
            data=lambda data_class_tuple: data_class_tuple[0],
            class_name=lambda data_class_tuple: data_class_tuple[1],
            **kwargs
        )

    def get(self, *args, data, class_name, class_names, is_label=False, **kwargs):
        if is_label:
            return dt.LoadImage(path=label).resolve()
        else:
            return dict(
                [(item_name, int(class_name == item_name)) for item_name in class_names]
            )
