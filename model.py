"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
def load_fashion_mnist(n_train=10000, n_test=2000):
    # TODO: download the four idx gz files once, parse with np.frombuffer, return float tensors in 0-1 and int64 labels.
    # from functools import partial
    # dataset = {
    #     'train-images-idx3-ubyte.gz': 'train-labels-idx1-ubyte.gz', 
    #     't10k-images-idx3-ubyte.gz': 't10k-labels-idx1-ubyte.gz'
    # }
    # offsets = { 16 : 8 }
    # os.makedirs(tempfile.gettempdir(), exist_ok=True)
    # def full_path(p): return os.path.join(tempfile.gettempdir(), p)
    
    # for fname in [*dataset.keys(), *dataset.values()]:
    #     f_path = full_path(fname)
    #     if not os.path.exists(f_path):
    #         urllib.request.urlretrieve(base_url + fname, f_path)
    
    # def gzip_open(fname): return gzip.open(fname, 'rb').read()
    # def load_file_to_npbuffer(o, f): return np.frombuffer(f, offset=o, dtype=np.uint8)
    # def copy(arr): return arr.copy() if isinstance(arr, np.ndarray) else arr.clone()
    
    # def reshape(arr): return arr.reshape(-1, 28, 28)
    # def asf32dtype(arr): return arr.astype(np.float32)
    # def asi64dtype(arr): return arr.astype(np.int64)
    # def rescale(arr): return arr / 255

    # image_pipeline = [
    #     full_path,
    #     gzip_open,
    #     partial(load_file_to_npbuffer, list(offsets.keys())[0]),
    #     copy,
    #     reshape,
    #     rescale,
    #     asf32dtype,
    #     torch.from_numpy,
    # ]
    # label_pipeline = [
    #     full_path,
    #     gzip_open,
    #     partial(load_file_to_npbuffer, list(offsets.values())[0]),
    #     copy,
    #     asi64dtype,
    #     torch.from_numpy,
    # ]

    # def run_pipeline(fname: str, pipeline: list, n: int):
    #     data = fname
    #     for step in pipeline: data = step(data)
    #     return copy(data[:n])

    # return {
    #     'X_train': run_pipeline('train-images-idx3-ubyte.gz', image_pipeline, n_train),
    #     'y_train': run_pipeline('train-labels-idx1-ubyte.gz', label_pipeline, n_train),
    #     'X_test' : run_pipeline('t10k-images-idx3-ubyte.gz',  image_pipeline, n_test),
    #     'y_test' : run_pipeline('t10k-labels-idx1-ubyte.gz',  label_pipeline, n_test)
    # }
    base_url = (
        "https://storage.googleapis.com"
        "/tensorflow/tf-keras-datasets/"
    )
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img" : "t10k-images-idx3-ubyte.gz",
        "test_lbl" : "t10k-labels-idx1-ubyte.gz"
    }
    IMAGE_OFFSET, LABEL_OFFSET = 16, 8
    tmp_dir = tempfile.gettempdir()
    local_files = {}

    for key, filename in files.items():
        local_path = os.path.join(tmp_dir, filename)
        local_files[key] = local_path
        
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(base_url + filename, local_path)

    def parse_images(filepath, n_samples):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=IMAGE_OFFSET)

        images_reshaped = data.reshape(-1, 28, 28)
        images_sliced   = images_reshaped[:n_samples]
        images_tensor   = torch.tensor(images_sliced, dtype=torch.float32)
        images_rescaled = images_tensor / 255.0
        return images_rescaled

    def parse_labels(filepath, n_samples):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=LABEL_OFFSET)

        labels_sliced = data[:n_samples]
        labels_tensor = torch.tensor(labels_sliced, dtype=torch.int64)
        return labels_tensor

    return {
        "X_train": parse_images(local_files["train_img"], n_train),
        "y_train": parse_labels(local_files["train_lbl"], n_train),
        "X_test" : parse_images(local_files["test_img"] , n_test),
        "y_test" : parse_labels(local_files["test_lbl"] , n_test)
    }

    pass

# Step 2 - FashionDataset (not yet solved)
# TODO: implement

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

