import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import gzip

def read_idx_file(file_path, is_image=True):
    if not os.path.exists(file_path):
        if not file_path.endswith('.gz'):
            gz_path = file_path + '.gz'
            if os.path.exists(gz_path):
                file_path = gz_path
            else:
                raise FileNotFoundError(f"Neither {file_path} nor {gz_path} found")
    
    opener = gzip.open if file_path.endswith('.gz') else open
    with opener(file_path, 'rb') as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), 'big')
        # Read number of items
        num_items = int.from_bytes(f.read(4), 'big')
        
        if is_image:
            # Read rows and columns for images
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            # Read pixel data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, rows, cols)
        else:
            # Read label data
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        return data

def load_mnist_data(data_dir, prefix):
    image_files = [
        os.path.join(data_dir, f'{prefix}-images-idx3-ubyte'),
        os.path.join(data_dir, f'{prefix}-images-idx3-ubyte.gz')
    ]
    label_files = [
        os.path.join(data_dir, f'{prefix}-labels-idx1-ubyte'),
        os.path.join(data_dir, f'{prefix}-labels-idx1-ubyte.gz')
    ]
    
    image_file = next((f for f in image_files if os.path.exists(f)), None)
    label_file = next((f for f in label_files if os.path.exists(f)), None)
    
    if image_file is None or label_file is None:
        available_files = "\n".join(os.listdir(data_dir))
        raise FileNotFoundError(
            f"Could not find MNIST {prefix} files in {data_dir}\n"
            f"Available files:\n{available_files}"
        )
    
    images = read_idx_file(image_file, is_image=True)
    labels = read_idx_file(label_file, is_image=False)
    
    # normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    return images, labels

def load_mnist_train(data_dir):
    return load_mnist_data(data_dir, 'train')

def load_mnist_test(data_dir):
    return load_mnist_data(data_dir, 't10k')

def encode_one_hot(labels, num_classes=10):
    encoder = OneHotEncoder(categories=[range(num_classes)], sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot = encoder.fit_transform(labels)
    return one_hot

def augment_data(images, labels=None, augment=False):
    if not augment:
        return images, labels
    
    augmented_images = []
    augmented_labels = []
    
    for img, lbl in zip(images, labels):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        
        # random rotation ±10 degrees
        angle = np.random.uniform(-10, 10)
        rotated_img = pil_img.rotate(angle)
        
        # random translation ±2 pixels
        translate_x = np.random.randint(-2, 3)
        translate_y = np.random.randint(-2, 3)
        translated_img = rotated_img.transform(
            rotated_img.size, 
            Image.AFFINE, 
            (1, 0, translate_x, 0, 1, translate_y)
        )
        
        # transform back to numpy array and normalize
        augmented_img = np.array(translated_img) / 255.0
        augmented_images.append(augmented_img)
        augmented_labels.append(lbl)
    
    return np.array(augmented_images), np.array(augmented_labels)

def visualize_samples(images, labels=None, title="Sample Images", save_dir="data_preprocess_visualization"):
    os.makedirs(save_dir, exist_ok=True)
    
    # select 20 random samples
    sample_indices = np.arange(min(20, len(images)))
    sampled_images = images[sample_indices]
    
    # if images are 1D, reshape to 2D (28x28)
    if sampled_images[0].ndim == 1:
        sampled_images = sampled_images.reshape(-1, 28, 28)
    
    # create a grid of images
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, ax in enumerate(axes):
        ax.imshow(sampled_images[i], cmap='gray')
        ax.axis('off')
        if labels is not None:
            if len(labels.shape) == 2 and labels.shape[1] > 1:  # one-hot
                class_label = np.argmax(labels[sample_indices[i]])
            else:
                class_label = labels[sample_indices[i]]
            ax.set_title(f"Label: {class_label}", fontsize=8)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def preprocess_and_save(data_dir, save_dir, augment=False):
    print(f"Loading data from {data_dir}...")
    X_train, y_train = load_mnist_train(data_dir)
    X_test, y_test = load_mnist_test(data_dir)
    
    # visualize original training samples
    print("Visualizing original training samples...")
    visualize_samples(X_train, y_train, title="Original Training Samples", save_dir=save_dir)
    
    # One-hot encoding
    print("Encoding labels...")
    y_train_one_hot = encode_one_hot(y_train)
    y_test_one_hot = encode_one_hot(y_test)
    
    if augment:
        print("Augmenting training data...")
        # apply data augmentation
        X_train_aug, y_train_aug = augment_data(X_train, y_train_one_hot, augment=True)
        
        # visualize augmented samples
        print("Visualizing augmented samples...")
        visualize_samples(X_train_aug, y_train_aug, title="Augmented Training Samples", save_dir=save_dir)
        
        # concatenate original and augmented data
        X_train = np.concatenate((X_train, X_train_aug))
        y_train_one_hot = np.concatenate((y_train_one_hot, y_train_aug))
    
    # flatten images for innput (28x28 -> 784)
    print("Reshaping data...")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # save preprocessed data
    print(f"Saving data to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train_one_hot)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test_one_hot)
    
    print(f"Data saved to {save_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset", "MNIST")
    save_dir = os.path.join(base_dir, "dataset", "preprocessed")
    
    print("Starting MNIST data preprocessing...")
    try:
        preprocess_and_save(data_dir, save_dir, augment=True)
        print("MNIST data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print("Please ensure:")
        print(f"1. The MNIST dataset files are in: {data_dir}")
        print("2. The files have one of these names:")
        print("   - train-images-idx3-ubyte or train-images-idx3-ubyte.gz")
        print("   - train-labels-idx1-ubyte or train-labels-idx1-ubyte.gz")
        print("   - t10k-images-idx3-ubyte or t10k-images-idx3-ubyte.gz")
        print("   - t10k-labels-idx1-ubyte or t10k-labels-idx1-ubyte.gz")
        print("3. The files are not corrupted")