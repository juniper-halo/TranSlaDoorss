import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def visualize_dataset(dataset, class_names_search=None, split="train", num_images=10):
    """
    Custom visualization for specific dataset structure
    Pipeline:
    Dataset Storage → sample['image'] → matplotlib.imshow() → Display
            ↓               ↓                   ↓
    Raw PIL Image   Same PIL Image    Still the same
                        object        PIL Image object
    """
    # add or remove columns here depending on the type of dataset we are using
    image_col = "image"
    label_col = "label"

    # get class names
    class_names = dataset[split].features["label"].names
    print(f"Available classes: {class_names}")
    print(f"Total classes: {len(class_names)}")

    # if searching for specific class
    if class_names_search is not None:
        # find the class index
        if class_names_search not in class_names:
            print(
                f"Class '{class_names_search}' not found. Available classes: {class_names}"
            )
            return

        class_idx = class_names.index(class_names_search)

        # find all images of this class
        class_image_indices = []
        for i in range(len(dataset[split])):
            if dataset[split][i][label_col] == class_idx:
                class_image_indices.append(i)

        if not class_image_indices:
            print(f"No images found for class '{class_names_search}'")
            return

        # limit to requested number of images
        display_indices = class_image_indices[:num_images]
        actual_num_images = len(display_indices)

        print(
            f"Found {len(class_image_indices)} images of class '{class_names_search}', showing {actual_num_images}"
        )

        # create appropriate subplot grid
        rows = int(np.sqrt(actual_num_images))
        cols = int(np.ceil(actual_num_images / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

        # handle single image case
        if actual_num_images == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        # display images
        for i, img_idx in enumerate(display_indices):
            sample = dataset[split][img_idx]
            axes[i].imshow(sample[image_col])
            axes[i].set_title(f"Class: {class_names_search}\nIndex: {img_idx}")
            axes[i].axis("off")

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
        return

    # original logic for showing random images from all classes
    # create subplots
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))

    # abstract scaling factor
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # edge case only one image exists
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.ravel()  # flatten the axes array

    # add images in each subplot
    for i in range(num_images):
        if i >= len(dataset[split]):
            print("Tried to access an invalid image.")
            break

        sample = dataset[split][i]  # this fetches the raw data

        # pass that directly to matplotlib
        axes[i].imshow(sample[image_col])

        # fetch the actual class name instead of just the number
        label_idx = sample[label_col]
        label_name = class_names[label_idx]

        # set an informative title
        axes[i].set_title(f"Class: {label_name}")
        axes[i].axis("off")

    # hide unused subplots also boxes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # example usage for manual exploration
    from datasets import load_dataset

    ds = load_dataset("aliciiavs/sign_language_image_dataset")
    visualize_dataset(ds, "X", "train", 100)
