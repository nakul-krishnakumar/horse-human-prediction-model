import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_random_data(train_horse_dir, train_horse_names, train_human_dir, train_human_names):
    # Parameters for your graph; you will output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 3, nrows * 3)

    next_horse_pix = [os.path.join(train_horse_dir, fname)
                    for fname in random.sample(train_horse_names, k=8)]
    next_human_pix = [os.path.join(train_human_dir, fname)
                    for fname in random.sample(train_human_names, k=8)]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


def plot_train_accuracy(history):
    # Plot the training accuracy for each epoch

    acc = history.history['accuracy']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend(loc=0)
    plt.show()