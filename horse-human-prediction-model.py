import os
import tensorflow as tf

from utils import display_random_data, plot_train_accuracy

TRAIN_DIR = 'horse-or-human'

print('-----------------------------------------------------------------------------------')

# You should see a `horse-or-human` folder here
print(f"\nfiles in current directory: {os.listdir()}")

# Check the subdirectories
print(f"\nsubdirectories within '{TRAIN_DIR}' dir: {os.listdir(TRAIN_DIR)}")

# Directory with the training horse pictures
train_horse_dir = os.path.join(TRAIN_DIR, 'horses')

# Directory with the training human pictures
train_human_dir = os.path.join(TRAIN_DIR, 'humans')

# Check the filenames
train_horse_names = os.listdir(train_horse_dir)
print(f"5 files in horses subdir: {train_horse_names[:5]}")
train_human_names = os.listdir(train_human_dir)
print(f"5 files in humans subdir:{train_human_names[:5]}")

print(f"total training horse images: {len(os.listdir(train_horse_dir))}")
print(f"total training human images: {len(os.listdir(train_human_dir))}")

# display batch of 8 horses and 8 humans
display_random_data(train_horse_dir, train_horse_names, train_human_dir, train_human_names)


# ConvNet architecture
model = tf.keras.models.Sequential([
    # first convolution layer with pooling
    tf.keras.Input((300, 300, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # second convolution layer with pooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # third convolution layer with pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # fourth convolution layer with pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # fifth convolution layer with pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    # output will be a value between 0 and 1, 
    # 0 -> horses, 1 -> human ( set according to alphabetical order, refer DATA PREPROCESSING part below)
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Analysis of the model
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# DATA PREPROCESSING 
# Reading the pics from source folder and converting them into tensors
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(300, 300),
    batch_size=32,
    label_mode='binary'
)

# Check the type
dataset_type = type(train_dataset)
print(f'train_dataset inherits from tf.data.Dataset: {issubclass(dataset_type, tf.data.Dataset)}')

# Rescaling image by normalizing for better perfomance
rescale_layer = tf.keras.layers.Rescaling(scale=1./255)

train_dataset_scaled = train_dataset.map(lambda image, label : (rescale_layer(image), label))

# Configuring the dataset
SHUFFLE_BUFFER_SIZE = 1000 #it will first select a sample from the first 1,000 elements, 
                           # then keep filling this buffer until all elements have been selected.
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = (train_dataset_scaled
                       .cache() #stores elements in mem for faster access if you need it again
                       .shuffle(SHUFFLE_BUFFER_SIZE) #shuffles dataset randomly according to the buffer size
                       )

history = model.fit(
    train_dataset_final,
    epochs=15,
    verbose=2
)

# Plot the training accuracy for each epoch
plot_train_accuracy(history)
