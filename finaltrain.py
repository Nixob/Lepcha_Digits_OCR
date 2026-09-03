import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


DATASET_DIR     = r"D:\college\A Proj\datasetmakes\dataset\processed"
MODEL_SAVE_PATH = r"D:\college\A Proj\final trained model\lepcha_ocr.keras"
OUTPUT_DIR      = r"D:\college\A Proj\final trained model"
IMG_SIZE        = (28, 28)
BATCH_SIZE      = 32
EPOCHS          = 50
NUM_CLASSES     = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

#data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

common_args = dict(
    directory=DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

train_gen = train_datagen.flow_from_directory(**common_args, subset="training",  shuffle=True)
val_gen   = val_datagen.flow_from_directory(**common_args,   subset="validation", shuffle=False)

print(f"Classes : {train_gen.class_indices}")
print(f"Train   : {train_gen.samples}  |  Val: {val_gen.samples}")


#CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


#training part
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)


#result
val_gen.reset()
y_pred  = np.argmax(model.predict(val_gen, verbose=1), axis=1)
y_true  = val_gen.classes
labels  = list(val_gen.class_indices.keys())

print("\n" + classification_report(y_true, y_pred, target_names=labels))


#graphs
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved → {path}")
    plt.show()
    plt.close()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Training Performance", fontsize=14, fontweight='bold')

epochs_ran = range(1, len(history.history["accuracy"]) + 1)

ax1.plot(epochs_ran, history.history["accuracy"],     label="Train",  color="#2196F3", lw=2)
ax1.plot(epochs_ran, history.history["val_accuracy"], label="Val",    color="#FF9800", lw=2, ls='--')
ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
ax1.legend(); ax1.grid(alpha=0.3)

best_ep  = np.argmax(history.history["val_accuracy"]) + 1
best_acc = max(history.history["val_accuracy"])
ax1.annotate(f"Best: {best_acc*100:.2f}% (ep {best_ep})",
             xy=(best_ep, best_acc), xytext=(best_ep + 1, best_acc - 0.1),
             arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=9)

ax2.plot(epochs_ran, history.history["loss"],     label="Train",  color="#2196F3", lw=2)
ax2.plot(epochs_ran, history.history["val_loss"], label="Val",    color="#FF9800", lw=2, ls='--')
ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
ax2.legend(); ax2.grid(alpha=0.3)

best_loss_ep = np.argmin(history.history["val_loss"]) + 1
best_loss    = min(history.history["val_loss"])
ax2.annotate(f"Best: {best_loss:.4f} (ep {best_loss_ep})",
             xy=(best_loss_ep, best_loss), xytext=(best_loss_ep + 1, best_loss + 0.2),
             arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=9)

plt.tight_layout()
save(fig, "training_curves.png")



cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
plt.tight_layout()
save(fig, "confusion_matrix.png")

print(f"\nDone! Results saved to: {OUTPUT_DIR}")