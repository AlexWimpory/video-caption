import seaborn as sns
from matplotlib import pyplot


def plot_history(history):
    pyplot.title('Model Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='val')
    pyplot.legend(loc='upper left')
    pyplot.show()

    pyplot.title('Model Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend(loc='upper right')
    pyplot.show()


def plot_confusion_matrix(dataframe):
    pyplot.figure(figsize=(8, 8))
    sns.heatmap(dataframe, annot=True, cmap=pyplot.cm.Blues)
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.show()