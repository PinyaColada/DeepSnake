import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Enables interactive mode
# display.display(plt.gcf())


class Visualizer:

    def __init__(self):
        self.title = "Training..."

    def plot(self, scores, mean_scores, loss):
        display.clear_output(wait=True)
        plt.clf()
        plt.title(self.title)
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(mean_scores)
        plt.plot(scores)
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)
