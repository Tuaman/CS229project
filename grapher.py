import numpy as np
import matplotlib.pyplot as plt

class Grapher:
    def __init__(self, title, action_labels=[]):
        self.title = title
        self.filename = '_'.join(map(lambda x: x[:10], title.lower().split()))
        self.action_labels = action_labels
        self.reset()

    def add(self, cash, nown, price, action, reward, date=None, **kwarg):
        self.points.append((cash, nown, price, action, reward, date or len(self.points)))

    def plot_interactions(self, ax1, dates, prices, actions, rewards):
        ax1.plot(dates, prices, color='darkslategray')
        ax1.set_ylabel('Prices', color='darkslategray')

        for i in range(len(self.action_labels)):
            ax1.plot(dates[actions==i], prices[actions==i], marker='.', linestyle='None', label=self.action_labels[i])
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(dates, np.cumsum(rewards), color='lightslategray')
        ax2.set_ylabel('Cumulative Reward', color='lightslategray')

    def plot_positions(self, ax1, dates, cashes, nowns, prices):
        ax1.plot(dates, cashes+prices*nowns, color='darkslategray')
        ax1.set_xlabel('Timestep (Day)')
        ax1.set_ylabel('Total Asset', color='darkslategray')

        stock_values = prices*nowns
        ax1.bar(dates, cashes, color='goldenrod', label='Cash', alpha=0.3)
        ax1.bar(dates[stock_values>0], stock_values[stock_values>0], bottom=cashes[stock_values>0], color='darkseagreen', label='Stock Value (+)', alpha=0.3)
        ax1.bar(dates[stock_values<0], stock_values[stock_values<0], color='#ef264b', label='Stock Value (-)', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax1.legend()

    def show(self, **kwargs):
        rmdot = lambda x: str(x).replace('.', '')
        rounder = lambda x: round(x,2) if isinstance(x, float) else x

        cashes, nowns, prices, actions, rewards, dates = list(map(np.array, zip(*self.points)))

        fig, (ax1, ax2) = plt.subplots(figsize = (10,5), nrows=2)
        attrs = ["{0}={1}".format(key, rounder(value)) for key, value in kwargs.items()]
        fig.suptitle("{} ({})".format(self.title, ' '.join(attrs)))

        self.plot_interactions(ax1, dates, prices, actions, rewards)
        self.plot_positions(ax2, dates, cashes, nowns, prices)

        attrs = ["{0}{1}".format(key, rmdot(rounder(value))) for key, value in kwargs.items()]
        fig.savefig('output/{}_{}.png'.format(self.filename, ''.join(attrs)))

    def reset(self):
        self.points = []

