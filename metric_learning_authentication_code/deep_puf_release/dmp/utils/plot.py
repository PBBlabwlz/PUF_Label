import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

import dmp
from ..utils.typing import *


class PlotMixin:
    def get_accuracy_roc(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        dmp.info(roc_auc_score(y_true, y_pred))

    def evaluate(self, sheet_dict, y_true_dict, y_pred_dict):
        for key in sheet_dict.keys():
            self.get_accuracy_roc(y_pred_dict[key], y_true_dict[key])
            heat = np.array(sheet_dict[key])
            self.plot_results(heat, y_true_dict[key], y_pred_dict[key], key)

    def plot_results(self, heat, y_true, y_pred, key_name):
        self.make_heatmap(heat, os.path.join(args.TEST_RESULT_DIR, f'heat_{key_name}.jpg'), y_pred, y_true)
        self.make_heatmap(heat, os.path.join(args.TEST_RESULT_DIR, f'heat_{key_name}_minmax.jpg'), y_pred, y_true,
                          bar_mode='min_max')
        index1, index2 = self.group_true_and_false_indices(y_true)
        self.plot_histograms(y_pred, index1, index2, key_name)

        max_val_index2 = max([y_pred[i] for i in index2])
        min_val_index1 = min([y_pred[i] for i in index1])
        dmp.info(f"Max value of y_pred for index1: {max_val_index2}")
        dmp.info(f"Min value of y_pred for index2: {min_val_index1}")

    def group_true_and_false_indices(self, y_true):
        index1 = [i for i, y in enumerate(y_true) if y == 1]
        index2 = [i for i, y in enumerate(y_true) if y == 0]
        return index1, index2

    def make_heatmap(self, x, results_path, y_pred, y_true, bar_mode='01'):
        '''
        :param x: input numpy array table with size MxN
        :return:
        '''
        plt.figure(figsize=(10, 7))
        ti = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        plt.title('roc_auc:%.5f, ap: %.5f' % (ti, ap), fontsize='large', fontweight='bold')
        if bar_mode == '01':
            sns_plot = sns.heatmap(x, vmin=0, vmax=1, annot=False, cmap="OrRd")
        else:
            sns_plot = sns.heatmap(x, annot=False, cmap="OrRd")
        plt.savefig(results_path, dpi=100)

    def plot_histograms(self, y_pred, index1, index2, key_name):
        x1 = [y_pred[i] for i in index1]
        x2 = [y_pred[i] for i in index2]
        x1 = np.array(x1)
        x2 = np.array(x2)
        results_path = os.path.join(args.TEST_RESULT_DIR, f'hist_{key_name}.png')

        plt.figure(figsize=(7, 4))
        bins = []
        i = 0
        while i < 0.9:
            bins.append(i)
            i += 0.02
        # min_val = min(x1.min(), x2.min())
        # max_val = max(x1.max(), x2.max())
        # bins = np.arange(min_val, max_val, 0.01)
        weights1 = np.ones_like(x1) / float(len(x1))
        kwargs1 = dict(rwidth=0.8, histtype='barstacked', alpha=0.4, bins=bins, weights=weights1, color='b')

        weights2 = np.ones_like(x2) / float(len(x2))
        kwargs2 = dict(rwidth=0.8, histtype='barstacked', alpha=0.4, weights=weights2, bins=bins, color='r')
        plt.hist(x1, **kwargs1)
        plt.hist(x2, **kwargs2)

        plt.savefig(results_path, dpi=100)