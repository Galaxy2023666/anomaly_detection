
import os
import warnings
import argparse
from tools.shaplets_base.time2graph_plus.config import *
# from pyts.classification import SAXVSMClassifier
from pyts.classification import SAXVSM as SAXVSMClassifier
from pyts.bag_of_words import BagOfWords
from pyts.datasets import load_gunpoint

from tools.shaplets_base.time2graph_plus.time2graph.utils.deep_utils import *
from tools.shaplets_base.time2graph_plus.time2graph.utils.base_utils import evaluate_performance

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--cache', action='store_true', default=False)

    args = parser.parse_args()
    Debugger.info_print('run with options: {}'.format(args.__dict__))
    if args.cache and os.path.isfile('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset)):
        x_train = np.load('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset))
        y_train = np.load('{}/scripts/cache/{}_y_train.npy'.format(module_path, args.dataset))
        x_test = np.load('{}/scripts/cache/{}_x_test.npy'.format(module_path, args.dataset))
        y_test = np.load('{}/scripts/cache/{}_y_test.npy'.format(module_path, args.dataset))
    else:
        x_train, x_test, y_train, y_test = load_gunpoint(return_X_y=True)
        # raise NotImplementedError()

    # x_train = x_train.reshape(x_train.shape[0], -1)
    # x_test = x_test.reshape(x_test.shape[0], -1)

    model = SAXVSMClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=y_pred)
    Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(accu, prec, recall, f1))

    bow = BagOfWords(window_size=0.5, word_size=0.5, n_bins=4, strategy='normal')
    arr = bow.fit_transform(x_train)
    len(np.unique(' '.join(arr).split(), return_counts=True)[0])

    print(model.vocabulary_)
