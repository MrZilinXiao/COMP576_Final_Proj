"""
Only for HuggingFace Trainer usage...
"""
import numpy as np
from sklearn import metrics

# temp solution MELD classes
classes = ['neutral',
           'surprise',
           'fear',
           'sadness',
           'joy',
           'disgust',
           'anger']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_metrics(eval_predictions) -> dict:
    """
    Return f1_weighted, f1_micro, and f1_macro scores.
    For huggingface Trainer usage.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1).tolist()

    weighted_prec, weigheted_rec, weighted_f1, _ = metrics.precision_recall_fscore_support(label_ids, preds,
                                                                                           average='weighted')

    # f1_weighted = f1_score(label_ids, preds, average='weighted')
    f1_micro = metrics.f1_score(label_ids, preds, average='micro')
    f1_macro = metrics.f1_score(label_ids, preds, average='macro')
    # class_prec, class_rec, class_f1, class_occur = metrics.precision_recall_fscore_support(label_ids, preds,
    #                                                                                        average=None)
    # class_report = metrics.classification_report(label_ids, preds, labels=[0, 1, 2, 3, 4, 5, 6], target_names=classes)

    return {'f1_weighted': weighted_f1, 'f1_micro': f1_micro, 'f1_macro': f1_macro,
            # 'class_report': class_report
            }


def general_compute_metrics(gt_list, pred_list, labels=None, text_labels=None) -> dict:
    weighted_prec, weighted_rec, weighted_f1, _ = metrics.precision_recall_fscore_support(gt_list, pred_list,
                                                                                          labels=labels,
                                                                                          average='weighted')
    f1_micro = metrics.f1_score(gt_list, pred_list, labels=labels, average='micro')
    f1_macro = metrics.f1_score(gt_list, pred_list, labels=labels, average='macro')
    class_report = metrics.classification_report(gt_list, pred_list, labels=labels, target_names=text_labels)
    return {
        'weighted_f1': weighted_f1,
        'weighted_prec': weighted_prec,
        'weighted_rec': weighted_rec,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'class_report': class_report
    }
