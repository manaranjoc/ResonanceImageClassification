from sklearn.metrics import confusion_matrix

def metrics(y_true, y_pred):
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy(tn, fp, fn, tp)
    prec = precision(tp, fp)
    sens = sensitivity(tp, fn)
    spec = specificity(tn, fp)
    f1_score = f1(fp,fn, tp)

    return [acc, prec, sens, spec, f1_score]

def accuracy(tn, fp, fn, tp):
    return (tp+tn)/(tp+fp+fn+tn)

def precision(tp, fp):
    return tp/(tp+fp)

def sensitivity(tp, fn):
    return tp/(tp+fn)

def specificity(tn, fp):
    return tn/(tn+fp)

def f1(fp, fn, tp):
    sens = sensitivity(tp, fn)
    prec = precision(tp, fp)
    return 2*(sens*prec)/(sens+prec)
    
def print_metrics(metrics):
    print('Accuracy: ', metrics[0])
    print('Precision: ', metrics[1])
    print('Sensitivity: ', metrics[2])
    print('Specificity: ', metrics[3])
    print('F1 score: ', metrics[4])
