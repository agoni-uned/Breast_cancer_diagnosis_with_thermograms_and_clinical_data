import tensorflow
import numpy as np

# ----------------------------------------------------------- CUSTOM METRICS -----------------------------------------------------------#
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

class Specificity(tensorflow.keras.metrics.Metric):
  """Computes the specificity of the predictions with respect to the labels.

  This metric creates two local variables, `true_negatives` and
  `false_positives`, that are used to compute the specificity. This value is
  ultimately returned as `specificity`, an idempotent operation that simply divides
  `true_negatives` by the sum of `true_negatives` and `false_positives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `top_k` is set, recall will be computed as how often on average a class
  among the labels of a batch entry is in the top-k predictions.

  If `class_id` is specified, we calculate specificity by considering only the
  entries in the batch for which `class_id` is in the label, and computing the
  fraction of them for which `class_id` is above the threshold and/or in the
  top-k predictions.
  """

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name='specificity',
               dtype=None):
    """Creates a `specificity` instance.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate recall with `thresholds=0.5`.
      top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating recall.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Specificity, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self.true_negatives = self.add_weight(
        name='true_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        name='false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates true negative and false positive statistics.

    Args:
      y_true: The ground truth values, with the same dimensions as `y_pred`.
        Will be cast to `bool`.
      y_pred: The predicted values. Each element must be in the range `[0, 1]`.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.true_negatives,
                                 self.true_negatives + self.false_positives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_state(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Specificity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class WeightedError(tensorflow.keras.metrics.Metric):
  """Computes the Weighted Error of the predictions with respect to the labels: FP + rate * FN

  This metric creates two local variables, `false_negatives` and
  `false_positives`, that are used to compute the weighted error. This value is
  ultimately returned as `weighted-error`, an idempotent operation that simply multiplies
  `false_negatives` by 'rate' and sums this value with `false_positives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `top_k` is set, weighted error will be computed as how often on average a class
  among the labels of a batch entry is in the top-k predictions.

  If `class_id` is specified, we calculate recall by considering only the
  entries in the batch for which `class_id` is in the label, and computing the
  fraction of them for which `class_id` is above the threshold and/or in the
  top-k predictions.
  """

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               rate=None,
               name='weighted-error',
               dtype=None):
    """Creates a `weighted-error` instance.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate recall with `thresholds=0.5`.
      top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating recall.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
       rate: Integer class ID. The rate of the error.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(WeightedError, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id
    self.rate = rate

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self.false_negatives = self.add_weight(
        name='false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        name='false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates true positive and false negative statistics.

    Args:
      y_true: The ground truth values, with the same dimensions as `y_pred`.
        Will be cast to `bool`.
      y_pred: The predicted values. Each element must be in the range `[0, 1]`.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    result = self.false_positives + math_ops.multiply_no_nan(self.false_negatives, self.rate)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_state(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id,
        'rate': self.rate
    }
    base_config = super(WeightedError, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    


def weighted_error(y_test, y_test_pred):
    WE = 20

    # Number of elements for which y_test > y_test_pred, i.e., y_test = 1 and y_test_pred = 0 (FN)
    fn = np.sum(np.greater(y_test, y_test_pred))

    # Number of elements for which y_test < y_test_pred, i.e., y_test = 0 and y_test_pred = 1 (FP)
    fp = np.sum(np.less(y_test, y_test_pred))

    return fn*WE + fp



# ----------------------------------------------------------- MODEL EVALUATION -----------------------------------------------------------#

def evaluate_model_skl(preds_raw, ground_truth):

    from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score #, accuracy_score, precision_score, recall_score, f1_score, make_scorer
    
    # BCE loss
    bce = log_loss(ground_truth, preds_raw)

    # Confusion matrix
    TN, FP, FN, TP = confusion_matrix(ground_truth, np.round(preds_raw)).ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # accuracy = accuracy_score(ground_truth, np.round(preds_raw))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    #gmean = np.sqrt((TP/(TP+FN))*(TN/(TN+FP)))
    precision = TP / (TP + FP)  # precision = precision_score(ground_truth, np.round(preds_raw))
    F1 = 2 * TP / (2*TP + FP + FN)  # F1 = f1_score(ground_truth, np.round(preds_raw))
    
    # ROC AUC
    auc = roc_auc_score(ground_truth, preds_raw)
    
    # Weighted error    
    we = weighted_error(ground_truth, np.round(preds_raw))
    
    results = {
        'BCELoss':bce,
        'Accuracy':accuracy,
        'TP':TP,
        'FP':FP,
        'TN':TN,
        'FN':FN,
        'Sensitivity':sensitivity,
        'Specificity':specificity,
        #'G-mean':gmean,
        'Precision':precision,
        'Recall':sensitivity,
        'F1':F1,
        'ROC_AUC':auc,
        'WE':we
    }
    return results

def evaluate_model_tf(preds_raw, ground_truth):

    bce = tf.keras.metrics.BinaryCrossentropy()
    bce.update_state(ground_truth, preds_raw)
    bce = bce.result().numpy()
    
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(ground_truth, preds_raw.round())
    accuracy = accuracy.result().numpy()
    
    sensitivity = tf.keras.metrics.Recall()
    sensitivity.update_state(ground_truth, preds_raw)
    sensitivity = sensitivity.result().numpy()

    specificity = Specificity()
    specificity.update_state(ground_truth, preds_raw)
    specificity = specificity.result().numpy()

    auc = tf.keras.metrics.AUC()
    auc.update_state(ground_truth, preds_raw)
    auc = auc.result().numpy()
    
    precision = tf.keras.metrics.Precision()
    precision.update_state(ground_truth, preds_raw)
    precision = precision.result().numpy()

    F1 = tf.keras.metrics.F1Score()
    F1.update_state(np.expand_dims(ground_truth,-1), np.expand_dims(preds_raw,-1))
    F1 = F1.result().numpy()
    
    we = WeightedError(rate=20)
    we.update_state(ground_truth, preds_raw)
    we = we.result().numpy()
    #we = weighted_error(ground_truth, np.round(preds_raw))
    
    results = {
        'BCELoss':bce,
        'Accuracy':accuracy,
        'sensitivity':sensitivity,
        'specificity':specificity,
        'precision':precision,
        'recall':sensitivity,
        'F1':F1,
        'ROC AUC':auc,
        'WE':we
    }
    return results


def store_results(num_paranms, trainTime, results_train, results_test):
    results = {}
    
    results['Parameters'] = num_paranms
    results['trainTime'] = trainTime
    
    # Train results
    for metric, value in results_train.items():
        results['train_' + metric] = value

    # Test results
    for metric, value in results_test.items():
        results['test_' + metric] = value
    
    return results


# ----------------------------------------------------------- MODEL COMPARISON -----------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt

## Show boxplots comparing different metrics for all models
def visualize_boxplots(df, metrics, savefigures = False, fname = ''):
    import seaborn as sns
    
    M = len(metrics)
    R = M//2 + int(M % 2 > 0)
    plt.figure(figsize=(5*R, 20))
    for i,metric in enumerate(metrics):
        plt.subplot(R,2,i+1)
        sns.boxplot(x='classifier', y=metric, data=df, palette='Set2')
        plt.xticks(rotation=25, ha='right')  # Rotate labels 25 degrees
        plt.grid(axis='y')
        plt.xlabel('')
        if 'Loss' in metric or 'WE' in metric:
            plt.ylim([0,df[metric].values.max()+0.05*df[metric].values.max()])
        else:
            plt.ylim([0,1])
        plt.title(metric, fontsize=16)
        #plt.tight_layout()
    if savefigures:
        plt.savefig(fname)
    plt.subplots_adjust(hspace=0.3)  # Increase hspace for more vertical spacing
    plt.show()

## Show boxplot for a single model
def visualize_boxplot_onemodel(df, metrics, savefigures = False, fname = ''):
    import seaborn as sns
    
    df_long = pd.melt(df[metrics], value_vars=metrics, var_name='metric', value_name='value')
    
    plt.figure(figsize=(12, 10))
    sns.boxplot(y='value',x='metric',data=df_long, palette='Set2', fliersize=10)
    new_labels = [l.replace('test_','') for l in metrics]
    plt.xticks(ticks=range(len(new_labels)), labels=new_labels, fontsize=18)#, rotation=25, ha='right')
    plt.yticks(fontsize=18)
    plt.grid(axis='y')
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.ylim([-0.05,1.05])
    if savefigures:
        plt.savefig(fname)
    plt.show()



################## CHECK CRITERIA FOR PARAMETRIC TESTS ##################

def check_var_correlation(df, metrics):
    # Check for multicollinearity using the correlation matrix
    correlation_matrix = df[metrics].corr()
    return correlation_matrix

def check_no_variance(df, metrics):
    var = df[metrics].var().to_numpy()
    adjusted_metrics = [metric for metric,v in zip(metrics,var) if not v == 0]
    return adjusted_metrics

# Q-Q plot for a specific model and metric
def plot_qq(data, model, metric):
    from scipy.stats import probplot
    probplot(data, dist='norm', plot=plt)
    plt.title(model)
    #plt.title(f"Q-Q Plot for {model} - {metric}")
    #plt.show()


################## APPLY STATISTICAL TESTS ##################

## MANOVA (parametric)
def apply_manova(df,metrics):
    from statsmodels.multivariate.manova import MANOVA
    dependent_variables = ' + '.join(metrics)
    formula = f'{dependent_variables} ~ classifier'
    manova = MANOVA.from_formula(formula, data=df)
    return manova.mv_test()


## Tukey’s HSD Test (parametric)
def apply_tukey(df,metrics):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = {}
    for metric in metrics:
        print(metric)
        tukey[metric] = pairwise_tukeyhsd(df[metric],   # Metric data
                                          df['classifier'],   # Grouping variable (model)
                                          alpha=0.05)   # Significance level
        print(tukey[metric])
    return tukey


################## MODEL COMPARISON PIPELINES ##################

def compare_models(df):
    
    models = df.classifier.unique()
    
    ## 1. Select independent metrics
    #These should be independent. Use the correlation to find dependent variables to remove, if necessary 
    compared_metrics = ['test_BCELoss','test_Accuracy','test_F1','test_ROC_AUC','test_WE']
    print('Check that the metrics are independent:'), print(check_var_correlation(df,compared_metrics)), print()
    compared_metrics = check_no_variance(df,compared_metrics)  # Remove metrics with zero variance across repetitions
    
    
    ## 3. Statistical test: MANOVA
    manova_test = apply_manova(df,compared_metrics)
    print(manova_test)

    ## 4. Post-hoc test: Tukey's HSD
    if manova_test.results['classifier']['stat']['Pr > F']['Pillai\'s trace'] < 0.05:
        tukey = apply_tukey(df, compared_metrics)
    else:
        print(f"ANOVA is not significant for {metric}, no need to apply Tukey's test.")
    print(), print()

