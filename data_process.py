
# run sequence
# 1. recode_tu_attr
# 2. (cat_var) optional, called in later functions
# 3. missing_obs_summary, missing_strategy_exec
# 4. column_encoder
# 5. linear_resample

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

def cat_var(df):
    """Identify categorical features
    Params
    ------
    df: original df after missing operations

    Return
    ------
    cat_var_df: summary df with col index and col name for all categorical vars
    """
    col_type = df.dtypes
    col_names = list(df)

    cat_var_index = [i for i, x in enumerate(col_type) if x == 'object']
    cat_var_name = [x for i, x in enumerate(col_names) if i in cat_var_index]

    cat_var_df = pd.DataFrame({'cat_ind': cat_var_index,
                               'cat_name': cat_var_name})

    return cat_var_df


def missing_obs_summary(df, pct_drop_col=0.5, pct_drop_row=0.01):
    """Summarize missing status in the data frame
    Params
    ------
    df: original data frame, first row as header
    pct_drop_col: minimum pct threshold to drop a column,
            if missing_pct > pct_drop, drop the column
            default value at 0.5
    pct_drop_row: maximum pct threshold to drop rows,
            if missing_pct < pct_drop, drop rows in the column
            default value at 0.01

    Return
    ------
    missing_num: data frame as a summary of missing obs, including
            missing col index, col name, pct missing, and strategy
    """

    # get missing column name and number of observations
    missing_row = df.isnull().sum(axis=0)
    missing_col = [i for i, x in enumerate(missing_row) if x > 0]

    # iterate through missing cols for col name, index, and missing amt, pct
    missing_amt = []
    missing_col_name = []
    pct = []

    for key, value in missing_row.iteritems():
        if value > 0:
            missing_col_name.append(key)
            missing_amt.append(value)
            pct.append(value / df.shape[0])

    # identify strategy for missing values
    strategy = []

    for k in pct:
        if k >= pct_drop_col:
            strategy.append("drop column")
        elif k < pct_drop_row:
            strategy.append("drop row")
        else:
            strategy.append("filling")

    # concat info into a df
    missing_num = pd.DataFrame({'col_name': missing_col_name,
                                'col_ind': missing_col,
                                'row_missing': missing_amt,
                                'pct_missing': pct,
                                'strategy': strategy})

    return missing_num

# execute from missing_obs_summary


def missing_strategy_exc(df, missing_sum_df, fill_num=0, fill_cat='missing'):
    """Executing based on identified missing strategy
    Params
    ------
    df: original data frame with missing values
    missing_sum_df: missing summary data frame from missing_obs_summary function

    Return
    ------
    new_df: new df after taking the missing strategy

    """
    # setup (empty) lists to categorize different strategy
    drop_col = []
    drop_row = []
    cat_var_list = cat_var(df)
    fill_in_missing = []
    fill_in_zero = []

    # loop through each missing col/row, categorize each col/row accordingly
    for index, row in missing_sum_df.iterrows():
        # drop column first
        if row['strategy'] == "drop column":
            drop_col.append(row['col_name'])
        # drop row next
        elif row['strategy'] == "drop row":
            drop_row.append(row['col_name'])
        elif row['strategy'] == 'filling' \
                and row['col_ind'] in cat_var_list.loc[:, 'cat_ind'].tolist():
            fill_in_missing.append(row['col_name'])
        elif row['strategy'] == 'filling' \
                and row['col_ind'] not in cat_var_list.loc[:, 'cat_ind'].tolist():
            fill_in_zero.append(row['col_name'])

    # drop rows first
    new_df = df.dropna(subset=drop_row)
    # drop cols then
    new_df = new_df.drop(drop_col, axis=1)
    # filling in zeros to numeric features
    new_df.loc[:, fill_in_zero] = new_df.loc[:, fill_in_zero].fillna(value=fill_num, axis=1)
    # filling missing (str) to object features
    new_df.loc[:, fill_in_missing] = new_df.loc[:, fill_in_missing].fillna(value=str(fill_cat), axis=1)

    return new_df


def column_encoder(df):
    """Encoding categorical feature in the data frame
    Params
    ------
    df: input data frame
    require loading cat_var functions

    Return
    ------
    df: new data frame where categorical features are encoded
    label_list: classes_ attribute for all encoded features
    """

    # set up input space and categorical variable check
    label_list = []
    label_name = []
    cat_var_df = cat_var(df)
    cat_list = cat_var_df.loc[:, 'cat_name']

    # loop through all categorical variable, label encode each and store classes info
    for index, cat_feature in enumerate(cat_list):
        le = LabelEncoder()
        le.fit(df.loc[:, cat_feature])
        # store each label encoder classes for future references
        label_list.append(list(le.classes_))
        label_name.append(cat_feature)
        # change the original feature into label encoding ones
        df.loc[:, cat_feature] = le.transform(df.loc[:, cat_feature])

    # getting categorical feature and their corresponding value into a data frame
    feature_list = pd.DataFrame({'Feature': label_name,
                                 'Value': label_list})

    return df, feature_list


def linear_resample(df, col_ind, verbose=False):
    """Re-sample linearly based on time bucket index
    Params
    ------
    df: original input data frame
    col_ind: str, column name for time bucket index
    verbose: boolean, print more detailed information during execution

    Return
    ------
    new_df: new data frame after re-sampling
    """
    ind_value = df.loc[:, str(col_ind)]    # extract time index column
    num_col = np.unique(ind_value.values)    # how many time bucket in the data frame
    increment_pct = 100 / np.max(num_col)    # incremental percentage from bucket to bucket
    grouped = df.groupby([str(col_ind)])    # create grouped object
    pct = np.zeros(grouped[str(col_ind)].count().shape)    # empty array to set up

    # get percentage for each time bucket
    for index, value in enumerate(pct):
        pct[index] = 100 - (index * increment_pct)

    # using percentage to calculate actual sample size for each bucket
    group_cnt = round(pct * grouped[str(col_ind)].count() / 100)
    # set up new empty data frame
    new_df = pd.DataFrame()

    # for each bucket, random sample based on pre-set sample sizes, attach all into the new data frame
    for index, value in group_cnt.iteritems():
        temp = df.loc[df[str(col_ind)] == index, :].sample(n=int(value))
        if verbose:
                print('Week {} has selected {} observations.'.format(index, value))
        new_df = pd.concat([new_df, temp])

    return new_df


def detect_extreme_value(data, cols=None, method='pctile'):
    """
    return column name with potential of having extreme values on max end
    :param data: data frame
    :param cols: column list to check for, default is all columns
    :param method: either 'std' for 75% + 5 std or 'pctile' for 97 percentile
    :return: a list of feature name that potentially contains extreme values to look closely at
    """
    # if not specified, using all columns
    if cols is None:
        cols = data.columns
    # set up return list
    extreme_cols = []
    if method == 'std':
        # getting summary data
        summary = data.describe()
        # loop through columns, return if max value is 5 std away from 75th percentile
        for col_name in cols:
            benchmark = summary.loc[r'75%', col_name] + 5 * summary.loc[r'std', col_name]
            if summary.loc[r'max', col_name] > benchmark:
                extreme_cols.append(col_name)
    elif method == 'pctile':
        # loop through columns, return if max value is 20% more than 97th percentile
        for col_name in cols:
            # set up bench mark values, 97 percentile
            benchmark = data[col_name].quantile(.97)
            if (data[col_name].max() - benchmark) > benchmark * 0.2:
                extreme_cols.append(col_name)

    return extreme_cols


def get_dist_point(data, method='median'):
    """
    Return specified distribution metrics for data frame
    :param data: input data frame, training set, could be ndarray as well
    :param method: metrics to choose to represent the distribution, i.e. average, median, etc.
    :return: ndarray with feature names and feature value (dist representation)
    """
    dist_point = []
    if method == 'average':
        dist_point = np.average(data, axis=0)
    elif method == 'median':
        dist_point = np.median(data, axis=0)

    return dist_point


def obs_level_importance(estimator, X_test, y_test, median, metric):
    """
    Get observation level importance for all input features
    :param estimator: model estimator
    :param X_test: 1-row obs, 1d array
    :param y_test: 1-row target feature
    :param median: median value for each feature
    :param metric: metric function to evaluate result
    :return: results of list with feature importances
    """
    # for 1-row X_test
    results = []

    for feature_ind, feature_value in enumerate(X_test):
        # save original data for references
        save = X_test.copy()
        baseline = metric(y_test, estimator.predict_proba(save)[:, y_test])
        # replace X_test with median value
        X_test[feature_ind] = median[feature_ind]
        new_score = metric(y_test, estimator.predict_proba(X_test)[:, y_test])
        # append the difference between new score and baseline score
        results.append(new_score - baseline)
        X_test = save

    return results


def get_top_n_feature(X_test, n):
    """
    get top n features from obs_level_importances
    :param X_test: testing obs
    :param n: top number of features to include in the result
    :return:
    """
    feature_imp = obs_level_importance(estimator, X_test, y_test, median, metric)



def plot_precision_recall(y_test, y_pred_score, alpha_value, pos_label, color_value):
    """
    plotting 2-class precision-recall curve
    require precision_recall_curve, average_precision_score from sklearn.metrics
    :param y_test:
    :param y_pred_score:
    :param alpha_value:
    :param pos_label:
    :param color_value:
    :return:
    """
    avg_precision = average_precision_score(y_test, y_pred_score)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_score, pos_label=pos_label)
    plt.step(recall, precision, color=str(color_value), alpha=alpha_value, where='post')
    plt.fill_between(recall, precision, color=str(color_value), alpha=alpha_value, step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(avg_precision))

    return plt


def sklearn_classifier():


def sklearn_cv_method(method, n_split):
    """
    Specify cross-validation splitting method and params
    note: not returning splitted data set
    :param method: string, method name from splitting/cv methods from sklearn
    :param n_split: number of splits
    :return: sklearn object, either kfold or splitting method
    """




def cv_roc_curve(cv_method, classifier, X, y,
                 base_label='Luck', base_color='r', base_linstyle='--', base_alpha=0.8,
                 mean_color='b', mean_lw=2, mean_alpha=0.8,
                 title='Receiver operation characteristic Plot', loc='lower right',
                 show=True):
    """
    plotting roc curve with auc number by different cv
    :param cv_method: sklearn cv/split object, predefined with sklearn_cv_method()
    :param classifier: sklearn classifier object, predefined with sklearn_classifier()
    :param X: input data, either numpy array or pandas data frame
    :param y: target data, either numpy array or pandas data frame
    :param base_label: base label on plot
    :param base_color: color style for base label line
    :param base_linstyle: line style for base label
    :param base_alpha:
    :param mean_color:
    :param mean_lw:
    :param mean_alpha:
    :param title:
    :param loc:
    :param show:
    :return: plot of cv roc plot
    """

    tprs = []
    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv_method.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.2,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle=base_linstyle, lw=2, color=base_color, label=base_label,
             alpha=base_alpha)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=mean_color,
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=mean_lw, alpha=mean_alpha)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc=loc)
    if show:
        plt.show()
