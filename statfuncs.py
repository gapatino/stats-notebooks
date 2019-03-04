# Loading Python libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as multi
from statsmodels.formula.api import ols
from IPython.display import Markdown
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#pd.options.display.float_format = '{:.3f}'.format
#np.set_printoptions(precision=3, suppress=True)


# Statistics functions
def parammct(data=None, independent=None, dependent=None):

    independent = str(independent)
    dependent = str(dependent)
    if input_check_numerical_categorical(data, independent, dependent):
        return

    parammct_df = pd.DataFrame()
    for value in pd.unique(data[independent]):
        mean = data[dependent][data[independent]==value].mean()
        stdev = data[dependent][data[independent]==value].std()
        n = data[dependent][data[independent]==value].count()
        sdemean = stdev/np.sqrt(n)
        ci = 1.96*sdemean
        lowerboundci = mean-ci
        upperboundci = mean+ci
        parammct_df[value] = pd.Series([mean, stdev, n, sdemean, lowerboundci, upperboundci],
                                       index = ['Mean','SD','n','SEM','Lower bound CI', 'Upper bound CI'])

    return parammct_df


def non_parammct(data=None, independent=None, dependent=None):

    independent = str(independent)
    dependent = str(dependent)
    if input_check_numerical_categorical(data, independent, dependent):
        return

    non_parammct_df = pd.DataFrame()
    for value in pd.unique(data[independent]):
        median = data[dependent][data[independent]==value].median()
        minimum = data[dependent][data[independent]==value].quantile(0)
        q25 = data[dependent][data[independent]==value].quantile(0.25)
        q75 = data[dependent][data[independent]==value].quantile(0.75)
        maximum = data[dependent][data[independent]==value].quantile(1)
        n = data[dependent][data[independent]==value].count()
        non_parammct_df[value] = pd.Series([median, minimum, q25,q75, maximum, n],
                                           index = ['Median', 'Minimum', 'Lower bound IQR', 'Upper bound IQR',
                                                    'Maximum', 'n'])

    return non_parammct_df


def histograms(data=None, independent=None, dependent=None):

    independent = str(independent)
    dependent = str(dependent)
    if input_check_numerical_categorical(data, independent, dependent):
        return

    for value in pd.unique(data[independent]):
        sns.distplot(data[dependent][data[independent]==value], fit=stats.norm, kde=False)
        plt.title(dependent + ' by ' + independent + '(' + str(value).lower() + ')',
                  fontweight='bold', fontsize=16)
        plt.ylabel('Frequency', fontsize=14)
        plt.xlabel(dependent, fontsize=14)
        plt.show()

    return


def t_test(data=None, independent=None, dependent=None):

    pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)
    independent_groups = pd.unique(data[independent])
    if len(independent_groups)>2:
        print('There are more than 2 groups in the independent variable')
        print('t-test is not the correct statistical test to run in that circumstance,')
        print('consider running an ANOVA')
        return

    mct = parammct(data=data, independent=independent, dependent=dependent)

    t_test_value, p_value = stats.ttest_ind(data[dependent][data[independent] == independent_groups[0]],
                                            data[dependent][data[independent] == independent_groups[1]])

    difference_mean = np.abs(mct.loc['Mean'][0] - mct.loc['Mean'][1])
    pooled_sd = np.sqrt( ( ((mct.loc['n'][0]-1)*mct.loc['SD'][0]**2) + ((mct.loc['n'][1]-1)*mct.loc['SD'][1]**2) ) /
                         (mct.loc['n'][0] + mct.loc['n'][1] - 2) )
    sedifference = pooled_sd * np.sqrt( (1/mct.loc['n'][0]) + (1/mct.loc['n'][1]) )
    difference_mean_ci1 = difference_mean + (t_test_value * sedifference)
    difference_mean_ci2 = difference_mean - (t_test_value * sedifference)
    if difference_mean_ci1>difference_mean_ci2:
        difference_mean_cilower = difference_mean_ci2
        difference_mean_ciupper = difference_mean_ci1
    else:
        difference_mean_cilower = difference_mean_ci1
        difference_mean_ciupper = difference_mean_ci2
    cohend = difference_mean / pooled_sd
    t_test_result= pd.DataFrame ([difference_mean, sedifference, t_test_value, p_value,
                                  difference_mean_cilower, difference_mean_ciupper, cohend],
                                 index = ['Difference between means', 'SE difference', 't-test', 'p-value',
                                          'Lower bound difference CI', 'Upper bound difference CI', 'Cohen\'s d'],
                                 columns=['Value'])

    return t_test_result


def anova(data=None, independent=None, dependent=None):

    pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)

    independent = str(independent)
    dependent = str(dependent)
    if input_check_numerical_categorical(data, independent, dependent):
        return

    formula = dependent + ' ~ ' + independent
    model = ols(formula, data=data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    aov_table.rename(columns={'PR(>F)':'p'}, inplace=True)
    aov_table['F'] = pd.Series([aov_table['F'][0], ''], index = [independent, 'Residual'])
    aov_table['p'] = pd.Series([aov_table['p'][0], ''], index = [independent, 'Residual'])
    eta_sq = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
    aov_table['Eta squared'] = pd.Series([eta_sq, ''], index = [independent, 'Residual'])

    return aov_table


def tukey(data=None, independent=None, dependent=None):

    pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)

    independent = str(independent)
    dependent = str(dependent)
    if input_check_numerical_categorical(data, independent, dependent):
        return

    test = multi.MultiComparison(data[dependent], data[independent])
    res = test.tukeyhsd()
    display(res.summary())
    res.plot_simultaneous()

    return


def chi_square(data=None, variable1=None, variable2=None):

    pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)

    variable1 = str(variable1)
    variable2 = str(variable2)
    if input_check_categorical_categorical(data, variable1, variable2):
        return

    values_var1=pd.unique(data[variable1])
    values_var2=pd.unique(data[variable2])

    problem_found=False
    for variable in [values_var1, values_var2]:
        if len(variable)<2:
            print(variable, 'has less than two categories. It has:', len(variable))
            problem_found=True
    if problem_found:
        return

    contingency_table = pd.crosstab(data[variable1], data[variable2])
    contingency_table = pd.DataFrame(contingency_table)
    display(Markdown('**Contingency Table**'))
    display(contingency_table)

    chi2_test=stats.chi2_contingency(contingency_table, correction=False)

    chi2_result = pd.Series ([chi2_test[0], chi2_test[1], chi2_test[2], chi2_test[3]],
                            index = ['Chi-square value', 'p-value', 'Degrees of freedom', 'Expected frequencies'])
    chi2_result = pd.DataFrame(chi2_result, columns=['Value'])
    display(Markdown('**Results Chi-square test**'))
    display(chi2_result)

    return


def logistic_reg(data=None, independent=None, dependent=None):

    pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)

    independent = str(independent)
    dependent = str(dependent)
    if input_check_categorical(data, independent, dependent):
        return

    if not len(pd.unique(data[dependent]))==2:
        print('Dependent variable must have two categories')
        print(dependent, 'variable has', len(pd.unique(data[dependent])), 'categories')
        return

    data['interceptant']=1
    independent=[independent, 'interceptant']
    logReg = sm.Logit(data[dependent], data[independent])
    regression = logReg.fit()
    display(regression.summary())
    display(Markdown('**Coefficients confidence intervals**'))
    display(regression.conf_int())

    predicted_values =regression.predict()
    plt.plot(data[independent[0]], data[dependent], 'o', label='Actual values')
    plt.plot(data[independent[0]], predicted_values, 'ok', label='Predicted probabilities')
    plt.xlabel(independent[0], fontsize=14)
    plt.ylabel('Probability '+dependent, fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.show()

    return


# Functions to validate statistical functions inputs
def input_check_numerical_categorical(data, independent, dependent):

    problem_found=check_input_dataframe(data)
    if check_variable_specified(independent):
        print ('An independent variable was not specified')
        problem_found=True
    if check_variable_specified(dependent):
        print ('A dependent variable was not specified')
        problem_found=True
    if problem_found:
        return problem_found

    if check_variables_are_columns(data, independent, dependent):
        return True

    if check_variable_types(data, dependent, ['int', 'float']):
        problem_found=True
    if check_variable_types(data, independent, ['bool', 'category']):
        problem_found=True

    return problem_found


def input_check_numerical_numerical(data, variable1, variable2):

    problem_found=check_input_dataframe(data)
    if check_variable_specified(variable1) or check_variable_specified(variable2):
        print ('Two variables must be specified')
        problem_found=True
    if problem_found:
        return problem_found

    if check_variables_are_columns(data, variable1, variable2):
        return True

    for variable in [variable1, variable2]:
        if check_variable_types(data, variable, ['int', 'float']):
            problem_found=True

    return problem_found


def input_check_categorical_categorical(data, variable1, variable2):

    problem_found=check_input_dataframe(data)
    if check_variable_specified(variable1) or check_variable_specified(variable2):
        print ('Two variables must be specified')
        problem_found=True
    if problem_found:
        return problem_found

    if check_variables_are_columns(data, variable1, variable2):
        return True

    for variable in [variable1, variable2]:
        if check_variable_types(data, variable, ['bool', 'category']):
            problem_found=True

    return problem_found


def input_check_categorical(data, independent, dependent):

    problem_found=check_input_dataframe(data)
    if check_variable_specified(independent):
        print ('An independent variable was not specified')
        problem_found=True
    if check_variable_specified(dependent):
        print ('A dependent variable was not specified')
        problem_found=True
    if problem_found:
        return problem_found

    if check_variables_are_columns(data, independent, dependent):
        return True

    if check_variable_types(data, dependent, ['bool', 'category']):
        problem_found=True

    return problem_found


# Functions to validate individual inputs
def check_input_dataframe(data):

    if not str(type(data))=='<class \'pandas.core.frame.DataFrame\'>':
        print (data, 'is not a DataFrame')
        return True
    else:
        return False


def check_variable_specified(variable):

    if variable==None:
        return True
    else:
        return False


def check_variable_is_column(data, variable):

    if variable not in data.columns:
        print (variable, 'is not a column of', data, 'dataset')
        return True
    else:
        return False


def check_variables_are_columns(data, variable1, variable2):

    problem_found=False
    for variable in [variable1, variable2]:
        if check_variable_is_column(data, variable):
            problem_found=True
    return problem_found


def check_variable_types(data, variable, data_types):

    if data[variable].dtypes not in data_types:
        print (variable, 'is not of', data_types, 'type')
        return True
    else:
        return False
