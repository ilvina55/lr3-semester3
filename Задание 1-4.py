#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Задание 1 – Сортировка признаков
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as sk
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


pd.set_option("display.width", 100)
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.1, random_state=13
)


# In[7]:


params = {
 "n_estimators": 500,
 "max_depth": 4,
 "min_samples_split": 5,
 "learning_rate": 0.01,
 "loss": "squared_error",
}


# In[8]:


reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print(
 "The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[10]:


test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)


# In[13]:


fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
np.arange(params["n_estimators"]) + 1,
reg.train_score_,
"b-",
label="Training Set Deviance",
)
plt.plot(
np.arange(params["n_estimators"]) + 1,
test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()


# In[14]:


feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5


# In[15]:


fig = plt.figure(figsize=(12, 6))


# In[16]:


plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title("Feature Importance (MDI)")


# In[17]:


result = permutation_importance(
 reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()


# In[18]:


plt.subplot(1, 2, 2)
plt.boxplot(
 result.importances[sorted_idx].T,
 vert=False,
 labels=np.array(diabetes.feature_names)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


# In[ ]:


#Задание 2 – Оценка информативности


# In[19]:


from sklearn.feature_selection import mutual_info_classif


# In[20]:


mutual_info = mutual_info_classif(X_train, y_train)
sorted_idx = np.argsort(mutual_info)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.barh(pos, mutual_info[sorted_idx], align="center")
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title("Mutual info gain")
plt.xlabel("Information gain")
plt.ylabel("Feature")
fig.tight_layout()
plt.show()


# In[29]:


from sklearn.preprocessing import KBinsDiscretizer
kbins = KBinsDiscretizer(
    n_bins=10,
    encode='ordinal',
    strategy='uniform')
data_trans = kbins.fit_transform(X_train)
mutual_info = mutual_info_classif(data_trans, y_train)
sorted_idx = np.argsort(mutual_info)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.barh(pos, mutual_info[sorted_idx], align="center")
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title("Mutual info gain")
plt.xlabel("Information gain")
plt.ylabel("Feature")
fig.tight_layout()
plt.show()


# In[30]:


#Задание 3 – Определение уровня вариации
from sklearn.metrics import explained_variance_score


# In[31]:


train_score = np.zeros(
    (params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_train)):
    train_score[i] = explained_variance_score(y_train, y_pred)
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = explained_variance_score(y_test, y_pred)


# In[32]:


fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
        plt.title("Variance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    train_score,
    "b-",
    label="Training Set Variance")


# In[27]:


plt.plot(
    np.arange(
        params["n_estimators"]) + 1, test_score,
        "r-", label="Test Set Variance")
plt.legend(loc="lower right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Variance")
fig.tight_layout()
plt.show()


# In[33]:


#Задание 4 – Определение источника вариации
def permutation_variation_gain(
    samples, X_train,
    y_train, X_test, y_test):
    n_features = X_train.shape[-1]
    variance_scores = np.zeros(
        (n_features, samples), dtype=np.float64)
    for sm in range(samples):
        for i in range(1, n_features):
            reg = ensemble.RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_split=5,)
            x_train_cut, x_test_cut = X_train[
                :, :i], X_test[:, :i]
            reg.fit(x_train_cut, y_train)
            
            
            y_pred = reg.predict(x_test_cut)
            variance_scores[i][sm] = explained_variance_score(
                y_test,
                y_pred)

    return variance_scores


# In[34]:


variation_gain = permutation_variation_gain(
    1, kbins.fit_transform(X_train), y_train,
    kbins.transform(X_test), y_test)
permutation_variation_gain = permutation_variation_gain(
    100, kbins.fit_transform(X_train), y_train,
    kbins.transform(X_test), y_test)


# In[35]:


sorted_idx = np.argsort(variation_gain[:, 0], axis=0)
residual_variation = variation_gain[sorted_idx][:, 0]
pos = np.arange(sorted_idx.shape[-1]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, residual_variation, align="center")
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title("Feature Variation Score")
plt.subplot(1, 2, 2)
print(permutation_variation_gain.shape,
np.array(diabetes.feature_names)[sorted_idx].shape)
plt.boxplot(
    permutation_variation_gain.T,
    vert=False,
    labels=np.array(diabetes.feature_names)[sorted_idx],)
plt.title("Permutation Variation")
fig.tight_layout()
plt.show()


# In[ ]:




