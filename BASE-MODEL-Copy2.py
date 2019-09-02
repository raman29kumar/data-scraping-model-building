import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
print(os.getcwd())
jobs=pd.read_csv("jobs_above10.csv")
exe=pd.read_csv("exe_above10.csv")
print(jobs.shape)

# changing the format of job completion and execution time
def convert_time(df):
    return (df.SubmissionTime.map(lambda x:x.split("T",1)[1].replace('GMT','')), df.completionTime.map(lambda x:x.split("T",1)[1].replace('GMT','') if type(x)==str else x))

jobs['subTime'], jobs['compTime'] = convert_time(jobs)


def entry_exit_time_diff(x):
    """Return the job execution time in minutes"""
    # x is a pandas series consist of two rows
    entry, exit=x[0],x[1]
    if type(entry)!=str or type(exit)!=str:
        return 0
    h_entry, m_entry, s_entry=map(float,entry.split(":"))
    h_exit, m_exit, s_exit=map(float,exit.split(":"))
    hdiff=h_exit-h_entry
    mdiff=m_exit-m_entry
    sdiff=s_exit-s_entry
    # scaling of the time(in min)
    return (hdiff*60+mdiff+(sdiff/60))

jobs['jobTime']=jobs.loc[:,['subTime','compTime']].apply(entry_exit_time_diff,axis=1)

# Selecting Successful applications only
success_apps=pd.read_csv('success_apps_last.csv')
print(success_apps.shape)
#success_apps = success_apps.drop([0, 850])
success_apps

# creating summary columns of executor instances and its storage memory
imp_exe=exe.groupby('application_id').agg({'executor_Id':'count', 'storage_memory':'sum'})
# merging the above summary with driver's storage memory as well
imp_exe=pd.merge(imp_exe,exe.loc[exe.executor_Id=='driver',['application_id','storage_memory']],on='application_id',how='inner')

imp_jobs=jobs.groupby('application_id').agg({'jobTime':'sum','jobId':'count','numCompletedTasks':'sum'})
jobs_exe=pd.merge(imp_exe,imp_jobs,on='application_id',how='inner')
# intersection of the actual applications with succcessful apps
agg=pd.merge(jobs_exe,success_apps[['appName','application_id']],on='application_id',how='inner')
agg.shape

# label encoding of type of application
def app_name_mapping(x):
    if 'Risk' in x:
        return 0
    elif 'Measure' in x:
        return 1
    elif 'Claim' in x:
        return 2
    elif 'Deduplicator' in x:
        return 3
agg['appName']=agg['appName'].map(app_name_mapping)

# renaming the columns
agg.rename(columns={'executor_Id':'executor_instances','storage_memory_x':'storage_memory_executor','storage_memory_y':'storage_memory_driver','jobId':'total_jobs', 'numCompletedTasks':'num_completed_tasks','appName':'app_name','jobTime':'job_time'},inplace=True)

# feature creation
agg['storage_memory_executor']=agg['storage_memory_executor']-agg['storage_memory_driver']
agg['executor_instances']=agg['executor_instances']-1
agg['appName_target_median']=agg.groupby('app_name')['job_time'].transform('median')
agg['appName_target_std']=agg.groupby('app_name')['job_time'].transform('std')
agg['median_time_job_ratio']=agg['appName_target_median']/agg['total_jobs']

# seperating the input features and target label
input_params=agg.drop(['application_id', 'job_time'],axis=1)
target_feature=agg.job_time

# train test split for model selection and hyper parameters selection

x_train, x_test, y_train, y_test = train_test_split(input_params, target_feature, test_size=.2, random_state=21)

dt=DecisionTreeRegressor(min_samples_leaf=4)

dt.fit(x_train,y_train)
print('RMSE using decision Tree on training data is', np.sqrt(mean_squared_error(y_train ,dt.predict(x_train))))

print('RMSE using Decision Tree on test data is', np.sqrt(mean_squared_error(y_test ,dt.predict(x_test))))

# plotting the Desired and Predicted values of the target label
temp=pd.DataFrame([x_train.index, y_train, dt.predict(x_train)]).T
temp.sort_values(0,inplace=True)
plt.figure(figsize=(10,10))
plt.plot(temp[0], temp[1])
plt.plot(temp[0],temp[2])
plt.legend(['Actual values on training data','predicted values on training data'], loc='upper left')
plt.xlabel('Indexes of training data')
plt.ylabel('Time taken(in min)')
plt.show()

temp=pd.DataFrame([x_test.index, y_test, dt.predict(x_test)]).T
temp.sort_values(0,inplace=True)
plt.figure(figsize=(10,10))
plt.plot(temp[0], temp[1])
plt.plot(temp[0],temp[2])
plt.legend(['actual values on test data','predicted values on test data'], loc='upper left')
plt.xlabel('Indexes of test data')
plt.ylabel('Time taken(in min)')
plt.show()

# model training on whole data after hyper parameter tuning
dt.fit(input_params, target_feature)
# dumping the trained model in a file named dt.joblib
#dump(dt,'dt.joblib')

# feature importance calculated by the Decision Tree
pd.DataFrame(dt.feature_importances_,index=x_train.columns).plot(kind='bar')

# from sklearn.linear_model import LinearRegression
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error
# import numpy as np
# lr=linear_model.Ridge(alpha=0.1)
# lr.fit(x_train,y_train)
# print('RMSE using Linear regression on training data is',np.sqrt(mean_squared_error(y_train ,lr.predict(x_train))))

# print('RMSE using Linear regression on test data is',np.sqrt(mean_squared_error(y_test ,lr.predict(x_test))))
# print('coefficients of the linear regression are:', lr.coef_)




