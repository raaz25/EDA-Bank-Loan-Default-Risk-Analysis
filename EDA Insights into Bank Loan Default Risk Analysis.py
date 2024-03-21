#!/usr/bin/env python
# coding: utf-8

# ## Import Python Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

# setting up plot style 
style.use('seaborn-poster')
style.use('fivethirtyeight')


# ### Supress Warnings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### Adjust Jupyter Views

# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


# ## Reading & Understanding the data

# ### Importing the input files

# In[4]:


applicationDF = pd.read_csv(r"E:\Projects\EDA Insights into Bank Loan Default Risk Analysis\application_data.csv")
previousDF = pd.read_csv("E:\Projects\EDA Insights into Bank Loan Default Risk Analysis\previous_application.csv")
applicationDF.head()


# In[5]:


previousDF.head()


# ### Inspect Data Frames

# In[6]:


# Database dimension
print("Database dimension - applicationDF     :",applicationDF.shape)
print("Database dimension - previousDF        :",previousDF.shape)

#Database size
print("Database size - applicationDF          :",applicationDF.size)
print("Database size - previousDF             :",previousDF.size)


# In[7]:


# Database column types
applicationDF.info(verbose=True)


# In[8]:


previousDF.info(verbose=True)


# In[9]:


# Checking the numeric variables of the dataframes
applicationDF.describe()


# In[10]:


previousDF.describe()


# ## Data Cleaning & Manipulation

# ### Null Value Calculation

# #### applicationDF Missing values

# In[11]:


import missingno as mn
mn.matrix(applicationDF)


# Insight:
# * Based on the above Matrix, it is evidednt that the dataset has many missing values. Let's check for 
#   each column what is the % of missing values

# In[12]:


# % null value in each column
round(applicationDF.isnull().sum() / applicationDF.shape[0] * 100.00,2)


# Insight:
# * There are many columns in applicationDF dataframe where missing value is more than 40%. Let's plot
#   the columns vs missing value % with 40% being the cut-off marks

# In[13]:


null_applicationDF = pd.DataFrame((applicationDF.isnull().sum())*100/applicationDF.shape[0]).reset_index()
null_applicationDF.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_applicationDF,color='blue')
plt.xticks(rotation =90,fontsize =7)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in application data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# Insight:
# * From the plot we can see the columns in which percentage of null values more than 40% are marked
#   above the red line and the columns which have less than 40 % null values below the red line. Let's
#   check the columns which has more than 40% missing values

# In[14]:


# more than or equal to 40% empty rows columns
nullcol_40_application = null_applicationDF[null_applicationDF["Null Values Percentage"]>=40]
nullcol_40_application


# In[15]:


# How many columns have more than or euqal to 40% null values ?
len(nullcol_40_application)


# Insight:
# * Total of 49 columns are there which have more than 40% null values.Seems like most of the columns
#   with high missing values are related to different area sizes on apartment owned/rented by the loan
#   applicant

# #### previousDF Missing Values

# In[16]:


mn.matrix(previousDF)


# In[17]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# Insight:
# * There are many columns in previousDF dataframe where missing value is more than 40%. Let's plot the
#   columns vs missing value % with 40% being the cut-off marks

# In[18]:


null_previousDF = pd.DataFrame((previousDF.isnull().sum())*100/previousDF.shape[0]).reset_index()
null_previousDF.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_previousDF,color ='blue')
plt.xticks(rotation =90,fontsize =7)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in previousDF data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# Insight:
# * From the plot we can see the columns in which percentage of null values more than 40% are marked
#   above the red line and the columns which have less than 40 % null values below the red line. Let's
#   check the columns which has more than 40% missing values

# In[19]:


# more than or equal to 40% empty rows columns
nullcol_40_previous = null_previousDF[null_previousDF["Null Values Percentage"]>=40]
nullcol_40_previous


# In[20]:


# How many columns have more than or euqal to 40% null values ?
len(nullcol_40_previous)


# Insight:
# * Total of 11 columns are there which have more than 40% null values. These columns can be deleted.
#   Before deleting these columns, let's review if there are more columns which can be dropped or not[]
#   (http://)

# ### Analyze & Delete Unnecessary Columns in applicationDF

# #### EXT_SOURCE_X

# In[21]:


# Checking correlation of EXT_SOURCE_X columns vs TARGET column
Source = applicationDF[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]
source_corr = Source.corr()
ax = sns.heatmap(source_corr,
            xticklabels=source_corr.columns,
            yticklabels=source_corr.columns,
            annot = True,
            cmap ="RdYlGn")


# Insight:
# * Based on the above Heatmap, we can see there is almost no correlation between EXT_SOURCE_X columns
#   and target column, thus we can drop these columns. EXT_SOURCE_1 has 56% null values, where as
#   EXT_SOURCE_3 has close to 20% null values

# In[22]:


# create a list of columns that needs to be dropped including the columns with >40% null values
Unwanted_application = nullcol_40_application["Column Name"].tolist()+ ['EXT_SOURCE_2','EXT_SOURCE_3'] 
# as EXT_SOURCE_1 column is already included in nullcol_40_application 
len(Unwanted_application)


# #### Flag Document

# In[23]:


# Checking the relevance of Flag_Document and whether it has any relation with loan repayment status
col_Doc = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
df_flag = applicationDF[col_Doc+["TARGET"]]

length = len(col_Doc)

df_flag["TARGET"] = df_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})

fig = plt.figure(figsize=(21, 24))

for i, j in itertools.zip_longest(col_Doc, range(length)):
    plt.subplot(5, 4, j + 1)
    ax = sns.countplot(x=i, hue="TARGET", data=df_flag, palette=["r", "g"])
    plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# Insight:
# * The above graph shows that in most of the loan application cases, clients who applied for loans has
#   not submitted FLAG_DOCUMENT_X except FLAG_DOCUMENT_3. Thus, Except for FLAG_DOCUMENT_3, we can delete
#   rest of the columns. Data shows if borrower has submitted FLAG_DOCUMENT_3 then there is a less chance 
#   of defaulting the loan.

# In[24]:


# Including the flag documents for dropping the Document columns
col_Doc.remove('FLAG_DOCUMENT_3') 
Unwanted_application = Unwanted_application + col_Doc
len(Unwanted_application)


# #### Contact Parameters

# In[25]:


# checking is there is any correlation between mobile phone, work phone etc, email, Family members and Region rating
contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','TARGET']
Contact_corr = applicationDF[contact_col].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(Contact_corr,
            xticklabels=Contact_corr.columns,
            yticklabels=Contact_corr.columns,
            annot = True,
            cmap ="RdYlGn",
            linewidth=1)


# Insight:
# * There is no correlation between flags of mobile phone, email etc with loan repayment; thus these
#   columns can be deleted

# In[26]:


# including the 6 FLAG columns to be deleted
contact_col.remove('TARGET') 
Unwanted_application = Unwanted_application + contact_col
len(Unwanted_application)


# Insight:
# * Total 76 columns can be deleted from applicationDF

# In[27]:


# Dropping the unnecessary columns from applicationDF
applicationDF.drop(labels=Unwanted_application,axis=1,inplace=True)


# In[28]:


# Inspecting the dataframe after removal of unnecessary columns
applicationDF.shape


# In[29]:


# inspecting the column types after removal of unnecessary columns
applicationDF.info()


# Insight:
# * After deleting unnecessary columns, there are 46 columns remaining in applicationDF

# ### Analyze & Delete Unnecessary Columns in previousDF

# In[30]:


# Getting the 11 columns which has more than 40% unknown
Unwanted_previous = nullcol_40_previous["Column Name"].tolist()
Unwanted_previous


# In[31]:


# Listing down columns which are not needed
Unnecessary_previous = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']


# In[32]:


Unwanted_previous = Unwanted_previous + Unnecessary_previous
len(Unwanted_previous)


# Insight:
# * Total 15 columns can be deleted from previousDF

# In[33]:


# Dropping the unnecessary columns from previous
previousDF.drop(labels=Unwanted_previous,axis=1,inplace=True)
# Inspecting the dataframe after removal of unnecessary columns
previousDF.shape


# In[34]:


# inspecting the column types after after removal of unnecessary columns
previousDF.info()


# Insight:
# * After deleting unnecessary columns, there are 22 columns remaining in applicationDF

# ### Standardize Values

# Strategy for applicationDF:
# * Convert DAYS_DECISION,DAYS_EMPLOYED, DAYS_REGISTRATION,DAYS_ID_PUBLISH from negative to positive as
#   days cannot be negative.
# * Convert DAYS_BIRTH from negative to positive values and calculate age and create categorical bins
#   columns
# * Categorize the amount variables into bins
# * Convert region rating column and few other columns to categorical

# In[35]:


# Converting Negative days to positive days

date_col = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']

for col in date_col:
    applicationDF[col] = abs(applicationDF[col])


# In[36]:


# Binning Numerical Columns to create a categorical column

# Creating bins for income amount
applicationDF['AMT_INCOME_TOTAL']=applicationDF['AMT_INCOME_TOTAL']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

applicationDF['AMT_INCOME_RANGE']=pd.cut(applicationDF['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[37]:


applicationDF['AMT_INCOME_RANGE'].value_counts(normalize=True)*100


# Insight:
# * More than 50% loan applicants have income amount in the range of 100K-200K. Almost 92% loan 
#   applicants have income less than 300K

# In[38]:


# Creating bins for Credit amount
applicationDF['AMT_CREDIT']=applicationDF['AMT_CREDIT']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

applicationDF['AMT_CREDIT_RANGE']=pd.cut(applicationDF['AMT_CREDIT'],bins=bins,labels=slots)


# In[39]:


#checking the binning of data and % of data in each category
applicationDF['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100


# Insight:
# * More Than 16% loan applicants have taken loan which amounts to more than 1M.

# In[40]:


# Creating bins for Age
applicationDF['AGE'] = applicationDF['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']

applicationDF['AGE_GROUP']=pd.cut(applicationDF['AGE'],bins=bins,labels=slots)


# In[41]:


#checking the binning of data and % of data in each category
applicationDF['AGE_GROUP'].value_counts(normalize=True)*100


# Insight:
# * 31% loan applicants have age above 50 years. More than 55% of loan applicants have age over 40 years.

# In[42]:


# Creating bins for Employement Time
applicationDF['YEARS_EMPLOYED'] = applicationDF['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

applicationDF['EMPLOYMENT_YEAR']=pd.cut(applicationDF['YEARS_EMPLOYED'],bins=bins,labels=slots)


# In[43]:


#checking the binning of data and % of data in each category
applicationDF['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100


# Insight:
# * More than 55% of the loan applicants have work experience within 0-5 years and almost 80% of them
#   have less than 10 years of work experience

# In[44]:


#Checking the number of unique values each column possess to identify categorical columns
applicationDF.nunique().sort_values()


# ### Data Type Conversion

# In[45]:


# inspecting the column types if they are in correct data type using the above result.
applicationDF.info()


# Insight:
# * Numeric columns are already in int64 and float64 format. Hence proceeding with other columns.

# In[46]:


#Conversion of Object and Numerical columns to Categorical Columns
categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]
for col in categorical_columns:
    applicationDF[col] =pd.Categorical(applicationDF[col])


# In[47]:


# inspecting the column types if the above conversion is reflected
applicationDF.info()


# #### Standardize Values for previousDF

# Strategy for previousDF:
# * Convert DAYS_DECISION from negative to positive values and create categorical bins columns.
# * Convert loan purpose and few other columns to categorical.

# In[48]:


#Checking the number of unique values each column possess to identify categorical columns
previousDF.nunique().sort_values() 


# In[49]:


# inspecting the column types if the above conversion is reflected
previousDF.info()


# In[50]:


#Converting negative days to positive days 
previousDF['DAYS_DECISION'] = abs(previousDF['DAYS_DECISION'])


# In[51]:


#age group calculation e.g. 388 will be grouped as 300-400
previousDF['DAYS_DECISION_GROUP'] = (previousDF['DAYS_DECISION']-(previousDF['DAYS_DECISION'] % 400)).astype(str)+'-'+ ((previousDF['DAYS_DECISION'] - (previousDF['DAYS_DECISION'] % 400)) + (previousDF['DAYS_DECISION'] % 400) + (400 - (previousDF['DAYS_DECISION'] % 400))).astype(str)


# In[52]:


previousDF['DAYS_DECISION_GROUP'].value_counts(normalize=True)*100


# Insight:
# * Almost 37% loan applicatants have applied for a new loan within 0-400 days of previous loan decision

# In[53]:


#Converting Categorical columns from Object to categorical 
Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE','DAYS_DECISION_GROUP']

for col in Catgorical_col_p:
    previousDF[col] =pd.Categorical(previousDF[col])


# In[54]:


# inspecting the column types after conversion
previousDF.info()


# ### Null Value Data Imputation

# #### Imputing Null Values in applicationDF

# Strategy for applicationDF:
# * To impute null values in categorical variables which has lower null percentage, mode() is used to impute the most frequent items.
# * To impute null values in categorical variables which has higher null percentage, a new category is created.
# * To impute null values in numerical variables which has lower null percentage, median() is used as
#      * There are no outliers in the columns
#      * Mean returned decimal values and median returned whole numbers and the columns were number of requests

# In[55]:


# checking the null value % of each column in applicationDF dataframe
round(applicationDF.isnull().sum() / applicationDF.shape[0] * 100.00,2)


# ##### Impute categorical variable 'NAME_TYPE_SUITE' which has lower null percentage(0.42%) with the most frequent category using mode()[0]:

# In[56]:


applicationDF['NAME_TYPE_SUITE'].describe()


# In[57]:


applicationDF['NAME_TYPE_SUITE'].fillna((applicationDF['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


# ##### Impute categorical variable 'OCCUPATION_TYPE' which has higher null percentage(31.35%) with a new category as assigning to any existing category might influence the analysis:

# In[58]:


applicationDF['OCCUPATION_TYPE'] = applicationDF['OCCUPATION_TYPE'].cat.add_categories('Unknown')
applicationDF['OCCUPATION_TYPE'].fillna('Unknown', inplace =True)


# ##### Impute numerical variables with the median as there are no outliers that can be seen from results of describe() and mean() returns decimal values and these columns represent number of enquiries made which cannot be decimal:

# In[59]:


applicationDF[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# ##### Impute with median as mean has decimals and this is number of requests

# In[60]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

for col in amount:
    applicationDF[col].fillna(applicationDF[col].median(),inplace = True)


# In[61]:


# checking the null value % of each column in previousDF dataframe
round(applicationDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# Insight:
# * We still have few null values in the columns: AMT_GOODS_PRICE, OBS_30_CNT_SOCIAL_CIRCLE,
#   DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE. We can ignore as this
#   percentage is very less.

#  #### Imputing Null Values in previousDF

# Strategy for applicationDF:
# * To impute null values in numerical column, we analysed the loan status and assigned values.
# * To impute null values in continuous variables, we plotted the distribution of the columns and used
#      * median if the distribution is skewed
#      * mode if the distribution pattern is preserved.

# In[62]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# ##### Impute AMT_ANNUITY with median as the distribution is greatly skewed:

# In[63]:


plt.figure(figsize=(6,6))
sns.kdeplot(previousDF['AMT_ANNUITY'])
plt.show()


# Insight:
# * There is a single peak at the left side of the distribution and it indicates the presence of outliers
#   and hence imputing with mean would not be the right approach and hence imputing with median.

# In[64]:


previousDF['AMT_ANNUITY'].fillna(previousDF['AMT_ANNUITY'].median(),inplace = True)


# ##### Impute AMT_GOODS_PRICE with mode as the distribution is closely similar:

# In[65]:


plt.figure(figsize=(6,6))
sns.kdeplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])])
plt.show()


# There are several peaks along the distribution. Let's impute using the mode, mean and median and see
# if the distribution is still about the same.

# In[66]:


statsDF = pd.DataFrame() # new dataframe with columns imputed with mode, median and mean
statsDF['AMT_GOODS_PRICE_mode'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0])
statsDF['AMT_GOODS_PRICE_median'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].median())
statsDF['AMT_GOODS_PRICE_mean'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mean())

cols = ['AMT_GOODS_PRICE_mode', 'AMT_GOODS_PRICE_median','AMT_GOODS_PRICE_mean']

plt.figure(figsize=(18,10))
plt.suptitle('Distribution of Original data vs imputed data')
plt.subplot(221)
sns.distplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])]);
for i in enumerate(cols): 
    plt.subplot(2,2,i[0]+2)
    sns.distplot(statsDF[i[1]])


# Insight:
# * The original distribution is closer with the distribution of data imputed with mode in this case

# In[67]:


previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0], inplace=True)


# ##### Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these indicate that most of these loans were not started:

# In[68]:


previousDF.loc[previousDF['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[69]:


previousDF['CNT_PAYMENT'].fillna(0,inplace = True)


# In[70]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# Insight:
# * We still have few null values in the PRODUCT_COMBINATION column. We can ignore as this percentage is
#   very less.

# ### Identifying the outliers

# #### Finding outlier information in applicationDF

# In[71]:


plt.figure(figsize=(22,10))

app_outlier_col_1 = ['AMT_ANNUITY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','DAYS_EMPLOYED']
app_outlier_col_2 = ['CNT_CHILDREN','DAYS_BIRTH']
for i in enumerate(app_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=applicationDF[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(app_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=applicationDF[i[1]])
    plt.title(i[1])
    plt.ylabel("")


# Insight:
# * It can be seen that in current application data -
#    * AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE,CNT_CHILDREN have some number of outliers.
#    * AMT_INCOME_TOTAL has huge number of outliers which indicate that few of the loan applicants have  high income when compared to the others.
#    * DAYS_BIRTH has no outliers which means the data available is reliable.
#    * DAYS_EMPLOYED has outlier values around 350000(days) which is around 958 years which is impossible and hence this has to be incorrect entry.

# ###### We can see the stats for these columns below as well.

# In[72]:


applicationDF[['AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_BIRTH','CNT_CHILDREN','DAYS_EMPLOYED']].describe()


# ##### Finding outlier information in previousDF

# In[73]:


plt.figure(figsize=(22,8))

prev_outlier_col_1 = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','SELLERPLACE_AREA']
prev_outlier_col_2 = ['SK_ID_CURR','DAYS_DECISION','CNT_PAYMENT']
for i in enumerate(prev_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=previousDF[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(prev_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=previousDF[i[1]])
    plt.title(i[1])
    plt.ylabel("") 


# Insight: 
# * It can be seen that in previous application data -
#     * AMT_ANNUITY, AMT_APPLICATION, AMT_CREDIT, AMT_GOODS_PRICE, SELLERPLACE_AREA have huge number of outliers.
#     * CNT_PAYMENT has few outlier values.
#     * SK_ID_CURR is an ID column and hence no outliers.
#     * DAYS_DECISION has little number of outliers indicating that these previous applications decisions were taken long back.

# ###### We can see the stats for these columns below as well.

# In[74]:


previousDF[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION']].describe()


# ### Data Analysis

# Strategy:
# The data analysis flow has been planned in following way :
# 
# * Imbalance in Data
# * Categorical Data Analysis
#   * Categorical segmented Univariate Analysis
#   * Categorical Bi/Multivariate analysis
# * Numeric Data Analysis
#   * Bi-furcation of databased based on TARGET data
#   * Correlation Matrix
#   * Numerical segmented Univariate Analysis
#   * Numerical Bi/Multivariate analysis

#  #### Imbalance Analysis

# In[75]:


Imbalance = applicationDF["TARGET"].value_counts().reset_index()

plt.figure(figsize=(7,4))
x= ['Repayer','Defaulter']
sns.barplot(x=x,y="TARGET",data = Imbalance,palette= ['g','r'])
plt.xlabel("Loan Repayment Status")
plt.ylabel("Count of Repayers & Defaulters")
plt.title("Imbalance Plotting")
plt.show()


# In[76]:


count_0 = Imbalance.iloc[0]["TARGET"]
count_1 = Imbalance.iloc[1]["TARGET"]
count_0_perc = round(count_0/(count_0+count_1)*100,2)
count_1_perc = round(count_1/(count_0+count_1)*100,2)

print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))


# #### Plotting Functions

# ###### Following are the common functions customized to perform uniform anaysis that is called for all plots:

# In[77]:


# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(feature,ylog=False,label_rotation=False,horizontal_layout=True):
    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=applicationDF,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
    
    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 
    ax1.legend(['Repayer','Defaulter'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.show();


# In[78]:


# function for plotting repetitive countplots in bivariate categorical analysis

def bivariate_bar(x,y,df,hue,figsize):
    
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                  y=y,
                  data=df, 
                  hue=hue, 
                  palette =['g','r'])     
        
    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels = ['Repayer','Defaulter'])
    plt.show()


# In[79]:


# function for plotting repetitive rel plots in bivaritae numerical analysis on applicationDF

def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):
    
    plt.figure(figsize=figsize)
    sns.relplot(x=x, 
                y=y, 
                data=applicationDF, 
                hue="TARGET",
                kind=kind,
                palette = ['g','r'],
                legend = False)
    plt.legend(['Repayer','Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


# In[80]:


#function for plotting repetitive countplots in univariate categorical analysis on the merged df

def univariate_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, 
                  data=df,
                  hue= hue,
                  palette= palette,
                  order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     
    else:
        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       

    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


# In[81]:


# Function to plot point plots on merged dataframe

def merged_pointplot(x,y):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=x, 
                  y=y, 
                  hue="TARGET", 
                  data=loan_process_df,
                  palette =['g','r'])
   # plt.legend(['Repayer','Defaulter'])


# #### Categorical Variables Analysis

# ##### Segmented Univariate Analysis

# In[82]:


# Checking the contract type based on loan repayment status
univariate_categorical('NAME_CONTRACT_TYPE',True)


# Inferences:
# * Contract type: Revolving loans are just a small fraction (10%) from the total number of loans; in the
#   same time, a larger amount of Revolving loans, comparing with their frequency, are not repaid.

# In[83]:


# Checking the type of Gender on loan repayment status
univariate_categorical('CODE_GENDER')


# Inferences:
# * The number of female clients is almost double the number of male clients. Based on the percentage of
#   defaulted credits, males have a higher chance of not returning their loans (~10%), comparing with 
#   women (~7%)

# In[84]:


# Checking if owning a car is related to loan repayment status
univariate_categorical('FLAG_OWN_CAR')


# Inferences:
# * Clients who own a car are half in number of the clients who dont own a car. But based on the
#   percentage of deault, there is no correlation between owning a car and loan repayment as in both
#   cases the default percentage is almost same.

# In[85]:


# Checking if owning a realty is related to loan repayment status
univariate_categorical('FLAG_OWN_REALTY')


# Inferences:
# * The clients who own real estate are more than double of the ones that don't own. But the defaulting
#   rate of both categories are around the same (~8%). Thus there is no correlation between owning a
#   reality and defaulting the loan.

# In[86]:


# Analyzing Housing Type based on loan repayment status
univariate_categorical("NAME_HOUSING_TYPE",True,True,True)


# Inferences:
# * Majority of people live in House/apartment
# * People living in office apartments have lowest default rate
# * People living with parents (~11.5%) and living in rented apartments(>12%) have higher probability of
#   defaulting

# In[87]:


# Analyzing Family status based on loan repayment status
univariate_categorical("NAME_FAMILY_STATUS",False,True,True)


# Inferences:
# * Most of the people who have taken loan are married, followed by Single/not married and civil marriage
# * In terms of percentage of not repayment of loan, Civil marriage has the highest percent of not
#   repayment (10%), with Widow the lowest (exception being Unknown).

# In[88]:


# Analyzing Education Type based on loan repayment status
univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)


# Inferences:
# * Majority of the clients have Secondary / secondary special education, followed by clients with Higher
#   education. Only a very small number having an academic degree
# * The Lower secondary category, although rare, have the largest rate of not returning the loan (11%). 
# * The people with Academic degree have less than 2% defaulting rate.

# In[89]:


# Analyzing Income Type based on loan repayment status
univariate_categorical("NAME_INCOME_TYPE",True,True,False)


# Inferences:
# * Most of applicants for loans have income type as Working, followed by Commercial associate, Pensioner
#   and State servant.
# * The applicants with the type of income Maternity leave have almost 40% ratio of not returning loans,
#   followed by Unemployed (37%). The rest of types of incomes are under the average of 10% for not
#   returning loans.
# * Student and Businessmen, though less in numbers do not have any default record. Thus these two
#   category are safest for providing loan.

# In[90]:


# Analyzing Region rating where applicant lives based on loan repayment status
univariate_categorical("REGION_RATING_CLIENT",False,False,True)


# Inferences:
# * Most of the applicants are living in Region_Rating 2 place.
# * Region Rating 3 has the highest default rate (11%)
# * Applicant living in Region_Rating 1 has the lowest probability of defaulting, thus safer for
#   approving loans

# In[91]:


# Analyzing Occupation Type where applicant lives based on loan repayment status
univariate_categorical("OCCUPATION_TYPE",False,True,False)


# Inferences:
# * Most of the loans are taken by Laborers, followed by Sales staff. IT staff take the lowest amount of
#   loans.
# * The category with highest percent of not repaid loans are Low-skill Laborers (above 17%), followed by
#   Drivers and Waiters/barmen staff, Security staff, Laborers and Cooking staff.

# In[92]:


# Checking Loan repayment status based on Organization type
univariate_categorical("ORGANIZATION_TYPE",True,True,False)


# Inferences:
# * Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self employed people have relative high defaulting rate, and thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of defaulting.
# * Most of the people application for loan are from Business Entity Type 3
# * For a very high number of applications, Organization type information is unavailable(XNA)
# It can be seen that following category of organization type has lesser defaulters thus safer for providing loans:
#   * Trade Type 4 and 5
#   * Industry type 8

# In[93]:


# Analyzing Flag_Doc_3 submission status based on loan repayment status
univariate_categorical("FLAG_DOCUMENT_3",False,False,True)


# Inferences:
# * There is no significant correlation between repayers and defaulters in terms of submitting document 3
#   as we see even if applicants have submitted the document, they have defaulted a slightly more (~9%)
#   than who have not submitted the document (6%)

# In[94]:


# Analyzing Age Group based on loan repayment status
univariate_categorical("AGE_GROUP",False,False,True)


# Inferences:
# * People in the age group range 20-40 have higher probability of defaulting
# * People above age of 50 have low probability of defailting

# In[95]:


# Analyzing Employment_Year based on loan repayment status
univariate_categorical("EMPLOYMENT_YEAR",False,False,True)


# Inferences:
# * Majority of the applicants have been employeed in between 0-5 years. The defaulting rating of this
#   group is also the highest which is 10%
# * With increase of employment year, defaulting rate is gradually decreasing with people having 40+ year
#   experience having less than 1% default rate

# In[96]:


# Analyzing Amount_Credit based on loan repayment status
univariate_categorical("AMT_CREDIT_RANGE",False,False,False)


# Inferences:
# * More than 80% of the loan provided are for amount less than 900,000
# * People who get loan for 300-600k tend to default more than others.

# In[97]:


# Analyzing Amount_Income Range based on loan repayment status
univariate_categorical("AMT_INCOME_RANGE",False,False,False)


# Inferences:
# * 90% of the applications have Income total less than 300,000
# * Application with Income less than 300,000 has high probability of defaulting
# * Applicant with Income more than 700,000 are less likely to default

# In[98]:


# Analyzing Number of children based on loan repayment status
univariate_categorical("CNT_CHILDREN",True)


# Inferences:
# * Most of the applicants do not have children
# * Very few clients have more than 3 children.
# * Client who have more than 4 children has a very high default rate with child count 9 and 11 showing
#   100% default rate

# In[99]:


# Analyzing Number of family members based on loan repayment status
univariate_categorical("CNT_FAM_MEMBERS",True, False, False)


# Inferences:
# * Family member follows the same trend as children where having more family members increases the risk
#   of defaulting

# ##### Categorical Bi/Multivariate Analysis

# In[100]:


applicationDF.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()


# In[101]:


# Income type vs Income Amount Range
bivariate_bar("NAME_INCOME_TYPE","AMT_INCOME_TOTAL",applicationDF,"TARGET",(18,10))


# Inferences:
# * It can be seen that business man's income is the highest and the estimated range with default 95%
#   confidence level seem to indicate that the income of a business man could be in the range of slightly
#   close to 4 lakhs and slightly above 10 lakhs

# #### Numeric Variables Analysis

# #####  Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

# In[102]:


applicationDF.columns


# In[108]:


# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis
cols_for_correlation = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                        'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                        'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 
                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']


Repayer_df = applicationDF.loc[applicationDF['TARGET']==0, cols_for_correlation] # Repayers
Defaulter_df = applicationDF.loc[applicationDF['TARGET']==1, cols_for_correlation] # Defaulters


# #### Correlation between numeric variable

# In[109]:


applicationDF.head()


# In[111]:


# Getting the top 10 correlation for the Repayers data
corr_repayer = Repayer_df.corr()
corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape), k=1).astype(bool))  # Use 'bool' instead of 'np.bool'
corr_df_repayer = corr_repayer.unstack().reset_index()
corr_df_repayer.columns = ['VAR1', 'VAR2', 'Correlation']
corr_df_repayer.dropna(subset=["Correlation"], inplace=True)
corr_df_repayer["Correlation"] = corr_df_repayer["Correlation"].abs()
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_repayer.head(10)


# In[112]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# Inferences:
# * Correlating factors amongst repayers:
#   * Credit amount is highly correlated with
#     * amount of goods price
#     * loan annuity
#     * total income
#   
#   
# We can also see that repayers have high correlation in number of days employed.

# In[115]:


# Getting the top 10 correlation for the Defaulter data
corr_Defaulter = Defaulter_df.corr()
corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape), k=1).astype(bool))  # Use 'bool' instead of 'np.bool'
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns = ['VAR1', 'VAR2', 'Correlation']
corr_df_Defaulter.dropna(subset=["Correlation"], inplace=True)
corr_df_Defaulter["Correlation"] = corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_Defaulter.head(10)


# In[116]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Defaulter_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# Inferences:
# * Credit amount is highly correlated with amount of goods price which is same as repayers.
# * But the loan annuity correlation with credit amount has slightly reduced in defaulters(0.75) when
#   compared to repayers(0.77)
# * We can also see that repayers have high correlation in number of days employed(0.62) when compared to
#   defaulters(0.58).
# * There is a severe drop in the correlation between total income of the client and the credit
#   amount(0.038) amongst defaulters whereas it is 0.342 among repayers.
# * Days_birth and number of children correlation has reduced to 0.259 in defaulters when compared to
#   0.337 in repayers.
# * There is a slight increase in defaulted to observed count in social circle among defaulters(0.264)
#   when compared to repayers(0.254)

# #### Numerical Univariate Analysis

# In[117]:


# Plotting the numerical columns related to amount as distribution plot to see density
amount = applicationDF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']]

fig = plt.figure(figsize=(16,12))

for i in enumerate(amount):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(Defaulter_df[i[1]], hist=False, color='r',label ="Defaulter")
    sns.distplot(Repayer_df[i[1]], hist=False, color='g', label ="Repayer")
    plt.title(i[1], fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    
plt.legend()

plt.show() 


# Inferences:
# * Most no of loans are given for goods price below 10 lakhs
# * Most people pay annuity below 50000 for the credit loan
# * Credit amount of the loan is mostly less then 10 lakhs
# * The repayers and defaulters distribution overlap in all the plots and hence we cannot use any of
#   these variables in isolation to make a decision

# #### Numerical Bivariate Analysis

# In[118]:


# Checking the relationship between Goods price and credit and comparing with loan repayment staus
bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',applicationDF,"TARGET", "line", ['g','r'], False,(15,6))


# Inferences:
# * When the credit amount goes beyond 3M, there is an increase in defaulters.

# In[119]:


# Plotting pairplot between amount variable to draw reference against loan repayment status
amount = applicationDF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE','TARGET']]
amount = amount[(amount["AMT_GOODS_PRICE"].notnull()) & (amount["AMT_ANNUITY"].notnull())]
ax= sns.pairplot(amount,hue="TARGET",palette=["g","r"])
ax.fig.legend(labels=['Repayer','Defaulter'])
plt.show()


# Inferences:
# * When amt_annuity >15000 amt_goods_price> 3M, there is a lesser chance of defaulters
# * AMT_CREDIT and AMT_GOODS_PRICE are highly correlated as based on the scatterplot where most of the
#   data are consolidated in form of a line
# * There are very less defaulters for AMT_CREDIT >3M
# * Inferences related to distribution plot has been already mentioned in previous distplot graphs
#   inferences section

# ### Merged Dataframes Analysis

# In[120]:


#merge both the dataframe on SK_ID_CURR with Inner Joins
loan_process_df = pd.merge(applicationDF, previousDF, how='inner', on='SK_ID_CURR')
loan_process_df.head()


# In[121]:


#Checking the details of the merged dataframe
loan_process_df.shape


# In[122]:


# Checking the element count of the dataframe
loan_process_df.size


# In[123]:


# checking the columns and column types of the dataframe
loan_process_df.info()


# In[124]:


# Checking merged dataframe numerical columns statistics
loan_process_df.describe()


# In[125]:


# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers
L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters


# ***Plotting Contract Status vs purpose of the loan:***

# In[126]:


univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))


# Inferences:
# * Loan purpose has high number of unknown values (XAP, XNA)
# * Loan taken for the purpose of Repairs seems to have highest default rate
# * A very high number application have been rejected by bank or refused by client which has purpose as
#   repair or other. This shows that purpose repair is taken as high risk by bank and either they are
#   rejected or bank offers very high loan interest rate which is not feasible by the clients, thus they
#   refuse the loan.

# In[127]:


# Checking the Contract Status based on loan repayment status and whether there is any business loss or financial loss
univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"TARGET",['g','r'],False,(12,8))
g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]
df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
df1['Percentage'] = df1['Percentage'].astype(str) +"%" # adding percentage symbol in the results for understanding
print (df1)


# Inferences:
# * 90% of the previously cancelled client have actually repayed the loan. Revisiting the interest rates
#   would increase business opoortunity for these clients
# * 88% of the clients who have been previously refused a loan has payed back the loan in current case.
# * Refual reason should be recorded for further analysis as these clients would turn into potential
#   repaying customer.

# In[128]:


# plotting the relationship between income total and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')


# Inferences:
# * The point plot show that the people who have not used offer earlier have defaulted even when there
#   average income is higher than others

# In[129]:


# plotting the relationship between people who defaulted in last 60 days being in client's social circle and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')


# Inferences:
# * Clients who have average of 0.13 or higher DEF_60_CNT_SOCIAL_CIRCLE score tend to default more and
#   hence client's social circle has to be analysed before providing the loan.

# ### Conclusions

# After analysing the datasets, there are few attributes of a client with which the bank would be able to identify if they will repay the loan or not. The analysis is consised as below with the contributing factors and categorization:

# Decisive Factor whether an applicant will be Repayer:
#    * NAME_EDUCATION_TYPE: Academic degree has less defaults.
#    * NAME_INCOME_TYPE: Student and Businessmen have no defaults.
#    * REGION_RATING_CLIENT: RATING 1 is safer.
#    * ORGANIZATION_TYPE: Clients with Trade Type 4 and 5 and Industry type 8 have defaulted less than 3%
#    * DAYS_BIRTH: People above age of 50 have low probability of defaulting
#    * DAYS_EMPLOYED: Clients with 40+ year experience having less than 1% default rate
#    * AMT_INCOME_TOTAL:Applicant with Income more than 700,000 are less likely to default
#    * NAME_CASH_LOAN_PURPOSE: Loans bought for Hobby, Buying garage are being repayed mostly.
#    * CNT_CHILDREN: People with zero to two children tend to repay the loans.

# Decisive Factor whether an applicant will be Defaulter:
# * CODE_GENDER: Men are at relatively higher default rate
# * NAME_FAMILY_STATUS : People who have civil marriage or who are single default a lot.
# * NAME_EDUCATION_TYPE: People with Lower Secondary & Secondary education
# * NAME_INCOME_TYPE: Clients who are either at Maternity leave OR Unemployed default a lot.
# * REGION_RATING_CLIENT: People who live in Rating 3 has highest defaults.
# * OCCUPATION_TYPE: Avoid Low-skill Laborers, Drivers and Waiters/barmen staff, Security staff, Laborers
#   and Cooking staff as the default rate is huge.
# * ORGANIZATION_TYPE: Organizations with highest percent of loans not repaid are Transport: type 3
#   (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self
#   -employed people have relative high defaulting rate, and thus should be avoided to be approved for
#   loan or provide loan with higher interest rate to mitigate the risk of defaulting.
# * DAYS_BIRTH: Avoid young people who are in age group of 20-40 as they have higher probability of
#   defaulting
# * DAYS_EMPLOYED: People who have less than 5 years of employment have high default rate.
# * CNT_CHILDREN & CNT_FAM_MEMBERS: Client who have children equal to or more than 9 default 100% and
#   hence their applications are to be rejected.
# * AMT_GOODS_PRICE: When the credit amount goes beyond 3M, there is an increase in defaulters.

# The following attributes indicate that people from these category tend to default but then due to the number of people and the amount of loan, the bank could provide loan with higher interest to mitigate any default risk thus preventing business loss:
# 
# * NAME_HOUSING_TYPE: High number of loan applications are from the category of people who live in
#   Rented apartments & living with parents and hence offering the loan would mitigate the loss if any of
#   those default.
# * AMT_CREDIT: People who get loan for 300-600k tend to default more than others and hence having higher
#   interest specifically for this credit range would be ideal.
# * AMT_INCOME: Since 90% of the applications have Income total less than 300,000 and they have high
#   probability of defaulting, they could be offered loan with higher interest compared to other income
#   category.
# * CNT_CHILDREN & CNT_FAM_MEMBERS: Clients who have 4 to 8 children has a very high default rate and
#   hence higher interest should be imposed on their loans.
# * NAME_CASH_LOAN_PURPOSE: Loan taken for the purpose of Repairs seems to have highest default rate. A
#   very high number applications have been rejected by bank or refused by client in previous
#   applications as well which has purpose as repair or other. This shows that purpose repair is taken as
#   high risk by bank and either they are rejected, or bank offers very high loan interest rate which is
#   not feasible by the clients, thus they refuse the loan. The same approach could be followed in future
#   as well.

# Other suggestions:
# * 90% of the previously cancelled client have actually repayed the loan. Record the reason for
#   cancellation which might help the bank to determine and negotiate terms with these repaying customers
#   in future for increase business opportunity.
# * 88% of the clients who were refused by bank for loan earlier have now turned into a repaying client.
#   Hence documenting the reason for rejection could mitigate the business loss and these clients could
#   be contacted for further loans.

# In[ ]:




