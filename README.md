<H3>NAME: DHARSHINI K</H3>
<H3>REG NO: 212223230047</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:

Hardware – PCs <BR>
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle:**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing:**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:

<B>STEP 1:</B> Importing the libraries<BR>
<B>STEP 2:</B> Importing the dataset<BR>
<B>STEP 3:</B> Taking care of missing data<BR>
<B>STEP 4:</B> Encoding categorical data<BR>
<B>STEP 5:</B> Normalizing the data<BR>
<B>STEP 6:</B> Splitting the data into test and train<BR>

##  PROGRAM:

``` python
# IMPORT LIBRARIES
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# READ THE DATASET
df = pd.read_csv("Churn_Modelling.csv")

# HANDLING MISSING VALUES
print("Missing Values:\n", df.isnull().sum(), '\n')

# HANDLE DUPLICATES (IF ANY)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}\n")

# CHECK FOR OUTLIERS USING DESCRIBE()
print("Outliers (Summary Statistics):\n", df.describe(), '\n')

# DROP UNNECESSARY COLUMNS (LIKE 'SURNAME', 'GEOGRAPHY', AND 'GENDER')
df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)

# NORMALIZE THE DATASET USING MINMAXSCALER
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df.drop('Exited', axis=1)), columns=df.columns[:-1])

# NORMALIZED DATASET
print("Normalized dataset:\n", df_normalized.head(), "\n")

# DEFINE FEATURES(X) AND TARGET(Y)
X = df_normalized.values
y = df['Exited'].values

# INPUT & OUTPUT VALUES
print("Input Values (Features):\n", X[:5])
print("\nOutput Values (Target):\n", y[:5])

# SPLIT THE DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SPLITTING THE DATA FOR TRAINING & TESTING
print(f"\nTraining data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")
```

## OUTPUT:

![image](https://github.com/user-attachments/assets/73e55433-8192-4600-99d2-07ef9f291061)

![image](https://github.com/user-attachments/assets/3d1ed10a-6063-478e-bdf0-e9e430494db7)

![image](https://github.com/user-attachments/assets/3640d816-8713-4d2e-a631-9534e2da55c0)

![image](https://github.com/user-attachments/assets/92b27176-965a-487b-a131-3c65f3df42b0)

![image](https://github.com/user-attachments/assets/4891f82a-a5c4-46be-9964-1dd571af0d4d)

![image](https://github.com/user-attachments/assets/75caf5fb-b3ba-4e04-aebc-5630ee044113)


## RESULT:

Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
