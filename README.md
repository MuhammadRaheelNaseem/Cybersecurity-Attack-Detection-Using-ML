---
jupyter:
  kernelspec:
    display_name: Python
    language: python
    name: conda-env-python-py
  language_info:
  nbformat: 4
  nbformat_minor: 4
---

::: {#a88b0ee0-e3c6-4c60-a2b0-4ebc14826a8a .cell .markdown}
```{=html}
<p style="text-align:center">
    <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>
```
:::

::: {#99a013db-1bdd-4a5e-87a2-cccbdff27e7a .cell .markdown}
# **The Art and Science of Cybersecurity Attack Detection: A Hybrid Approach**
:::

::: {#466dd3cb-63e6-4b11-991d-d18feea9fee4 .cell .markdown}
Estimated time needed: **90** minutes
:::

::: {#03fa1eff-879e-41d9-ba04-f78bd5022b44 .cell .markdown}
This project aims to improve cyber security by developing a machine
learning and rule-based approach to detect cyber attacks. The approach
involves analyzing network data to identify potential attacks by
identifying correlations between various variables. By completing this
project, you will be able to understand how to analyze network data and
identify the variables associated with cyber attacks. By leveraging
machine learning algorithms and rule-based approaches, this project
helps to improve the accuracy and efficiency of cyber attack detection,
thereby enhancing the security of digital networks and systems. This
project is a valuable first step towards becoming a cyber security
expert.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/CC_CybersecurityDetection.png" width="600" alt="Cyber attack image">`{=html}
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

::: {#c44aae55-746c-4000-8994-7ae1cfb5ee41 .cell .markdown}
## **Table of Contents**

```{=html}
<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
        </ol>
    </li>
    <li><a href="#3.-Strategies-to-Detect-Cyber-Attacks">Strategies to Detect Cyber Attacks</a></li>
    <li>
        <a href="#4.Cyber Attack Data">Cyber Attacks Data</a>
        <ol>
            <li><a href="#Data Exploration">Data Exploration</a></li>
        </ol>
    </li>
    <li>
        <a href="#5.Rule-Based System">Rule-Based System</a>
        <ol>
            <li><a href="#Evaluation Metric">Evaluation Metric</a></li>
            <li><a href="#Introducing Snort">Introducing Snort For Rule-Based System</a></li>
        </ol>
    </li>
     <li>
        <a href="#6.Machine Learning Model For Cyber Attack Detection">Machine Learning Model For Cyber Attack Detection</a>
        <ol>
            <li><a href="#Building a RandomForest Model">Building a RandomForest Model</a></li>
        </ol>
    </li>
    <li>
        <a href="#7.Human Analysis">Human Analysis</a>
        <ol>
            <li><a href="#7.1. Correlations In The Dataset">Correlations In The Dataset</a></li>
            <li><a href="#7.2 Feature Ranking From Random Forest">Feature Ranking From Random Forest</a></li>
            <li><a href="#7.3 Discussing The Network Features">Discussing The Network Features</a></li>
        </ol>
    </li>
     <li>
        <a href="#8.Cyber Security for Cloud Services">Cyber Security for Cloud Services</a>
    </li>
     <li>
        <a href="#9.List of All Features With Descriptions">List of All Features With Descriptions</a>
    </li>
</ol>
```
:::

::: {#00038d0a-234b-4463-806a-d6b16075c6fe .cell .markdown}
## 1. Objectives {#1-objectives}

Our main goal is to understand how attacks happen and what are the
important indicators of attack. by knowing that, we can implement a
monitoring system for attack detection. By completing this project, you
will be able to apply your learnings to real-world scenarios and
contribute to the ongoing effort to secure the cyber realm.

After completing this lab you will be able to:

-   Understand how cyber attacks occur and identify important indicators
    of attacks.
-   Implement a monitoring system for attack detection using both
    rule-based and machine learning approaches.
-   Learn how to visualize variables in network data.
-   Gain experience in using machine learning algorithms such as Random
    Forest for classification and feature ranking.
-   Enhance your knowledge and skills in cybersecurity and introducing
    powerful tools to equipped to detect and prevent cyber attacks.
:::

::: {#9cec4049-b8f3-4de9-b694-4daa6df30871 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#e6af415f-378e-4049-8fe1-2d07ce6e834d .cell .markdown}
## 2. Setup {#2-setup}
:::

::: {#67f587e3-f358-4663-b264-93a317d61e00 .cell .markdown}
### 2.1 Installing Required Libraries {#21-installing-required-libraries}

The following required libraries are pre-installed in the Skills Network
Labs environment. However, if you run this notebook commands in a
different Jupyter environment (e.g. Watson Studio or Ananconda), you
will need to install these libraries in the code cell below.
:::

::: {#01707fbd-a8e0-4690-88aa-b7c6e843b157 .cell .code}
``` python
%%capture 
!pip install -U 'skillsnetwork' 'seaborn' 'nbformat' 
```
:::

::: {#75bef560-a4c3-421b-be55-09fd50534b69 .cell .code}
``` python
%%capture 
!pip install scikit-learn==1.0.0
!pip install dtreeviz
```
:::

::: {#e17f7d5a-bdfe-4ae0-a90d-2244e2779386 .cell .markdown}
> *YOU NEED TO `<span style="color:red">`{=html} **RESTART THE KERNEL**
> `</span>`{=html} by going to the `Kernel` menu and clicking on
> `Restart Kernel`.*
:::

::: {#3e9f9232-6f96-4213-8312-2a335a52e049 .cell .markdown}
### 2.2 Importing Required Libraries {#22-importing-required-libraries}

*import some essential libraries*
:::

::: {#5f23d57d-6c52-4aee-a805-881b1899ee7e .cell .code}
``` python
# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

#import shap
import skillsnetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_context('notebook')
sns.set_style('white')
```
:::

::: {#33d8b47d-3aa5-463b-a25c-be55134c1067 .cell .markdown}

------------------------------------------------------------------------

## 3. Strategies to Detect Cyber Attacks {#3-strategies-to-detect-cyber-attacks}

1.  The first approach to detecting cyber attacks is to use **rule-based
    system**. These systems use a set of predefined rules to identify
    potential attacks based on known attack patterns. For example, a
    rule might flag an attack if the source to destination time to live
    (sttl) value is less than 10 and the count of states time to live
    (ct_state_ttl) value is greater than 100. While rule-based systems
    can be effective in detecting known attacks, they may also produce
    false positives, so it\'s important to validate the alerts generated
    by these systems.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/images/simple_rule_system.png" width="300" alt="Simple Rule-Based System">`{=html}`</img>`{=html}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                           *Simple Rule-Based System*

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1.  Another approach to detecting cyber attacks is to use **machine
    learning algorithms**, such as Random Forest and adaboost. These
    algorithms are trained on a large dataset of network packets and can
    be used to identify anomalies in real-time network traffic that
    might indicate an attack. For example, a machine learning model
    might detect an attack if the destination to source transaction
    bytes (rate) value is greater than 10,000.

  ---------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cdn.pixabay.com/photo/2019/01/22/10/58/pixel-cells-3947912_1280.png" width="200" alt="Cyber attack image">`{=html}
  ---------------------------------------------------------------------------------------------------------------------------------------
                                                  *image credit: <https://pixabay.com/>*

  ---------------------------------------------------------------------------------------------------------------------------------------

1.  In addition to these automated methods, **human analysis** can play
    a critical role in identifying cyber attacks. Human analysts can use
    their expertise to interpret the data and understand the context in
    which the attack is taking place. They can also validate the alerts
    generated by automated systems and take into account the broader
    context of the organization when analyzing data. For example, they
    may understand that a particular system is undergoing maintenance
    and can disregard anomalies in the data that might otherwise
    indicate an attack.

  --------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cdn.pixabay.com/photo/2018/03/11/06/15/cyber-security-3216076_1280.jpg" width="200" alt="human analysis">`{=html}
  --------------------------------------------------------------------------------------------------------------------------------------
                                                  *image credit: <https://pixabay.com/>*

  --------------------------------------------------------------------------------------------------------------------------------------

Therefore, our strategy involves utilizing establishing a rule-based
system as the first layer of detection. Then, we utilize a machine
learning algorithm to pinpoint attacks. Finally, we delve into the
variables to understand their significance and examine their importance
as indicators of cyber attacks. This will contribute to developing cyber
security knowledge for human analysis.
:::

::: {#d28c2f8a-b926-4cad-9168-8b2048a9d81a .cell .markdown}

------------------------------------------------------------------------
:::

::: {#217405fa-30c3-4c3b-9ebd-a323d0ab0fbc .cell .markdown}
## 4. Cyber Attack Data {#4-cyber-attack-data}

The data is collected by the [University of New South Wales
(Australia)](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01).
That includes records of different types of cyber attacks. The dataset
contains network packets captured in the Cyber Range Lab of UNSW
Canberra. The data is provided in two sets of training and testing data.
We combine them to create one set of larger data.
:::

::: {#aced1ace-f594-438a-9633-fe90b7a60107 .cell .code}
``` python
## loading the data
training = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/UNSW_NB15_training-set.csv")
testing = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/UNSW_NB15_testing-set.csv")
print("training ",training.shape)
print("testing ",testing.shape)
```
:::

::: {#fd69a7ee-9100-4ec9-ad0c-9defd3cde574 .cell .markdown}
To achieve a better performance, we will create a larger dataset and
assign 70% for training and 30% to testing.
:::

::: {#565aedd0-c285-4ec7-ab00-d3f552747ab4 .cell .code}
``` python
# checking if all the columns are similar
all(training.columns == testing.columns)
```
:::

::: {#e36df897-32dd-4edb-9bf4-b22577996649 .cell .code}
``` python
# creating one-whole dataframe which contains all data and drop the 'id' column
df = pd.concat([training,testing]).drop('id',axis=1)
df = df.reset_index(drop=True)

# print one attack sample
df.head(2)
```
:::

::: {#7976a39a-3319-463d-8c4a-9a59c3882392 .cell .markdown}
The dataset includes 43 variables regarding monitoring the network and 2
variables that define if an attack happens (`label`) and the types of
attacks (`attack_cat`). The description of all the variables is
available at the end of this notebook.

Lets quick look on the types of attacks.
:::

::: {#6e8b6547-6aac-48de-9e5e-048cd3b4f333 .cell .code}
``` python
# getting the attack category column 
df.attack_cat.unique()
```
:::

::: {#17839221-6814-4ce1-943c-542aad3d79c9 .cell .markdown}
  ---------------------------------------------------------------------------------------------------------------------
   `<img src="https://cdn.pixabay.com/photo/2022/03/15/16/52/scam-7070718_1280.png" width="600" alt="hacking">`{=html}
  ---------------------------------------------------------------------------------------------------------------------
                                         *image credit: <https://pixabay.com/>*

  ---------------------------------------------------------------------------------------------------------------------

The dataset includes nine types of attacks, including:

1.  `Fuzzers`: Attack that involves sending random data to a system to
    test its resilience and identify any vulnerabilities.

2.  `Analysis`: A type of attack that involves analyzing the system to
    identify its weaknesses and potential targets for exploitation.

3.  `Backdoors`: Attack that involves creating a hidden entry point into
    a system for later use by the attacker.

4.  `DoS (Denial of Service)`: Attack that aims to disrupt the normal
    functioning of a system, making it unavailable to its users.

5.  `Exploits`: Attack that leverages a vulnerability in a system to
    gain unauthorized access or control.

6.  `Generic`: A catch-all category that includes a variety of different
    attack types that do not fit into the other categories.

7.  `Reconnaissance`: Attack that involves gathering information about a
    target system, such as its vulnerabilities and potential entry
    points, in preparation for a future attack.

8.  `Shellcode`: Attack that involves executing malicious code,
    typically in the form of shell scripts, on a target system.

9.  `Worms`: A type of malware that spreads itself automatically to
    other systems, often causing harm in the process.

These nine categories cover a wide range of attack types that can be
used to exploit a system, and it is important to be aware of them to
protect against potential security threats.
:::

::: {#6b9e5a48-4df5-4423-90cf-ff954010e084 .cell .markdown}
### 4.1. Data Exploration {#41-data-exploration}
:::

::: {#9083142c-9527-4129-8685-87bf1333e902 .cell .markdown}
In this section, we briefly explore our dataset.
:::

::: {#4cb81ed6-9b57-4e92-83a2-4205424ca0d8 .cell .code}
``` python
# exploring the types of variables
df.info()
```
:::

::: {#fb03ef01-9eca-45cd-91d3-6a5b034c19b9 .cell .markdown}
As we can see, some variables, that are categorical, are defined as
strings. In the following cell we convert them into categorical type
provided by `pandas`.
:::

::: {#5957d7bf-d410-492d-b1e6-e8a1494b6a6e .cell .code}
``` python
# some columns should be change from string to categoriacal
for col in ['proto', 'service', 'state']:
    df[col] = df[col].astype('category').cat.codes
    df[col] = df[col].astype('category').cat.codes
    
df['attack_cat'] = df['attack_cat'].astype('category') # keep the nomical info for attack info
```
:::

::: {#b82a1ddd-166a-40ae-9e7b-00235901f152 .cell .markdown}
Exploring how many records of different types of attacks are in the
dataset.
:::

::: {#99a2c4e1-3dcd-4362-8d66-3dfab09ccaae .cell .code}
``` python
# explore different types of attackes
print(df[df['label']==1]
     ['attack_cat']
     .value_counts()
)
# plot the pie plot of attacks
df[df['label']==1]['attack_cat'].value_counts()\
    .plot\
    .pie(autopct='%1.1f%%',wedgeprops={'linewidth': 2, 'edgecolor': 'white', 'width': 0.50})
```
:::

::: {#73a225a7-13f4-4c4a-9f01-0671a142d596 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#50cafc29-652f-4877-ab03-42ed1529c9fa .cell .markdown}
## 5. Rule-Based System {#5-rule-based-system}

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/images/simple_rule_system.png" width="400" alt="Simple Rule-Based System">`{=html}`</img>`{=html}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                           *Simple Rule-Based System*

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Both **rule-based systems and machine learning systems** have their own
strengths and weaknesses, and using both together can provide a more
comprehensive and effective approach to detecting cyber attacks. Here
are a few reasons why:

1.  Explainability: Rule-based systems provide clear and concise rules
    that can be easily understood and interpreted by human experts. This
    makes it easier to understand how the system is making its
    predictions and to validate the results.

2.  Robustness: Rule-based systems are less likely to be affected by
    unexpected changes in the data distribution compared to machine
    learning models. They can still provide accurate results even when
    the data changes, as long as the rules remain valid.

3.  Speed: Rule-based systems can be much faster than machine learning
    models, especially for simple problems. This can be important in
    real-time monitoring systems where the response time needs to be
    fast.

4.  Complementary strengths: Rule-based systems and machine learning
    models can complement each other. Rule-based systems can be used to
    detect simple, well-defined attacks, while machine learning models
    can be used to detect more complex, subtle attacks.

In our project, we first employ rule-based model and then we utilize
machine learning model. By combining rule-based systems and machine
learning models, it is possible to take advantage of the strengths of
each approach to create a more effective and comprehensive system for
detecting cyber attacks.
:::

::: {#853ccd95-4b91-462f-a19c-653a12b4d068 .cell .markdown}
### 5.1. Evaluation Metric {#51-evaluation-metric}

In the rule-based model, we are looking for higher recall rate because
we are sensitive to alarm potential threats, and we can not afford to
miss attacks (FALSE NEGATIVE). Recall (or True Positive Rate) is
calculated by dividing the true positives (actual attacks) by anything
that should have been predicted as positive (detected and non-detected
attacks).

  ----------------------------------------------------------------------------------------------------------------------
   `<img src="https://keytodatascience.com/wp-content/uploads/2019/09/values.jpg" width="400" alt="IBM Watson">`{=html}
  ----------------------------------------------------------------------------------------------------------------------
          Learn more about confusion matrix (and image credit): <https://keytodatascience.com/confusion-matrix/>

  ----------------------------------------------------------------------------------------------------------------------
:::

::: {#c2769d8e-2e9a-482b-adca-77a173212239 .cell .code}
``` python
# separating the target columns in the training and testing data 
from sklearn.model_selection import train_test_split

# Split the data into variables and target variables
# let's exclude label columns 
X = df.loc[:, ~df.columns.isin(['attack_cat', 'label'])]
y = df['label'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# Getting the list of variables
feature_names = list(X.columns)

# print the shape of train and test data
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
```
:::

::: {#e7f04e06-2f2d-4fb7-a200-50c7f6009a15 .cell .markdown}
We use a decision tree model to create a set of criteria for detecting
cyber attacks in our rule-based system. The goal of this first layer of
protection is to have a high recall rate, so we conduct a grid search to
optimize the model toward maximizing recall.
:::

::: {#8f1cbcc4-3c13-4e84-bf1b-5c5854f3aa7c .cell .code}
``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Use GridSearchCV to search for the best parameters
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best recall score:", grid_search.best_score_)
```
:::

::: {#5475eb4f-f2af-40db-86aa-d82beba90ff8 .cell .markdown}
Using the parameters above, adjust the decision tree for high recall
rate.
:::

::: {#1418d03d-f9f5-4ee3-8960-2e48d5616aa7 .cell .code}
``` python
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

clf=grid_search.best_estimator_
#same as
#clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, min_samples_split=2, criterion= 'entropy')
#clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the mean absolute error of the model
recall = recall_score(y_test, y_pred)
print("Recall: ", recall)
```
:::

::: {#168ab5d6-e3d1-49d6-8d3d-9beeee8e21ef .cell .markdown}
One of the strengths of a decision tree is to present the sets of rules
than can be utilized for rule-based systems. Here, we visualize the
rules.
:::

::: {#305c7f4c-ff01-4c32-95dd-d17559648315 .cell .code}
``` python
# plot the tree 
from sklearn.tree import export_text
import dtreeviz

print(":::::::> The RULES FOR HIGH RECALL RATE <::::::: \n" ,export_text(clf,feature_names=feature_names))

# visualizing the tree
viz_model = dtreeviz.model(clf,
                           X_train=X_train, y_train=y_train, 
                           feature_names=feature_names)

v = viz_model.view(fancy=True)     # render as SVG into internal object 
v
```
:::

::: {#e874cb8a-8ba5-494b-9a19-ed4d465afa07 .cell .markdown}
We create rules for those that are identified as potential attacks
(`class 1`) in the decision tree. Then, filter out the testing set.
:::

::: {#73080c9b-3e2f-43d6-94f1-88a1f8bbd311 .cell .markdown}
We apply our rules to the testing data and call them `test_2`.
:::

::: {#49aab9a5-e57e-4473-869a-1a32b3c6d532 .cell .code}
``` python
X_test = X_test.reset_index(drop=True)

# filter out testing part based on our rules
rules= "(sttl <= 61.00 & sinpkt<= 0.00) | (sttl >  61.00 )"

# getting the index of records to keep
ind = X_test.query(rules).index

# filtering test set (both X_test and y_test)
X_test_2 = X_test.loc[ind,:]
y_test_2 = y_test[ind]

print(X_test.shape)
print(X_test_2.shape)
print("filtered data" , (1- np.round(X_test_2.shape[0] / X_test.shape[0],2))*100, "%")
```
:::

::: {#839bcf2a-2970-475b-a186-bbc65e93664c .cell .markdown}
Our simple rule-based system filtered 23% of network traffic for further
analysis, demonstrating its efficacy in detecting non-threatening
network activity. In practice, rule-based systems are more complex and
capable of detecting the vast majority of non-threatening network
traffic.

The next step involves using machine learning to detect cyber attacks by
applying the trained model to the filtered data (`test_2`) from the
previous step. It may be useful to introduce Snort, which is a powerful
open-source detection software that can be utilized for network
security.
:::

::: {#9f873781-78b2-4947-9aba-5719cda71f95 .cell .markdown}
### 5.2. Introducing Snort For Rule-Based System {#52-introducing-snort-for-rule-based-system}

`Snort` is a free and open-source rule-based system for network
intrusion detection and prevention system (NIDPS) developed by Cisco. It
uses rules to analyze network traffic and identify potential security
threats based on specific patterns or behaviors. `Snort` comes with a
set of pre-defined rules that can be used for basic intrusion detection.
These rules are included in the \"rules\" directory in the `Snort`
installation and can be enabled in the configuration file. The default
rules cover a range of attack types, such as buffer overflows, SQL
injection, and network scanning, and can be a good starting point for
building a more customized intrusion detection system. However, it\'s
important to note that the default rules are not comprehensive and may
not provide complete coverage for all possible attack scenarios.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0Q8REN/images/snort.png" width="300" alt="Cyber attack image">`{=html}
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                      *source: <https://www.snort.org/>*

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Covering the implementation of `Snort` will be beyond the scope of this
project; However, here are some general steps to get started with
`Snort`:

> 1.  Install Snort: You can [download Snort from the official
>     website](https://www.snort.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01),
>     and install it on your system. Make sure to follow the
>     instructions carefully.
> 2.  Configure Snort: Once Snort is installed, you will need to
>     configure it. This includes setting up the network interface to
>     monitor, defining the rules that Snort should use to detect
>     suspicious traffic, and specifying the logging and alerting
>     options.
> 3.  Create Rules: Snort uses rules to detect various types of
>     suspicious activity. Rules are typically written in a specific
>     format, and include information such as the type of traffic to
>     monitor, the conditions for detection, and the alerting options.
>     You can write your own custom rules, or use pre-defined rules that
>     come with Snort.
> 4.  Start Snort: Once Snort is installed and configured, you can start
>     it to begin monitoring traffic on the specified network interface.
>     Snort will run in the background, and will generate alerts or log
>     events when suspicious activity is detected.
> 5.  Analyze Alerts: When Snort detects suspicious activity, it will
>     generate alerts or log events, depending on your configuration.
>     You can then use these alerts to investigate and respond to the
>     detected activity.

It is important to note that using Snort effectively requires a solid
understanding of networking, security, and the various threats that can
be detected. It is recommended that you invest time in learning these
topics and building your skills before using Snort in a production
environment.

***However, keep in mind that rule-based models may not be enough to
protect against cyber attacks, especially in cloud services where more
sophisticated strategies are needed. I will elaborate a cloud Security
tool call `Qradar` in part 8.***
:::

::: {#cb51e3d6-5453-4adf-a92b-9dfa2514d366 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#1ca53687-96db-4423-8964-13131e5b16f5 .cell .markdown}
## 6. Machine Learning Model For Cyber Attack Detection {#6-machine-learning-model-for-cyber-attack-detection}

  ---------------------------------------------------------------------------------------------------------------------------------------
   `<img src="https://cdn.pixabay.com/photo/2019/01/22/10/58/pixel-cells-3947912_1280.png" width="300" alt="Cyber attack image">`{=html}
  ---------------------------------------------------------------------------------------------------------------------------------------
                                                  *image credit: <https://pixabay.com/>*

  ---------------------------------------------------------------------------------------------------------------------------------------
:::

::: {#deea384b-e18e-4f50-9e82-113fd17de4d3 .cell .markdown}
The combination of machine learning and rule-based models offers several
advantages in detecting cyber attacks:

1.  Improved accuracy: Machine learning models can identify complex
    patterns and relationships in data, whereas rule-based models are
    limited by the explicit rules defined.
2.  Enhanced interpretability: Rule-based models are easier to
    understand and interpret, making it easier to validate the results
    generated by machine learning models.
3.  Increased speed: Machine learning models can quickly analyze large
    amounts of data, while rule-based models can make decisions faster
    in real-time.
4.  Better scalability: Machine learning models can be easily updated
    and retrained on new data, while rule-based models can be difficult
    to update as the threat landscape changes.
5.  Enriched data utilization: Both methods can complement each other by
    using different data sources and types, leading to a more
    comprehensive analysis.
:::

::: {#81cfb519-5cba-4b66-8571-f8d78f58373a .cell .markdown}
### 6.1. Building a RandomForest Model {#61-building-a-randomforest-model}

Random Forest is a good choice for cyber attack detection due to its
high accuracy in classifying complex data patterns. The ability to
interpret the results of Random Forest models also makes it easier to
validate and understand the decisions it makes, leading to more
effective and efficient cyber security measures.
:::

::: {#a49695fd-ba27-4d63-b8e2-a279725958ec .cell .code}
``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Create a Random Forest model
rf = RandomForestClassifier(random_state=123)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test_2)

# Calculate the mean absolute error of the model
acc = accuracy_score(y_test_2, y_pred)
rec = recall_score(y_test_2, y_pred)
per = precision_score(y_test_2, y_pred)
print("Recall: ", rec)
print("Percision: ", per)
print("Accuracy: ", acc)
```
:::

::: {#c65367f9-1dd9-4b46-8c59-1956f4060b80 .cell .markdown}
As we can see, the random forest algorithm showed strong performance in
cyber attack detection. To gain better insight into the performance of
our prediction model, let\'s plot a confusion matrix. It is important to
note that the majority of our data contains actual attack information,
as we filtered out some portion of non-threatening traffic in the
previous step.
:::

::: {#182d1dda-2c20-44e4-9f81-3ac5426d5903 .cell .code}
``` python
# plot confusion matrix
cross = pd.crosstab(pd.Series(y_test_2, name='Actual'), pd.Series(y_pred, name='Predicted'))
plt.figure(figsize=(5, 5))
sns.heatmap(cross, annot=True,fmt='d', cmap="YlGnBu")
plt.show()
```
:::

::: {#d78666a8-6d19-45c5-a60e-4e2d08d033b1 .cell .markdown}
To understand the functioning of the final tree in the random forest, we
will print the rules present in the 100th tree to a file named
`Tree_output.txt`. You can access to the file by clicking `file browser`
located in the left panel or pressing `ctrl + shift + f` (in Windows)
and `command + shift + f` (in Mac).

This will allow us to have a visual representation of the tree and help
to better understanding of how the model is making decisions to detect
cyber attacks. The rules present in the tree can also be used as a
reference for developing a rule-based system or for fine-tuning the
model for better results. The output will also highlight the most
important factors considered by the model for attack detection, which
can be useful for further analysis and optimization.
:::

::: {#961f4222-ae10-45dd-a916-28a2f54231a2 .cell .code}
``` python
# save the 100th tree sample in random forest in the file 
from sklearn.tree import export_text
feature_names = list(X.columns)

# Create a file and write to it
with open("Tree_output.txt", "w") as file:
    print(export_text(rf.estimators_[99], 
            spacing=3, decimals=2,
            feature_names=feature_names), file=file)
```
:::

::: {#6d8f9a3a-94a0-417a-ab12-e99154626e50 .cell .markdown}
### Exercise: try GBM classifier with grid search on the parameters
:::

::: {#2b57a1ca-7bb9-4fe9-85d4-94f68e209282 .cell .code}
``` python
# try grid search on GBM parameters first
# write your code here
```
:::

::: {#3abfe09e-67fb-4bff-95ce-e9e69393fbff .cell .markdown}
```{=html}
<details>
    <summary>Click here for a Hint</summary>
    
```python

from sklearn.ensemble import GradientBoostingClassifier

# Define the Gradient Boosting Classifier model
gbc = GradientBoostingClassifier()

# Define the hyperparameters for Grid Search
# we purposely reduce the number of possible parameters due to the process time
param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [1, 3, 5]
}

# Perform Grid Search on the GBC model with 5-fold cross-validation
grid_search = GridSearchCV(gbc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

```
</details>
```
:::

::: {#47135550-5ee7-4b9c-9773-ac5ac842c867 .cell .code}
``` python
# apply the parameters to GBM model
# write your code here
```
:::

::: {#17da7ba1-a559-421d-995d-a05c75d373bc .cell .markdown}
```{=html}
<details>
    <summary>Click here for a Hint</summary>
    
```python

# tryin GBM
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print(clf.score(X_test_2, y_test_2))

y_pred = gbc.predict(X_test_2)

# Calculate the mean absolute error of the model
acc = accuracy_score(y_test_2, y_pred)
rec = recall_score(y_test_2, y_pred)
per = precision_score(y_test_2, y_pred)
print("Recall: ", rec)
print("Percision: ", per)
print("Accuracy: ", acc)
```
</details>
```
:::

::: {#0c9e8197-df02-4af8-9322-e37bc675ef26 .cell .markdown}
## 7. Human Analysis {#7-human-analysis}

In addition to these automated methods, human analysis can play a
critical role in identifying cyber attacks. Human analysis is important
in identifying cyber attacks. Analysts use their expertise to interpret
data and understand the context of an attack. Understanding key
variables in network data is crucial for effective human analysis in
detecting cyber attacks.
:::

::: {#4791fcee-54f8-4f26-bf9d-e7dac62baa78 .cell .markdown}
### 7.1. Correlations In The Dataset {#71-correlations-in-the-dataset}

To improve our understanding of the variables involved in cyber attack
detection, we need to analyze the network data. Correlation diagrams can
be helpful in visualizing how different variables are associated with
each other and with cyber attacks. Additionally, random forest models
can help identify the importance of different features in predicting the
target variable (cyber attacks). We can compare the feature rankings
from the random forest with the results of the correlation analysis to
gain a better understanding of the key features to focus on for
effective cyber attack detection.
:::

::: {#5d88d8d0-ad61-492c-95fa-135767d9571e .cell .code}
``` python
# creating the correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
sns.heatmap(df.corr(),vmin=-1, vmax=1,cmap='BrBG', mask=mask)
```
:::

::: {#2f7388ea-b5bc-4060-aa80-31104b37bb0f .cell .markdown}
The heatmap visualizes the correlation between variables in the dataset.
It shows that certain features are highly correlated, such as `tcprtt`
with `ackdat` and `synack`. This is because these variables measure
different aspects of the same TCP connection setup process.
Specifically, `tcprtt` is the round-trip time it takes for the TCP
connection to be established, while `ackdat` measures the time between
the `SYN_ACK` and `ACK` packets, and `synack` measures the time between
the SYN and `SYN_ACK` packets. Since these variables are all related to
the same underlying process of establishing a TCP connection, they are
highly correlated.
:::

::: {#9d4afb48-5743-4eb5-ad63-7154d93cd1da .cell .markdown}
Let\'s have a look at the correlation of variables with the cyber attack
(label column):
:::

::: {#61062fac-9f43-420c-af7c-4f6aef74c8ba .cell .code}
``` python
# modify the headmap plot to show correlation variables to the label 
plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(df.corr()[['label']].sort_values(by='label', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with the Label', fontdict={'fontsize':18}, pad=16);
```
:::

::: {#777442e7-bfb7-4dc6-8daa-12d46c490dc8 .cell .markdown}
The following variables are positively correlated with cyber attacks:

> -   `sttl`: Source to destination time to live value. Attackers may
>     use techniques such as packet fragmentation or tunneling to avoid
>     detection or bypass security measures, which can increase the
>     number of hops or decrease the TTL value. A higher value for sttl
>     may be indicative of such techniques.\
> -   `ct_state_ttl` and `state`: These features reflect various stages
>     of TCP connections and may be related to port scanning, SYN flood,
>     or DDoS attacks. Attackers may exploit the state of TCP
>     connections using different techniques, which may be reflected in
>     the values of ct_state_ttl and state.
> -   `ct_dst_sport_ltm`: This feature measures the number of
>     connections from the same source IP to the same destination port
>     in a short time period. Attackers may initiate multiple
>     connections to the same port in a short time period to exploit
>     vulnerabilities or launch attacks against a particular service or
>     application, which may be reflected in a higher value for
>     ct_dst_sport_ltm.
> -   `rate`: This feature may represent various types of traffic rates
>     or frequencies. Attackers may generate high traffic rates or
>     bursts of traffic to overwhelm or bypass security measures, which
>     may be reflected in a higher value for rate.

In contrast, the following variables are negatively correlated with
cyber attacks:

> -   `swin`: The size of the TCP window may decrease during an attack
>     when attackers try to flood the network with traffic. A lower
>     value for swin may be indicative of such attacks.
> -   `dload`: A decrease in the download speed may be indicative of an
>     attack that consumes network bandwidth, such as DDoS attacks or
>     worm propagation. A lower value for dload may be reflective of
>     such attacks.
:::

::: {#bb5afa0f-df33-41e7-8e4b-1cb786d4641b .cell .markdown}
### 7.2. Feature Ranking From Random Forest {#72-feature-ranking-from-random-forest}

The random forest provides a list of features based on their
contributions to the prediction model. The feature ranking can be
accessed through RandomForest object (in our example `rf`) using
`feature_importances_` attribute.
:::

::: {#9e4bd85d-264b-4964-a087-3606dc398548 .cell .code}
``` python
# creating of ranking data frame
feature_imp = pd.DataFrame({'Name':X.columns, 'Importance':rf.feature_importances_})

# sorting the features based on their importance value
feature_imp = feature_imp.sort_values('Importance',ascending=False).reset_index(drop=True)

# show only 10 most important feature in style of gradien of colores
feature_imp[:10].style.background_gradient()
```
:::

::: {#c0090cbd-bf09-4611-a9ca-6ee7f0cc208b .cell .code}
``` python
# plot the important features
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh',color=['g','b']*5)
```
:::

::: {#5ce0a8f0-2977-40e8-86eb-2c3c34106905 .cell .markdown}
As we can see, the feature importance ranking is aligned with
correlation result. This highlights the importance of top features such
`sttl`, `ct_stat_ttl`, `rate`, and `dload`.

Following is a brief description of some of these important features (a
full list of features is available at the end of this notebook).

  -----------------------------------------------------------------------------------------------
  No.   Name               Type      Description
  ----- ------------------ --------- ------------------------------------------------------------
  10    sttl               Integer   Source to destination time to live value

  37    ct_state_ttl       Integer   No. for each state (6) according to specific range of values
                                     for source/destination time to live (10) (11)(see the full
                                     list at the end of this project to find no 6,10,11).

  9     rate               Integer   Destination to source transaction bytes

  16    Dload              Float     Destination bits per second

  15    Sload              Float     Source bits per second

  47    ct_dst_src_ltm     integer   No of connections of the same source (1) and the destination
                                     (3) address in in 100 connections according to the last time
                                     (26).

  23    smeansz            integer   Mean of the ?ow packet size transmitted by the src

  8     sbytes             Integer   Source to destination transaction bytes

  22    dtcpb              integer   Destination TCP base sequence number

  42    ct_srv_dst         integer   No. of connections that contain the same service (14) and
                                     destination address (3) in 100 connections according to the
                                     last time (26).

  6     state              nominal   Indicates to the state and its dependent protocol, e.g. ACC,
                                     CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD,
                                     URH, URN, and (-) (if not used state)

  46    ct_dst_sport_ltm   integer   No of connections of the same destination address (3) and
                                     the source port (2) in 100 connections according to the last
                                     time (26).

  7     dur                Float     Record total duration
  -----------------------------------------------------------------------------------------------
:::

::: {#ddbcc589-a888-4900-8e13-bb7aeadfdee9 .cell .markdown}
### Exercise: Select These Top 10 Features And Compare the Performance
:::

::: {#a4e1aa66-80d4-45b8-baac-d2b4eeb9d0bc .cell .code}
``` python
# use Random Forest to run train and test on data of only top 10 features
```
:::

::: {#0e9876e1-541d-4fcc-98c4-ef0d78a2da9a .cell .markdown}
```{=html}
<details>
```
    <summary>Click here for Solution</summary>

``` python

top10= feature_imp.Name[:10].tolist()

X = df.loc[:, df.columns.isin(top10)]
y = df['label'].values

rf_top10 = RandomForestClassifier(random_state=11)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# Train the model on the training data
rf_top10.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_top10.predict(X_test_2)

# Calculate the mean absolute error of the model
acc = accuracy_score(y_test_2, y_pred)
print("Accuracy: ", acc)

#  Accuracy:  0.91  
```
:::

::: {#d7f944c2-685d-412e-9f23-66836391da75 .cell .markdown}
Let\'s select only the top 10 features and find their associations with
the type of cyber attack.
:::

::: {#c009e42a-ae62-47ab-a2a4-849b0a8f801d .cell .code}
``` python
# get the names of top 10 features
top10= feature_imp.Name[:10].tolist()

# get the attack names
attack_names = np.array(df['attack_cat'].unique())

# selecting only top 10 features
X_top = df.loc[:, df.columns.isin(top10)]
# need to convert the catagorical data into numbers (e.g. normal ->0, Blackdoor ->2)
y_top = pd.factorize(df['attack_cat'])[0]


# for the purpose of visualization we set max_depth to 6 in order to be shown in the notebook
clf_top10 = DecisionTreeClassifier(max_depth=6)

# Split the data into train and test sets
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y_top, test_size=0.3, random_state=11)

# Train the model on the training data
clf_top10.fit(X_train_top, y_train_top)

# visualizing the tree
viz_model = dtreeviz.model(clf_top10,
                           X_train=X_train_top, y_train=y_train_top, 
                           class_names=attack_names,
                           feature_names=top10)

v = viz_model.view(fancy=False,scale=1) # render as SVG into internal object 
v
#v.save("The_100th_tree.svg") # if you willing to save the 
```
:::

::: {#ed954db8-1ac0-46a3-b3d1-a1a8216420a5 .cell .markdown}
For a better understanding, we can randomly select a point and visualize
the path for prediction.
:::

::: {#dd90f067-96de-4ae0-9471-669af430bf19 .cell .code}
``` python
# get a random point
rand = np.random.randint(0, len(X))
sample_point = X.iloc[rand,:].values

# visualizing the path for the point
v = viz_model.view(fancy=True,scale=1.5,x=sample_point,show_just_path=True)
v
```
:::

::: {#c295d4be-3455-4bbf-8b55-9e5d6936a55f .cell .markdown}
please keep in mind that we utilize a simple decision tree for
visualization (above cells), and random forest can outperform decision
tree in predicting the type of attack.
:::

::: {#d0790839-a96c-4a4c-b10f-939e919cef94 .cell .markdown}
### Exercise: Run Random Forest With Attack Category As the Prediction Labels And Plot The Confusion Matrix
:::

::: {#7a3a8696-0479-40eb-a5c6-147247e417bb .cell .code}
``` python
# write your code here
```
:::

::: {#0e5c8687-720d-48d5-9ff8-23182fd6b6bc .cell .markdown}
```{=html}
<details>
    <summary>Click here for solution</summary>
    
```python 

top10= feature_imp.Name[:10].tolist()

X = df.loc[:, df.columns.isin(top10)]

# X = df.loc[:, ~df.columns.isin(['attack_cat', 'label'])] # if you like to use all features use this line
y = df['attack_cat'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# Create a Random Forest model
rf = RandomForestClassifier(random_state=123,min_samples_leaf= 1, min_samples_split= 5, n_estimators= 100)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Calculate the mean absolute error of the model
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

cross = pd.crosstab(y_test,  y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cross, annot=True,fmt='d', cmap="YlGnBu")

```
    
</details>
```
:::

::: {#70d9a618-432d-469b-b446-c6d3a069e444 .cell .markdown}
### 7.3 Discussing The Network Variables and Their Role in Detecting Different Types of Cyber Attacks. {#73-discussing-the-network-variables-and-their-role-in-detecting-different-types-of-cyber-attacks}
:::

::: {#6fb57c23-7e8f-437c-8cb0-7323f9b8c59d .cell .markdown}
Let\'s discuss about some of the important features for detecting the
type of cyber attack.

`sttl`: Source to destination time to live value can be used to detect
attacks such as packet fragmentation or tunneling that can increase the
number of hops or decrease the TTL value. These techniques are often
used by attackers to avoid detection or bypass security measures. A
higher value for sttl may indicate the presence of such techniques.

`ct_state_ttl` and `state`: These features reflect the various stages of
TCP connections and can be related to port scanning, SYN flood, or DDoS
attacks. Attackers can exploit the state of TCP connections using
different techniques, which may be reflected in the values of
ct_state_ttl and state.

`rate`: This feature can represent various types of traffic rates or
frequencies. Attackers may generate high traffic rates or bursts of
traffic to overwhelm or bypass security measures, which may be reflected
in a higher value for rate.

`dload`: A decrease in the download speed may indicate an attack that
consumes network bandwidth, such as DDoS attacks or worm propagation. A
lower value for dload may be reflective of such attacks.

The different types of attacks can have different characteristics that
can be detected using network variables. For example, DoS attacks aim to
disrupt the normal functioning of a system, so an increase in the rate
of traffic or a decrease in the download speed may indicate the presence
of such an attack. Port scanning, SYN flood, and DDoS attacks can be
reflected in the values of ct_state_ttl and state. Fuzzers and analysis
attacks may involve generating large amounts of traffic, which can be
reflected in the value of rate. Reconnaissance attacks involve gathering
information about a target system, which can potentially be detected by
analyzing network traffic. Finally, shellcode and worm attacks can be
detected by analyzing the content of network packets.
:::

::: {#d3da833a-a685-495b-aded-be13eb789f75 .cell .markdown}
### Bonus Exercise - Run Deep Neural Network On The Dataset
:::

::: {#67a74182-6375-4bae-9421-0917fa6c75e9 .cell .code}
``` python
 # write your code here
```
:::

::: {#396f97b9-cbf8-4f8c-a3e6-1706d34cbcda .cell .markdown}
```{=html}
<details>
    <summary>Click here for Solution</summary>

```python
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error , auc
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a DNN model

model = Sequential()
# The Input Layer
model.add(Dense(512, activation='relu', input_dim = X_train.shape[1]))

# The Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# The Output Layer

model.add(Dense(2, activation='softmax'))

# Compile the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=5)

```

</details>
```
:::

::: {#1b0b5f1d-7f84-4032-84ee-8c4c9acc8f5c .cell .markdown}
## 8. Cyber Security for Cloud Services {#8-cyber-security-for-cloud-services}

We may scratch the surface, but as you start implementing your system,
you will inevitably encounter complex issues. However, there are
powerful cybersecurity tools available that you should consider.

The complexities of cybersecurity in cloud services include shared
responsibility, data privacy, complex architecture, multi-tenancy,
regulatory compliance, and vulnerability to attacks. To mitigate these
risks, effective cybersecurity strategies must be in place.

Implementing cybersecurity measures for cloud computing can be
particularly challenging due to several reasons, such as:

> 1.  Shared responsibility: In cloud computing, the responsibility for
>     security is shared between the cloud provider and the customer,
>     which can lead to confusion and a lack of clear ownership over
>     security issues.
> 2.  Complex architecture: Cloud environments typically have a complex
>     and dynamic architecture, making it difficult to implement and
>     manage effective security controls.
> 3.  Multi-tenancy: Cloud providers often use multi-tenant
>     infrastructure, where multiple customers share the same physical
>     and virtual resources. This can lead to security risks, such as
>     the accidental or intentional exposure of one customer\'s data to
>     another.
> 4.  Regulatory compliance: Organizations must comply with regulations
>     such as the General Data Protection Regulation (GDPR) or the
>     Health Insurance Portability and Accountability Act (HIPAA), which
>     can be difficult to achieve in a cloud environment.
> 5.  Vulnerability to attacks: Cloud environments are vulnerable to
>     attacks such as distributed denial of service (DDoS) attacks,
>     malware, and unauthorized access, making it critical to implement
>     appropriate measures to mitigate the risks.

Therefore, implementing effective cybersecurity measures in cloud
computing requires a comprehensive and multi-layered approach to address
these challenges and secure sensitive data and systems.

### 8.1 IBM QRadar {#81-ibm-qradar}

[IBM Security® QRadar® Security Information and Event Management
(SIEM)](https://www.ibm.com/products/qradar-siem?utm_source=Exinfluencer&utm_content=000026UJ&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01&utm_medium=Exinfluencer&utm_term=10006555)
helps security teams detect, prioritize and respond to threats across
the enterprise. As an integral part of your XDR and zero trust
strategies, it automatically aggregates and analyzes log and flow data
from thousands of devices, endpoints and apps across your network,
providing single, prioritized alerts to speed incident analysis and
remediation. QRadar SIEM is available for on-premises and cloud
environments.

`<a href="https://www.ibm.com/products/qradar-siem?utm_source=Exinfluencer&utm_content=000026UJ&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01&utm_medium=Exinfluencer&utm_term=10006555">`{=html}
![image.png](https://1.cms.s81c.com/sites/default/files/2022-11/pulse-summary-view-dashboard.png)`</a>`{=html}
:::

::: {#f02081c1-fc90-4acb-80b7-578742a5cbf6 .cell .markdown}
## 9. List of all features with description {#9-list-of-all-features-with-description}
:::

::: {#6a29a11a-9732-4359-b98e-6c7fe592fb88 .cell .markdown}
The data was processed using Argus and Bro-ID tools, resulting in 49
features with class labels.

  -------------------------------------------------------------------------------------------------
  No.   Name               Type        Description
  ----- ------------------ ----------- ------------------------------------------------------------
  1     srcip              nominal     Source IP address

  2     sport              integer     Source port number

  3     dstip              nominal     Destination IP address

  4     dsport             integer     Destination port number

  5     proto              nominal     Transaction protocol

  6     state              nominal     Indicates to the state and its dependent protocol, e.g. ACC,
                                       CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD,
                                       URH, URN, and (-) (if not used state)

  7     dur                Float       Record total duration

  8     sbytes             Integer     Source to destination transaction bytes

  9     dbytes             Integer     Destination to source transaction bytes

  10    sttl               Integer     Source to destination time to live value

  11    dttl               Integer     Destination to source time to live value

  12    sloss              Integer     Source packets retransmitted or dropped

  13    dloss              Integer     Destination packets retransmitted or dropped

  14    service            nominal     http, ftp, smtp, ssh, dns, ftp-data ,irc and (-) if not much
                                       used service

  15    Sload              Float       Source bits per second

  16    Dload              Float       Destination bits per second

  17    Spkts              integer     Source to destination packet count

  18    Dpkts              integer     Destination to source packet count

  19    swin               integer     Source TCP window advertisement value

  20    dwin               integer     Destination TCP window advertisement value

  21    stcpb              integer     Source TCP base sequence number

  22    dtcpb              integer     Destination TCP base sequence number

  23    smeansz            integer     Mean of the ?ow packet size transmitted by the src

  24    dmeansz            integer     Mean of the ?ow packet size transmitted by the dst

  25    trans_depth        integer     Represents the pipelined depth into the connection of http
                                       request/response transaction

  26    res_bdy_len        integer     Actual uncompressed content size of the data transferred
                                       from the serverís http service.

  27    Sjit               Float       Source jitter (mSec)

  28    Djit               Float       Destination jitter (mSec)

  29    Stime              Timestamp   record start time

  30    Ltime              Timestamp   record last time

  31    Sintpkt            Float       Source interpacket arrival time (mSec)

  32    Dintpkt            Float       Destination interpacket arrival time (mSec)

  33    tcprtt             Float       TCP connection setup round-trip time, the sum of ísynackí
                                       and íackdatí.

  34    synack             Float       TCP connection setup time, the time between the SYN and the
                                       SYN_ACK packets.

  35    ackdat             Float       TCP connection setup time, the time between the SYN_ACK and
                                       the ACK packets.

  36    is_sm_ips_ports    Binary      If source (1) and destination (3)IP addresses equal and port
                                       numbers (2)(4) equal then, this variable takes value 1 else
                                       0

  37    ct_state_ttl       Integer     No. for each state (6) according to specific range of values
                                       for source/destination time to live (10) (11).

  38    ct_flw_http_mthd   Integer     No. of flows that has methods such as Get and Post in http
                                       service.

  39    is_ftp_login       Binary      If the ftp session is accessed by user and password then 1
                                       else 0.

  40    ct_ftp_cmd         integer     No of flows that has a command in ftp session.

  41    ct_srv_src         integer     No. of connections that contain the same service (14) and
                                       source address (1) in 100 connections according to the last
                                       time (26).

  42    ct_srv_dst         integer     No. of connections that contain the same service (14) and
                                       destination address (3) in 100 connections according to the
                                       last time (26).

  43    ct_dst_ltm         integer     No. of connections of the same destination address (3) in
                                       100 connections according to the last time (26).

  44    ct_src\_ ltm       integer     No. of connections of the same source address (1) in 100
                                       connections according to the last time (26).

  45    ct_src_dport_ltm   integer     No of connections of the same source address (1) and the
                                       destination port (4) in 100 connections according to the
                                       last time (26).

  46    ct_dst_sport_ltm   integer     No of connections of the same destination address (3) and
                                       the source port (2) in 100 connections according to the last
                                       time (26).

  47    ct_dst_src_ltm     integer     No of connections of the same source (1) and the destination
                                       (3) address in in 100 connections according to the last time
                                       (26).

  48    attack_cat         nominal     The name of each attack category. In this data set , nine
                                       categories e.g. Fuzzers, Analysis, Backdoors, DoS Exploits,
                                       Generic, Reconnaissance, Shellcode and Worms

  49    Label              binary      0 for normal and 1 for attack records
  -------------------------------------------------------------------------------------------------
:::

::: {#d5d7f103-8305-4d1f-bce3-30da7f3bf213 .cell .markdown}
## Authors
:::

::: {#a8497672-718d-42b3-b790-5cfe3591b6e5 .cell .markdown}
[Sina Nazeri (Linkedin
profile)](https://www.linkedin.com/in/sina-nazeri?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMSkillsNetworkGPXX0Q8REN2117-2023-01-01)

> `<i>`{=html} As a data scientist in IBM, I have always been passionate
> about sharing my knowledge and helping others learn about the field. I
> believe that everyone should have the opportunity to learn about data
> science, regardless of their background or experience level. This
> belief has inspired me to become a learning content provider, creating
> and sharing educational materials that are accessible and engaging for
> everyone.
:::

::: {#5bd40682-9e68-410c-95b6-60fbb57df593 .cell .markdown}
Joseph Santarcangelo
:::

::: {#d03e58d6-f925-4cde-a7bd-1281c2ec9cee .cell .markdown}
### Other Contributors
:::

::: {#63b21348-182d-4983-a35a-1c8368c02169 .cell .markdown}
Sheng-Kai Chen

J.C.(Junxing) Chen

Artem Arutyunov

Roxanne Li
:::

::: {#b60305ea-b17f-46b7-a258-95f4db0dd27d .cell .markdown}
## Change Log
:::

::: {#49611d3e-22eb-4191-a5ee-ac92ccf64b0f .cell .markdown}
  Date (YYYY-MM-DD)   Version   Changed By    Change Description
  ------------------- --------- ------------- ---------------------
  2023-03-03          0.1       Sina Nazeri   Create Lab Template
:::

::: {#534358f2-e703-41f2-8915-e1759f74d88d .cell .markdown}
Copyright © 2022 IBM Corporation. All rights reserved.
:::
