import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
@purpose: Plotting different graphs of the data for analyzing features and helping with feature selection
"""
print("Reading dataset...")
heartDisease_df = pd.read_csv("heart_cleveland_upload.csv") # there are 297 rows and 14 columns in the dataset


# Count plot for occurrence counts of sex affected vs condition to find any correlations - sex
print("Plotting Counts vs Sex Affected with Heart Disease")
fig1 = plt.figure(figsize=(8, 6))
ax1 = sns.countplot(x="sex", data=heartDisease_df, hue="condition")

ax1.set_xticklabels(["Female", "Male"])
ax1.set_title("Counts of Sex Affected vs the Development of Heart Disease")

ax1.set_xlabel("Sex")
ax1.set_ylabel("Count")

ax1.legend(["Not Developed", "Developed"])
plt.show()

"""
The plot seems to show that there are more cases of heart disease in males than females, but there is also
more data on males than on females. Thus, this feature may not be a huge decision making factor. 
"""


# Count plot for counts of chest pain vs condition to find any correlations - cp
print("Plotting Counts of Chest Pain vs Condition")
fig2 = plt.figure(figsize=(8, 6))
ax2 = sns.countplot(x="cp", data=heartDisease_df, hue="condition")

ax2.set_xticklabels(["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
ax2.set_title("Counts of Chest Pain Types vs the Development of Heart Disease")

ax2.set_xlabel("Chest Pain Type")
ax2.set_ylabel("Counts")

ax2.legend(["Not Developed", "Developed"])
plt.show()

"""
There are significantly more occurrences in the dataset for those who have typical angina, compared to the other pain types.
There is slightly more non-anginal pain than the 2 anginal pains, but that is insignificant compared to those that
have typical angina. There are not that many cases of heart disease from those who are asymptomatic.
"""


# Count plot for counts of those with fasting blood sugar above or below 120 mg/dl vs condition - fbs
print("Plotting Counts of Those with Fasting Blood Sugar Above or Below 120 mg/dl vs Condition")
fig3 = plt.figure(figsize=(8, 6))
ax3 = sns.countplot(x="fbs", data=heartDisease_df, hue="condition")

ax3.set_title("Counts of Fasting Blood Sugar Values vs the Development of Heart Disease")
ax3.set_xticklabels(["Value < 120 mg/dl", "Value > 120 mg/dl"])

ax3.set_xlabel("Fasting Blood Sugar Value")
ax3.set_ylabel("Counts")

ax3.legend(["Not Developed", "Developed"])
plt.show()

"""
There is a significant increase in the amount of data for those with a blood sugar value < 120 mg/dl compared to its
counterpart. There is slightly more people who don't have heart disease when looking at the blood sugar value < 120 mg/dl.
The counts for the presence of heart disease in the 2 categories are relatively close. 
"""


# Count plot for resting electrocardiographic results vs condition - restecg
print("Plotting Resting Electrocardiographic Results vs Condition")
fig4 = plt.figure(figsize=(9, 7))
ax4 = sns.countplot(x="restecg", data=heartDisease_df, hue="condition")

ax4.set_title("Counts of Different Resting Electrocardiographic Results vs Development of Heart Disease")
ax4.set_xticklabels(["Normal", "Presence of ST-T Wave \nAbnormality", "Showing Possible/Definite Left \nVentricular Hypertrophy"])

ax4.set_xlabel("Resting ECG Results", labelpad=10)
ax4.set_ylabel("Counts", labelpad=10)

ax4.legend(["Not Developed", "Developed"])
plt.show()

"""
There are significantly few cases of ST-T Wave Abnormality in the dataset. Comparing the cases with
left ventricular hypertrophy results, there are more cases of heart disease than cases with no heart disease, but the 
difference isn't huge (~10 more cases).
Comparing the cases with normal results, there are more cases of absent heart disease than cases with heart disease.
The difference here is bigger (~45 more cases).
"""


# Count plot for exercise induced angina vs condition - exang
print("Plotting Exercise Induced Angina vs Condition")
fig5 = plt.figure(figsize=(8, 6))
ax5 = sns.countplot(x="exang", data=heartDisease_df, hue="condition")

ax5.set_title("Presence of Exercise Induced Angina vs Development of Heart Disease")
ax5.set_xticklabels(["Absent", "Present"])

ax5.set_xlabel("Presence of Exercise Induced Angina", labelpad=10)
ax5.set_ylabel("Counts", labelpad=10)

ax5.legend(["Not Developed", "Developed"])
plt.show()

"""
There is a significant difference when comparing the cases of absent angina cases; there are more cases with no
heart disease than cases with heart disease (~75 more cases). When comparing present angina cases, there are more cases
with heart disease than no heart disease (~50 more cases). 
"""


# Count plot for peak exercise ST segment vs condition - slope
print("Plotting Peak Exercise ST Segment vs Condition")
fig6 = plt.figure(figsize=(8, 6))
ax6 = sns.countplot(x="slope", data=heartDisease_df, hue="condition")

ax6.set_title("Peak Exercise ST Segment Slope vs Development of Heart Disease")
ax6.set_xticklabels(["Upsloping", "Flat", "Downsloping"])

ax6.set_xlabel("Type of ST Segment Slope", labelpad=10)
ax6.set_ylabel("Counts", labelpad=10)

ax6.legend(["Not Developed", "Developed"])
plt.show()

"""
There are not that many cases of downsloping ST segment. When comparing cases with upsloping ST segment, there
are more cases with no heart disease than with heart disease (~70 more cases). When comparing cases with a flat 
ST segment, there are more cases with heart disease than without heart disease (~40 more cases).
"""


# Count plot for the number of major blood vessels with abnormalities vs condition - ca
print("Plotting the Number of Abnormal Blood Vessels vs Condition")
fig7 = plt.figure(figsize=(8, 6))
ax7 = sns.countplot(x="ca", data=heartDisease_df, hue="condition")

ax7.set_title("Number of Structurally Abnormal Blood Vessels vs the Development of Heart Disease")
ax7.set_xticklabels(["0", "1", "2", "3"])

ax7.set_xlabel("Number of Structurally Abnormal Blood Vessels")
ax7.set_ylabel("Counts")

ax7.legend(["Not Developed", "Developed"])
plt.show()

"""
There is relatively lots of data for those who have 0 structurally abnormal blood vessels;
there are much more cases of no heart disease than cases with heart disease (~85 more cases). 
As the number of structurally abnormal blood vessels increases, it seems like there is a fewer number of cases 
without heart disease.
There are also slightly fewer cases with heart disease, but there is still a difference between cases with heart
disease and cases without heart disease, with case counts with heart disease being higher (~20 more cases). 
"""


# Count plot for Ability to Absorb Thallium vs Condition - thal
print("Plotting Ability to Absorb Thallium vs Condition")
fig8 = plt.figure(figsize=(8, 6))
ax8 = sns.countplot(x="thal", data=heartDisease_df, hue="condition")

ax8.set_title("Ability to Absorb Thallium vs Development of Heart Disease")
ax8.set_xticklabels(["Normal", "Fixed Defect", "Reversible Defect"])

ax8.set_xlabel("Ability to Absorb Thallium")
ax8.set_ylabel("Counts")

ax8.legend(["Not Developed", "Developed"])
plt.show()

"""
There are not many cases of a fixed defect. With a normal ability, there are much more cases of no heart disease
than cases with heart disease (~90 more cases). With a reversible defect, there are more cases with heart disease
than cases without heart disease (~60 more cases). 
"""


# Histogram of counts of age - age
fig9 = plt.figure(figsize=(12, 8))
axes9 = fig9.subplots(nrows=1, ncols=2)
ax9 = sns.histplot(x="age", data=heartDisease_df[heartDisease_df["condition"] == 0], ax=axes9[0])

ax9.set_title("Ages of People Without Heart Disease")
ax9.set_xlabel("Age")

ax9.set_ylabel("Counts")
ax9.xaxis.set_ticks(np.arange(25, max(heartDisease_df["age"]), 5))

ax10 = sns.histplot(x="age", data=heartDisease_df[heartDisease_df["condition"] == 1], ax=axes9[1])
ax10.set_title("Ages of People With Heart Disease")

ax10.set_xlabel("Age")
ax10.set_ylabel("Counts")

ax10.set_xticks(np.arange(25, max(heartDisease_df["age"]), 5))
plt.show()

"""
The histogram showed that the highest counts of those with heart diseases are around 60 years old. As a range,
55-65 years old has a relatively high number of counts. The youngest case with heart disease is about 35 years old, 
while the oldest case is about 75 years old. 
The other histogram showed that the highest counts of those without heart disease are around 50-55 years old. The
range for the highest counts of cases without heart disease would be about 40-60 years old. 
"""

# Histogram of resting blood pressure vs condition - trestbps
fig11 = plt.figure(figsize=(12, 8))
axes11 = fig11.subplots(nrows=1, ncols=2)
ax11 = sns.histplot(x="trestbps", data=heartDisease_df[heartDisease_df["condition"] == 0], ax=axes11[0])

ax11.set_title("Resting Blood Pressure for Cases Without Heart Disease")
ax11.set_xticks(np.arange(min(heartDisease_df["trestbps"]), max(heartDisease_df["trestbps"]), 10))

ax11.set_xlabel("Resting Blood Pressure (mmHg)")
ax11.set_ylabel("Counts")

ax12 = sns.histplot(x="trestbps", data=heartDisease_df[heartDisease_df["condition"] == 1], ax=axes11[1])
ax12.set_title("Resting Blood Pressure for Cases With Heart Disease")

ax12.set_xticks(np.arange(min(heartDisease_df["trestbps"]), max(heartDisease_df["trestbps"]), 10))
ax12.set_xlabel("Resting Blood Pressure (mmHg)")

ax12.set_ylabel("Counts")
plt.show()

"""
For the cases with heart disease, the higher counts come from the range of 118-145 mmHg (~80 cases). For the
cases without heart disease, the higher counts come from the ranges of 115-122 mmHg and 130-145 mmHg (~90 cases). 
These ranges are similar for patients with or without heart disease. As for the other values, there are not 
that many datapoints. 
Based on these limited datapoints, there are slightly more cases with heart disease 
and a resting blood pressure greater than 154 mmHg (~5 more cases). Comparing the cases with a resting blood pressure
less than 115 mmHg, there are slightly more cases without heart disease (~10 more cases). 
The general trend seems to be that cases with heart disease may have a higher resting blood pressure range, 
but more datapoints would be needed to confirm this.
"""


# Histogram of serum cholesterol amount vs condition - chol
print("Plotting Serum Cholesterol Amount vs Condition")
fig13 = plt.figure(figsize=(12, 8))
axes13 = fig13.subplots(nrows=1, ncols=2)

ax13 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 0], x="chol", ax=axes13[0])
ax13.set_title("Amount of Serum Cholesterol for Those \nWithout Heart Disease")

ax13.set_xlabel("Amount of Serum Cholesterol (mg/dl)")
ax13.set_ylabel("Counts")

ax13.set_xticks(np.arange(min(heartDisease_df["chol"]), max(heartDisease_df["chol"]), 20))
ax13.set_xticklabels(ax13.get_xticks(), rotation=45)

ax14 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 1], x="chol", ax=axes13[1])
ax14.set_title("Amount of Serum Cholesterol for Those \nWith Heart Disease")

ax14.set_xlabel("Amount of Serum Cholesterol (mg/dl)")
ax14.set_ylabel("Counts")

ax14.set_xticks(np.arange(min(heartDisease_df["chol"]), max(heartDisease_df[heartDisease_df["condition"] == 1]["chol"]), 20))
ax14.set_xticklabels(ax14.get_xticks(), rotation=45)
plt.show()

"""
Most of the occurring values range from about 186-286 mg/dl for cases without heart disease. There are also 3 cases
that have a serum cholesterol value greater than 400; one is at ~546 mg/dl.
For cases with heart disease, most of the occurring values range from about 186-306 mg/dl. There is just 1 case 
that has a relatively high serum cholesterol value of ~386, compared to the rest of the points. 
Without considering these higher serum cholesterol values, it seems that the range of the values that can be found
for both kinds of cases is from about 126-366. 
It seems that for cases with heart disease, patients may have slightly higher serum cholesterol values, 
but more data is needed to confirm this. 
"""


# Histogram of maximum heart rate vs condition - thalach
print("Plotting Maximum Heart Rate vs Condition")
fig15 = plt.figure(figsize=(12, 8))
axes15 = fig15.subplots(nrows=1, ncols=2)

ax15 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 0], x="thalach", ax=axes15[0])
ax15.set_title("Maximum Heart Rate for Those Without Heart Disease")

ax15.set_xlabel("Value for Maximum Heart Rate (beats/min)")
ax15.set_ylabel("Counts")

ax15.set_xticks(np.arange(min(heartDisease_df["thalach"]), max(heartDisease_df["thalach"]), 10))
ax15.set_xticklabels(ax15.get_xticks(), rotation=45)

ax16 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 1], x="thalach", ax=axes15[1])
ax16.set_title("Maximum Heart Rate for Those With Heart Disease")

ax16.set_xlabel("Value for Maximum Heart Rate (beats/min)")
ax16.set_ylabel("Counts")

ax16.set_xticks(np.arange(min(heartDisease_df["thalach"]), max(heartDisease_df["thalach"]), 10))
ax16.set_xticklabels(ax16.get_xticks(), rotation=45)
plt.show()

"""
For cases without heart disease, the maximum heart rates range from about 145-185 beats/min. For cases with
heart disease, the maximum heart rates range from about 111-171 beats/min. 
The highest heart rate was about 201 beats/min for 2 cases without heart disease, and the highest heart rate 
for cases with heart disease was about 191 beats/min. 
It seems that cases with heart disease would generally have a lower maximum heart rate range, but more data
is needed to confirm that. 
"""


# Histogram of ST depression values induced by exercise relative to rest vs condition - oldpeak
print("ST Depression Values Induced by Exercise Relative to Rest vs Condition")
fig17 = plt.figure(figsize=(12, 8))
axes17 = fig17.subplots(nrows=1, ncols=2)

ax17 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 0], x="oldpeak", ax=axes17[0])
ax17.set_title("ST Depression Values Induced by Exercise Relative \nto Rest for Cases Without Heart Disease")

ax17.set_xlabel("ST Depression Values (mm)")
ax17.set_ylabel("Counts")
ax17.set_xticks(np.arange(0, max(heartDisease_df["oldpeak"]), 0.5))

ax18 = sns.histplot(data=heartDisease_df[heartDisease_df["condition"] == 1], x="oldpeak", ax=axes17[1])
ax18.set_title("ST Depression Values Induced by Exercise Relative \nto Rest for Cases With Heart Disease")

ax18.set_xlabel("ST Depression Values (mm)")
ax18.set_ylabel("Counts")

ax18.set_xticks(np.arange(0, max(heartDisease_df["oldpeak"]), 0.5))
plt.show()

"""
For cases without heart disease, the ST depression values range from about 0-1.5 mm. For cases with heart disease,
the ST depression values range from about 0-3.5mm. 
There are many cases with about 0-0.5 mm ST depression values in the cases without heart disease, compared to the 
cases with heart disease (~45 more cases). 
It also seems like that cases with heart disease can have a higher ST depression value, as shown by the increase in 
value range. 
The highest ST depression value for a case without heart disease is about 4 mm, whereas the highest ST depression 
value for a case with heart disease is about 6 mm. There are also at least 10 cases with a ST depression value of 
about 4 mm. 
"""


# Heatmap of the features to find any linear relationships between the categorical features
# (eg. features with binary numbers)

fig19 = plt.figure(figsize=(10, 8), tight_layout=True)
ax19 = sns.heatmap(data=heartDisease_df.corr(), annot=True, vmin=0, vmax=1, cmap="coolwarm", linewidth=0.5, fmt=".2f")

ax19.set_title("Heatmap of the Correlation Between Different Features", pad=15)
ax19.set_xlabel("Dataset Feature Labels", labelpad=20)

ax19.set_ylabel("Dataset Feature Labels", labelpad=20)
ax19.set_xticklabels(ax19.get_xticklabels(), rotation=45)
plt.show()

"""
The highest correlated feature against condition is the thal feature, which is for the heart's ability
to absorb thallium. The next highest correlated feature against condition is the ca feature, which is for the number
of abnormally structured blood vessels. 
After, the next relatively high correlated features include exang, oldpeak, cp, and slope. However, oldpeak is based on
measurements, not on categorical values, so it's excluded from this heatmap analysis. The exang, cp, and slope, are for 
exercise induced angina, chest pain, and type of peak exercise ST segment slope, respectively.  
"""


"""
Based on the count plots, the histograms, and the heatmap, the following features won't be used as they do not 
seem helpful: fbs and restecg (fasting blood sugar and resting ECG results). 

Features that are being considered to remove include age, sex, and trestbps (resting blood pressure)
"""
