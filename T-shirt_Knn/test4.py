import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

def div_conversion(df, column, div):
    return df[column].apply(lambda x: x/div)

male_df = pd.read_csv('male.csv')
male_shirt = male_df[['chestcircumference', 'waistcircumference']]
male_shirt.to_csv('T-Shirt_Male.csv', index=False)
male_shirt_with_size = pd.read_csv("T-Shirt_Male.csv")
male_shirt_with_size.insert(2, column = "size", value="-")
male_shirt_with_size.tail()
#male_df = pd.read_csv("male.csv")
#male_pants = male_df[["crotchheight", "hipbreadth"]]
#male_pants.to_csv('T-Shirt_Male.csv', index=False)
#male_pants_with_size = pd.read_csv("male_pants.csv")

female_df = pd.read_csv('female.csv')
female_shirt = female_df[['chestcircumference', 'waistcircumference']]
female_shirt.to_csv('T-Shirt_Female.csv', index=False)
female_shirt_with_size = pd.read_csv("T-Shirt_Female.csv")
female_shirt_with_size.insert(2, column = "size", value="-")
female_shirt_with_size.tail()
#female_df = pd.read_csv("female.csv")
#female_pants = female_df[["crotchheight", "hipbreadth"]]
#female_pants.to_csv('female_pants.csv', index=False)
#female_pants_with_size = pd.read_csv("female_pants.csv")

#print("Male chestcircumference\n", male_shirt['chestcircumference'].describe(), "\nNull values:\t", any(male_shirt['chestcircumference'].isnull()),"\n")
#print("Male waistcircumference\n", male_shirt['waistcircumference'].describe(), "\nNull values:\t", any(male_shirt['waistcircumference'].isnull()))
#print("Male crotchheight\n", male_pants['crotchheight'].describe(), "\nNull values:\t", any(male_pants['crotchheight'].isnull()),"\n")
#print("Male hipbreadth\n", male_pants['hipbreadth'].describe(), "\nNull values:\t", any(male_pants['hipbreadth'].isnull()))

#print("Female chestcircumference\n", female_shirt['chestcircumference'].describe(), "\nNull values:\t", any(female_shirt['chestcircumference'].isnull()),"\n")
#print("Female waistcircumference\n", female_shirt['waistcircumference'].describe(), "\nNull values:\t", any(female_shirt['waistcircumference'].isnull()))
#print("Female crotchheight\n", female_pants['crotchheight'].describe(), "\nNull values:\t", any(female_pants['crotchheight'].isnull()),"\n")
#print("Female hipbreadth\n", female_pants['hipbreadth'].describe(), "\nNull values:\t", any(female_pants['hipbreadth'].isnull()))

#male_shirt.loc[male_shirt.Size == "XXL"]
def create_sample(df, gender, columns: list, sample_size):
        [list(df.columns).index(column) for column in columns]
        df = df.iloc[:sample_size, [list(df.columns).index(column) for column in columns]].copy()
        df.to_csv(f'{gender}_sample{sample_size}.csv', index=False)

def backup_sample(df, gender, sample_size):
    df.to_csv(f'{gender}_sample{sample_size}.csv', index=False)

    create_sample(male_shirt, 'male', ['chestcircumference', 'waistcircumference'], 100)

    # Male shirt definition
    male_shirt.loc[
        (male_shirt["chestcircumference"] <= 761) & (male_shirt["waistcircumference"] <= 710), "SizeShirt"] = "XXS"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 762) & (male_shirt["waistcircumference"] >= 711), "SizeShirt"] = "XS"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 863) & (male_shirt["waistcircumference"] >= 762), "SizeShirt"] = "S"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 965) & (male_shirt["waistcircumference"] >= 812), "SizeShirt"] = "M"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 1066) & (male_shirt["waistcircumference"] >= 838), "SizeShirt"] = "L"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 1168) & (male_shirt["waistcircumference"] >= 914), "SizeShirt"] = "XL"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 1219) & (male_shirt["waistcircumference"] >= 1016), "SizeShirt"] = "XXL"
    male_shirt.loc[
        (male_shirt["chestcircumference"] >= 1270) & (male_shirt["waistcircumference"] >= 1117), "SizeShirt"] = "XXXL"
    male_shirt.to_csv('T-Shirt_Male.csv', index=False)
    # Female shirt definition
    female_shirt.loc[
        (female_shirt["chestcircumference"] <= 710) & (female_shirt["waistcircumference"] <= 583), "SizeShirt"] = "XXS"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 711) & (female_shirt["waistcircumference"] >= 584), "SizeShirt"] = "XS"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 762) & (female_shirt["waistcircumference"] >= 635), "SizeShirt"] = "S"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 812) & (female_shirt["waistcircumference"] >= 685), "SizeShirt"] = "M"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 914) & (female_shirt["waistcircumference"] >= 762), "SizeShirt"] = "L"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 1016) & (female_shirt["waistcircumference"] >= 838), "SizeShirt"] = "XL"
    female_shirt.loc[
        (female_shirt["chestcircumference"] >= 1117) & (female_shirt["waistcircumference"] >= 914), "SizeShirt"] = "XXL"
    female_shirt.loc[(female_shirt["chestcircumference"] >= 1193) & (
                female_shirt["waistcircumference"] >= 990), "SizeShirt"] = "XXXL"
    female_shirt.to_csv('T-Shirt_Female.csv', index=False)

     #male_shirt

    # Male test definition
    male_test = pd.read_csv("male_sample100.csv")
    male_test.loc[
        (male_test["chestcircumference"] <= 761) & (male_test["waistcircumference"] <= 710), "SizeShirt"] = "XXS"
    male_test.loc[
        (male_test["chestcircumference"] >= 762) & (male_test["waistcircumference"] >= 711), "SizeShirt"] = "XS"
    male_test.loc[
        (male_test["chestcircumference"] >= 863) & (male_test["waistcircumference"] >= 762), "SizeShirt"] = "S"
    male_test.loc[
        (male_test["chestcircumference"] >= 965) & (male_test["waistcircumference"] >= 812), "SizeShirt"] = "M"
    male_test.loc[
        (male_test["chestcircumference"] >= 1066) & (male_test["waistcircumference"] >= 838), "SizeShirt"] = "L"
    male_test.loc[
        (male_test["chestcircumference"] >= 1168) & (male_test["waistcircumference"] >= 914), "SizeShirt"] = "XL"
    male_test.loc[
        (male_test["chestcircumference"] >= 1219) & (male_test["waistcircumference"] >= 1016), "SizeShirt"] = "XXL"
    male_test.loc[
        (male_test["chestcircumference"] >= 1270) & (male_test["waistcircumference"] >= 1117), "SizeShirt"] = "XXXL"
    male_test.to_csv('male_sample100.csv', index=False)

    hue_tshirt_size = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

    sns.lmplot(data=female_shirt, x="chestcircumference", y="waistcircumference", hue="SizeShirt", hue_order=hue_tshirt_size,
               legend=True)
    sns.lmplot(data=male_test, x="chestcircumference", y="waistcircumference", hue="SizeShirt", hue_order=hue_tshirt_size,
               legend=True)

    import numpy as np

    dataset = pd.read_csv("male_sample100.csv")
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    dataset.head()

    dataset.info()

    X = dataset.iloc[:, 0:2]

    X=pd.get_dummies(X)

    #X=np.isnan(X.values.any())

    X

    Y = dataset.iloc[:, 2].values

    Y

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    from sklearn.neighbors import KNeighborsClassifier

    KNN = KNeighborsClassifier(n_neighbors=5,
                               weights="uniform",
                               algorithm="kd_tree",
                               leaf_size=30,
                               p=2,
                               metric="minkowski",
                               n_jobs=-1
                               )

    KNN.fit(X_train, Y_train)

    Y_pred = KNN.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(Y_test.reshape(-1, 1), Y_pred))

    from sklearn.model_selection import cross_val_score

    print("Cross val", cross_val_score(KNN, Y_test.reshape(-1, 1), Y_pred, cv=10))
    print("Cross val", np.mean(cross_val_score(KNN, Y_test.reshape(-1, 1), Y_pred)))

    NewKNN = pd.read_csv("male_sample100.csv")

    NewKNN._data.shape

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(X_test.shape)

    print(Y_train.shape)
    print(Y_test.shape)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    k_range = range(1, 26)
    scores = {}
    scores_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, Y_pred)
        scores_list.append(metrics.accuracy_score(Y_test, Y_pred))

        import matplotlib.pyplot as plt

        plt.plot(k_range, scores_list)
        plt.xlabel("Value of K for KNN")
        plt.ylabel("Testing accuracy")

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, Y)

        KNN = KNeighborsClassifier(n_neighbors=5,
                                   weights="uniform",
                                   algorithm="kd_tree",
                                   leaf_size=30,
                                   p=2,
                                   metric="minkowski",
                                   n_jobs=-1
                                   )
        # chestcircumference
        chest = input("Enter your chestcircumference (cm):")
        # waistcircumference
        waist = input("Enter your waistcircumference (cm):")

        gender = input("Are you male or female?")

        df = pd.read_csv('female_sample100.csv') if gender == "female" else pd.read_csv('T-Shirt_Male.csv')

        classes = {0: "test1", 1: "test2", 2: "test3"}

        x_new = [[chest, waist]]

        y_predict = knn.predict(x_new)

        y_predict[0]
        # y_predict[1]
        size = y_predict[0]
        print(f"Your predicted clothing size: {size}")

