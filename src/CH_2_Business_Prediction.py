# import all of the required libraries and classes right here
import pandas as pd
import re
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")

df = pd.read_excel("Coffee_shop_data.xlsx")
population = pd.read_csv("population.csv", skiprows=[0])

population.head()
df.head()  # checking first five rows

# check for data info
df.info()
# our data types checks out

# check the number of records and features
df.shape
population.shape

# get basic stats about the data
df.describe()
# see avergae latte price and salary

ax = df["City"].value_counts().head(5).plot(kind="bar")
ax.set_title("Top 5 cities with most cofee shops")
plt.show()

ax = df["Business Name"].value_counts().head(10).plot(kind="bar")
ax.set_title("Top 10 most famous brands")
plt.show()

df.isna().sum()
# no null values
# if we have null values we would impute it. If we have numberical replace mean. Missing values - replace it with the mode (most occuring values)

# converting zipcode to object data (str) - We need to join the zip code with the population data. Converting the coffee shop data. In order to store it into alphanumerical value, it should be string.
df["Zip Code"] = df["Zip Code"].astype(str)

# extract zip code from population
# Find all of the zipcode that has a 5 digit pattern. Getting the last 5 digits from the population zip code. Creating a new column called zip code


def find_zip_code(geocode):
    pattern = r"\d{5}$"

    match = re.search(pattern, geocode)

    if match:
        zip_code = match.group(0)
    return zip_code


# The actual coversion is below. The above is the function

population["Zip Code"] = population["Geography"].apply(find_zip_code)

cafe_data = df.copy()
# merging the population via zip code as population is an important feature to determing the price / locations
df = pd.merge(cafe_data, population)
# notice that the data size is reduced afer a join

# keeping only Total from population. In the pop dataset, keeping total population column and other columns.
columns = cafe_data.columns.values.tolist() + ["Total"]
df = df[columns]
# rename Total to Population
df = df.rename(columns={"Total": "Population"})

df

# keeping only relevant features
df = df[["Zip Code", "Rating", "Median Salary", "Latte Price", "Population"]]
# df.shape

df.columns

# Calculate the total number of coffee shops for each zip code
coffee_shop_counts = df["Zip Code"].value_counts().reset_index()
coffee_shop_counts.columns = ["Zip Code", "CoffeeShopCount"]

# Ensure 'Zip Code' is of type string in both DataFrames
df["Zip Code"] = df["Zip Code"].astype(str)
coffee_shop_counts["Zip Code"] = coffee_shop_counts["Zip Code"].astype(str)

# Merge the counts back into the original DataFrame
df = df.merge(coffee_shop_counts, on="Zip Code", how="left")

# Print the updated DataFrame
print(df)

# Criteria:
# a. High population
# b. Low total number of coffee shops
# c. Low ratings
# d. High median salary

# Sorting the DataFrame based on the criteria
sorted_df = df.sort_values(
    by=["Population", "CoffeeShopCount", "Rating", "Median Salary"],
    ascending=[False, True, True, False],
).reset_index(drop=True)

# Created a list - if length of list 5, if the zip code is already present, it will not add that into the list.
# Deduping zip code column and displaying all of the records for the top 5.
lst = []
for i in range(len(sorted_df)):
    if len(lst) != 5:
        if (sorted_df["Zip Code"][i]) not in lst:
            lst.append(sorted_df["Zip Code"][i])

# Filter 'sorted_df' to include only rows where 'Zip Code' is in 'lst'
top_5_zip_codes_df = sorted_df[sorted_df["Zip Code"].isin(lst)]

top_5_zip_codes_df

X = df.drop(
    ["Latte Price", "Zip Code"], axis=1
)  # Features excluding 'Latte Price' and 'Zip Code'
y = df["Latte Price"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
