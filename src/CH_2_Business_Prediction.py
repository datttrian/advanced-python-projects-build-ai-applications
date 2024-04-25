# import all of the required libraries and classes right here
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Model Selection
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
}

# Hyperparameter Tuning
param_grid = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10],
    },
}

for model_name, model in models.items():
    if model_name in param_grid:
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(
            model,
            param_grid[model_name],
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_search.fit(X, y)

        # Set the best hyperparameters to the model
        models[model_name] = grid_search.best_estimator_

# Model Training
for model_name, model in models.items():
    # Train the model on the training set
    model.fit(X_train, y_train)

# Model Evaluation
for model_name, model in models.items():
    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    print(f"{model_name} Metrics:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print()

# We want this dataframe to be same as the training data so that model can predict the value
zip_codes_df = top_5_zip_codes_df.drop(["Zip Code", "Latte Price"], axis=1)
zip_codes_df = sc.transform(zip_codes_df)

for model_name, model in models.items():
    # Predict the prices for lattes in the top 5 zip codes
    predicted_prices = model.predict(zip_codes_df)
    print(f"{model_name} Predicted Prices for Top 5 Zip Codes:")
    print(predicted_prices)
    print()

predictions = {}

for model_name, model in models.items():
    # Predict the prices for lattes in the top 5 zip codes
    predicted_prices = model.predict(zip_codes_df)
    predictions[model_name] = predicted_prices

# Convert the predictions dictionary to a DataFrame
predictions_df = pd.DataFrame(predictions)
# Add the zip codes to the predictions DataFrame
predictions_df["Zip Code"] = top_5_zip_codes_df["Zip Code"].values

# Rearrange the columns to have 'Zip Code' as the first column
cols = ["Zip Code"] + [
    col for col in predictions_df.columns if col != "Zip Code"
]
predictions_df = predictions_df[cols]

predictions_df

agg_df = (
    predictions_df.groupby("Zip Code")["Gradient Boosting"]
    .agg([("Highest", "max"), ("Lowest", "min")])
    .reset_index()
)
agg_df.columns = ["Zip Code", "Highest", "Lowest"]
print(agg_df)
