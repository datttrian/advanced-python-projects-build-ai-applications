# import all of the required libraries and classes right here
import pandas as pd
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
