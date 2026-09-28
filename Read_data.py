import csv
import matplotlib.pyplot as plt
import pandas as pd

# Reading data from csv file.
DATA_PATH = "Spotpriser.csv"

# Price data
df = pd.read_csv(DATA_PATH, quoting=csv.QUOTE_NONE) 

# Remove quotation marks from column names and data
df.columns = df.columns.str.replace('"', '', regex=False)
df = df.replace('"', '', regex=True)

# Selecting relevant columns
df = df[["MTU (CET/CEST)", "Day-ahead Price (EUR/MWh)"]]
df.columns = ["Time", "Price"] # Renaming columns

# Convert price to number and time to datetime format
df["Price"] = pd.to_numeric(df["Price"])
df["Time"] = pd.to_datetime(df["Time"].str[:19], dayfirst=True)

#Checking for missing values and duplicates: 
print("Number of quarters:", len(df))
print("Missing values in Price column:", df["Price"].isna().sum())
print("Duplicate values in Time column:", df["Time"].duplicated().sum())

# Summary statistics
print(df["Price"].describe())

#Plotting daily curves against each other

df["Date"] = df["Time"].dt.date #Finding the date for each quarter
df["Time_of_day"] = df["Time"].dt.hour + df["Time"].dt.minute / 60 #Finding the time of day for each quarter in hours

for date, day in df.groupby("Date"):
    plt.plot(day["Time_of_day"], day["Price"], label=str(date))
plt.xlabel("Hour of the day")
plt.ylabel("Price [EUR/MWh]")
plt.title("Day-ahead price profiles NO3")
plt.legend()
plt.grid()
plt.show()

# Histogram of prices
plt.hist(df["Price"], bins=30) #dividing the price range into 30 bins
plt.xlabel("Price [EUR/MWh]")
plt.ylabel("Frequency")
plt.title("Distribution of day-ahead prices")
plt.grid()
plt.show()

# Boxplot of prices
plt.boxplot(df["Price"])
plt.ylabel("Price [EUR/MWh]")
plt.title("Boxplot of day-ahead prices")
plt.grid()
plt.show()



#Finding the correlation between the daily price profiles using the corr() function.
def correlation(temp_df):
    cor = temp_df.corr()
    print(cor.round(2))

correlation(df.pivot(index="Time_of_day", columns="Date", values="Price"))



