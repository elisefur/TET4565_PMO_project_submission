import csv
import matplotlib.pyplot as plt
import pandas as pd

#------------
#File paths
#------------

# Reading data from csv file.
DATA_PATH = "Spotpriser.csv"

#------------
#Scenario settings
#------------
STAGE1_DAY = 18                      #Day in January used as the known first stage
STAGE2_DAYS = [1, 14, 16, 30, 31]  #Days in January used as the five stage-2 scenarios


#------------
#Read data
#------------
def read_spot_prices(path): 

    # Price data
    df = pd.read_csv(path, quoting=csv.QUOTE_NONE) 

    # Remove quotation marks from column names and data
    df.columns = df.columns.str.replace('"', '', regex=False)
    df = df.replace('"', '', regex=True)

    # Selecting relevant columns
    df = df[["MTU (CET/CEST)", "Day-ahead Price (EUR/MWh)"]]
    df.columns = ["Time", "Price"] # Renaming columns

    # Convert price to number and time to datetime format
    df["Price"] = pd.to_numeric(df["Price"])
    df["Time"] = pd.to_datetime(df["Time"].str[:19], dayfirst=True)

    #Add columns for date, hour of day and time of day, and day
    df["Date"] = df["Time"].dt.date
    df["Hour"] = df["Time"].dt.hour + df["Time"].dt.minute / 60
    df["Time_of_day"] = df["Time"].dt.time
    df["Day"] = df["Time"].dt.day 
    return df

#------------
#Check data
#------------

def check_data(df):
    
    #Checking for missing values and duplicates: 
    print("Number of quarters:", len(df))
    print("Missing values in Price column:", df["Price"].isna().sum())
    print("Duplicate values in Time column:", df["Time"].duplicated().sum())

    # Summary statistics
    print(df["Price"].describe())



#------------
#Plotting
#------------


def plot_daily_profiles(df):
    # Plotting daily curves against each other

    plt.figure(figsize=(9, 5))

    for date, day in df.groupby("Date"):
        plt.step(day["Hour"], day["Price"], where="post", label=str(date))

    plt.xlabel("Hour of the day", fontsize=12)
    plt.ylabel("Price [EUR/MWh]", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        labelspacing=0.3,
        borderpad=0.3,  
        handlelength=1.5,
        handletextpad=0.4
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    
def plot_histogram(df):

    # Histogram of prices
    plt.hist(df["Price"], bins=30)  # Dividing the price range into 30 bins

    plt.xlabel("Price [EUR/MWh]")
    plt.ylabel("Frequency")

    # Remove top and right borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.show()

def plot_boxplot(df):
    # Boxplot of prices
    plt.boxplot(df["Price"])
    plt.ylabel("Price [EUR/MWh]")

    # Remove top and right borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.show()



def correlation(df):
    #Finding the correlation between the daily price profiles using the corr() function.
    #pivot: one row per quarter of the day, one column per date
    daily_df=df.pivot(index="Time_of_day", columns="Date", values="Price")
    cor = daily_df.corr()
    print(cor.round(2))




#------------
#Stage 1 and Stage 2 Price-scenarios
#------------


def get_stage1_prices(df,day): 
    df_stage1= df[df["Day"] ==day] #Returns stage1 prices for one day.
    return df_stage1

def get_stage2_prices(df,days): 

    scenarios_price= {} #Making a dictonary fro each day: price. 

    for i in range(len(days)): 
        scenarios_price[i+1]=df[df["Day"] ==days[i]]

    return scenarios_price

def get_average_prices(df,days): 
    #Create the average price for stage 2-deterministic case. 
    df_scenarios = df[df["Day"].isin(days)]
    average_price = df_scenarios.groupby(df_scenarios["Time_of_day"])["Price"].mean()
    return average_price


#Plotting stage 1+ average price

def plot_stage1_and_average(df_stage1, average_price):
    # Plotting stage 1 prices followed by the average stage-2 price
    plt.step(df_stage1["Hour"], df_stage1["Price"], where="post", label="Stage 1: 18 January")
    plt.step(df_stage1["Hour"] + 24, average_price.values, where="post", label="Stage 2: average of the scenarios")
    plt.axvline(24, color="red", linestyle=":")

    plt.xlabel("Hour")
    plt.ylabel("Price [EUR/MWh]")
    plt.legend()
    plt.show()


#----------
#MAIN
#----------

if __name__== "__main__": # Runs only when this file is run directly, not importet 

    df=read_spot_prices(DATA_PATH)
    check_data(df)

    correlation(df)
    plot_daily_profiles(df)
    plot_histogram(df)
    plot_boxplot(df)

    df_stage1= get_stage1_prices(df,STAGE1_DAY)
    average_price= get_average_prices(df,STAGE2_DAYS)
    scenarios_price= get_stage2_prices(df, STAGE2_DAYS)

    #Plotting the average price: 
    plot_stage1_and_average(df_stage1, average_price)