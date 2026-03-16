import pandas as pd

# Load the dataset
try:
    housing = pd.read_csv("california_housing.csv")

    # Find houses with more than 40 rooms
    large_houses = housing[housing["AveRooms"] > 40]

    # Print the results
    if not large_houses.empty:
        print("Found houses with more than 40 rooms:")
        print(large_houses)
    else:
        print("No houses found with more than 40 rooms.")

except FileNotFoundError:
    print("Error: california_housing.csv not found.")
