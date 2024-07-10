import pandas as pd
from datetime import datetime, timedelta
import re


#############################################################################
###                                Options                                ###
#############################################################################
folder_1 = "5cmx5cm_square/1-continuos/"
folder_2 = "5cmx5cm_square/2-separate/"
folder_3 = "LHB-DotBox/"
folder_4 = "LHC-DotBox/"

filename = "pydotbot.log"

# choose which dataset to process
folder = folder_3

# Global variable marking which sweep  (first or second) has been already, and when
sweep_slot = [{"exist":False, "time":0., "index":0}, \
            {"exist":False, "time":0., "index":0}]

#############################################################################
###                             Functions                                 ###
#############################################################################

def select_sweep(data: dict[str, str | int]) -> int:
    global sweep_slot

    selected_sweep = 0

    # If the DotBot timer resets, depend on the lfsr index to detect which sweep it is
    if (data['db_time'] - sweep_slot[0]["time"]) < 0 or (data['db_time'] - sweep_slot[1]["time"]) < 0:
        
        idiff_0 =  abs(data['lfsr_index'] - sweep_slot[0]["index"])
        idiff_1 =  abs(data['lfsr_index'] - sweep_slot[1]["index"])

        # Use the one that is closest to 20ms
        if (idiff_0 <= idiff_1):
            selected_sweep = 0
        else:
            selected_sweep = 1
    else:
        # both sweep_slots are empty
        if not sweep_slot[0]["exist"] and not sweep_slot[1]["exist"]:
            # use first slot
            selected_sweep = 0

        # first sweep_slots is empty
        if not sweep_slot[0]["exist"] and sweep_slot[1]["exist"]:
            diff:int = (data["db_time"] - sweep_slot[1]["time"]) % 20000
            if not (diff < 20000 - diff):
                diff = 20000 - diff

            if (diff < 1000): selected_sweep = 1
            else: selected_sweep = 0

        # second sweep_slots is empty
        if sweep_slot[0]["exist"] and not sweep_slot[1]["exist"]:
            diff:int = (data["db_time"] - sweep_slot[0]["time"]) % 20000
            if not (diff < 20000 - diff):
                diff = 20000 - diff     

            if (diff < 1000): selected_sweep = 0
            else: selected_sweep = 1   

        # Both sweep_slote are full
        if sweep_slot[0]["exist"] and sweep_slot[1]["exist"]:
            # How far away is this new pulse from the already stored data
            diff_0:int = (data["db_time"] - sweep_slot[0]["time"]) % 20000
            if not (diff_0 < 20000 - diff_0):
                diff_0 = 20000 - diff_0

            diff_1:int = (data["db_time"] - sweep_slot[1]["time"]) % 20000
            if not (diff_1 < 20000 - diff_1):
                diff_1 = 20000 - diff_1

            # Use the one that is closest to 20ms
            if (diff_0 <= diff_1):
                selected_sweep = 0
            else:
                selected_sweep = 1
    

    # save the current state of the global variable
    sweep_slot[selected_sweep]["time"] = data["db_time"]
    sweep_slot[selected_sweep]["exist"] = True
    sweep_slot[selected_sweep]["index"] = data['lfsr_index']

    return selected_sweep


def check_timestamp(data: dict[str, str | int], log_data:list[dict[str, str | int]]) -> float | None:
    """
    return the relative milisecond error between the computer timestamp and the nRF timestamp.
    if a reset is detected on the nRF (negative delta on db_time), return None.
    If a big gap (>500ms) is detected, return None.
    """
    # First data point, return None
    if log_data == []: return None

    # Get last data point of the data log
    prev_data = log_data[-1]

    # Get time difference
    db_time_diff: int = data["db_time"] - prev_data["db_time"]
    pc_time_diff: int = (datetime.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.strptime(prev_data["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")).microseconds

    # Check for nRF reset.
    if (db_time_diff < 0): return None

    # Check for large gap
    if (db_time_diff > 500e3): return None

    # return time delta error
    return (pc_time_diff - db_time_diff) / 1000.0

#############################################################################
###                                Code                                   ###
#############################################################################

# Updated regular expression to capture the desired values more precisely
log_pattern = re.compile(
    r"timestamp=(?P<timestamp>[^ ]+) .* source=(?P<source>[^ ]+) .* poly=(?P<poly>[^ ]+) lfsr_index=(?P<lfsr_index>[^ ]+) db_time=(?P<db_time>[^ ]+)"
)

# Create an empty list to store the extracted data
log_data:list[dict[str, str | int]] = []
time_diff: list[float] = []

# If the single test works, you can then apply the same to the file reading
with open(folder + filename, "r") as log_file:
    for line in log_file:
        match = log_pattern.search(line)
        if match and int(match.group("lfsr_index")) < 120e3:

            data = {
                # "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                "timestamp": match.group("timestamp"),
                "source": match.group("source"),
                "poly": int(match.group("poly")),
                "lfsr_index": int(match.group("lfsr_index")),
                "db_time": int(match.group("db_time")),
            }

            # check time difference between dotbot and pc
            data_time_diff = check_timestamp(data, log_data)
            if data_time_diff is not None:
                time_diff.append(data_time_diff)

            # Estimate if the data is the first or second sweep 
            data["sweep"] = select_sweep(data)

            log_data.append(data)


print(f"{time_diff}")

df = pd.DataFrame(log_data)
# convert the string timestamp, to a dattime object
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ")

# Remove outliers detected in polynomials other than 0 and 1. Only mode 1 was used in the experiment
df = df[df['poly'] < 2]
# Reset the index
df.reset_index(drop=True, inplace=True)


## Detect repeated data and remove.
duplicate_mask = df.duplicated(subset=['lfsr_index', 'db_time', 'poly'], keep=False)
# Remove both instances of the duplicates
df = df[~duplicate_mask]
print(duplicate_mask.sum()/2)
# Reset the index
df.reset_index(drop=True, inplace=True)


## Sort small out-of-order data lines
# run it a few times to ensure no stragglers remain
for y in range(5):
    ## Organize data, some rows are inverted according to db_time
    df['diff'] = df['db_time'].diff()
    condition = (df['diff'] > -60000) & (df['diff'] < 0)
    # Iterate through the rows that need swapping
    for i in df.index[condition]:
        if i > 0:  # Ensure we're not at the first row
            # Swap the current row with the row above it
            df.iloc[i-1], df.iloc[i] = df.iloc[i].copy(), df.iloc[i-1].copy()
    # Remove the temporary 'time_diff' column
    df = df.drop(columns=['diff'])
    # Reset the index
    df.reset_index(drop=True, inplace=True)


## Recalculate the python timestamp based on the dotbot-timer timestamp
# Fix the timestamp for the LH data
base_timestamp = df.iloc[0]['timestamp']
base_db_time   = df.iloc[0]['db_time']

prev_db_time   = df.iloc[0]['db_time']

for index, row in df.iterrows():
    current_timestamp = df.at[index, 'timestamp']
    current_db_time   = df.at[index, 'db_time']

    # A lh-minimote reset was detected. Reset the base for the timestamp calculation.
    if (current_db_time < prev_db_time):
        base_db_time = current_db_time
        base_timestamp = current_timestamp
        # Update the previous value
        prev_db_time  = current_db_time
        continue

    # Estimate the real timestamp from the initial timestamp + the dotbot timer 
    df.at[index, 'timestamp'] = base_timestamp + timedelta(microseconds=float(current_db_time - base_db_time))

    # Update the previous value
    prev_db_time  = current_db_time

# sorted_df.to_csv(folder + 'data.csv', index=True)
df.to_csv(folder + 'lh_data.csv', index=True, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

