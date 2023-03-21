import pandas as pd
import pyreadr
import os

def start():
    Dataset_Name_List = ["GEFCOM2012", "GEFCOM2014", "GEFCOM2017"]  # "GEFCOM2012", "GEFCOM2014", "GEFCOM2017"
    for dataset_name in Dataset_Name_List:
        os.makedirs("./Dataset/" + dataset_name, exist_ok=True)
        if dataset_name == "GEFCOM2012":
            data = pd.read_csv("./Public_Data/"+dataset_name+"/Load_history.csv", engine="python", thousands = ",")
            zone_id_list = list(data["zone_id"].unique())
            for zone_id in zone_id_list:
                current_data = data.loc[data["zone_id"] == zone_id,:]
                current_data = current_data.sort_values(["year", "month", "day"])

                time_dataframe = []
                for year in range(current_data["year"].to_list()[0], current_data["year"].to_list()[-1]+1):
                    for month in range(1,13):
                        if month in [1, 3, 5, 7, 8, 10, 12]:
                            day_length = 31
                        elif month in [4, 6, 9, 11]:
                            day_length = 30
                        else:
                            if year % 4 == 0:
                                day_length = 29
                            else:
                                day_length = 28
                        for day in range(1, day_length+1):
                            for hour in range(1,25):
                                time_dataframe.append([year, month, day, hour])
                time_dataframe = pd.DataFrame(time_dataframe, columns=["Year", "Month", "Day", "Hour"])

                start_index = time_dataframe.loc[(time_dataframe["Year"] == current_data["year"].to_list()[0]) &
                                                 (time_dataframe["Month"] == current_data["month"].to_list()[0]) &
                                                 (time_dataframe["Day"] == current_data["day"].to_list()[0]) &
                                                 (time_dataframe["Hour"] == 1),].index[0]
                end_index = time_dataframe.loc[(time_dataframe["Year"] == current_data["year"].to_list()[-1]) &
                                                 (time_dataframe["Month"] == current_data["month"].to_list()[-1]) &
                                                 (time_dataframe["Day"] == current_data["day"].to_list()[-1]) &
                                                 (time_dataframe["Hour"] == 24),].index[0]

                time_dataframe = time_dataframe.iloc[start_index:(end_index+1),:].reset_index(drop=True)
                current_dataframe = time_dataframe.copy()
                current_dataframe["Load"] = list(current_data.loc[:,"h1":"h24"].values.reshape(1,-1)[0])
                current_dataframe.to_csv("./Dataset/" + dataset_name + "/" + dataset_name + "_Zone_" + str(zone_id) + ".csv", index=False)

        elif dataset_name == "GEFCOM2014":
            task_list = list(filter(lambda x: x.startswith("Task"), os.listdir("./Public_Data/" + dataset_name + "/Load/")))
            task_id_list = sorted([int(task.split(" ")[1]) for task in task_list])

            total_data = pd.DataFrame()
            for task_id in task_id_list:
                current_data = pd.read_csv("./Public_Data/" + dataset_name + "/Load/Task " + str(task_id) + "/" + "L" + str(task_id) + "-train.csv", engine="python")
                total_data = pd.concat([total_data, current_data], axis=0)
            total_data['datetime'] = pd.date_range('2001-01-01', '2011-12-01', freq='H')[1:]
            total_data["Year"] = total_data["datetime"].dt.year
            total_data["Month"] = total_data["datetime"].dt.month
            total_data["Day"] = total_data["datetime"].dt.day
            total_data["Hour"] = total_data["datetime"].dt.hour
            total_data = total_data[["Year", "Month", "Day", "Hour", "LOAD"]]
            total_data.columns = ["Year", "Month", "Day", "Hour", "Load"]
            total_data = total_data[~pd.isnull(total_data.Load)].reset_index(drop=True)
            total_data.to_csv("./Dataset/"+dataset_name+"/" + dataset_name + "_Zone_1.csv", index=False)

        elif dataset_name == "GEFCOM2017":
            total_data = pyreadr.read_r("./Public_Data/GEFCOM2017/gefcom.rda")["gefcom"]
            zone_list = list(total_data["zone"].unique())

            for zone in zone_list:
                current_data = total_data.loc[(total_data["zone"] == zone),:]
                current_data["Year"] = current_data["ts"].dt.year
                current_data["Month"] = current_data["ts"].dt.month
                current_data["Day"] = current_data["ts"].dt.day
                current_data["Hour"] = current_data["ts"].dt.hour
                current_data = current_data.sort_values(["Year", "Month", "Day", "Hour"])
                current_data = current_data[["Year", "Month", "Day", "Hour", "demand"]]
                current_data.columns = ["Year", "Month", "Day", "Hour", "Load"]
                current_data.to_csv("./Dataset/" + dataset_name + "/" + dataset_name + "_Zone_" + zone + ".csv", index=False)