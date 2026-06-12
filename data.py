import os
import shutil
import kagglehub


path = kagglehub.dataset_download("simonezappatini/body-fat-extended-dataset")

print(path)


csv_file = None
for f in os.listdir(path):
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)
        break

if csv_file is None:
    raise FileNotFoundError("no CSV file found in downloaded dataset")


output_path = os.path.join(os.getcwd(), "BodyFat.csv")
shutil.copy(csv_file, output_path)

print(output_path)