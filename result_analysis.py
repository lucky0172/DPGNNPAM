import pandas as pd
import csv

def extract_matching_values(file1_path, file2_path, output_file_path):
    df1 = pd.read_csv(file1_path, sep='\t')

    df2 = pd.read_csv(file2_path)

    matching_values = df1[df1['Index'].isin(df2['Node_Index'])]['Gene']

    matching_values.to_csv(output_file_path, index=False, header=False)

extract_matching_values(file1_path, file2_path, output_file_path)


def find_intersection(file1, file2, output_file):
    with open(file1, 'r') as f1:
        data1 = {line.strip() for line in f1}

    with open(file2, 'r') as f2:
        data2 = {line.strip() for line in f2}
 
    intersection = data1.intersection(data2)

    with open(output_file, 'w') as outfile:
        for item in intersection:
            outfile.write(f"{item}\n")


file1 = 'matching_values.txt' 
file2 = 'drought_known(2024.5.30).txt' 
output_file = 'result.txt'  
find_intersection(file1, file2, output_file)

print(f"Matching values have been saved to {output_file}")