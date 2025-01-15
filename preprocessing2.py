import pandas as pd

# Load the original CSV file
file_path = "Normalized_Image_Paths.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Create the first CSV file: Convert all 1 and 2 labels to 1, leave 0 as 0
df_infection = df.copy()
df_infection['Labels'] = df_infection['Labels'].apply(lambda x: 1 if x in [1, 2] else x)
df_infection.to_csv("infection_labels.csv", index=False)

# Create the second CSV file: Include only 1 and 2 labels, switch 1 to 0 and 2 to 1
df_type = df[df['Labels'].isin([1, 2])].copy()
df_type['Labels'] = df_type['Labels'].apply(lambda x: 0 if x == 1 else 1)
df_type.to_csv("infection_type_labels.csv", index=False)

print("CSV files have been successfully created:")
print("- infection_labels.csv (1 and 2 converted to 1, 0 remains as 0)")
print("- infection_type_labels.csv (1 and 2 only, with 1 -> 0 and 2 -> 1)")