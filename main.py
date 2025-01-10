import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# File path settings
input_folder = 'datasInput'  # Folder containing XPT files
output_folder = 'datasOutput'  # Folder to save CSV files

# Step 1: Batch read .xpt files and convert them to .csv files
def convert_xpt_to_csv(input_folder, output_folder):
    # Get all XPT files
    xpt_files = [f for f in os.listdir(input_folder) if f.endswith('.xpt')]
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for xpt_file in xpt_files:
        # Build file paths
        xpt_path = os.path.join(input_folder, xpt_file)
        csv_path = os.path.join(output_folder, xpt_file.replace('.xpt', '.csv'))
        
        # Read the XPT file
        data = pd.read_sas(xpt_path, format='xport')
        
        # Save as a CSV file
        data.to_csv(csv_path, index=False)
        print(f"Converted {xpt_file} to CSV and saved to {csv_path}")

# Step 2: Batch read CSV files and concatenate them vertically
def merge_csv_files(output_folder):
    csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
    data_frames = [pd.read_csv(os.path.join(output_folder, f)) for f in csv_files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    
    # Ensure key columns for target variable exist
    required_columns = ['LBXGLU', 'LBXGH']
    for col in required_columns:
        if col not in merged_data.columns:
            raise ValueError(f"Missing required column: {col} for target variable creation.")

    # Handle missing values
    print("Missing values before handling:")
    print("LBXGLU:", merged_data['LBXGLU'].isnull().sum())
    print("LBXGH:", merged_data['LBXGH'].isnull().sum())
    
    # Option 1: Impute missing values
    merged_data['LBXGLU'] = merged_data['LBXGLU'].fillna(merged_data['LBXGLU'].median())
    merged_data['LBXGH'] = merged_data['LBXGH'].fillna(merged_data['LBXGH'].median())
    
    print("Missing values after handling:")
    print("LBXGLU:", merged_data['LBXGLU'].isnull().sum())
    print("LBXGH:", merged_data['LBXGH'].isnull().sum())
    
    # Define target variable
    merged_data['target'] = (
        (merged_data['LBXGLU'] >= 126) | 
        (merged_data['LBXGH'] >= 6.5)
    ).astype(int)

    print("Target variable counts:")
    print(merged_data['target'].value_counts())

    return merged_data



# Step 3: Fill missing and zero values
def fill_missing_and_zero_values(data):
    for column in data.columns:
        missing_count = data[column].isnull().sum()
        zero_count = (data[column] == 0).sum()
        print(f"Column {column}: {missing_count} missing values, {zero_count} zero values")

        # Calculate the median
        median_value = data[column].median()

        # Fill missing values
        data[column] = data[column].fillna(median_value)

        # Fill zero values
        data[column] = data[column].replace(0, median_value)

    print("Missing and zero values have been filled!")
    return data


# Step 4: Data standardization
def standardize_data(data):
    # Select only numeric feature columns (exclude 'target')
    numeric_columns = data.drop(columns=['target']).select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data


# Step 5: Split data into training and testing sets
def split_data(data):
    # Ensure output folder exists
    output_folder = 'processed_data'
    os.makedirs(output_folder, exist_ok=True)

    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data has been split into training and testing sets!")

    # Save training and testing sets
    X_train.to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)

    return X_train, X_test, y_train, y_test



# Main process
def main():
    print("Step 1: Converting .xpt files to .csv")
    convert_xpt_to_csv(input_folder, output_folder)

    print("Step 2: Merging CSV files and creating target variable")
    merged_data = merge_csv_files(output_folder)

    print("Step 3: Filling missing and zero values")
    merged_data = fill_missing_and_zero_values(merged_data)

    print("Step 4: Standardizing numeric data")
    standardized_data = standardize_data(merged_data)

    print("Step 5: Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = split_data(standardized_data)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print("Data preparation completed!")


if __name__ == '__main__':
    main()

