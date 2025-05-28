import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Global variables to hold the dataset and model
df = None
model = None

# Function to preprocess the dataset
def preprocess_data(df):
    """Clean and preprocess the dataset"""
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Handle 'varies' and 'unknown' values in study_hours_per_day
    if 'study_hours_per_day' in df.columns:
        # Convert 'varies' to NaN, then fill with median
        df['study_hours_per_day'] = df['study_hours_per_day'].replace('varies', pd.NA)
        df['study_hours_per_day'] = pd.to_numeric(df['study_hours_per_day'], errors='coerce')
        df['study_hours_per_day'].fillna(df['study_hours_per_day'].median(), inplace=True)
    
    # Handle 'unknown' values in age
    if 'age' in df.columns:
        df['age'] = df['age'].replace('unknown', pd.NA)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'].fillna(df['age'].median(), inplace=True)
    
    # Handle exam_score outliers (200 seems to be a placeholder for missing data)
    if 'exam_score' in df.columns:
        df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
        # Replace scores of 200 with NaN and fill with median
        df.loc[df['exam_score'] == 200, 'exam_score'] = pd.NA
        df['exam_score'].fillna(df['exam_score'].median(), inplace=True)
    
    # Handle missing values in categorical columns
    categorical_columns = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                          'internet_quality', 'extracurricular_participation']
    
    for col in categorical_columns:
        if col in df.columns:
            # Fill missing values with mode (most frequent value)
            if df[col].mode().empty:
                df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Fill remaining numeric columns with median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df

# Loads a dataset from a file selected by the user
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"),
("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            df = preprocess_data(df)
             
            messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# Trains a Random Forest model using the provided features and target variable
def train_model(df, features, target):
    global model
    if df is None:
        messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
        return None
    try:
         # Clean feature names (remove whitespace)
        features = [f.strip() for f in features]
        target = target.strip()
        
        # Check if columns exist
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Features not found in dataset: {missing_cols}")
            return None
        
        if target not in df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found in dataset.")
            return None
        
        X = df[features]
        y = df[target]
        
        # Handle categorical variables with LabelEncoder
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Handle target variable if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
        return None

# Makes predictions using the trained model and the provided features
def make_predictions(model, df, features):
    if model is None:
        messagebox.showerror("Error", "Model not trained. Please train the model first.")
        return None
    if df is None:
        messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
        return None
    try:
        # Clean feature names (remove whitespace)
        features = [f.strip() for f in features]
        
        X_new = df[features]
        
        # Handle categorical variables with LabelEncoder (same as training)
        for col in X_new.columns:
            if X_new[col].dtype == 'object':
                le = LabelEncoder()
                X_new[col] = le.fit_transform(X_new[col].astype(str))
                
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Creates the main application window and sets up the UI components
root = tk.Tk()
root.title("Student Predictive Grades")

# loads the dataset when the user clicks the button
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

# Features input section
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# target input section
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Train model button
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df,
features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# make predictions button
predict_button = tk.Button(root, text="Make Predictions", command=lambda:
make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Text widget to display results
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Run the application
root.mainloop()