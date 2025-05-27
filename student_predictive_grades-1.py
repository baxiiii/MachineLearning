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

#add funtion

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
        X = df[features]
        y = df[target]
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
        X_new = df[features]
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Please add funtion comment
root = tk.Tk()
root.title("Student Predictive Grades")

# Please add funtion comment
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

#Please add funtion comment
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Please add funtion comment
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Please add funtion comment
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df,
features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Please add funtion comment
predict_button = tk.Button(root, text="Make Predictions", command=lambda:
make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Please add funtion comment
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Please add funtion comment
root.mainloop()
