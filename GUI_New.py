import pickle
import tkinter as tk
from tkinter import ttk
import unicodeit
from PIL import Image, ImageTk
from tkinter import font
import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt

# from lime.lime_tabular import LimeTabularExplainer
# Load and preprocess dataset
data = pd.read_excel("Dataset.xlsx")
X = data.iloc[:, :-2]
y = data.iloc[:, -2:]


# Function to load your model

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


pcc = 'P{}'.format(get_sub('cc'))
ecc = 'ε{}'.format(get_sub('cc'))


def load_model():
    with open('NN_Pcc_ecc.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Predict function to handle GUI input and show output
def predict():
    try:
        # Convert input values from strings to appropriate types (floats here)
        inputs = [float(entry.get()) for entry in entries]
        print("Inputs:", inputs)  # Debugging statement
        # Load your model
        model = load_model()

        # Make predictions using the model
        # Note: model.predict() generally expects a 2D array of inputs
        results = model.predict([inputs])[0]  # Get the first prediction from the results
        print("Results:", results)  # Debugging statement

        # Update the output label with the prediction results
        output_label_pcc.config(text=f'{pcc} (kN): {results[0]:.2f}')
        output_label_ecc.config(text=f'{ecc} (mm/mm): {results[1]:.2f}')

        # Generate LIME explanations
        # explain(inputs, model, results)


    except ValueError as ve:

        # If there is an error in input conversion, notify the user

        output_label_pcc.config(text=f'ValueError: {str(ve)}')

        output_label_ecc.config(text=f'ValueError: {str(ve)}')
    except Exception as e:
        # Catch other potential errors from prediction or loading model
        output_label_pcc.config(text=f'Error: {str(e)}')
        output_label_ecc.config(text=f'Error: {str(e)}')


# Function to generate LIME explanations
def explain(inputs, model, results):
    # Create a LIME explainer
    explainer = LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['Pcc', 'ecc'], mode='regression')

    # Explain the instance
    instance_explanations = []
    for j in range(2):  # Loop over each output
        exp = explainer.explain_instance(np.array(inputs), model.predict, labels=[j], num_features=5)
        instance_explanations.append(exp)

        # Plot the explanation
        fig = exp.as_pyplot_figure(label=j)
        plt.title(f'LIME Explanation for {["Pcc", "ecc"][j]}')
        plt.tight_layout()
        plt.savefig(f"lime_explanation_output_{j + 1}.png", dpi=600)
        plt.close(fig)

    # Export explanations to Excel
    explanation_data = []
    for j, exp in enumerate(instance_explanations):
        for feature, weight in exp.as_list(label=j):
            explanation_data.append({'Output': ["Pcc", "ecc"][j], 'Feature': feature, 'Weight': weight})

    explanation_df = pd.DataFrame(explanation_data)
    explanation_df.to_excel("lime_explanations.xlsx", index=False, engine='openpyxl')


# Set up the main window
root = tk.Tk()
root.title(f'Prediction of {pcc} and {ecc}')

# Define a style for the button
style = ttk.Style()
style.configure('TButton', background='blue', foreground='black', font=('Times New Roman', 12, 'bold'))

# Function to convert to subscript


# Display subscript
print('f{}'.format(get_sub('co')))  # H₂SO₄

# Load and display an image
file_path = (r'C:\Users\rupes\OneDrive - charusat.edu.in\Upcoming Research Work\Collabration\China\New Papers and '
             r'Reports (18-01-2024)\ML Reports\Current\Topic 21\ML model\GUI\FRP Wrapping Final.png')
exists = os.path.exists(file_path)
print("File exists:", exists)

image = Image.open(file_path)
photo = ImageTk.PhotoImage(image)

image_label = ttk.Label(root, image=photo)
image_label.grid(row=0, column=2, rowspan=10)

# Input labels and entries
labels = [
    'X1',  # No subscript needed
    'X2',  # No subscript needed
    'X3',  # No subscript needed
    'X4',  # No subscript needed
    'X5',  # No subscript needed
    'X6',  # No subscript needed
    'X7',  # No subscript needed
    'X8',  # No subscript needed
    'X9',  # No subscript needed
]

entries = []
# Italic font setup
italic_font = font.Font(root, ('Times New Roman', 10, 'italic'))
for i, label in enumerate(labels):
    ttk.Label(root, text=label, font=italic_font).grid(row=i, column=0)
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Predict button
pcc = 'P{}'.format(get_sub('cc'))
ecc = 'ε{}'.format(get_sub('cc'))
predict_button = ttk.Button(root, text=f'Calculate {pcc} & {ecc}', command=predict, style='TButton')
predict_button.grid(row=len(labels), column=0, columnspan=2)

# Output labels
output_label_pcc = ttk.Label(root, text=f'{pcc} (kN): ', font=('Times New Roman', 18, 'bold'))
output_label_pcc.grid(row=len(labels) + 1, column=0, columnspan=2)
output_label_ecc = ttk.Label(root, text=f'{ecc} (mm/mm): ', font=('Times New Roman', 18, 'bold'))
output_label_ecc.grid(row=len(labels) + 2, column=0, columnspan=2)

# Run the main loop
root.mainloop()
