import sys
import os

# Try to import heavy dependencies and provide a clear message if any are missing. This
# helps when users click "Run" in an IDE and nothing appears to happen because an
# import fails immediately (e.g. numpy not installed).
try:
    import argparse
    import pickle  # Used for serializing and deserializing Python objects (e.g., data, models)
    import numpy as np  # Library for handling arrays and performing numerical computations
    from sklearn.ensemble import RandomForestClassifier  # Machine learning model: Random Forest Classifier
    from sklearn.model_selection import train_test_split  # Used for splitting the dataset into training and testing sets
    from sklearn.metrics import accuracy_score  # Metric to evaluate model performance (accuracy)
except Exception as e:
    print('Import error:', e)
    print('\nOne or more Python packages required by this project are not installed.')
    print('Install dependencies with:')
    print('  pip install -r requirements.txt')
    # If running from an IDE, the above message will appear in the run console.
    sys.exit(1)


def main(no_preview: bool = False):
    # Resolve paths relative to this file so "Run" in an IDE works regardless of CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data.pickle')
    model_path = os.path.join(base_dir, 'model.p')

    # Load the dataset from a pickle file ('rb' means reading the file in binary mode)
    if not os.path.exists(data_path):
        print(f"ERROR: dataset not found at {data_path}\nMake sure you've created 'data.pickle' by running 'data creation.py' from the project folder.")
        sys.exit(2)

    try:
        with open(data_path, 'rb') as fh:
            data_dict = pickle.load(fh)
    except Exception as e:
        print(f"ERROR: could not read data.pickle ({data_path}): {e}")
        sys.exit(3)

    # Convert the data and labels into NumPy arrays for easier manipulation in machine learning
    data = np.asarray(data_dict.get('data', []))  # Feature data (e.g., hand landmarks)
    labels = np.asarray(data_dict.get('labels', []))  # Labels (class indices for each sample)

    if data.size == 0 or labels.size == 0:
        print(f"ERROR: data.pickle appears to be empty or malformed (data size: {data.size}, labels size: {labels.size}).")
        sys.exit(4)

    # Split the dataset into training and testing sets
    # 80% of the data will be used for training, and 20% will be used for testing
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    # Initialize the RandomForestClassifier model
    model = RandomForestClassifier()

    # Train the model using the training data (fit the model to the data)
    model.fit(x_train, y_train)

    # Use the trained model to predict labels for the test data
    y_predict = model.predict(x_test)

    # Calculate the accuracy of the model's predictions
    score = accuracy_score(y_predict, y_test)

    # Print the accuracy score as a percentage, formatted to two decimal places
    print(f'{score * 100:.2f}% of samples were classified correctly!')

    # Save the trained model to a file using pickle ('wb' means writing the file in binary mode)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model}, f)
        print(f'Trained model saved to {model_path}')
    except Exception as e:
        print(f'ERROR: could not save model to {model_path}: {e}')
        sys.exit(5)

    # After training and saving the model, optionally launch the camera preview
    if not no_preview:
        try:
            import testing

            # Pass the trained model to the camera preview so it uses the freshly trained model
            print('Launching camera preview with trained model...')
            testing.run_camera(model=model)
        except Exception as e:
            print(f'Could not start camera preview from training.py: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the RandomForest model from data.pickle (paths resolved relative to this file).')
    parser.add_argument('--no-preview', action='store_true', help='Do not launch the live camera preview after training')
    args = parser.parse_args()
    main(no_preview=args.no_preview)
