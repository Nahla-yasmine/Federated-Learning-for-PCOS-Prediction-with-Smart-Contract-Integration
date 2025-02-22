import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

class Party:
    def __init__(self, name, features):
        self.name = name
        self.features = features
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()

    def set_data(self, data):
        """Set and scale the data"""
        self.data = data[self.features]

    def split_data(self, train_idx, test_idx):
        """Split data using provided indices and scale"""
        self.train_data = self.data.iloc[train_idx]
        self.test_data = self.data.iloc[test_idx]

        # Scale the data
        self.train_data = pd.DataFrame(
            self.scaler.fit_transform(self.train_data),
            columns=self.features
        )
        self.test_data = pd.DataFrame(
            self.scaler.transform(self.test_data),
            columns=self.features
        )

def preprocess_data(df):
    """Preprocess the dataset"""
    try:
        # Binary columns encoding
        binary_columns = ['Menstrual Regularity', 'Hirsutism', 'Family History of PCOS',
                          'Insulin Resistance', 'Urban/Rural', 'Diagnosis']
        for col in binary_columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        # One-hot encoding
        df = pd.get_dummies(df, columns=['BMI', 'Socioeconomic Status', 'Ethnicity'], drop_first=True)

        # Label Encoding for remaining categorical columns (except Country)
        categorical_columns = df.select_dtypes(include=['object']).columns
        if 'Country' in categorical_columns:
            categorical_columns = categorical_columns.drop('Country')
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

        # Scale numerical columns
        scaler = MinMaxScaler()
        numeric_columns = ['Lifestyle Score', 'Stress Levels', 'Undiagnosed PCOS Likelihood']
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    except Exception as e:
        print(f"Error during preprocessing: {e}")

    return df

def setup_vertical_split(df):
    """Split features between parties"""
    # Party A: Medical measurements and symptoms
    party_a_features = [
        'Menstrual Regularity', 'Hirsutism', 'Acne Severity',
        'Family History of PCOS', 'Insulin Resistance',
        'BMI_Obese', 'BMI_Overweight', 'BMI_Underweight'
    ]

    # Party B: Lifestyle and demographic data
    party_b_features = [
        'Lifestyle Score', 'Stress Levels', 'Urban/Rural',
        'Socioeconomic Status_Low', 'Socioeconomic Status_Middle',
        'Ethnicity_Asian', 'Ethnicity_Caucasian', 'Ethnicity_Hispanic', 'Ethnicity_Other'
    ]

    party_a = Party("Medical Center", party_a_features)
    party_b = Party("Demographics Center", party_b_features)

    # Set data for each party
    party_a.set_data(df)
    party_b.set_data(df)

    return party_a, party_b

class VFLCoordinator:
    def __init__(self, party_a, party_b):
        self.party_a = party_a
        self.party_b = party_b
        self.global_model = None

    def generate_intermediate_features(self, data_a, data_b, epsilon=0.5):
        """Simulate secure feature sharing with differential privacy"""
        # Convert to numpy arrays if they're pandas DataFrames
        if isinstance(data_a, pd.DataFrame):
            data_a = data_a.values
        if isinstance(data_b, pd.DataFrame):
            data_b = data_b.values

        # Add scaled Laplace noise for privacy
        noise_scale_a = np.std(data_a, axis=0) * (1 / epsilon) * 0.01  # Reduced noise
        noise_scale_b = np.std(data_b, axis=0) * (1 / epsilon) * 0.01  # Reduced noise

        noise_a = np.random.laplace(0, noise_scale_a, size=data_a.shape)
        noise_b = np.random.laplace(0, noise_scale_b, size=data_b.shape)

        # Combine features with scaled noise
        return np.hstack([data_a + noise_a, data_b + noise_b])

    def train(self, X_train, X_test, y_train, y_test, epsilon=0.5, n_rounds=10):
        """Train the VFL model with multiple rounds"""
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        # Initialize global model with SGDClassifier
        self.global_model = SGDClassifier(
            loss='log_loss',
            max_iter=1000,
            random_state=42,
            class_weight=class_weight_dict,
            alpha=0.001  # Tuned regularization
        )

        # Mini-batch training over multiple rounds
        batch_size = len(X_train) // n_rounds
        for round in range(n_rounds):
            start = round * batch_size
            end = (round + 1) * batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Train the model on the current batch
            self.global_model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

        # Evaluate
        train_pred = self.global_model.predict(X_train)
        test_pred = self.global_model.predict(X_test)

        # Compute ROC-AUC
        test_probs = self.global_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, test_probs)

        # Detailed metrics
        train_report = classification_report(y_train, train_pred, output_dict=True)
        test_report = classification_report(y_test, test_pred, output_dict=True)

        return {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'roc_auc': roc_auc,
            'train_report': train_report,
            'test_report': test_report
        }

def run_vfl_simulation(df, test_size=0.2, epsilon=0.5, n_rounds=10):
    """Run the complete VFL simulation"""
    # Preprocess data
    df_processed = preprocess_data(df)

    # Setup parties
    party_a, party_b = setup_vertical_split(df_processed)

    # Split indices for consistent splitting across parties
    X = df_processed.drop('Diagnosis', axis=1)
    y = df_processed['Diagnosis']

    # Get train/test indices
    train_idx, test_idx = train_test_split(
        np.arange(len(df_processed)),
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    # Split data for each party
    party_a.split_data(train_idx, test_idx)
    party_b.split_data(train_idx, test_idx)

    # Initialize coordinator
    coordinator = VFLCoordinator(party_a, party_b)

    # Generate training and test features
    X_train = coordinator.generate_intermediate_features(
        party_a.train_data, 
        party_b.train_data, 
        epsilon
    )
    X_test = coordinator.generate_intermediate_features(
        party_a.test_data, 
        party_b.test_data, 
        epsilon
    )

    # Split labels
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Train and evaluate
    global_results = coordinator.train(X_train, X_test, y_train, y_test, epsilon, n_rounds)

    # Party-specific results
    party_a_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
    party_a_model.fit(party_a.train_data, y_train)
    party_a_pred = party_a_model.predict(party_a.test_data)
    party_a_accuracy = accuracy_score(y_test, party_a_pred)

    party_b_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
    party_b_model.fit(party_b.train_data, y_train)
    party_b_pred = party_b_model.predict(party_b.test_data)
    party_b_accuracy = accuracy_score(y_test, party_b_pred)

    return coordinator, global_results, party_a_accuracy, party_b_accuracy

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("pcos_prediction_dataset.csv")

    # Run VFL simulation
    coordinator, global_results, party_a_accuracy, party_b_accuracy = run_vfl_simulation(df, test_size=0.2, epsilon=0.5, n_rounds=10)

    # Print detailed results
    print("\nGlobal VFL Results:")
    print(f"Training Accuracy: {global_results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {global_results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {global_results['roc_auc']:.4f}")

    print("\nParty-Specific Results:")
    print(f"Party A (Medical Center) Test Accuracy: {party_a_accuracy:.4f}")
    print(f"Party B (Demographics Center) Test Accuracy: {party_b_accuracy:.4f}")

    # Feature importance visualization
    features = coordinator.party_a.features + coordinator.party_b.features
    importance = np.abs(coordinator.global_model.coef_[0])

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(features)), importance)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Absolute Coefficient Value')
    plt.title('Feature Importance in VFL Model')
    plt.tight_layout()
    plt.show()