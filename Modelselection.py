import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = 'Dataset.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Select columns and preprocess data
selected_columns = [
    'EF', 'Crop_Type', 'Tmp', 'Prec', 'BD', 'Clay', 'SOC', 'pH', 'Fertilizer',
    'Tillage practice', 'Irrigation', 'Fertilizer placement', 'Nrate'
]
data = data[selected_columns]
data = pd.get_dummies(data, columns=['Crop_Type', 'Fertilizer'], drop_first=False)

# Separate features (X) and target variable (y)
X = data.drop(columns=['EF'])
y = data['EF']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features for models that require it
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Model 1: SVR ------------------

# Initialize and train the SVR model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions and evaluate the model
svm_pred = svm_model.predict(X_test)
svm_r2 = r2_score(y_test, svm_pred)
svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))

print(f"SVR R²: {svm_r2:.4f}")
print(f"SVR RMSE: {svm_rmse:.4f}")

# ------------------ Model 2: Ridge Regression ------------------

# Initialize and train the Ridge regression model
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Make predictions and evaluate the model
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

print(f"Ridge R²: {ridge_r2:.4f}")
print(f"Ridge RMSE: {ridge_rmse:.4f}")

# ------------------ Model 3: FNN (MLPRegressor) ------------------

# Initialize and train the FNN model (MLPRegressor)
fnn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, random_state=42)
fnn_model.fit(X_train, y_train)

# Make predictions and evaluate the model
fnn_pred = fnn_model.predict(X_test)
fnn_r2 = r2_score(y_test, fnn_pred)
fnn_rmse = np.sqrt(mean_squared_error(y_test, fnn_pred))

print(f"FNN R²: {fnn_r2:.4f}")
print(f"FNN RMSE: {fnn_rmse:.4f}")

# ------------------ Model 4: GBDT (Gradient Boosting) ------------------

# Set up parameter grid for GridSearchCV (Hyperparameter tuning)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5]
}

# Initialize and train the Gradient Boosting model
gbdt = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Get best parameters and make predictions
best_gbdt = grid_search.best_estimator_
gbdt_pred = best_gbdt.predict(X_test)
gbdt_r2 = r2_score(y_test, gbdt_pred)
gbdt_rmse = np.sqrt(mean_squared_error(y_test, gbdt_pred))

print(f"GBDT R²: {gbdt_r2:.4f}")
print(f"GBDT RMSE: {gbdt_rmse:.4f}")

# ------------------ Model 5: ResNet ------------------

# Define the ResNet model
def build_resnet(input_dim, hidden_units=64, learning_rate=0.001):
    inputs = Input(shape=(input_dim,))
    x = Dense(hidden_units, activation='relu')(inputs)
    
    # First residual block
    x_shortcut = x
    x = Dense(hidden_units, activation='relu')(x)
    x = Dense(hidden_units)(x)
    x = Add()([x, x_shortcut])  # Residual connection

    # Second residual block
    x_shortcut = x
    x = Dense(hidden_units, activation='relu')(x)
    x = Dense(hidden_units)(x)
    x = Add()([x, x_shortcut])  # Residual connection
    
    # Output layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Initialize and train the ResNet model
resnet_model = build_resnet(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
resnet_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Make predictions and evaluate the model
resnet_pred = resnet_model.predict(X_test)
resnet_r2 = r2_score(y_test, resnet_pred)
resnet_rmse = np.sqrt(mean_squared_error(y_test, resnet_pred))

print(f"ResNet R²: {resnet_r2:.4f}")
print(f"ResNet RMSE: {resnet_rmse:.4f}")

# ------------------ Model 6: Random Forest ------------------

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"Random Forest R²: {rf_r2:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}")

# ------------------ Save the results ------------------

# Save results for all models to CSV
results_df = pd.DataFrame({
    'Actual_EF': y_test,
    'SVR_Predicted_EF': svm_pred,
    'Ridge_Predicted_EF': ridge_pred,
    'FNN_Predicted_EF': fnn_pred,
    'GBDT_Predicted_EF': gbdt_pred,
    'ResNet_Predicted_EF': resnet_pred.flatten(),
    'RF_Predicted_EF': rf_pred
})

output_file = 'model_comparison_results.csv'
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
