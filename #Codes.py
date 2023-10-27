from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have already loaded and preprocessed your data
X = # Independent variables
y = # Dependent variable (accident severity)

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model for future use
import joblib
joblib.dump(model, 'accident_severity_model.pkl')
# Load the saved model
loaded_model = joblib.load('accident_severity_model.pkl')

# Create a hypothetical set of independent variables
hypothetical_data = # Your data in the same format as the training data

# Make a prediction
predicted_severity = loaded_model.predict([hypothetical_data])

print(f"Predicted Accident Severity: {predicted_severity[0]}")
