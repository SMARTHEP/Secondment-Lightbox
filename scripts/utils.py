import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_dataframe(df,cols_to_use,n_past=14,n_future=1):
    updated_df = df.copy()

    updated_df = updated_df[cols_to_use] 
    updated_df_scaled = updated_df.copy()

    scalers = {}
    for column in updated_df.columns:
        scaler = MinMaxScaler()
        column_data = updated_df[column].values.reshape(-1, 1)  # Reshape the column to fit the scaler
        scalers[column] = scaler.fit(column_data)
        scaled_column = scaler.transform(column_data)
        updated_df_scaled[column] = scaled_column.flatten()

    # upd_scaler = MinMaxScaler()
    # upd_scaler = upd_scaler.fit(updated_df)
    # updated_df_scaled = upd_scaler.transform(updated_df)

    X = []
    Y = []
    n_past = n_past  # Number of past days we want to use to predict the future.
    n_future = n_future   # Number of days we want to look into the future based on the past days.
    for i in range(n_past, len(updated_df_scaled) - n_future + 1,n_future):
        X.append(updated_df_scaled.iloc[i - n_past:i, :].values)
        Y.append(updated_df_scaled.iloc[i:i + n_future, 3].values) #index 3 is Close

    return np.array(X), np.array(Y), scalers




def build_lstm_model(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_output1 = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    dropout_output1 = tf.keras.layers.Dropout(0.3)(lstm_output1)  # Adding dropout with a rate of 0.2
    lstm_output2 = tf.keras.layers.LSTM(128, return_sequences=True)(dropout_output1)
    dropout_output2 = tf.keras.layers.Dropout(0.3)(lstm_output2)  
    lstm_output3 = tf.keras.layers.LSTM(128, return_sequences=False)(dropout_output2)
    dropout_output3 = tf.keras.layers.Dropout(0.3)(lstm_output3)  

    flat_mean_output = tf.keras.layers.Dense(5, activation='relu')(dropout_output3) 
    output_mean = tf.keras.layers.Dense(1)(flat_mean_output)  # Output for mean prediction

    model = tf.keras.models.Model(inputs=inputs, outputs=output_mean)

    return model
