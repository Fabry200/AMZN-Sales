from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
import os


project_path = os.getcwd() 
mlflow.set_tracking_uri(f"file:///{project_path}/mlruns")
mlflow.set_experiment("prediction2")

def scaling():
    X = pd.read_csv('to_normalize.csv')
    
    mask = (X['time_to_start_shipping'] >= 0) & (X['time_to_ship'] >= 0)
    X = X[mask].copy() 
    
    X['order_date'] = pd.to_datetime(X['order_date'])
    X['order_day'] = X['order_date'].dt.day
    X['order_month'] = X['order_date'].dt.month
    
    y = X['time_to_arrive']
    X = X.drop(columns=['time_to_arrive', 'order_date', 'ship_date', 'delivery_date', 'time_to_start_shipping', 'time_to_ship'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    return X_train, y_train.to_numpy().reshape(-1,1), X_test, y_test.to_numpy().reshape(-1,1), scaler_X


def model_fitting(X_train, y_train):
    mlflow.autolog()

    dt_regressor= DecisionTreeRegressor(random_state=0)
    dt_regressor= dt_regressor.fit(X_train,y_train)

    dnn = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
    dnn.compile(optimizer='adam',
                loss='mae',
                metrics=['mae'])
    dnn.fit(X_train, y_train, epochs=30)





def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(14,)))

  for i in range(hp.Int("num_layers", 1,4)):
    model.add(
      tf.keras.layers.Dense(
      units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
      activation=hp.Choice(f"activation_{i}", ["relu","tanh"])      )
    )
    
  if hp.Boolean("dropout"):
    model.add(tf.keras.layers.Dropout(0.2))
  
  model.add(tf.keras.layers.Dense(1, ))
  
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mae',
                metrics=['mae'])
  return model


def custom_dnn(X_train, y_train):
    tuner = kt.Hyperband(model_builder,
                    objective='val_mae',
                    max_epochs=30,
                    factor=3,
                    project_name='arrival_date_dnn_trial2'
                    )
            
    tuner.search(X_train, y_train, epochs=30, validation_split=0.2)
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.summary()
    return model 


def save_model(X_train,y_train,X_test, y_test):
  model=custom_dnn(X_train, y_train)
  model.fit(
      X_train,
      y_train,
      epochs=27,
      validation_split=0.2,
      verbose=0
  )
  model.evaluate(X_test, y_test)
  tf.keras.models.save_model(model, 'custom_dnn_model.keras')






if __name__ == "__main__":
  X_train, y_train, X_test, y_test, scaler=scaling()
  #print(y_train)
  #model_fitting(X_train, y_train)
  #save_model(X_train, y_train, X_test, y_test)



  run_id = "720cbc1f282a4084a1e9ee76be453231"
  model_uri = f"runs:/{run_id}/model"


  decision_tree_model = mlflow.sklearn.load_model(model_uri)   
  dtm_predict = decision_tree_model.predict(X_test)
  print(mean_absolute_error(dtm_predict, y_test.ravel()))


  dnn_id = "dc4c12dc117d47ab82c22067749fe15c"
  dnn_uri = f"runs:/{dnn_id}/model"

  dnn_model = mlflow.pyfunc.load_model(dnn_uri)

  dnn_predict=dnn_model.predict(X_test)

  print(mean_absolute_error(dnn_predict, y_test.ravel()))
