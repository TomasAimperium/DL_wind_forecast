import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from lstm_wind.functions import prepro,agg
from lstm_wind.config import config
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from tensorflow import keras
import joblib


from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent






def train():
	print("############")
	print("loading data")
	print(" ")




	file = Path(BASE_DIR).joinpath(config.data_path)



	all_data = pd.read_csv(file,sep=",")
	trunc_days = config.days
	trunc_mins = trunc_days*24*60



	my_data = pd.DataFrame(prepro(all_data))


	station_name,mape_train,mape_val,mape_test,epochs,batch_size,train_size,test_size,r2_test,mse_test = [],[],[],[],[],[],[],[],[],[]




	print("############")
	print("training models")
	print(" ")




	for k,j in enumerate(my_data.columns):

		#data reshape
		station = my_data.iloc[:trunc_mins,k].values.reshape(trunc_mins,1)
		n_past = config.n_past
		y_col = station.shape[1]-1
		n_future = config.n_future	 

		#initialization
		data_X = []
		data_Y = []
		data_p_X = []
		data_p_Y = []	

		for i in range(n_past, len(station) - n_future + 1):
			data_X.append(station[i - n_past:i, 0:station.shape[1]])
			data_Y.append(station[i:i + n_future, y_col])
		data_X, data_Y = np.array(data_X), np.array(data_Y)	

		
		train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.33, random_state=42, shuffle=False)



		print(j)
		try:
			keras.backend.clear_session()
			lstm_model = Sequential()
			lstm_model.add(Input(shape=[train_X.shape[-2], train_X.shape[-1]]))
			lstm_model.add(Dense(10))
			lstm_model.add(LSTM(100, activation='tanh', input_shape=(train_X.shape[-1], train_X.shape[-2]), return_sequences=True))
			lstm_model.add(Dense(30))
			lstm_model.add(Dropout(0.2))
			lstm_model.add(LSTM(100, activation='tanh', return_sequences=False))
			lstm_model.add(Dense(10))
			lstm_model.add(Dense(train_Y.shape[1]))
			lstm_model.compile(optimizer='adam', loss='mape')
			history = lstm_model.fit(train_X, train_Y, epochs = config.epochs, batch_size = config.batch, validation_data=(test_X, test_Y), verbose=1)


			model_file_ = config.model_file + j + ".joblib"

			joblib.dump(lstm_model,Path(BASE_DIR).joinpath(model_file_))


			station_test = my_data.iloc[trunc_mins:,k]
			my_data_test = station_test.values.reshape(len(station_test),1)


			data_p_X = []
			data_p_Y = []



			for i in range(n_past, len(my_data_test) - n_future + 1, n_future):
			    data_p_X.append(my_data_test[i - n_past:i, 0:my_data_test.shape[1]])
			    data_p_Y.append(my_data_test[i:i + n_future, y_col])
			data_p_X, data_p_Y = np.array(data_p_X), np.array(data_p_Y)

			__forecast = lstm_model.predict(data_p_X)
			_forecast = __forecast
			_y_test = data_p_Y
			_forecast = _forecast.reshape(_forecast.shape[0]*_forecast.shape[1], 1)
			forecast_dummy = np.zeros((_forecast.shape[0], my_data_test.shape[-1]-1))
			_forecast = np.append(_forecast, forecast_dummy, axis=1)
			_forecast = _forecast[:,0]

			_y_test = _y_test.reshape(_y_test.shape[0]*_y_test.shape[1], 1)
			y_test_dummy = np.zeros((_y_test.shape[0], my_data_test.shape[-1]-1))
			_y_test = np.append(_y_test, y_test_dummy, axis=1)
			_y_test = _y_test[:,0]

			mape = keras.losses.MeanAbsolutePercentageError()


			station_name.append(j)
			mape_train.append(history.history['loss'][-1])
			mape_val.append(history.history['val_loss'][-1])
			mape_test.append(mape(_y_test, _forecast).numpy())
			r2_test.append(r2_score(_y_test, _forecast))
			mse_test.append(mean_squared_error(_y_test, _forecast))	
			epochs.append(config.epochs)
			batch_size.append(config.batch)
			train_size.append(len(station))
			test_size.append(len(station_test))
			           
			keras.backend.clear_session()
		except Exception as e: print(e) 
			





		


	outputs = {
	"station_name":station_name,
	"mape_train":mape_train,
	"mape_val": mape_val,
	"mape_test":mape_test,
	"r2_test":r2_test,
	"mse_test":mse_test,
	"epochs": epochs,
	"batch_size":batch_size,
	"train_size":train_size,
	"test_size": test_size
	}

	print("############")
	print("output")
	print(" ")
	print(outputs)

	return outputs

def predict(inputs):
	inp = pd.DataFrame(list(inputs.values())[1:]).T
	inp.columns = list(inputs.values())[0]

	station_name,forecast = [],[]

	for i,j in enumerate(inp.columns):
		try:
			modelo = config.model_file + j + ".joblib"
			lstm_model = joblib.load(Path(BASE_DIR).joinpath(modelo))
			output = lstm_model.predict(np.array(inp.loc[:,j]).reshape(-1,1))
			station_name.append(j)
			forecast.append(output[-1])
		except:
			print("Modelo no encontrado")
	print("funciona")

	prediction_list = pd.DataFrame(forecast,columns = station_name).to_dict()

#	outputs = {
#	"station_name":station_name,    
#	"forecast":forecast
#	}
    
	return prediction_list








