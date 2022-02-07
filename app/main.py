from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from lstm_wind.model import train,predict
from lstm_wind.model import predict

app = FastAPI()

class StockIn(BaseModel):
    inputs: dict
    

#class StockOut(StockIn):
#    forecast: dict




@app.get("/")
async def root():
	return {"message":"Todo correcto con lstm"}



@app.post("/predict")#, response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    inputs = payload.inputs

    prediction_list = predict(inputs)

    print(prediction_list)
#    if not prediction_list:
#        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {"inputs": inputs, "lstm_forecast": prediction_list}

    #print(response_object)

    return response_object
    #return prediction_list


# @app.post("/train")#, response_model=StockOut, status_code=200)
# def get_ptrain():
#     return train()


#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)

#uvicorn app.main:app --reload
