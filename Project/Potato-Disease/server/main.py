from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, mai zinda hu bsdk"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    print(file)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)