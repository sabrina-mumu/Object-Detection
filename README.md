# From Image/Video to Object
## About:
- This project contains the api for object detection

## Steps:
### Install necessary plugins
- Install virtual environment in the command prompt  `pip install virtualenv`
- Make a virtual environment in the directory  `python -m venv .venv`      (Here the environment name is .venv)
- Activate the environment  
	- For Windows `.venv\Scripts\activate`
	- For Unix `source .venv/bin/activate`
 - Download and install the necessary files  `pip install -r requirements.txt`
 - For linux os, ignore "python-magic-bin==0.4.14" installing
 - Download [yolov8l-oiv7.pt](https://drive.google.com/file/d/1NVJQjbxVMrWEcsqZahnIsBdoLfrLzqrE/view?usp=sharing) and put it in the root directory

>  If you want to use GPU acceleration you will need to
>  install pytorch with cuda enabled. For that go to this
> [website](https://pytorch.org/get-started/locally/) and install the cuda version  your device require.

 ### Run the server
 <!-- - Run cmmand `uvicorn api:api --reload --timeout-keep-alive 600 --limit-max-requests=52428800` -->
 - Run cmmand `uvicorn object_detection_api:app --reload`
 - Go to this link `localhost:8000/upload` [post method]
 - UI can be load from here:  `localhost:8000/` & `localhost:8000/docs`
 - On Postman go to the body and the Key parameter has to be `file`.

