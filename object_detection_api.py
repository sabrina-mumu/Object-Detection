from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from object_detection import ObjectDetection, InvalidVideoError, DownloadFailedError

app = FastAPI(title="Object Detection API")
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ObjectDetection instance with the model path
object_detector = ObjectDetection('yolov8l-oiv7.pt')

class VideoURL(BaseModel):
    """
    Model for the request body of the /object_detection/ endpoint.

    Attributes:
        url (str): The URL of the video to be processed.
    """
    url: str

@app.post("/object_detection/")
async def process_video(request: VideoURL):
    """
    Endpoint to process a video for object detection.

    This endpoint accepts a URL pointing to a video, verifies its accessibility, 
    and then processes the video using the ObjectDetection instance.

    **Args:**
        request (VideoURL): The request body containing the video URL.

    **Returns:**
        **dict:** A dictionary containing the status and result of the object detection.

            - If successful, returns {"status": 1, "data": output_data}

            - If the URL is invalid or inaccessible, returns {"status": 0, "error_type": "invalid_video", "detail": "Invalid URL or the file is not accessible"}
            
            - If the video file type is invalid, returns {"status": 0, "error_type": "invalid_video", "detail": "Invalid video file type."}
            
            - If the video download fails, returns {"status": 0, "error_type": "download_failed", "detail": "Failed to download video from URL."}
            
            - For other HTTP exceptions, returns {"status": 0, "error_type": "error", "detail": "<HTTPException message>"}
            

            - For network-related exceptions, returns {"status": 0, "error_type": "error", "detail": "<RequestException message>"}
            
            - For value errors, returns {"status": 0, "error_type": "error", "detail": "<ValueError message>"}
            
            - For unexpected exceptions, returns {"status": 0, "error_type": "error", "detail": "An unexpected error occurred: <Exception message>"}
    """
    file_url = request.url

    try:
        # Check if the URL is valid and accessible
        response = requests.head(file_url, timeout=10)
        if response.status_code != 200:
            return {
                "status": 0,
                "status_code": response.status_code,
                "error_type": "invalid_video",
                "detail": "Invalid URL or the file is not accessible"
            }

        # Run the prediction function
        output_data = object_detector.predict(file_url, notFromGui=True)
        
        if "error" in output_data:
            raise HTTPException(status_code=400, detail=output_data.get("error_message", "Unknown error"))

        return {"status": 1, "data": output_data}

    except InvalidVideoError:
        return {
            "status": 0,
            "error_type": "invalid_video",
            "detail": "Invalid video file type."
        }
    
    except DownloadFailedError:
        return {
            "status": 0,
            "error_type": "download_failed",
            "detail": "Failed to download video from URL."
        }
    
    # except NoObjectsDetectedError:
    #     return {
    #         "status": 0,
    #         "error_type": "no_objects",
    #         "detail": "No objects detected from the image or video."
    #     }

    except HTTPException as e:
        return {
            "status": 0,
            "error_type": "error",
            "detail": str(e)
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": 0,
            "error_type": "error",
            "detail": str(e)
        }

    except ValueError as e:
        return {
            "status": 0,
            "error_type": "error",
            "detail": str(e)
        }

    except Exception as e:
        return {
            "status": 0,
            "error_type": "error",
            "detail": f"An unexpected error occurred: {str(e)}"
        }
