import os
import boto3
import cv2
import requests
from ultralytics import YOLO
from typing import Optional, List, Set
import numpy as np
# from dotenv import load_dotenv
from object_detection_file_operations import FileOperationsObject
import botocore.exceptions
import boto3.exceptions
import shutil
from collections import defaultdict

# Custom Exceptions
class InvalidVideoError(Exception):
    """Exception raised for invalid video file types."""
    def __init__(self, message="Invalid video file type."):
        self.message = message
        super().__init__(self.message)

class DownloadFailedError(Exception):
    """Exception raised when video download fails."""
    def __init__(self, message="Failed to download video from URL."):
        self.message = message
        super().__init__(self.message)

# class NoObjectsDetectedError(Exception):
#     """Exception raised when no objects are detected."""
#     def __init__(self, message="No objects detected."):
#         self.message = message
#         super().__init__(self.message)

class ObjectDetection:
    """
    A class to handle object detection using a YOLO model with image and video support.

    :param model_path: Path to the YOLO model file
    :type model_path: str
    """
    def __init__(self, model_path: str):
        """
        Initializes the ObjectDetection class.

        :param model_path: Path to the YOLO model file
        :type model_path: str
        """
        self.model = self.load_model(model_path)
        # load_dotenv()
        self.s3_client = self.initialize_s3_client()
        self.s3_bucket_name = 'fanfaretcs'
        self.processed_frames_buffer: Set[int] = set()
        self.frame_count: int = 0
        # print(f"Initialized S3 client with bucket: {self.s3_bucket_name}")

    def load_model(self, model_path: str) -> YOLO:
        """
        Loads the YOLO model from the specified path.

        :param model_path: Path to the YOLO model file
        :type model_path: str
        :return: Loaded YOLO model
        :rtype: YOLO
        :raises ValueError: If the model fails to load
        """
        try:
            return YOLO(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    def initialize_s3_client(self) -> boto3.client:
        """
        Initializes the S3 client using credentials from environment variables.

        :return: Initialized S3 client
        :rtype: boto3.client
        :raises ValueError: If AWS credentials are missing or incomplete
        """
        try:
            return boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION'),
                config=boto3.session.Config(connect_timeout=5, read_timeout=15)
            )
        except botocore.exceptions.NoCredentialsError:
            raise ValueError("AWS credentials not found. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        except botocore.exceptions.PartialCredentialsError:
            raise ValueError("Incomplete AWS credentials. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        except boto3.exceptions.Boto3Error as e:
            raise ValueError(f"An error occurred while initializing the S3 client: {e}")

    def read_image_from_url(self, url: str) -> np.ndarray:
        """
        Reads an image from a URL.

        :param url: URL of the image
        :type url: str
        :return: Decoded image as a NumPy array
        :rtype: np.ndarray
        :raises ValueError: If the image fails to decode
        :raises DownloadFailedError: If the image fails to download
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode the image from the URL.")
            return img
        except requests.exceptions.RequestException as e:
            raise DownloadFailedError(f"Failed to fetch the image from the URL: {e}")

    def process_image(self, img: np.ndarray, folder: str, notFromGui: bool) -> dict:
        """
        Processes a single image to detect objects.

        :param img: Image as a NumPy array
        :type img: np.ndarray
        :param folder: Folder name to save the processed image
        :type folder: str
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: bool
        :return: Detection results
        :rtype: dict
        :raises NoObjectsDetectedError: If no objects are detected
        """
        print("Processing image...")
        FileOperationsObject().save_frame_to_s3(img, 0, folder)
        results = self.model.predict(source=img, save=False, stream=True)
        response = self.extract_unique_items(results, notFromGui)
        
        # # Check if response contains no items and raise an exception
        # if 'message' in response and response['message'] == "No objects detected.":
        #     raise NoObjectsDetectedError()
        
        response['duration'] = "0h 0m 0s"  # Duration for image
        response['total_frame_count']= 1
        return response

    def process_video(self, video_url: str, folder: str, notFromGui: bool, skip: Optional[int] = None, batch_size: int = 1) -> dict:
        """
        Processes a video to detect objects.

        :param video_url: URL of the video
        :type video_url: str
        :param folder: Folder name to save the processed video frames
        :type folder: str
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: bool
        :param skip: Number of frames to skip between processing
        :type skip: Optional[int]
        :param batch_size: Number of frames to process in a batch
        :type batch_size: int
        :return: Detection results
        :rtype: dict
        :raises NoObjectsDetectedError: If no objects are detected
        :raises ValueError: If the video stream fails to open
        """

        # Reset internal state before starting to process a new video
        self.frame_count = 0
        self.processed_frames_buffer = set()


        # Download the video from the URL and save it as a temporary file
        video_path = FileOperationsObject().download_video(video_url)

        # Use the base name of the downloaded video file (without extension) as the folder name
        folder = os.path.splitext(os.path.basename(video_path))[0]

        # Open the video to get its frame rate
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video stream. The stream might be corrupted or incomplete.")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Set the skip value to the frame rate if it was not provided
        if skip is None:
            skip = fps
        response = self.process_video_frames(video_path, folder, notFromGui, skip, batch_size)
        
        # # Check if response contains no items and raise an exception
        # if 'message' in response and response['message'] == "No objects detected.":
        #     raise NoObjectsDetectedError()
        
        return response

    def process_video_frames(self, video_path: str, folder: str, notFromGui: bool, skip: int, batch_size: int) -> dict:
        """
        Processes video frames to detect objects.

        :param video_path: Path to the video file
        :type video_path: str
        :param folder: Folder name to save the processed video frames
        :type folder: str
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: bool
        :param skip: Number of frames to skip between processing
        :type skip: int
        :param batch_size: Number of frames to process in a batch
        :type batch_size: int
        :return: Detection results
        :rtype: dict
        :raises ValueError: If the video stream fails to open
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video stream. The stream might be corrupted or incomplete.")
        
        frame_count = 0
        unique_items: Set[str] = set()
        max_counts = defaultdict(int)  # To store the maximum count of each detected object
        frames: List[np.ndarray] = []

        # Get video duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        duration_str = self.format_duration(duration_sec)

        try:
            processed_frame_count=0 
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                
                self.frame_count += 1  # Increment the frame count
                
                if frame_count % skip == 0:
                    # frame_hash = self.hash_frame(frame)
                    # if frame_hash not in self.processed_frames_buffer:
                        # FileOperationsObject().save_frame_to_s3(frame, frame_count, folder)
                    frames.append(frame)
                    # self.processed_frames_buffer.add(frame_hash)
                    processed_frame_count+=1
                

                    if len(frames) >= batch_size:
                        combined_results = self.process_frame_batch(frames)
                        
                        # Update max_counts with the counts from combined_results
                        for item, count in combined_results.items():
                            max_counts[item] = max(max_counts[item], count)
                        
                        print("Processing frame no: ", frame_count)

                        frames = []  # Clear the frames buffer

                    
                frame_count += 1

            
            # Process remaining frames
            if frames:
                combined_results = self.process_frame_batch(frames) or {}
                for item, count in combined_results.items():
                    max_counts[item] = max(max_counts[item], count)


            response = {"max_counts": dict(max_counts)}
            # response = self.prepare_response(unique_items, notFromGui)
            response['total_frame_count'] = self.frame_count  # Add frame count to response
            response['duration'] = duration_str  # Add duration to response
            response['frame_rate'] = int(fps)
            response['processed_frame_count']= processed_frame_count
            return response

        finally:
            cap.release()
            # Clean up: delete the temporary video file and its containing folder
            if os.path.exists(video_path):
                shutil.rmtree(os.path.dirname(video_path), ignore_errors=True)

    def format_duration(self, seconds: float) -> str:
        """
        Formats the duration from seconds to 'Xh Ym Zs'.

        :param seconds: Duration in seconds
        :type seconds: float
        :return: Formatted duration
        :rtype: str
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"

    def process_frame_batch(self, frames: List[np.ndarray]) -> Set[str]:
        """
        Processes a batch of video frames to detect objects.

        :param frames: List of video frames as NumPy arrays
        :type frames: List[np.ndarray]
        :return: Set of unique detected objects
        :rtype: Set[str]
        """
        
        combined_results = defaultdict(int)  # To store counts of detected objects
        threshold = 0.40  # Confidence threshold
        predictions = self.model.predict(source=frames, save=False, stream=True, verbose=False)
        
        for result in predictions:
            counts = defaultdict(int)
            for box in result.boxes:
                if box.conf.item() >= threshold:  # Apply the threshold
                    class_index = int(box.cls.item())
                    if class_index < len(result.names):
                        item_name = result.names[class_index]
                        counts[item_name] += 1
            for item, count in counts.items():
                combined_results[item] = max(combined_results[item], count)        
        return combined_results

    def prepare_response(self, unique_items: Set[str], notFromGui: bool) -> dict:
        """
        Prepares the response based on detected objects.

        :param unique_items: Set of unique detected objects
        :type unique_items: Set[str]
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: bool
        :return: Detection results
        :rtype: dict
        """
        # if not unique_items:
            # pass
            # message = "No objects detected."
            # print(message)
            # return {"message": message} if notFromGui else {"items": []}

        if notFromGui:
            print("Detected items:", unique_items)
            return {"items": list(unique_items)}
        else:
            return list(unique_items)

    def extract_unique_items(self, results, notFromGui: bool) -> dict:
        """
        Extracts unique items from detection results.

        :param results: Model prediction results
        :type results: list
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: bool
        :return: Detection results
        :rtype: dict
        """
        unique_items: Set[str] = set()
        for result in results:
            for box in result.boxes:
                class_index = int(box.cls.item())
                unique_items.add(result.names[class_index])

        response = self.prepare_response(unique_items, notFromGui)
        response['frame_count'] = self.frame_count  # Add frame count to response
        return response

    @staticmethod
    def hash_frame(frame: np.ndarray) -> int:
        """
        Computes a hash for a given video frame.

        :param frame: Video frame as a NumPy array
        :type frame: np.ndarray
        :return: Hash value of the frame
        :rtype: int
        """
        return hash(frame.tobytes())

    def predict(self, input_source: str, skip: Optional[int]= None, notFromGui: Optional[bool] = False,  batch_size: Optional[int] = 1) -> dict:
        """
        Predicts objects in the input source, which can be an image or a video URL.

        :param input_source: URL of the image or video
        :type input_source: str
        :param notFromGui: Whether the processing is not from GUI
        :type notFromGui: Optional[bool]
        :param skip: Number of frames to skip between processing (for videos)
        :type skip: Optional[int]
        :param batch_size: Number of frames to process in a batch (for videos)
        :type batch_size: Optional[int]
        :return: Detection results
        :rtype: dict
        :raises ValueError: If the file type is unsupported
        """
        file_extension = FileOperationsObject().get_file_extension(input_source)
        file_type = FileOperationsObject().check_file_type(file_extension)
        if file_type == 'unsupported':
            raise ValueError("Unsupported file type.")

        folder = os.path.splitext(os.path.basename(input_source))[0]

        if file_type == 'image':
            img = self.read_image_from_url(input_source)
            return self.process_image(img, folder, notFromGui)
        elif file_type == 'video':
            return self.process_video(input_source, folder, notFromGui, skip= skip, batch_size= batch_size)

# Example usage:
# object_detection = ObjectDetection(model_path="path_to_your_model.pt")
# result = object_detection.predict(input_source="https://example.com/image_or_video_url")


