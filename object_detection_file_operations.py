import os
import boto3
import cv2
import numpy as np
from urllib.parse import urlparse
# from dotenv import load_dotenv
import requests
from datetime import datetime
from uuid import uuid4


class FileOperationsObject:
    """
    A class for handling file operations with AWS S3 and local temporary files.

    Attributes:
        s3_client (boto3.client): The S3 client used for interacting with the S3 bucket.
        s3_bucket_name (str): The name of the S3 bucket.
    """

    def __init__(self):
        """
        Initializes the FileOperationsObject instance by loading environment variables,
        setting up the S3 client, and configuring the S3 bucket name.
        """
        # load_dotenv()
        self.s3_client = self.initialize_s3_client()
        self.s3_bucket_name = 'fanfaretcs'
        self.video_root_folder = 'obj_videos'
        os.makedirs(self.video_root_folder, exist_ok=True)

    def initialize_s3_client(self) -> boto3.client:
        """
        Initializes the S3 client with credentials from environment variables.

        Returns:
            boto3.client: The initialized S3 client.
        """
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )

    def save_frame_to_s3(self, frame: np.ndarray, frame_number: int, folder: str) -> str:
        """
        Encodes a frame as a JPEG image and uploads it to the specified S3 bucket.

        Args:
            frame (np.ndarray): The frame to be saved, represented as a NumPy array.
            frame_number (int): The number of the frame, used for naming the file.
            folder (str): The folder within the S3 bucket where the frame will be stored.

        Returns:
            str: The S3 key for the uploaded frame.

        Raises:
            ValueError: If the upload to S3 fails.
        """
        _, buffer = cv2.imencode('.jpg', frame)
        frame_key = f"ai-assets/{folder}/frame_{frame_number:06d}.jpg"
        print(f"Uploading frame {frame_number} to S3 with key: {frame_key}")
        try:
            self.s3_client.put_object(Bucket=self.s3_bucket_name, Key=frame_key, Body=buffer.tobytes())
            print(f"Successfully uploaded frame {frame_number} to S3")
        except Exception as e:
            print(f"Failed to upload frame to S3: {e}")
            raise ValueError(f"Failed to upload frame to S3: {e}")
        return frame_key
    

    
    def generateUUID():
        return str(uuid4())


    def download_video(self, url: str) -> str:
        """
        Downloads a video from the specified URL and saves it to the obj_videos folder.
        """
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            # Use UUID to avoid filename collisions
            unique_id = FileOperationsObject.generateUUID()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_root_folder, f"{timestamp}_{unique_id}.mp4")

            total_size = int(response.headers.get('Content-Length', 0))  # Get file size if available
            downloaded_size = 0
            
            with open(video_path, 'wb') as video_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        video_file.write(chunk)
                        downloaded_size += len(chunk)
            
            if total_size and downloaded_size != total_size:
                raise ValueError(f"Incomplete download: Expected {total_size} bytes, got {downloaded_size} bytes.")
            
            print(f"Video downloaded and saved as: {video_path}")
            return video_path
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download the video from the URL: {e}")

    def save_video_buffer(self, buffer: bytes) -> str:
        """
        Saves a video buffer to a file in the obj_videos folder and returns its path.

        Args:
            buffer (bytes): The video data to be saved.

        Returns:
            str: The path to the video file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_root_folder, f"{timestamp}.mp4")
        with open(video_path, 'wb') as f:
            f.write(buffer)
        return video_path

    @staticmethod
    def get_file_extension(url: str) -> str:
        """
        Extracts and returns the file extension from a given URL.

        Args:
            url (str): The URL from which to extract the file extension.

        Returns:
            str: The file extension, including the dot (e.g., '.jpg').
        """
        parsed_url = urlparse(url)
        return os.path.splitext(parsed_url.path)[1].lower()
    
    @staticmethod
    def check_file_type(file_extension: str) -> str:
        """
        Determines the type of file based on its extension.

        Args:
            file_extension (str): The file extension (including the dot, e.g., '.jpg').

        Returns:
            str: A string indicating the file type ('image', 'video', or 'unsupported').
        """
        image_extensions = {'.jpg', '.jpeg', '.png'}
        video_extensions = {'.mp4', '.avi', '.avm'}
        
        if file_extension in image_extensions:
            return 'image'
        elif file_extension in video_extensions:
            return 'video'
        else:
            return 'unsupported'
