import time
import requests
import os
import tempfile
import io
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO
# Import VideoFromFile class for ComfyUI video handling
try:
    # Try the newer import path first
    from comfy_extras.nodes_video import VideoFromFile
except ImportError:
    try:
        # Fallback to older import path
        from comfy.model_management import VideoFromFile
    except ImportError:
        # Final fallback - create a simple wrapper
        class VideoFromFile:
            def __init__(self, video_io):
                self.video_io = video_io
                self.video_data = video_io.getvalue() if hasattr(video_io, 'getvalue') else video_io
            
            def get_dimensions(self):
                # Return default dimensions if we can't determine them
                return (1280, 720)  # width, height

class Modelverse_WanAII2V:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "Convert to video","tooltip": "Text prompt to guide video generation"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE,),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "First frame image URL (use either this OR first_frame_image, not both)"}),
                "last_frame_image": (IO.IMAGE,),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "Optional: URL for the last frame of the video"}),
                "negative_prompt": (IO.STRING, {"multiline": True, "default": "low quality, blurry","tooltip": "Negative prompt to avoid unwanted content"}),
                "resolution": (["720P", "480P"], {"default": "720P", "tooltip": "Output video resolution"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Random seed for reproducible results"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_video"
    CATEGORY = "UCLOUD_MODELVERSE"

    def generate_video(self, client, prompt, first_frame_image=None, first_frame_url="", last_frame_image=None, last_frame_url="", negative_prompt="", resolution="720P", seed=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set in the client")
            
        mv_client = ModelverseClient(api_key)

        # Prepare the input data
        task_input = {"prompt": prompt}
        
        # Validate first frame input - must provide either image or URL, but not both
        has_url = first_frame_url and first_frame_url.strip()
        has_image = first_frame_image is not None
        
        if has_url and has_image:
            raise ValueError("Please provide either first_frame_image OR first_frame_url, not both")
        elif not has_url and not has_image:
            raise ValueError("Must provide either first_frame_image or first_frame_url")
        
        # Handle first frame
        if has_url:
            task_input["first_frame_url"] = first_frame_url.strip()
            print(f"Using first frame URL: {first_frame_url}")
        else:
            # Convert IMAGE tensor to base64
            first_frame_base64 = image_to_base64(first_frame_image)
            if not first_frame_base64:
                raise ValueError("Failed to convert first frame image to base64")
            task_input["first_frame_url"] = first_frame_base64
            print("Using first frame from IMAGE input (converted to base64)")

        # Handle last frame (optional)
        if last_frame_url and last_frame_url.strip():
            task_input["last_frame_url"] = last_frame_url.strip()
            print(f"Using last frame URL: {last_frame_url}")
        elif last_frame_image is not None:
            # Convert IMAGE tensor to base64
            last_frame_base64 = image_to_base64(last_frame_image)
            if last_frame_base64:
                task_input["last_frame_url"] = last_frame_base64
                print("Using last frame from IMAGE input (converted to base64)")

        # Add negative prompt if provided
        if negative_prompt and negative_prompt.strip():
            task_input["negative_prompt"] = negative_prompt.strip()

        # Set parameters
        parameters = {
            "resolution": resolution,
            "duration": 5,  # Fixed as per documentation
            "seed": seed,
        }

        print(f"Submitting I2V task with model: Wan-AI/Wan2.2-I2V")
        print(f"Parameters: resolution={resolution}, seed={seed}")

        # 1. Submit the task
        submit_res = mv_client.submit_task("Wan-AI/Wan2.2-I2V", task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res.get('request_id')}")

        print(f"Task submitted successfully with ID: {task_id}")

        # 2. Poll for the result
        video_url = ""
        max_retries = 120  # Maximum 10 minutes (120 * 5 seconds)
        retry_count = 0
        
        while True:
            try:
                status_res = mv_client.get_task_status(task_id)
                task_status = status_res.get("output", {}).get("task_status")

                if task_status == "Success":
                    urls = status_res.get("output", {}).get("urls", [])
                    if urls and len(urls) > 0:
                        video_url = urls[0]
                        print(f"Task completed successfully! Video URL: {video_url}")
                        break
                    else:
                        raise Exception("Task succeeded but no video URL was returned.")
                        
                elif task_status == "Failure":
                    error_message = status_res.get("output", {}).get("error_message", "Unknown error")
                    raise Exception(f"Task failed: {error_message}")
                    
                elif task_status in ["Pending", "Running"]:
                    print(f"Task {task_id} is {task_status}, waiting... ({retry_count + 1}/{max_retries})")
                    time.sleep(5)  # Wait for 5 seconds before polling again
                    retry_count += 1
                else:
                    raise Exception(f"Unknown task status: {task_status}")
                    
            except Exception as e:
                if "Task failed" in str(e) or "Unknown task status" in str(e):
                    raise e
                print(f"Error checking task status: {e}, retrying...")
                retry_count += 1
                time.sleep(5)
        
        if not video_url:
            raise Exception(f"Task timed out after {max_retries * 5} seconds")
        
        # 3. Download video
        print(f"Downloading video from: {video_url}")
        try:
            response = requests.get(video_url, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download video: {e}")
        
        # Extract video data
        video_data = response.content
        
        if not video_data:
            raise Exception("No video data was returned")
        
        print("Video generation completed successfully")
        print(f"Downloaded video data size: {len(video_data)} bytes")
        
        # Convert video data to BytesIO object
        video_io = io.BytesIO(video_data)
        
        # Ensure the BytesIO object is at the beginning
        video_io.seek(0)
        
        # Return VideoFromFile object
        try:
            video_obj = VideoFromFile(video_io)
            print("VideoFromFile object created successfully")
            return (video_obj,)
        except Exception as e:
            print(f"Error creating VideoFromFile object: {e}")
            # If VideoFromFile fails, try returning the raw video data
            # This is a fallback that might work with some ComfyUI versions
            return (video_data,)


NODE_CLASS_MAPPINGS = {
    "Modelverse_WanAII2V": Modelverse_WanAII2V
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_WanAII2V": "Modelverse Wan-AI I2V"
}
