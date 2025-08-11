import os
import io
import json
import requests
import runpod
import torch
import timm
import numpy as np
from typing import Any, Dict, Optional
from PIL import Image
from torchvision import transforms
import base64
import traceback

# Optional: Hugging Face for downloading private/public checkpoints
from huggingface_hub import snapshot_download

# Import GigaPath modules
import gigapath.slide_encoder as slide_encoder

# Global variables for model caching
TILE_ENCODER = None
SLIDE_ENCODER = None
DEVICE = None
TRANSFORM = None

# Environment variables
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/model")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODE = os.getenv("MODE", "inference")  # "inference" or "train"
USE_GPU = torch.cuda.is_available()

def get_device():
    """Get the compute device."""
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda" if USE_GPU else "cpu")
        print(f"Using device: {DEVICE}")
    return DEVICE

def get_transform():
    """Get the image transform pipeline."""
    global TRANSFORM
    if TRANSFORM is None:
        TRANSFORM = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return TRANSFORM

def lazy_load_tile_encoder():
    """Load the GigaPath tile encoder model once."""
    global TILE_ENCODER
    if TILE_ENCODER is not None:
        return TILE_ENCODER
    
    try:
        print("Loading GigaPath tile encoder...")
        # Load from HuggingFace hub
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            
        TILE_ENCODER = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        TILE_ENCODER = TILE_ENCODER.to(get_device())
        TILE_ENCODER.eval()
        
        print(f"Tile encoder loaded. Parameters: {sum(p.numel() for p in TILE_ENCODER.parameters())}")
        return TILE_ENCODER
    except Exception as e:
        print(f"Error loading tile encoder: {e}")
        raise

def lazy_load_slide_encoder():
    """Load the GigaPath slide encoder model once."""
    global SLIDE_ENCODER
    if SLIDE_ENCODER is not None:
        return SLIDE_ENCODER
    
    try:
        print("Loading GigaPath slide encoder...")
        # Load from HuggingFace hub with global pooling
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            
        SLIDE_ENCODER = slide_encoder.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", 
            "gigapath_slide_enc12l768d", 
            1536, 
            global_pool=True
        )
        SLIDE_ENCODER = SLIDE_ENCODER.to(get_device())
        SLIDE_ENCODER.eval()
        
        print(f"Slide encoder loaded. Parameters: {sum(p.numel() for p in SLIDE_ENCODER.parameters())}")
        return SLIDE_ENCODER
    except Exception as e:
        print(f"Error loading slide encoder: {e}")
        raise

def download_input(url: str) -> bytes:
    """Download input from URL."""
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content

def process_image(image_data: bytes, encoder_type: str = "tile") -> Dict[str, Any]:
    """Process an image through GigaPath encoder."""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if encoder_type == "tile":
            # Use tile encoder for single image patches
            encoder = lazy_load_tile_encoder()
            transform = get_transform()
            
            # Prepare input
            input_tensor = transform(image).unsqueeze(0).to(get_device())
            
            # Run inference
            with torch.no_grad():
                output = encoder(input_tensor)
                
            # Convert to numpy for JSON serialization
            features = output.cpu().numpy().tolist()
            
            return {
                "encoder_type": "tile",
                "features_shape": list(output.shape),
                "features": features,
                "device": str(get_device())
            }
            
        elif encoder_type == "slide":
            # For slide encoder, we'd need multiple tile features
            # This is a placeholder - real implementation would tile the WSI
            return {
                "encoder_type": "slide",
                "message": "Slide encoder requires WSI tiling - not implemented in this demo",
                "device": str(get_device())
            }
        else:
            return {"error": f"Unknown encoder type: {encoder_type}"}
            
    except Exception as e:
        return {"error": f"Processing error: {str(e)}", "traceback": traceback.format_exc()}

def handler(event):
    """
    RunPod handler function.
    
    Expected input:
    {
      "input": {
        "image_url": "https://.../image.png",  # URL to image
        "image_base64": "base64_string",       # OR base64 encoded image
        "encoder_type": "tile",                # "tile" or "slide" (default: "tile")
        "mode": "inference"                    # "inference" or "train" (default: "inference")
      }
    }
    """
    try:
        # Parse input
        data = event.get("input", {}) if isinstance(event, dict) else {}
        
        # Get mode
        mode = data.get("mode", MODE)
        if mode not in ("inference", "train"):
            return {"error": f"Unsupported mode '{mode}'"}
        
        if mode == "train":
            return {
                "error": "Training should run on Cloud GPUs/Clusters, not Serverless.",
                "suggestion": "Use RunPod GPU Cloud or Pods for training."
            }
        
        # Get encoder type
        encoder_type = data.get("encoder_type", "tile")
        
        # Get image data
        image_data = None
        if "image_url" in data:
            print(f"Downloading image from URL: {data['image_url']}")
            image_data = download_input(data["image_url"])
        elif "image_base64" in data:
            print("Decoding base64 image")
            image_data = base64.b64decode(data["image_base64"])
        else:
            return {"error": "Either 'image_url' or 'image_base64' is required"}
        
        # Process image
        result = process_image(image_data, encoder_type)
        
        return {
            "status": "success" if "error" not in result else "error",
            "result": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Test function for local testing
def test_handler():
    """Test the handler locally."""
    # Test with a sample image
    test_event = {
        "input": {
            "image_url": "https://raw.githubusercontent.com/prov-gigapath/prov-gigapath/main/images/prov_normal_000_1.png",
            "encoder_type": "tile"
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
    return result

# RunPod Serverless entrypoint
if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.getenv("RUNPOD_POD_ID"):
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Running test locally...")
        test_handler()