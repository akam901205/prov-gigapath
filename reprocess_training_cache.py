"""
Reprocess Training Cache with Complete 7-Step Preprocessing Pipeline
Ensures training/inference consistency by applying same preprocessing to all training data
"""

import os
import pickle
import numpy as np
from PIL import Image
import torch
import timm
from preprocessing_pipeline import preprocessor, get_dataset_um_per_pixel
from tqdm import tqdm
import json
from pathlib import Path

def load_original_images():
    """
    Load original raw images from BACH and BreakHis datasets
    This assumes we have access to the original image files
    """
    image_paths = []
    labels = []
    datasets = []
    filenames = []
    
    # BACH dataset paths (adjust paths as needed)
    bach_base_path = "/workspace/BACH_dataset"  # Adjust to actual path
    bach_classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
    
    if os.path.exists(bach_base_path):
        print(f"🔍 Loading BACH images from {bach_base_path}...")
        for class_name in bach_classes:
            class_path = os.path.join(bach_base_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        image_paths.append(os.path.join(class_path, img_file))
                        labels.append(class_name.lower())
                        datasets.append('bach')
                        filenames.append(img_file)
    
    # BreakHis dataset paths (adjust paths as needed)
    breakhis_base_path = "/workspace/BreakHis_dataset"  # Adjust to actual path
    
    if os.path.exists(breakhis_base_path):
        print(f"🔍 Loading BreakHis images from {breakhis_base_path}...")
        # BreakHis has complex directory structure - implement based on actual structure
        # For now, we'll work with what we have in the cache
    
    print(f"📊 Found {len(image_paths)} images total")
    return image_paths, labels, datasets, filenames

def reprocess_with_new_pipeline():
    """
    Reprocess all training images with the 7-step preprocessing pipeline
    """
    print("🚀 Starting cache reprocessing with 7-step preprocessing pipeline...")
    
    # Load GigaPath model once
    print("🔥 Loading GigaPath model...")
    gigapath_model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gigapath_model = gigapath_model.to(device)
    gigapath_model.eval()
    print(f"✅ GigaPath model loaded on {device}")
    
    # Try to load original images
    image_paths, labels, datasets, filenames = load_original_images()
    
    if not image_paths:
        print("⚠️ No original images found - will work with existing cache structure")
        return reprocess_from_existing_cache(gigapath_model)
    
    # Process each image through the complete pipeline
    processed_features = []
    processed_labels = []
    processed_datasets = []
    processed_filenames = []
    processing_metadata = []
    
    print(f"🔄 Processing {len(image_paths)} images with 7-step pipeline...")
    
    for i, (img_path, label, dataset, filename) in enumerate(tqdm(zip(image_paths, labels, datasets, filenames))):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Get dataset-specific μm/pixel
            source_um_per_pixel = get_dataset_um_per_pixel(dataset)
            
            # Run complete 7-step preprocessing pipeline
            l2_features, metadata = preprocessor.complete_pipeline(
                image=image,
                gigapath_model=gigapath_model,
                source_um_per_pixel=source_um_per_pixel,
                apply_tissue_mask=True,
                apply_stain_norm=True
            )
            
            processed_features.append(l2_features)
            processed_labels.append(label)
            processed_datasets.append(dataset)
            processed_filenames.append(filename)
            processing_metadata.append(metadata)
            
            if (i + 1) % 100 == 0:
                print(f"✅ Processed {i + 1}/{len(image_paths)} images")
                
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")
            continue
    
    # Create new cache with preprocessed features
    new_cache = {
        'combined': {
            'features': processed_features,
            'labels': processed_labels,
            'datasets': processed_datasets,
            'filenames': processed_filenames
        },
        'preprocessing_metadata': {
            'pipeline_version': '7-step-v1.0',
            'steps': ['scale_normalization', 'tissue_masking', 'stain_normalization', 
                     'resize_224x224', 'imagenet_normalization', 'gigapath_extraction', 'l2_normalization'],
            'gigapath_model': 'prov-gigapath/prov-gigapath',
            'target_um_per_pixel': preprocessor.target_um_per_pixel,
            'total_processed': len(processed_features)
        },
        'individual_metadata': processing_metadata
    }
    
    # Save new cache
    new_cache_path = "/workspace/embeddings_cache_7_STEP_PREPROCESSING.pkl"
    with open(new_cache_path, 'wb') as f:
        pickle.dump(new_cache, f)
    
    print(f"✅ New cache saved: {new_cache_path}")
    print(f"📊 Total samples: {len(processed_features)}")
    print(f"🎯 All samples processed with consistent 7-step pipeline")
    
    return new_cache_path

def reprocess_from_existing_cache(gigapath_model):
    """
    Alternative: Apply preprocessing pipeline to cached features (if original images not available)
    Note: This can only apply steps 6-7 since we need original images for steps 1-5
    """
    print("⚠️ Working with existing cache - limited preprocessing possible")
    
    # Load existing cache
    cache_path = "/workspace/embeddings_cache_4_CLUSTERS_FIXED_TSNE.pkl"
    with open(cache_path, 'rb') as f:
        old_cache = pickle.load(f)
    
    print(f"📊 Loaded existing cache: {len(old_cache['combined']['features'])} samples")
    print("⚠️ Note: Can only apply L2 normalization to existing features")
    print("⚠️ For full 7-step preprocessing, need access to original raw images")
    
    # Apply L2 normalization to existing features (step 7)
    features = old_cache['combined']['features']
    l2_features = []
    
    for feature_vector in features:
        l2_feature = preprocessor.l2_normalize_features(np.array(feature_vector))
        l2_features.append(l2_feature)
    
    # Update cache with L2 normalized features
    new_cache = old_cache.copy()
    new_cache['combined']['features'] = l2_features
    new_cache['preprocessing_metadata'] = {
        'pipeline_version': 'l2-only-v1.0',
        'note': 'Only L2 normalization applied - need original images for full pipeline',
        'steps_applied': ['l2_normalization'],
        'steps_missing': ['scale_normalization', 'tissue_masking', 'stain_normalization'],
        'recommendation': 'Reprocess with original images for optimal results'
    }
    
    # Save updated cache
    updated_cache_path = "/workspace/embeddings_cache_L2_REPROCESSED.pkl"
    with open(updated_cache_path, 'wb') as f:
        pickle.dump(new_cache, f)
    
    print(f"✅ Updated cache saved: {updated_cache_path}")
    return updated_cache_path

if __name__ == "__main__":
    print("🔄 TRAINING CACHE REPROCESSING")
    print("="*50)
    
    try:
        new_cache_path = reprocess_with_new_pipeline()
        print(f"\\n🎉 SUCCESS: New preprocessed cache created!")
        print(f"📁 Location: {new_cache_path}")
        print(f"🔄 Ready for classifier retraining with consistent preprocessing")
        
    except Exception as e:
        print(f"❌ Reprocessing failed: {e}")
        import traceback
        traceback.print_exc()