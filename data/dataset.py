import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.optical_flow import get_optical_flow_feature_bb, get_optical_flow_feature
import pickle
import logging
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)

class AVADataset(Dataset):
    def __init__(self, dataframe, root_dir, max_frames=16, flow_bins=16, transform=None, config=None,
                 cache_dir=None, max_memory_cache_size=1000, enable_disk_cache=True, 
                 cache_compression=True, precompute_all=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.flow_bins = flow_bins
        self.transform = transform
        self.config = config
        
        # Caching configuration
        self.cache_dir = cache_dir or os.path.join(root_dir, ".flow_cache")
        self.max_memory_cache_size = max_memory_cache_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_compression = cache_compression
        
        # Memory cache using OrderedDict for LRU behavior
        self.flow_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory if needed
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Optional: precompute all flow features
        if precompute_all:
            self._precompute_all_features()

    def _get_cache_key(self, folder_name, frame_paths):
        """Generate a unique cache key based on folder name and frame paths"""
        # Include key parameters in hash to ensure cache validity
        key_data = f"{folder_name}_{self.max_frames}_{self.flow_bins}"
        # Add file modification times for cache invalidation
        if len(frame_paths) > 0:
            try:
                mtime = str(max(os.path.getmtime(p) for p in frame_paths[:3]))  # Sample first 3 files
                key_data += f"_{mtime}"
            except (OSError, IndexError):
                pass
        
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_disk_cache_path(self, cache_key):
        """Get the disk cache file path for a given cache key"""
        filename = f"flow_{cache_key}.{'pkl.gz' if self.cache_compression else 'pkl'}"
        return os.path.join(self.cache_dir, filename)

    def _load_from_disk_cache(self, cache_key):
        """Load flow feature from disk cache"""
        if not self.enable_disk_cache:
            return None
            
        cache_path = self._get_disk_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
            
        try:
            if self.cache_compression:
                import gzip
                with gzip.open(cache_path, 'rb') as f:
                    flow_feat = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    flow_feat = pickle.load(f)
            
            logger.debug(f"Loaded flow feature from disk cache: {cache_key}")
            return flow_feat
        except Exception as e:
            logger.warning(f"Failed to load from disk cache {cache_path}: {e}")
            return None

    def _save_to_disk_cache(self, cache_key, flow_feat):
        """Save flow feature to disk cache"""
        if not self.enable_disk_cache:
            return
            
        cache_path = self._get_disk_cache_path(cache_key)
        try:
            if self.cache_compression:
                import gzip
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(flow_feat, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(flow_feat, f)
            
            logger.debug(f"Saved flow feature to disk cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to disk cache {cache_path}: {e}")

    def _manage_memory_cache(self, cache_key, flow_feat):
        """Manage memory cache with LRU eviction"""
        # If key already exists, move to end (most recently used)
        if cache_key in self.flow_cache:
            self.flow_cache.move_to_end(cache_key)
            return
        
        # Add new item
        self.flow_cache[cache_key] = flow_feat
        
        # Evict least recently used items if cache is full
        while len(self.flow_cache) > self.max_memory_cache_size:
            oldest_key, oldest_value = self.flow_cache.popitem(last=False)
            # Optionally save evicted item to disk
            if self.enable_disk_cache:
                self._save_to_disk_cache(oldest_key, oldest_value)
            logger.debug(f"Evicted from memory cache: {oldest_key}")

    def _get_flow_feature(self, folder_name, frame_paths):
        """Get flow feature with multi-level caching"""
        cache_key = self._get_cache_key(folder_name, frame_paths)
        
        # Check memory cache first
        if cache_key in self.flow_cache:
            self.cache_hits += 1
            self.flow_cache.move_to_end(cache_key)  # Mark as recently used
            logger.debug(f"Memory cache hit: {cache_key}")
            return self.flow_cache[cache_key]
        
        # Check disk cache
        flow_feat = self._load_from_disk_cache(cache_key)
        if flow_feat is not None:
            self.cache_hits += 1
            # Add to memory cache
            self._manage_memory_cache(cache_key, flow_feat)
            logger.debug(f"Disk cache hit: {cache_key}")
            return flow_feat
        
        # Cache miss - compute flow feature
        self.cache_misses += 1
        logger.debug(f"Cache miss, computing flow: {cache_key}")
        
        try:
            try:
                video_folder = os.path.dirname(frame_paths[0])
                bb_cache_dir = str(video_folder).replace(folder_name, ".bb_cache")
                flow_feat = get_optical_flow_feature_bb(video_folder, bins=self.flow_bins, bb_cache_dir=bb_cache_dir)
            except Exception as e:
                print("Fail", e, folder_name)
                flow_feat = get_optical_flow_feature(frame_paths, bins=self.flow_bins)
            flow_feat = torch.tensor(flow_feat, dtype=torch.float32)
            
            # Cache the computed feature
            self._manage_memory_cache(cache_key, flow_feat)
            if self.enable_disk_cache:
                self._save_to_disk_cache(cache_key, flow_feat)
                
            return flow_feat
        except Exception as e:
            logger.error(f"Failed to compute flow feature for {folder_name}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(self.flow_bins, dtype=torch.float32)

    def _precompute_all_features(self):
        """Precompute all flow features (useful for training)"""
        logger.info(f"Precomputing flow features for {len(self.df)} samples...")
        
        for idx in range(len(self.df)):
            try:
                video_info = self.df.iloc[idx]
                folder_name = video_info["file_name"]
                
                video_folder = os.path.join(self.root_dir, folder_name)
                if not os.path.exists(video_folder):
                    logger.warning(f"Video folder not found: {video_folder}")
                    continue
                    
                frame_files = sorted([
                    f for f in os.listdir(video_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                
                if len(frame_files) == 0:
                    logger.warning(f"No frames found in: {video_folder}")
                    continue
                
                # Apply same frame sampling logic as __getitem__
                if len(frame_files) >= self.max_frames:
                    frame_files = frame_files[-self.max_frames:]
                else:
                    indices = torch.linspace(0, len(frame_files) - 1, self.max_frames).round().long()
                    indices = indices.clamp(max=len(frame_files) - 1)
                    frame_files = [frame_files[i] for i in indices]
                
                frame_paths = [os.path.join(video_folder, f) for f in frame_files]
                
                # This will compute and cache the flow feature
                self._get_flow_feature(folder_name, frame_paths)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Precomputed {idx + 1}/{len(self.df)} features")
                    
            except Exception as e:
                logger.error(f"Error precomputing features for index {idx}: {e}")
        
        logger.info(f"Precomputation complete. Cache stats: {self.get_cache_stats()}")

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'memory_cache_size': len(self.flow_cache),
            'max_memory_cache_size': self.max_memory_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'disk_cache_enabled': self.enable_disk_cache,
            'cache_dir': self.cache_dir if self.enable_disk_cache else None
        }
        
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            disk_cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('flow_')]
            stats['disk_cache_files'] = len(disk_cache_files)
        
        return stats

    def clear_cache(self, memory_only=False):
        """Clear cache (memory and optionally disk)"""
        self.flow_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        if not memory_only and self.enable_disk_cache and os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Cleared both memory and disk cache")
        else:
            logger.info("Cleared memory cache")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_info = self.df.iloc[idx]
        folder_name = video_info["file_name"]
        label = int(video_info["risk"])

        video_folder = os.path.join(self.root_dir, folder_name)
        frame_files = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Pad or truncate to max_frames
        if len(frame_files) >= self.max_frames:
            frame_files = frame_files[-self.max_frames:]
        else:
            # frame_files += [frame_files[-1]] * (self.max_frames - len(frame_files))
            indices = torch.linspace(0, len(frame_files) - 1, self.max_frames).round().long()
            indices = indices.clamp(max=len(frame_files) - 1)  # avoid overflow
            frame_files = [frame_files[i] for i in indices]
            

        frame_paths = [os.path.join(video_folder, f) for f in frame_files]
        frames = [Image.open(p).convert("RGB") for p in frame_paths]

        # Apply synchronized transform
        video_tensor = self.transform(frames)  # [T, C, H, W]

        # Get flow feature using enhanced caching system
        flow_feat = self._get_flow_feature(folder_name, frame_paths)

        return video_tensor, torch.tensor(label), folder_name, flow_feat

