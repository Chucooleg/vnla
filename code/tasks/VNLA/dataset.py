import torch
from torch.utils.data import Dataset

class PanoramaDataset(Dataset):
    """Indoor panoramic view dataset"""

    def __init__(self, rm_labels, feature_ids, all_img_features):
        self.rm_labels = torch.tensor(rm_labels)
        self.feature_ids = feature_ids
        # already pytroch tensors
        self.all_img_features = all_img_features

    def __len__(self):
        return len(self.feature_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        long_id, viewix = self.feature_ids[idx][0], self.feature_ids[idx][1]
        feature = self.all_img_features[long_id][viewix, :]
        room_idx = self.rm_labels[idx]
        # sample = {'long_id': long_id, 'viewix': viewix, 'feature': torch.from_numpy(feature), 'room': torch.tensor(room_idx)}
        sample = {'long_id': long_id, 'viewix': viewix, 'feature': feature, 'room': room_idx}

        return sample
