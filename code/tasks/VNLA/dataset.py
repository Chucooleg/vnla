import torch
from torch.utils.data import Dataset

class PanoramaDataset(Dataset):
    """Indoor panoramic view dataset"""

    def __init__(self, rm_labels, feature_ids, all_img_features, image_extent):
        self.rm_labels = torch.tensor(rm_labels)
        self.feature_ids = feature_ids
        # already pytroch tensors
        self.all_img_features = all_img_features
        self.image_extent = image_extent

    def __len__(self):
        return len(self.feature_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        long_id, viewix = self.feature_ids[idx][0], self.feature_ids[idx][1]
        room_idx = self.rm_labels[idx]

        if isinstance(viewix, tuple):
            feature = torch.cat([self.all_img_features[long_id][vix, :] for vix in viewix])
        elif isinstance(viewix, int):
            feature = self.all_img_features[long_id][viewix, :]
        else:
            raise ValueError('viewIndex inputs are not parsable for dataset creation.')

        # sample = {'long_id': long_id, 'viewix': viewix, 'feature': torch.from_numpy(feature), 'room': torch.tensor(room_idx)}
        sample = {'long_id': long_id, 'viewix': viewix, 'feature': feature, 'room': room_idx}

        return sample
