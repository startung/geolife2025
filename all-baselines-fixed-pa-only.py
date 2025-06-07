import os
import torch
import tqdm
import rasterio
import numpy as np
import pandas as pd
import albumentations as A
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from time import time
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support


start_time = time()


# Hyperparameters
learning_rate = 0.0006
learning_rate_sen = 0.0004
num_epochs = 20
positive_weigh_factor = 1.0

top_n_species = 1000
num_classes = top_n_species
# num_classes = 11255  # Number of all unique classes within the PO and PA data.
batch_size = 128
seed = 113


# !kaggle competitions download -c geolifeclef-2025
# datazip = ZipFile('geolifeclef-2025.zip', 'r')
# datazip.extractall('data/')


# Check if cuda is available
num_workers = 60
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE =", device)


tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok=True)
submission_path = Path("submission")
submission_path.mkdir(exist_ok=True)

data_path = (
    Path("/kaggle/input/geolifeclef-2025")
    if os.environ["PWD"].startswith("/kaggle")
    else Path.cwd() / "baselines/data"
)


# Load PA Training data
pa_train_metadata_path = "train_metadata_updated.csv"
pa_train_bio_data_path = data_path / "BioclimTimeSeries/cubes/PA-train"
pa_train_lan_data_path = data_path / "SateliteTimeSeries-Landsat/cubes/PA-train"
pa_train_sen_data_path = data_path / "SatelitePatches/PA-train"

# Load PO Training data
po_train_metadata_path = "po_data_updated.csv"
po_train_bio_data_path = data_path / "BioclimTimeSeries/cubes/PO-train"
po_train_lan_data_path = data_path / "SateliteTimeSeries-Landsat/cubes/PO-train"
po_train_sen_data_path = data_path / "SatelitePatches/PO-train"

# Load Test metadata
test_metadata_path = "test_metadata_updated.csv"
test_bio_data_path = data_path / "BioclimTimeSeries/cubes/PA-test"
test_lan_data_path = data_path / "SateliteTimeSeries-Landsat/cubes/PA-test"
test_sen_data_path = data_path / "SatelitePatches/PA-test/"


def set_seed(seed):
    # Set seed for Python's built-in random number generator
    torch.manual_seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(seed)


# Prepare custom dataset loader
class TrainDatasetBio(Dataset):
    def __init__(self, data_dir, metadata, subset, po_pa, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        self.po_pa = po_pa
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata["speciesId"] = self.metadata["speciesId"].astype(int)
        self.label_dict = (
            self.metadata.groupby("surveyId")["speciesId"].apply(list).to_dict()
        )

        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        survey_id = self.metadata.surveyId[idx]
        # sub_dir = "" if self.po_pa == "PA" else "/".join([str(survey_id)[max(0,len(str(survey_id))-2-i):len(str(survey_id))-i] for i in range(0, len(str(survey_id)), 2)][0:2]) + '/'
        if self.po_pa == "PO":
            file_name = f"{"/".join([str(survey_id)[max(0,len(str(survey_id))-2-i):len(str(survey_id))-i] for i in range(0, len(str(survey_id)), 2)][0:2])}/GLC25-P0-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"
        else:
            file_name = (
                f"GLC25-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"
            )
        sample = torch.load(
            os.path.join(
                self.data_dir,
                file_name,
            ),
            weights_only=True,
        )
        species_ids = self.label_dict.get(
            survey_id, []
        )  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = (
                1  # Set the corresponding class index to 1 for each species
            )

        # Ensure the sample is in the correct format for the transform
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, label, survey_id


class TrainDatasetLan(Dataset):
    def __init__(self, data_dir, metadata, subset, po_pa, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        self.po_pa = po_pa
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata["speciesId"] = self.metadata["speciesId"].astype(int)
        self.label_dict = (
            self.metadata.groupby("surveyId")["speciesId"].apply(list).to_dict()
        )

        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        if self.po_pa == "PO":
            file_name = f"{"/".join([str(survey_id)[max(0,len(str(survey_id))-2-i):len(str(survey_id))-i] for i in range(0, len(str(survey_id)), 2)][0:2])}/GLC25-PO-{self.subset}-landsat_time_series_{survey_id}_cube.pt"
        else:
            file_name = (
                f"GLC25-PA-{self.subset}-landsat-time-series_{survey_id}_cube.pt"
            )
        sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.data_dir,
                    file_name,
                ),
                weights_only=True,
            )
        )

        species_ids = self.label_dict.get(
            survey_id, []
        )  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  # Initialize label tensor
        for species_id in species_ids:
            # label_id = self.species_mapping[species_id]  # Get consecutive integer label
            label_id = species_id
            label[label_id] = (
                1  # Set the corresponding class index to 1 for each species
            )

        # Ensure the sample is in the correct format for the transform
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()  # Convert tensor to numpy array
            # print(sample.shape)

        if self.transform:
            sample = self.transform(sample)

        return sample, label, survey_id


def construct_patch_path(data_path, survey_id):
    """Construct the patch file path based on plot_id as './CD/AB/XXXXABCD.jpeg'"""
    path = data_path
    for d in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, d)

    path = os.path.join(path, f"{survey_id}.tiff")

    return path


def quantile_normalize(band, low=2, high=98):
    sorted_band = np.sort(band.flatten())
    quantiles = np.percentile(sorted_band, np.linspace(low, high, len(sorted_band)))
    normalized_band = np.interp(band.flatten(), sorted_band, quantiles).reshape(
        band.shape
    )

    min_val, max_val = np.min(normalized_band), np.max(normalized_band)

    # Prevent division by zero if min_val == max_val
    if max_val == min_val:
        return np.zeros_like(
            normalized_band, dtype=np.float32
        )  # Return an array of zeros

    # Perform normalization (min-max scaling)
    return ((normalized_band - min_val) / (max_val - min_val)).astype(np.float32)


class TrainDatasetSen(Dataset):
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata["speciesId"] = self.metadata["speciesId"].astype(int)
        self.label_dict = (
            self.metadata.groupby("surveyId")["speciesId"].apply(list).to_dict()
        )

        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        species_ids = self.label_dict.get(
            survey_id, []
        )  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = (
                1  # Set the corresponding class index to 1 for each species
            )

        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array(
                [quantile_normalize(band) for band in image]
            )  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        image = self.transform(image)

        return image, label, survey_id


class TestDatasetBio(TrainDatasetBio):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        sample = torch.load(
            os.path.join(
                self.data_dir,
                f"GLC25-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt",
            ),
            weights_only=True,
        )

        if isinstance(sample, torch.Tensor):
            sample = sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, survey_id


class TestDatasetLan(TrainDatasetLan):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.data_dir,
                    f"GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt",
                ),
                weights_only=True,
            )
        )

        if isinstance(sample, torch.Tensor):
            sample = sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, survey_id


class TestDatasetSen(TrainDatasetSen):
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]

        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array(
                [quantile_normalize(band) for band in image]
            )  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.transform(image)
        return image, survey_id


class ModifiedResNet18Bio(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18Bio, self).__init__()

        self.norm_input = nn.LayerNorm([4, 19, 12])
        self.resnet18 = models.resnet18(weights=None)
        # We have to modify the first convolutional layer to accept 4 channels instead of 3
        self.resnet18.conv1 = nn.Conv2d(
            4, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, 2056)
        self.fc2 = nn.Linear(2056, num_classes)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ModifiedResNet18Lan(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18Lan, self).__init__()

        self.norm_input = nn.LayerNorm([6, 4, 21])
        self.resnet18 = models.resnet18(weights=None)
        # We have to modify the first convolutional layer to accept 4 channels instead of 3
        self.resnet18.conv1 = nn.Conv2d(
            6, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, 2056)
        self.fc2 = nn.Linear(2056, num_classes)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, num_epochs, device, name, lr):
    print(f"Training for {num_epochs} epochs started.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    for epoch in range(num_epochs):
        epoch_start_time = time()
        model.train()
        for batch_idx, (data, targets, _) in enumerate(train_loader):

            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            pos_weight = (
                targets * positive_weigh_factor
            )  # All positive weights are equal to 10
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if batch_idx % 2000 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
                )
        epoch_end_time = time()
        print(
            f"Epoch {epoch+1}/{num_epochs} completed in {(epoch_end_time - epoch_start_time)/60:.2f} minutes."
        )

        scheduler.step()
        print("Scheduler:", scheduler.state_dict())
        new_lr = scheduler.get_last_lr()

    # Save the trained model
    model.eval()
    torch.save(model.state_dict(), tmp_path / f"resnet18-with-{name}.pth")

    return model, new_lr


def test(model, test_loader, device):
    # Test Loop
    with torch.no_grad():
        all_predictions = []
        surveys = []
        top_k_indices = None
        for data, surveyID in tqdm.tqdm(test_loader, total=len(test_loader)):

            data = data.to(device)

            outputs = model(data)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)

            # Select top-25 values as predictions
            top_25 = np.argsort(-predictions, axis=1)[:, :25]
            if top_k_indices is None:
                top_k_indices = top_25
            else:
                top_k_indices = np.concatenate((top_k_indices, top_25), axis=0)

            surveys.extend(surveyID.cpu().numpy())

    return top_k_indices, surveys, all_predictions


def save_submission(
    top_k_indices, surveys, all_predictions, submission_path, tmp_path, name, epoch=0
):
    # Save prediction file
    # data_concatenated = [" ".join(map(str, row)) for row in top_k_indices]

    # top_speciesIds_dict_inverted = {v: k for k, v in top_speciesIds_dict.items()}

    top_25 = np.argsort(-np.array(all_predictions), axis=1)[:, :25]

    top_speciesIds_dict_inverted = {v: k for k, v in top_speciesIds_dict.items()}

    # map vaules in row to species_id using dict top_speciesIds_dict
    data_concatenated = [
        " ".join([str(int(top_speciesIds_dict_inverted[i])) for i in line])
        for line in top_25
    ]

    pd.DataFrame(
        {
            "surveyId": surveys,
            "predictions": data_concatenated,
        }
    ).to_csv(
        submission_path
        / f"submission-single-modality-baseline-with-{name}-e{epoch}-data-pa-only.csv",
        index=False,
    )

    all_predictions_np = np.asarray(all_predictions)
    np.save(
        tmp_path
        / f"all-predictions-single-modality-baseline-with-{name}-e{epoch}-data-pa-only.npy",
        all_predictions_np,
    )

    np.save(
        tmp_path
        / f"surveys-single-modality-baseline-with-{name}-e{epoch}-data-pa-only.npy",
        surveys,
    )


# Load metadata and prepare data loaders

# Transforms
transform = transforms.Compose([transforms.ToTensor()])

transform_sen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)
        ),
    ]
)

# Load Training metadata
pa_train_metadata = pd.read_csv(pa_train_metadata_path)

# filter to top N speciesIds in train_metadata
top_speciesIds = (
    pa_train_metadata.speciesId.value_counts().head(top_n_species).index.tolist()
)
pa_train_metadata = pa_train_metadata[pa_train_metadata.speciesId.isin(top_speciesIds)]
print(
    f"After filtering to top {top_n_species} species PA training has shape: {pa_train_metadata.shape}"
)

top_speciesIds = np.random.permutation(top_speciesIds).tolist()
top_speciesIds_dict = {species_ids: i for i, species_ids in enumerate(top_speciesIds)}
pa_train_metadata["speciesId"] = pa_train_metadata.speciesId.map(top_speciesIds_dict)

# Load PO Train data
po_train_metadata = pd.read_csv(po_train_metadata_path)
po_train_metadata = po_train_metadata[po_train_metadata.speciesId.isin(top_speciesIds)]
print(
    f"After filtering to top {top_n_species} species PO training has shape: {po_train_metadata.shape}"
)
po_train_metadata["speciesId"] = po_train_metadata.speciesId.map(top_speciesIds_dict)

# bio_cubes_exist = [int(p.name.split("_")[2]) for p in po_train_bio_data_path.rglob("*") if p.is_file() and p.name != ".DS_Store"]
# po_train_metadata = po_train_metadata[po_train_metadata.surveyId.isin(bio_cubes_exist)]
# print(
#     f"After filtering to {len(bio_cubes_exist)} valid cubes PO training has shape: {po_train_metadata.shape}"
# )

po_train_bio_dataset = TrainDatasetBio(
    po_train_bio_data_path,
    po_train_metadata,
    subset="train",
    po_pa="PO",
    transform=transform,
)
po_train_bio_loader = DataLoader(
    po_train_bio_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
po_train_lan_dataset = TrainDatasetLan(
    po_train_lan_data_path,
    po_train_metadata,
    subset="train",
    po_pa="PO",
    transform=transform,
)
po_train_lan_loader = DataLoader(
    po_train_lan_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
po_train_sen_dataset = TrainDatasetSen(
    po_train_sen_data_path, po_train_metadata, transform=transform_sen
)
po_train_sen_loader = DataLoader(
    po_train_sen_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


# Load Train data
pa_train_bio_dataset = TrainDatasetBio(
    pa_train_bio_data_path,
    pa_train_metadata,
    subset="train",
    po_pa="PA",
    transform=transform,
)
pa_train_bio_loader = DataLoader(
    pa_train_bio_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
pa_train_lan_dataset = TrainDatasetLan(
    pa_train_lan_data_path,
    pa_train_metadata,
    subset="train",
    po_pa="PA",
    transform=transform,
)
pa_train_lan_loader = DataLoader(
    pa_train_lan_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
pa_train_sen_dataset = TrainDatasetSen(
    pa_train_sen_data_path, pa_train_metadata, transform=transform_sen
)
pa_train_sen_loader = DataLoader(
    pa_train_sen_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

# Load Test metadata
test_metadata = pd.read_csv(test_metadata_path)

# Load Test data
test_bio_dataset = TestDatasetBio(
    test_bio_data_path, test_metadata, subset="test", transform=transform
)
test_bio_loader = DataLoader(
    test_bio_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)
test_lan_dataset = TestDatasetLan(
    test_lan_data_path, test_metadata, subset="test", transform=transform
)
test_lan_loader = DataLoader(
    test_lan_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_sen_dataset = TestDatasetSen(
    test_sen_data_path, test_metadata, transform=transform_sen
)
test_sen_loader = DataLoader(
    test_sen_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

print("PO Train data", len(po_train_bio_dataset))
print("PA Train data", len(pa_train_bio_dataset))
print("Test data", len(test_bio_dataset))


print("PO Train data", len(po_train_bio_dataset))


model_bio = ModifiedResNet18Bio(num_classes).to(device)
model_lan = ModifiedResNet18Lan(num_classes).to(device)
model_sen = models.resnet18(weights="IMAGENET1K_V1")
model_sen.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2))
model_sen.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model_sen.to(device)

# lr_bio = [learning_rate]
# lr_lan = [learning_rate]
# lr_sen = [learning_rate_sen]


# # PO
# for epoch in range(num_epochs):
#     print(f"****** PO: Epoch {epoch+1} of {num_epochs} ******")
#     model_bio, lr_bio = train(
#         model_bio,
#         po_train_bio_loader,
#         num_epochs=1,
#         name="bio",
#         device=device,
#         lr=lr_bio[0],
#     )

#     top_k_indices_bio, surveys_bio, all_predictions_bio = test(
#         model_bio, test_bio_loader, device=device
#     )
#     save_submission(
#         top_k_indices_bio,
#         surveys_bio,
#         all_predictions_bio,
#         submission_path=submission_path,
#         tmp_path=tmp_path,
#         name="bio-po",
#         epoch=epoch,
#     )

#     model_lan, lr_lan = train(
#         model_lan,
#         po_train_lan_loader,
#         num_epochs=1,
#         name="lan",
#         device=device,
#         lr=lr_lan[0],
#     )

#     top_k_indices_lan, surveys_lan, all_predictions_lan = test(
#         model_lan, test_lan_loader, device=device
#     )
#     save_submission(
#         top_k_indices_lan,
#         surveys_lan,
#         all_predictions_lan,
#         submission_path=submission_path,
#         tmp_path=tmp_path,
#         name="lan-po",
#         epoch=epoch,
#     )

#     model_sen, lr_sen = train(
#         model_sen,
#         po_train_sen_loader,
#         num_epochs=1,
#         name="sen",
#         device=device,
#         lr=lr_sen[0],
#     )
#     top_k_indices_sen, surveys_sen, all_predictions_sen = test(
#         model_sen, test_sen_loader, device=device
#     )
#     save_submission(
#         top_k_indices_sen,
#         surveys_sen,
#         all_predictions_sen,
#         submission_path=submission_path,
#         tmp_path=tmp_path,
#         name="sen-po",
#         epoch=epoch,
#     )

lr_bio = [learning_rate]
lr_lan = [learning_rate]
lr_sen = [learning_rate_sen]


for epoch in range(num_epochs):
    print(f"****** PA: Epoch {epoch+1} of {num_epochs} ******")
    model_bio, lr_bio = train(
        model_bio,
        pa_train_bio_loader,
        num_epochs=1,
        device=device,
        name="bio",
        lr=lr_bio[0],
    )

    top_k_indices_bio, surveys_bio, all_predictions_bio = test(
        model_bio, test_bio_loader, device=device
    )
    save_submission(
        top_k_indices_bio,
        surveys_bio,
        all_predictions_bio,
        submission_path=submission_path,
        tmp_path=tmp_path,
        name="bio-pa",
        epoch=epoch + 10,
    )

    model_lan, lr_lan = train(
        model_lan,
        pa_train_lan_loader,
        num_epochs=1,
        name="lan",
        device=device,
        lr=lr_lan[0],
    )

    top_k_indices_lan, surveys_lan, all_predictions_lan = test(
        model_lan, test_lan_loader, device=device
    )
    save_submission(
        top_k_indices_lan,
        surveys_lan,
        all_predictions_lan,
        submission_path=submission_path,
        tmp_path=tmp_path,
        name="lan-pa",
        epoch=epoch + 10,
    )

    model_sen, lr_sen = train(
        model_sen,
        pa_train_sen_loader,
        num_epochs=1,
        name="sen",
        device=device,
        lr=lr_sen[0],
    )
    top_k_indices_sen, surveys_sen, all_predictions_sen = test(
        model_sen, test_sen_loader, device=device
    )
    save_submission(
        top_k_indices_sen,
        surveys_sen,
        all_predictions_sen,
        submission_path=submission_path,
        tmp_path=tmp_path,
        name="sen-pa",
        epoch=epoch + 10,
    )

# ensure all three datasets are in the same order
assert len(np.setdiff1d(surveys_bio, surveys_lan)) == 0
assert len(np.setdiff1d(surveys_lan, surveys_sen)) == 0
assert len(np.setdiff1d(surveys_sen, surveys_bio)) == 0

# combine and find the top 25 values
combined_array = (
    np.array(all_predictions_bio)
    + np.array(all_predictions_lan)
    + np.array(all_predictions_sen)
)
top_k_indices = None
top_25 = np.argsort(-combined_array, axis=1)[:, :25]

top_speciesIds_dict_inverted = {v: k for k, v in top_speciesIds_dict.items()}

# map vaules in row to species_id using dict top_speciesIds_dict
data_concatenated = [
    " ".join([str(int(top_speciesIds_dict_inverted[i])) for i in line])
    for line in top_25
]


# save the submission file
with open("submission-all-baselines-pa-only.csv", "w") as f:
    f.write("surveyId,predictions\n")
    for survey_id, predictions in zip(surveys_bio, data_concatenated):
        f.write(f"{survey_id},{predictions}\n")

print("Submission file saved successfully!")


end_time = time()
print(f"Execution time: {(end_time - start_time)/60:.2f} minutes.")
