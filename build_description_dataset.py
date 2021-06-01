# Build description dataset
from kirby.run_params import RunParams
from kirby.data_manager import DataManager
from kirby.dataset_builder import DatasetBuilder
from datasets import load_dataset

run_params = RunParams(debug=True)
data_manager = DataManager(run_params)
ds_builder = DatasetBuilder()
block_size = 128

split = 'train'
split = f'{split}[:{run_params.batch_size*block_size if run_params.debug else f"{run_params.data_set_percentage}%"}]'
ds = load_dataset('text', data_files=run_params.data_files, split=split)
aug_train_ds = ds_builder.build(ds, dataset_type='description')

# Save
aug_train_ds.to_csv('data/augmented_datasets/description_train.csv')

split = 'valid'
split = f'{split}[:{run_params.batch_size*block_size if run_params.debug else f"{run_params.data_set_percentage}%"}]'
ds = load_dataset('text', data_files=run_params.data_files, split=split)
aug_valid_ds = ds_builder.build(ds, dataset_type='description')

# Save
aug_valid_ds.to_csv('data/augmented_datasets/description_valid.csv')
