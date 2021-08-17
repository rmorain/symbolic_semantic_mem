__all__ = ["RunParams"]

import multiprocessing

import torch


class RunParams:
    def __init__(
        self,
        model="gpt2",
        pretrained=False,
        data_dir="data/",
        db="db/wikidata.db",
        data_files={
            "train": ["data/wikitext_train.pkl"],
            "valid": ["data/wikitext_valid.pkl"],
        },
        data_file_type="text",
        max_epochs=1000,
        debug=True,
        batch_size=8,
        data_set_percentage=100,
        seq_length=128,
        knowledge_buffer=64,
        knowledge_tokenize=False,
        momentum=0.9,
        lr=1e-3,
        repo="wikitext-103-raw-v1",
        num_workers=multiprocessing.cpu_count(),
        num_gpus=torch.cuda.device_count(),
        kb_statements_file=None,
        run_name="test",
        project_name="kirby",
    ):

        self.model = model
        self.pretrained = pretrained
        self.data_dir = data_dir
        self.db = db
        self.data_files = data_files
        self.data_file_type = data_file_type
        self.max_epochs = max_epochs
        self.debug = debug
        self.batch_size = batch_size
        self.data_set_percentage = data_set_percentage
        self.seq_length = seq_length
        self.knowledge_buffer = knowledge_buffer
        self.knowledge_tokenize = knowledge_tokenize
        self.momentum = momentum
        self.lr = lr
        self.repo = repo
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.kb_statements_file = kb_statements_file
        self.run_name = run_name
        self.project_name = project_name
