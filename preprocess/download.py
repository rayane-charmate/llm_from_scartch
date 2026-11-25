import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

class DatasetDownloader:

    def download_dataset(self):

        return snapshot_download(
            "HuggingFaceFW/fineweb",
            repo_type="dataset",
            cache_dir=os.path.join(os.environ["HF_HOME"], "fineweb/"),
            allow_patterns="sample/10BT/*")
