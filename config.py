from pathlib import Path
import transformers
from torch.cuda.amp import GradScaler


class TrainConfig:
    data_dir = Path("../input/AI4Code")
    valid_ratio = 0.1
    max_len = 120
    num_epochs = 2
    batch_size = 32
    lr = 3e-4
    wandb = True


class BertLargeConfig:
    model_name = "bert-large-uncased"
    tokenizer = transformers.BertTokenizer.from_pretrained(
        model_name, do_lower_case=True
    )
    scaler = GradScaler()
    T_0 = 20
    min_eta = 1e-4


class DistillBertConfig:
    model_path = "../input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased"
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        model_path, do_lower_case=True
    )
    model = transformers.DistilBertModel.from_pretrained(model_path)
