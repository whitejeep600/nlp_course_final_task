import torch
from torch.nn import Linear
from transformers import BertModel


class ClassificationBert(torch.nn.Module):
    def __init__(self, bert_model_name: str, n_out_classes: int, device: torch.device):
        super(ClassificationBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classification_layer = Linear(self.bert.config.hidden_size, n_out_classes)

        self.bert.to(device)
        self.classification_layer.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled = bert_output.pooler_output
        logits = self.classification_layer(pooled)
        return logits
