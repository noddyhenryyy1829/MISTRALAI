import os
import re
import pathlib
import logging
import shutil
import label_studio_sdk
import dotenv

dotenv.load_dotenv(override=True)
model = None
tokenizer = None

import torch
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from transformers import T5Tokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)

class EntityExtractionModel(LabelStudioMLBase):
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    MODEL_DIR = os.getenv('MODEL_DIR', './results/entity-extraction')
    CURRENT_MODEL_VERSION = os.getenv('CURRENT_MODEL_VERSION', '1.0.0')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup(self):
        self.set("model_version", self.CURRENT_MODEL_VERSION)

    def getModelPath(self):
        return str(pathlib.Path(self.MODEL_DIR) / "flan_t5_entity_extractor")

    def loadModel(self):
        chk_path = self.getModelPath()
        logger.info(f"Loading model from {chk_path}")
        tokenizer = T5Tokenizer.from_pretrained(chk_path)
        model = T5ForConditionalGeneration.from_pretrained(chk_path).to(self.DEVICE)
        model.eval()
        return model, tokenizer

    def lazyInit(self):
        global model
        global tokenizer
        if model is None:
            model, tokenizer = self.loadModel()

    def prediction(self, texts: str, context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        self.lazyInit()

        if not isinstance(texts, str):
            request_id = texts[-1]["id"]
            text = texts[-1]["text"]
        else:
            request_id = ""  # default fallback
            text = texts

        prompt = f"Extract all entities from this text: {text.strip()}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        model_predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(model_predictions)

        result = [{
            "id": request_id,
            "from_name": "instruction",
            "to_name": "prompt",
            "type": "textarea",
            "value": {
                "text": [model_predictions]
            }
        }]

        predictions = [{
            "model_version": self.get("model_version"),
            "score": 0.99,
            "result": result
        }]

        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))
