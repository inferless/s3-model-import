import boto3
import os
import json
import numpy as np
import torch
from transformers import pipeline


AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.environ.get('REGION_NAME')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
FOLDER_NAME = os.environ.get('FOLDER_NAME')


class InferlessPythonModel:
    def initialize(self):
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=REGION_NAME
        )

        objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_NAME)
        # Download everything from the folder
        for obj in objects['Contents']:
            key = obj['Key']
            file_name = os.path.join('model', key.replace(FOLDER_NAME + '/', ''))

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            s3.download_file(BUCKET_NAME, key, file_name)

        self.generator = pipeline(
            "text-generation",
            model="model",
            tokenizer="model",
            # device=0
        )

    def infer(self, inputs):
        pipeline_output = self.generator(inputs['prompt'], do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]
        return generated_txt

    def finalize(self):
        self.generator = None

