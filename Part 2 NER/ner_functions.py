# ner_functions.py

import numpy as np
import pandas as pd
import re
import glob
from os import path
import os
import json
from tqdm.notebook import tqdm
from dateutil.parser import parse
from dateutil.tz import gettz
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


if torch.backends.mps.is_available():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)


def combineHeadlineText(row):
    if isinstance(row["Headline"], str):
        return row["Headline"] + ". " + row["Text"]
    else:
        return row["Text"]


def preprocess_dataframe(df, use_parse=False):
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    df = df.drop_duplicates(['Date', 'Headline'], keep='last')
    df['Text'] = df['Text'].astype(str)
    df['Text'] = df.apply(lambda row: combineHeadlineText(row), axis=1)

    if use_parse:
        df['Date'] = df['Date'].str.replace(r'Published: ', ' ')
        df['Date'] = df['Date'].str.replace(r'First', ' ')
        df['Date'] = df['Date'].apply(lambda date_str: parse(date_str, tzinfos={'ET': gettz('America/New_York')}))
        df['Date'] = df['Date'].dt.date
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.reset_index(drop=True).sort_values(by=['Date'], ascending=True)

    return df


def process_entities(ner_results):
    entities = [{'word': d['word'], 'entity': d['entity'], 'score': d['score']} for d in ner_results]
    processed_entities = []
    current_entity = []
    for entity in entities:
        if entity['entity'].startswith('B-') or (entity['entity'].startswith('I-') and not current_entity):
            if current_entity:
                processed_entities.append(current_entity)
            current_entity = [entity]
        elif entity['entity'].startswith('I-') and current_entity:
            current_entity.append(entity)
    if current_entity:
        processed_entities.append(current_entity)

    return processed_entities


def json_serializable(item):
    if isinstance(item, np.float32):
        return float(item)
    raise TypeError(f"Type {type(item)} not serializable")


def perform_ner_on_dataframe(df, country_name, country_aliases, threshold):
    count = []
    check = []
    ner_results_data_list = []

    for i in tqdm(range(len(df)), desc=f"Processing {country_name}"):
        ner_results = nlp(df["Text"].iloc[i])
        processed_entities = process_entities(ner_results)

        country_instances = []
        country_check = []
        aliases = country_aliases.get(country_name, [country_name])
        for entity_group in processed_entities:
            words = [entity['word'] for entity in entity_group]
            entity_name = ' '.join(words)
            entity_type = entity_group[0]['entity']
            entity_score = sum(entity['score'] for entity in entity_group) / len(entity_group)
            if entity_type in ["B-LOC", "B-ORG"] and entity_score > 0.98:
                country_check.append(entity_name)
                if any(alias in entity_name for alias in aliases):
                    country_instances.append(entity_name)

        count.append(len(country_instances))
        check.append(country_check)

        ner_result = {
            'Date': df['Date'].iloc[i],
            'Headline': df['Headline'].iloc[i],
            'Count': count[i],
            'NER': json.dumps(ner_results, default=json_serializable)
        }
        ner_results_data_list.append(ner_result)

    df['Count'] = count
    df = df[df['Count'] > threshold]
    df = df.drop(['Count'], axis=1)
    df_ner_results = pd.DataFrame(ner_results_data_list)

    return df, df_ner_results
