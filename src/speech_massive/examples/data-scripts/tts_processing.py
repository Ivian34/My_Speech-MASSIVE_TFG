from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from datasets import Dataset, DatasetDict, Audio, Features, Sequence, Value
import re

# Cargar modelo y tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-cat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)
model = model.to(device)
model.eval()

rate = int(model.config.sampling_rate)
print("Tasa de muestreo:", rate)

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-cat")

dict_list = []
batch_size = 8  # Puedes ajustar el tamaño del batch
sentences_batch = []
intents_batch = []
slots_batch = []

with open("./phrases.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sentence, intent_str, slots = line.split("\t", 2)
        slots = slots.split("\t")
        sentences_batch.append(sentence)
        intents_batch.append(intent_str)
        slots_batch.append(slots)
        

        # Cuando alcanzamos el batch_size, procesamos el batch
        if len(sentences_batch) == batch_size:
            print("Procesando batch de", batch_size, "frases...")
            inputs = tokenizer(sentences_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs).waveform 
            
            for i, sentence in enumerate(sentences_batch):
                audio = outputs[i].cpu().numpy()
                #audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
                slots = slots_batch[i]                
                intent_str = intents_batch[i]

                audio_dict = {
                    "path": None,
                    "array": audio,
                    "sampling_rate": rate,
                }
                slots_dict = {
                    slot.split(":")[0] : slot.split(":")[1] for slot in slots
                }
                print("slots_dict1:", slots_dict)
                print("utt:", sentence)
                dict_list.append({
                    "audio" : audio_dict,
                    'path': None,
                    'utt': sentence,
                    'intent_str': intent_str,
                    'tokens': re.sub(r"[^\w\s]", " ", sentence, flags=re.UNICODE).split(),
                    'labels': ["player" if label == slots_dict["player"] else "other" for label in re.sub(r"[^\w\s]", " ", sentence, flags=re.UNICODE).split()],
                    "slots": slots_dict
                })
                """
                if slots_dict["player"] == "Àlex":
                    print(slots_dict)
                """
            sentences_batch = []
            intents_batch = []
            slots_batch = []

# Procesamos el resto de frases que no forman un batch completo
if sentences_batch:
    inputs = tokenizer(sentences_batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs).waveform
    for i, sentence in enumerate(sentences_batch):
        audio = outputs[i].cpu().numpy()
        #audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767) si guardamos en wav
        slots = slots_batch[i]                
        intent_str = intents_batch[i]
        audio_dict = {
            "path": None,
            "array": audio,
            "sampling_rate": rate,
        }
        slots_dict = {
            slot.split(":")[0] : slot.split(":")[1] for slot in slots 
        }
        dict_list.append({
            "audio" : audio_dict,
            'path': None,
            'utt': sentence,
            'intent_str': intent_str,
            'tokens': re.sub(r"[^\w\s]", " ", sentence, flags=re.UNICODE).split(),
            'labels': ["player" if label == slots_dict["player"] else "other" for label in re.sub(r"[^\w\s]", " ", sentence, flags=re.UNICODE).split()],
            "slots": slots_dict
        })
        """
        if slots_dict["player"] == "Àlex":
            print(slots_dict)
        """

print("Total de ejemplos generados:", len(dict_list))

columns = {k: [row[k] for row in dict_list] for k in dict_list[0]}

features = Features({
    "audio":  Audio(sampling_rate=16000),            # columna especial
    "path":   Value("string"),
    "utt":    Value("string"),
    "intent_str": Value("string"),
    "tokens": Sequence(Value("string")),
    "labels": Sequence(Value("string")),
    "slots":  {                                      # STRUCT → totes les claus fixed
        "player":  Value("string"),
        "location_x": Value("string"),
        "location_y": Value("string"),
        "object": Value("string"),
    }
})


dataset = Dataset.from_dict(columns, features=features)

# Crear un DatasetDict (por ejemplo, asignando todos los datos al split "train")
# Puedes dividir el dataset en train, validation y test si lo deseas
split = 0.9
train_size = int(len(dataset) * split)
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

print("dataset_dict:", dataset_dict)

# Guardar el DatasetDict en disco
dataset_dict.save_to_disk("../Audios/dataset1")
print("Dataset guardado localmente en '../Audios/dataset1'")