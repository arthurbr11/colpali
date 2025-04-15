import torch
from PIL import Image

from colpali_engine.models import ColIdefics3, ColIdefics3Processor

print("loading base model")
model = ColIdefics3.from_pretrained(
    "arthurbresnu/ColSmolDocling-256M-preview-base",  # "vidore/colSmol-256M",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",  # or eager
).eval()
processor = ColIdefics3Processor.from_pretrained("arthurbresnu/ColSmolDocling-256M-preview-base")

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (32, 32), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)


# # Prerequisites:
# # pip install torch
# # pip install docling_core
# # pip install transformers

# import torch
# from docling_core.types.doc import DoclingDocument
# from docling_core.types.doc.document import DocTagsDocument
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from transformers.image_utils import load_image
# from pathlib import Path

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load images
# image = load_image("https://upload.wikimedia.org/wikipedia/commons/7/76/GazettedeFrance.jpg")

# # Initialize processor and model
# processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
# model = AutoModelForVision2Seq.from_pretrained(
#     "ds4sd/SmolDocling-256M-preview",
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
# ).to(DEVICE)

# # Create input messages
# messages = [
#     {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]},
# ]

# # Prepare inputs
# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(text=prompt, images=[image], return_tensors="pt")
# inputs = inputs.to(DEVICE)

# # Generate outputs
# generated_ids = model.generate(**inputs, max_new_tokens=8192)
# prompt_length = inputs.input_ids.shape[1]
# trimmed_generated_ids = generated_ids[:, prompt_length:]
# doctags = processor.batch_decode(
#     trimmed_generated_ids,
#     skip_special_tokens=False,
# )[0].lstrip()

# # Populate document
# doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
# print(doctags)
# # create a docling document
# doc = DoclingDocument(name="Document")
# doc.load_from_doctags(doctags_doc)

# # export as any format
# # HTML
# # Path("Out/").mkdir(parents=True, exist_ok=True)
# # output_path_html = Path("Out/") / "example.html"
# # doc.save_as_html(output_path_html)
# # MD
# print(doc.export_to_markdown())


# from datasets import load_dataset

# dataset = load_dataset("vidore/colpali_train_set", cache_dir="data_dir")
