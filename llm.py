from transformers import AutoTokenizer, pipeline as Pipeline
import torch


def chat(question):
    model = 'tiiuae/falcon-7b-instruct'

    tokenizer = AutoTokenizer.from_pretrained(
        model, device_map='auto', offload_folder='offload', offload_state_dict = True,
    )

    pipeline = Pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
        question,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences
