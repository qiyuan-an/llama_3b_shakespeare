"""
https://youtu.be/CbmTFTsbyPI?si=FQ7vaDYvZcTqw-S9
"""
import os
import requests
import torch
import random
from transformers import (LlamaTokenizer, LlamaForCausalLM, TextDataset, Trainer, 
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, PeftModel

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def generate_response(prompt_text, model, tokenizer, max_length=30, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to('mps')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to('mps')
    print(f'model on: {next(model.parameters()).device}')
    print(f'input_ids on: {input_ids.device}')
    print(f'attention_mask on: {attention_mask.device}')
    output_seqs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
    )

    responses = []
    for response_id in output_seqs:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    model_path = f'openlm-research/open_llama_3b_v2'
    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    base_model = LlamaForCausalLM.from_pretrained(model_path)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = PeftModel(base_model, lora_config, adapter_name='Shakespeare')
    device = torch.device('mps')
    model.to(device)
    print(f'model on: {next(model.parameters()).device}')

    file_name = 'data/shakespeare.txt'
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    if not os.path.isfile(file_name):
        data = requests.get(url)
        with open(file_name, 'w') as f:
            f.write(data.text)
    
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=file_name, block_size=128)[:256]
    training_args = TrainingArguments(
        output_dir=f'output',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        eval_strategy='no',
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    prompt_text = f'Uneasy lies the head that wears a crown.'
    responses = generate_response(prompt_text, model, tokenizer)  # before training
    for res in responses:
        print(res)

    trainer.train()

    responses = generate_response(prompt_text, model, tokenizer)  # after training
    for res in responses:
        print(res)
    
    save_path = f'merged_finetuned_open_llama2_3b_shakespeare'
    tokenizer.save_pretrained(save_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)

if __name__ == '__main__':
    main()
