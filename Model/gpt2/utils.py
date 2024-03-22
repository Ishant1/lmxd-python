import transformers as tr


def load_model_and_tokenizer(model_path: str):

    tokenizer = tr.AutoTokenizer.from_pretrained(
        model_path,
        additional_special_tokens=['<Document>', '</Document>', '<Task>', '</Task>', '<Extraction>', '</Extraction>']
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Loading model and resizing it for new tokens
    model = tr.AutoModelForCausalLM.from_pretrained(
        model_path
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4)

    return model, tokenizer
