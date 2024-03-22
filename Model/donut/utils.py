import os.path
import torch


def get_special_tokens(list_of_classes):
    special_extra_tokens = []
    for entity_class in list_of_classes:
        fields = list(entity_class.__fields__.keys())
        for f in fields:
            special_extra_tokens+=[f"<s_{f}>", f"</s_{f}>"]
    special_extra_tokens+=["<sep/>"]
    return special_extra_tokens


def load_model_and_tokenizer(path_or_model_id: str, new_special_tokens: list[str]):
    from transformers import DonutProcessor
    from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

    task_start_token, eos_token = "<s>", "</s>"
    # Load processor
    processor = DonutProcessor.from_pretrained(path_or_model_id)

    # add new special tokens to tokenizer
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]}
    )

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # resizing the image to smaller sizes from [1920, 2560] to [960,1280]
    processor.feature_extractor.size = [720, 960]  # should be (width, height)
    processor.feature_extractor.do_align_long_axis = False

    # Load model from huggingface.co
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    # Adjust our image size and output sequence lengths
    model.config.encoder.image_size = processor.feature_extractor.size[::-1]  # (height, width)
    model.config.decoder.max_length = 100

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([task_start_token])[0]

    return model, processor




