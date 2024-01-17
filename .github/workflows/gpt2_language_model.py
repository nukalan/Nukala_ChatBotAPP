# gpt2_language_model.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2LanguageModel:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_text(self, input_text, max_length=100, num_beams=5, temperature=0.8, do_sample=True):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = input_ids.ne(1)  # Create attention mask
        output = self.model.generate(input_ids,
                                     max_length=max_length,
                                     num_beams=num_beams,
                                     temperature=temperature,
                                     do_sample=do_sample,
                                     attention_mask=attention_mask,
                                     no_repeat_ngram_size=2,
                                     top_k=50,
                                     top_p=0.95)

        # Find the position of the EOS token and extract text up to that position
        eos_position = (output[0] == self.tokenizer.eos_token_id).nonzero()
        eos_position = eos_position[0].item() if eos_position.numel() > 0 else max_length
        generated_text = self.tokenizer.decode(output[0, :eos_position], skip_special_tokens=True)
        return generated_text
