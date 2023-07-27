import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig


class LLama:
    """
    LLama class represents an AI language model called LLama, based on the LlamaForCausalLM model from the Transformers library.

    Attributes:
        basemodel (str): The name or path of the base model to use for LLama.
        lora_weights (str): The name or path of the LORA (Lossy Audio Compression) model weights to use for LLama.

    Methods:
        response(query, input): Generates a text response based on the input query.
        process_response(text): Processes the generated response and performs any additional post-processing.
    """
    def __init__(self, base_model, lora_weights):
        """
        Initializes the LLama instance with the given base model and LORA model weights.

        Args:
            base_model (str): The name or path of the base model to use for LLama.
            lora_weights (str): The name or path of the LORA model weights to use for LLama.
        """
        self.config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.lora_model = lora_weights
        self.basemodel = base_model
        self.model = LlamaForCausalLM.from_pretrained(self.basemodel,
                                                      load_in_4bit=True,
                                                      torch_dtype=torch.float16,
                                                      quantization_config=self.config,
                                                      device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.basemodel)
        self.model = PeftModel.from_pretrained(self.model, self.lora_model)

        self.generation_config = GenerationConfig(
            temperature=0.01,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=5.0,
            encoder_repetition_penalty=5.0,
            top_k=40,
            renormalize_logits=True,
            do_sample=True,
            num_beams=2,
            num_return_sequences=1,
            remove_invalid_values=True
        )

    def response(self, query, input):
        """
        Generates a text response based on the given query and input.

        Args:
            query (str): The input query or prompt for the model.
            input (str): Additional input data required for generating the response.

        Returns:
            str: The generated text response.
        """
        inputs = self.tokenizer(
            PROMPT,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].cuda()

        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=256,
        )

        response = generation_output.sequences[0]
        return self.tokenizer.decode(response)

    def process_response(self, text):
        """
        Process the text to give response only.

        Args:
            text (str): The input text containing the response.

        Returns:
            str: The extracted text.
        """
        start = text.find("Response:") + len("Response:")
        end = text.find("Chat Doctor.")
        return text[start:end].strip()




