from typing import List, Optional, Union

import gradio as gr
import torch
import transformers
from llavajp.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llavajp.conversation import conv_templates
from llavajp.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llavajp.model.llava_llama import LlavaLlamaForCausalLM
from llavajp.train.dataset import tokenizer_image_token
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class LlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is not None:
            # pixel_values = self.image_processor(images, return_tensors=return_tensors, size={"height": 768, "width": 768})["pixel_values"]
            images = images.convert("RGB")
            pixel_values = self.image_processor(images, return_tensors=return_tensors)[
                "pixel_values"
            ]
        else:
            pixel_values = None

        question = None
        answer = None

        user_name = "ユーザー: "
        assistant_name = "システム: "

        if user_name in text:
            question = text.split(user_name)[-1].strip()

            if assistant_name in text:
                answer = text.split(assistant_name)[-1].strip()
                question = question.replace(assistant_name + answer, "").strip()
        else:
            question = text

        inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        print({"prompt": prompt})

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

        input_ids = input_ids[:, :-1]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# argparseがHfArgumentParserと干渉するので、inputで受け取る
model_path = "hibikaze/finetune-llava-v1.5-japanese-gpt2-small_test-checkpoint-1200"

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = LlavaGpt2ForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    torch_dtype=torch_dtype,
    device_map=device,
    attn_implementation="eager",
)

processor = LlavaProcessor.from_pretrained(
    "hibikaze/finetune-llava-v1.5-japanese-gpt2-small_test-checkpoint-1200",
    cache_dir="cache",
)

LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}これは好奇心旺盛なユーザーと人工知能システムのチャットです。システムはユーザーの質問に親切、詳細、丁寧に答える。 ユーザー: {% else %}システム: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>\n{% endif %}{% endfor %}{% if message['role'] == 'user' %}{% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}システム:  {% endif %}"""
processor.chat_template = LLAVA_CHAT_TEMPLATE

model.eval()


@torch.inference_mode()
def inference_fn(
    image,
    prompt,
    max_len,
    temperature,
    repetition_penalty,
    top_p,
):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        0, torch.bfloat16
    )
    input_ids = inputs["input_ids"]
    image_tensor = inputs["pixel_values"]

    print("------- input --------")
    print(input_ids)
    print(image_tensor)
    print("------- input --------")
    print(inputs["attention_mask"])

    # generate
    output_ids = model.generate(
        inputs=input_ids,
        images=image_tensor,
        do_sample=temperature != 0.0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_len,
        repetition_penalty=repetition_penalty,
        use_cache=False,
    )

    output_ids = [
        token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX
    ]

    print(output_ids)
    output = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    print(output)

    target = "システム: "
    idx = output.find(target)
    output = output[idx + len(target) :]

    return output


with gr.Blocks() as demo:
    gr.Markdown("# LLaVA-JP Demo")

    with gr.Row():
        with gr.Column():
            # input_instruction = gr.TextArea(label="instruction", value=DEFAULT_INSTRUCTION)
            input_image = gr.Image(type="pil", label="image")
            prompt = gr.Textbox(label="prompt (optional)", value="")
            with gr.Accordion(label="Configs", open=False):
                max_len = gr.Slider(
                    minimum=10,
                    maximum=256,
                    value=50,
                    step=5,
                    interactive=True,
                    label="Max New Tokens",
                )

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top p",
                )

                repetition_penalty = gr.Slider(
                    minimum=-1,
                    maximum=3,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Repetition Penalty",
                )
            # button
            input_button = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Output")

    inputs = [input_image, prompt, max_len, temperature, repetition_penalty, top_p]
    input_button.click(inference_fn, inputs=inputs, outputs=[output])
    prompt.submit(inference_fn, inputs=inputs, outputs=[output])
    img2txt_examples = gr.Examples(
        examples=[
            [
                "./imgs/sample1.jpg",
                "猫の隣には何がありますか？",
                32,
                0.1,
                1.0,
                0.9,
            ],
        ],
        inputs=inputs,
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")
