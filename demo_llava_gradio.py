"""
python demo_llava_gradio.py

Enter model path: your-username/your-repo-id
"""

import gradio as gr
import torch
import transformers
from llavajp.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llavajp.conversation import conv_templates
from llavajp.model.llava_llama import LlavaLlamaForCausalLM
from llavajp.train.dataset import tokenizer_image_token

# argparseがHfArgumentParserと干渉するので、inputで受け取る
model_path = input("Enter model path: ")

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    torch_dtype=torch_dtype,
    device_map=device,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=8192,
    padding_side="right",
    use_fast=False,
)
model.eval()
conv_mode = "v1"


@torch.inference_mode()
def inference_fn(
    image,
    prompt,
    max_len,
    temperature,
    repetition_penalty,
    top_p,
    no_repeat_ngram_size
):
    # prepare inputs
    # image pre-process
    image_size = model.get_model().vision_tower.image_processor.size["height"]
    if model.get_model().vision_tower.scales is not None:
        image_size = model.get_model().vision_tower.image_processor.size[
            "height"
        ] * len(model.get_model().vision_tower.scales)

    if device == "cuda":
        image_tensor = (
            model.get_model()
            .vision_tower.image_processor(
                image,
                return_tensors="pt",
                size={"height": image_size, "width": image_size},
            )["pixel_values"]
            .half()
            .cuda()
            .to(torch_dtype)
        )
    else:
        image_tensor = (
            model.get_model()
            .vision_tower.image_processor(
                image,
                return_tensors="pt",
                size={"height": image_size, "width": image_size},
            )["pixel_values"]
            .to(torch_dtype)
        )

    # create prompt
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    if device == "cuda":
        input_ids = input_ids.to(device)

    input_ids = input_ids[:, :-1]  # </sep>がinputの最後に入るので削除する

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
        no_repeat_ngram_size=no_repeat_ngram_size
    )

    output_ids = [
        token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX
    ]

    print(output_ids)
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

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
                    value=200,
                    step=5,
                    interactive=True,
                    label="Max New Tokens",
                )

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=1.0,
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

                no_repeat_ngram_size = gr.Slider(
                    minimum=0,
                    maximum=4,
                    value=0,
                    step=1,
                    interactive=True,
                    label="No Repeat Ngram Size",
                )
            # button
            input_button = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Output")

    inputs = [input_image, prompt, max_len, temperature, repetition_penalty, top_p, no_repeat_ngram_size]
    input_button.click(inference_fn, inputs=inputs, outputs=[output])
    prompt.submit(inference_fn, inputs=inputs, outputs=[output])
    img2txt_examples = gr.Examples(
        examples=[
            [
                "./imgs/sample1.jpg",
                "猫の隣には何がありますか？",
                200,
                0.0,
                1.0,
                1.0,
                3,
            ],
        ],
        inputs=inputs,
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")
